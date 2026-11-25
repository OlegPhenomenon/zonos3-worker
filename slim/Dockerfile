# Slim cold-start image for Vast (and RunPod, --build-arg RUNPOD=1): prebuilt CUDA binaries (release asset) + PTQ
# backbone + keep2 decoder + speaker encoder + steering bundle (emotion/style controls, 0.2 MB) + one-shot voice
# clips. No compiler, no PyTorch. Every new Vast host pulls the whole image before its first request, so each stage
# below ships only what the worker runs:
#   * CUDA libs: cuBLAS/cuBLASLt with GPU code for sm_80/86/89 only (workers are Ampere/Ada), a stub for NCCL (one GPU)
#   * ffmpeg: a 2 MB build with just the audio decoders the server needs (zonos3-server shells out to it for clips),
#     checked bit for bit against Ubuntu's ffmpeg on every voice clip before it is accepted
#   * Python: only what the PyWorker imports, no pip, no vastai CLI/SDK dependency tree
# Layers go from rarely to often changed, so a host that pulled an older tag re-downloads only the top layers.
ARG UBUNTU=ubuntu:22.04

# ---------------------------------------------------------------- CUDA libs (build only)
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS cuda
FROM ${UBUNTU} AS cudalibs
RUN apt-get update && apt-get install -y --no-install-recommends python3 binutils gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
# Named sonames are symlinks; COPY of a named symlink copies the target file under the soname.
COPY --from=cuda /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12 /out/
COPY prune_fatbin.py nccl_stub.c /src/
RUN python3 /src/prune_fatbin.py /out/libcublas.so.12 /out/libcublasLt.so.12 \
 && gcc -shared -fPIC -O2 -Wl,-soname,libnccl.so.2 -o /out/libnccl.so.2 /src/nccl_stub.c

# ---------------------------------------------------------------- zonos3-server (build only)
FROM ${UBUNTU} AS server
RUN apt-get update && apt-get install -y --no-install-recommends python3 binutils \
 && rm -rf /var/lib/apt/lists/*
COPY zonos3-ggml-cuda-sm86-89.tgz /tmp/bin.tgz
COPY prune_fatbin.py /src/
RUN mkdir -p /app && tar xzf /tmp/bin.tgz -C /app && rm /tmp/bin.tgz /app/build/zonos3-cli \
 && python3 /src/prune_fatbin.py --drop-ptx /app/build/ggml/src/ggml-cuda/libggml-cuda.so.0.27.0
# The server's own text normalizer as a shared library (release asset), for common.keep_latin_words.
COPY libzyphra_textnorm.so.gz /tmp/
RUN mkdir -p /app/lib && gunzip -c /tmp/libzyphra_textnorm.so.gz > /app/lib/libzyphra_textnorm.so && rm /tmp/libzyphra_textnorm.so.gz

# ---------------------------------------------------------------- models (build only)
FROM ${UBUNTU} AS models
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
ARG HF=https://huggingface.co/Zyphra/ZONOS3-Mini/resolve/main/gguf
RUN mkdir -p /app/models && cd /app/models \
 && for f in backbone-ptq.gguf zac-keep2.gguf spk-encoder.gguf steer.gguf; do curl -fsSL -o "$f" "$HF/$f"; done

# ---------------------------------------------------------------- ffmpeg (build only)
FROM ${UBUNTU} AS ffbuild
RUN apt-get update && apt-get install -y --no-install-recommends build-essential nasm curl xz-utils ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Same version as Ubuntu 22.04's ffmpeg (4.4.2) so the native Opus decoder and the resampler produce the same samples.
RUN curl -fsSL -o /tmp/ff.tar.xz https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.xz \
 && echo 'af419a7f88adbc56c758ab19b4c708afbcae15ef09606b82b855291f6a6faa93  /tmp/ff.tar.xz' | sha256sum -c - \
 && tar xf /tmp/ff.tar.xz -C /tmp && cd /tmp/ffmpeg-4.4.2 \
 && ./configure --prefix=/opt/ff --disable-everything --disable-autodetect --disable-doc --disable-debug \
      --disable-ffplay --disable-ffprobe --disable-avdevice --disable-swscale --disable-network --disable-postproc \
      --enable-static --disable-shared --enable-protocol=file,pipe \
      --enable-demuxer=ogg,wav,flac,mp3,mov,matroska,aac,aiff,w64,caf \
      --enable-decoder=opus,vorbis,flac,mp3,mp3float,aac,aac_fixed,alac,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_u8,pcm_f32le,pcm_f32be,pcm_f64le,pcm_f64be,pcm_alaw,pcm_mulaw \
      --enable-parser=opus,vorbis,flac,mpegaudio,aac,aac_latm \
      --enable-muxer=pcm_f32le --enable-encoder=pcm_f32le \
      --enable-filter=aresample,aformat,anull,abuffer,abuffersink,atrim,atempo \
      --enable-swresample --enable-avfilter \
 && make -j"$(nproc)" ffmpeg && strip ffmpeg && install -m 755 ffmpeg /opt/ffmpeg-min

# Gate: the server asks ffmpeg for `-ac 1 -ar 44100 -f f32le` of a clip. The minimal build must return the very same
# samples as Ubuntu's full ffmpeg for every voice clip and for the other containers a caller may send as reference.
FROM ${UBUNTU} AS ffcheck
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ffbuild /opt/ffmpeg-min /opt/ffmpeg-min
COPY voices/ /voices/
RUN set -e; mkdir /other; \
    s=/voices/en_us/Schedar.ogg; \
    ffmpeg -nostdin -v error -i $s -ar 24000 /other/a.wav; ffmpeg -nostdin -v error -i $s -ac 2 -ar 48000 /other/b.flac; \
    ffmpeg -nostdin -v error -i $s -ar 22050 /other/c.mp3; ffmpeg -nostdin -v error -i $s -ar 44100 -c:a pcm_s24le /other/d.wav; \
    ffmpeg -nostdin -v error -i $s -ar 44100 /other/e.m4a; ffmpeg -nostdin -v error -i $s -ar 16000 -c:a pcm_u8 /other/f.wav; \
    n=0; for f in $(find /voices /other -type f \( -name '*.ogg' -o -name '*.wav' -o -name '*.flac' -o -name '*.mp3' -o -name '*.m4a' \) | sort); do \
      ffmpeg -nostdin -v error -y -i $f -ac 1 -ar 44100 -t 30 -f f32le /tmp/a.raw; [ -s /tmp/a.raw ]; \
      /opt/ffmpeg-min -nostdin -v error -y -i $f -ac 1 -ar 44100 -t 30 -f f32le /tmp/b.raw; \
      cmp /tmp/a.raw /tmp/b.raw || { echo "ffmpeg mismatch: $f"; exit 1; }; n=$((n+1)); done; \
    echo "ffmpeg identical on $n files"; cp /opt/ffmpeg-min /ffmpeg-ok

# ---------------------------------------------------------------- Python dependencies (build only)
FROM ${UBUNTU} AS pydeps
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
# The PyWorker (vastai.serverless.server) needs aiohttp, pycryptodome, anyio and psutil; our code needs requests and
# cryptography. The `vastai` wheel also drags in its CLI/SDK (borb, fonttools, pillow, lxml, ...), imported by the
# package __init__ but unused by the worker, so install it without dependencies and cut those imports.
ARG RUNPOD=0
RUN pip3 install --no-cache-dir -U pip && pip3 install --no-cache-dir --target /opt/py --no-deps vastai==1.8.3 \
 && pip3 install --no-cache-dir --target /opt/py aiohttp requests cryptography pycryptodome anyio psutil \
 && if [ "$RUNPOD" = 1 ]; then pip3 install --no-cache-dir --target /opt/py runpod; fi \
 && python3 - <<'E'
p = '/opt/py/vastai/__init__.py'
s = open(p).read()
open(p, 'w').write(s[:s.index('from .sdk import VastAI')])   # keeps the logger setup, drops the SDK/CLI imports
open('/opt/py/vastai/serverless/__init__.py', 'w').write('')
E
RUN PYTHONPATH=/opt/py python3 -c "from vastai.serverless.server.worker import BenchmarkConfig, HandlerConfig, Worker, WorkerConfig; import cryptography, requests"

# ---------------------------------------------------------------- final image
FROM ${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
# The NVIDIA container runtime mounts the driver (libcuda, nvidia-smi) only when asked to.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Binaries run on older 12.x drivers through CUDA minor-version compatibility; don't let the image refuse them.
ENV NVIDIA_DISABLE_REQUIRE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-distutils libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=pydeps /opt/py /opt/py
ENV PYTHONPATH=/opt/py
COPY --from=cudalibs /out/ /usr/local/lib/
RUN ldconfig
COPY --from=server /app /app
COPY --from=models /app/models /app/models
ENV LD_LIBRARY_PATH=/app/build/ggml/src:/app/build/ggml/src/ggml-cuda:/app/build/ggml/src/ggml-cpu:/app/build/bin
# Build gate: every server library resolves inside the image (libcuda.so.1 comes from the host driver at run time).
RUN missing=$(for f in /app/build/zonos3-server $(find /app/build -name '*.so*'); do ldd "$f"; done | grep 'not found' | grep -v libcuda.so.1 | sort -u); \
    [ -z "$missing" ] || { echo "$missing"; exit 1; }
COPY --from=ffcheck /ffmpeg-ok /usr/local/bin/ffmpeg
COPY voices/ /app/voices/
RUN ffmpeg -nostdin -v error -i /app/voices/ru/Charon.ogg -ac 1 -ar 44100 -f f32le - >/dev/null
COPY common.py prosody.py handler.py vast_worker.py /app/
# `vast` runs the Vast PyWorker; anything else (RunPod) the RunPod handler.
ENTRYPOINT ["/bin/sh","-c","cd /app && if [ \"$0\" = vast ]; then exec python3 -u vast_worker.py; else exec python3 -u handler.py; fi"]
CMD ["runpod"]
