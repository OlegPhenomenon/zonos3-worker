# Slim cold-start image for RunPod and Vast: prebuilt CUDA binaries (release asset) + PTQ backbone + keep2 decoder
# + speaker encoder + one-shot voice clips. No compiler, no PyTorch.
# Plain Ubuntu plus only the CUDA libraries the server links (cudart, cublas, cublasLt, nccl, ~1.2 GB) instead of the
# full CUDA runtime image (3.1 GB): every new Vast host pulls the whole image before its first request.
# Layers go from rarely to often changed, so a host that pulled an older tag re-downloads only the top layers.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS cuda

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
# The NVIDIA container runtime mounts the driver (libcuda, nvidia-smi) only when asked to.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Binaries run on older 12.x drivers through CUDA minor-version compatibility; don't let the image refuse them.
ENV NVIDIA_DISABLE_REQUIRE=1
# ffmpeg: zonos3-server decodes the Opus voice clips with it when a speaker is registered.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir -U pip && pip3 install --no-cache-dir runpod requests vastai==1.8.3
# Named sonames are symlinks; COPY of a named symlink copies the target file under the soname.
COPY --from=cuda /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/lib/
RUN ldconfig
COPY zonos3-ggml-cuda-sm86-89.tgz /tmp/bin.tgz
RUN mkdir -p /app && tar xzf /tmp/bin.tgz -C /app && rm /tmp/bin.tgz
ARG HF=https://huggingface.co/Zyphra/ZONOS3-Mini/resolve/main/gguf
RUN mkdir -p /app/models && cd /app/models \
 && for f in backbone-ptq.gguf zac-keep2.gguf spk-encoder.gguf; do curl -fsSL -o "$f" "$HF/$f"; done
ENV LD_LIBRARY_PATH=/app/build/ggml/src:/app/build/ggml/src/ggml-cuda:/app/build/ggml/src/ggml-cpu:/app/build/bin
# Build gate: every server library resolves inside the image (libcuda.so.1 comes from the host driver at run time).
RUN missing=$(for f in /app/build/zonos3-server $(find /app/build -name '*.so*'); do ldd "$f"; done | grep 'not found' | grep -v libcuda.so.1 | sort -u); \
    [ -z "$missing" ] || { echo "$missing"; exit 1; }
# The server's own text normalizer as a shared library (release asset), for common.keep_latin_words.
COPY libzyphra_textnorm.so.gz /tmp/
RUN mkdir -p /app/lib && gunzip -c /tmp/libzyphra_textnorm.so.gz > /app/lib/libzyphra_textnorm.so && rm /tmp/libzyphra_textnorm.so.gz
COPY voices/ /app/voices/
RUN ffmpeg -v error -i /app/voices/ru/Charon.ogg -f null -
COPY common.py handler.py vast_worker.py /app/
# `vast` runs the Vast PyWorker; anything else (RunPod) the RunPod handler.
ENTRYPOINT ["/bin/sh","-c","cd /app && if [ \"$0\" = vast ]; then exec python3 -u vast_worker.py; else exec python3 -u handler.py; fi"]
CMD ["runpod"]
