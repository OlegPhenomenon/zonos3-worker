# Slim cold-start image for RunPod and Vast: prebuilt CUDA binaries (release asset) + PTQ backbone + keep2 decoder
# + speaker encoder + precomputed voice-bank embeddings. No compiler, no PyTorch (~1.2 GB weights).
# Layers go from rarely to often changed, so a host that pulled an older tag re-downloads only the top layers.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
# Binaries run on older 12.x drivers through CUDA minor-version compatibility; don't let the image refuse them.
ENV NVIDIA_DISABLE_REQUIRE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir -U pip && pip3 install --no-cache-dir runpod requests vastai==1.8.3
COPY zonos3-ggml-cuda-sm86-89.tgz /tmp/bin.tgz
RUN mkdir -p /app && tar xzf /tmp/bin.tgz -C /app && rm /tmp/bin.tgz
ARG HF=https://huggingface.co/Zyphra/ZONOS3-Mini/resolve/main/gguf
RUN mkdir -p /app/models && cd /app/models \
 && for f in backbone-ptq.gguf zac-keep2.gguf spk-encoder.gguf; do curl -fsSL -o "$f" "$HF/$f"; done
ENV LD_LIBRARY_PATH=/app/build/ggml/src:/app/build/ggml/src/ggml-cuda:/app/build/ggml/src/ggml-cpu:/app/build/bin:$LD_LIBRARY_PATH
# The server's own text normalizer as a shared library (release asset), for common.keep_latin_words.
COPY libzyphra_textnorm.so.gz /tmp/
RUN mkdir -p /app/lib && gunzip -c /tmp/libzyphra_textnorm.so.gz > /app/lib/libzyphra_textnorm.so && rm /tmp/libzyphra_textnorm.so.gz
COPY voices/ /app/voices/
COPY common.py handler.py vast_worker.py /app/
# `vast` runs the Vast PyWorker; anything else (RunPod) the RunPod handler.
ENTRYPOINT ["/bin/sh","-c","cd /app && if [ \"$0\" = vast ]; then exec python3 -u vast_worker.py; else exec python3 -u handler.py; fi"]
CMD ["runpod"]
