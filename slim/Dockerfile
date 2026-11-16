# Slim cold-start image: prebuilt CUDA binaries (release asset) + PTQ backbone + keep2 decoder + speaker encoder.
# No compiler, no PyTorch, no raw-reference encoders (~1.2 GB weights).
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir runpod requests
COPY zonos3-ggml-cuda-sm86-89.tgz /tmp/bin.tgz
RUN mkdir -p /app && tar xzf /tmp/bin.tgz -C /app && rm /tmp/bin.tgz
ARG HF=https://huggingface.co/Zyphra/ZONOS3-Mini/resolve/main/gguf
RUN mkdir -p /app/models && cd /app/models \
 && for f in backbone-ptq.gguf zac-keep2.gguf spk-encoder.gguf; do curl -fsSL -o "$f" "$HF/$f"; done
ENV LD_LIBRARY_PATH=/app/build/ggml/src:/app/build/ggml/src/ggml-cuda:/app/build/ggml/src/ggml-cpu:/app/build/bin:$LD_LIBRARY_PATH
COPY handler.py /app/handler.py
CMD ["python3","-u","/app/handler.py"]
