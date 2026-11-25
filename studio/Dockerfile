# studio-prep worker (docs/architecture/studio-dub-realtime.md): Vast PyWorker + Bandit v2 dialogue separation on ONNX Runtime
# (no PyTorch) + the studio mix library. Plain Ubuntu plus only the CUDA/cuDNN libraries ONNX Runtime loads for this model,
# instead of the CUDA runtime image: every new Vast host pulls the whole image before its first request.
# Layers go from rarely to often changed, so a host that pulled an older tag re-downloads only the top layers.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS cuda

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# The NVIDIA container runtime mounts the driver (libcuda, nvidia-smi) only when asked to.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Binaries run on older 12.x drivers through CUDA minor-version compatibility; don't let the image refuse them.
ENV NVIDIA_DISABLE_REQUIRE=1
# ffmpeg: decodes the source audio, encodes the opus speech stem and the AAC windows (libopus, native aac).
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U pip && pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt \
 && rm -f /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime_providers_tensorrt.so
# The libraries libonnxruntime_providers_cuda.so loads for this model's GRU/MatMul/FFT-free graph (checked with /proc/self/maps).
# Named sonames are symlinks; COPY of a named symlink copies the target file under the soname.
COPY --from=cuda /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so.11 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcurand.so.10 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
     /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9 \
     /usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9 /usr/local/lib/
RUN ldconfig
# Build gate: the CUDA provider's libraries resolve (libcuda.so.1 comes from the host driver at run time).
RUN missing=$(ldd /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so | grep 'not found' | grep -v libcuda.so.1); \
    [ -z "$missing" ] || { echo "$missing"; exit 1; }
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|aac' | wc -l | grep -qx 2 || { echo 'ffmpeg lacks libopus or aac'; exit 1; }
# Bandit v2 speech branch (CC BY-SA 4.0, see NOTICE-BANDIT.txt); CI downloads it from a release asset into models/.
COPY NOTICE-BANDIT.txt /app/
COPY models/ /app/models/
ENV STUDIO_MODEL=/app/models/bandit_speech.onnx STUDIO_CACHE_DIR=/var/studio-cache STUDIO_CACHE_GB=40
COPY __init__.py sep.py vast_worker.py /app/studio/
COPY mix/ /app/studio/mix/
ENTRYPOINT ["/bin/sh","-c","cd /app && exec python3 -u -m studio.vast_worker"]
