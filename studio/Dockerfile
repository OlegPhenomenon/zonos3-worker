# studio-prep worker (docs/architecture/studio-dub-realtime.md): Vast PyWorker + Bandit v2 dialogue separation and emotion2vec+
# large on ONNX Runtime (no PyTorch) + the lab's draft mix (lab/, copied from mobile-universal-dubber tool/studio_dub_lab by CI).
# Every new Vast host pulls the whole image before its first request, so each stage below
# ships only what the worker runs:
#   * CUDA/cuDNN: only the libraries ONNX Runtime's CUDA provider loads for this model (cudart, cuBLAS, cuBLASLt, cuFFT, cuRAND,
#     cuDNN core/ops/adv/graph/heuristic), with GPU code for sm_80/86/89 only (workers are Ampere: the Vast search is
#     compute_cap 860..869, scripts/studio_setup.sh); the provider library itself is pruned the same way
#   * Python: site-packages built in a throwaway stage, no pip in the image
# Layers go from rarely to often changed, so a host that pulled an older tag re-downloads only the top layers.
ARG UBUNTU=ubuntu:22.04

# ---------------------------------------------------------------- CUDA libs (build only)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS cuda
FROM ${UBUNTU} AS cudalibs
RUN apt-get update && apt-get install -y --no-install-recommends python3 binutils && rm -rf /var/lib/apt/lists/*
# Named sonames are symlinks; COPY of a named symlink copies the target file under the soname. The cuDNN engine libraries (and
# NVRTC for the runtime-compiled ones) serve the cuDNN frontend convolutions of emotion2vec's feature encoder: without them ONNX
# Runtime fails that Conv on CUDA and falls back to the CPU.
COPY --from=cuda /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so.11 \
     /usr/local/cuda/targets/x86_64-linux/lib/libcurand.so.10 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
     /usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9 \
     /usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9 \
     /usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12 \
     /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so.12.8 /out/
COPY prune_fatbin.py /src/
RUN python3 /src/prune_fatbin.py /out/libcublas.so.12 /out/libcublasLt.so.12 /out/libcufft.so.11 /out/libcurand.so.10 \
      /out/libcudnn_adv.so.9 /out/libcudnn_graph.so.9 /out/libcudnn_ops.so.9 /out/libcudnn_heuristic.so.9 \
      /out/libcudnn_engines_precompiled.so.9

# ---------------------------------------------------------------- Python dependencies (build only)
FROM ${UBUNTU} AS pydeps
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip binutils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
COPY prune_fatbin.py /src/
RUN pip3 install --no-cache-dir --target /py -r /tmp/requirements.txt \
 && rm -rf /py/bin /py/onnxruntime/capi/libonnxruntime_providers_tensorrt.so /py/onnxruntime/transformers /py/onnxruntime/tools \
 && find /py -name __pycache__ -prune -exec rm -rf {} + \
 && python3 /src/prune_fatbin.py /py/onnxruntime/capi/libonnxruntime_providers_cuda.so

# ---------------------------------------------------------------- image
FROM ${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/py
# The NVIDIA container runtime mounts the driver (libcuda, nvidia-smi) only when asked to.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Binaries run on older 12.x drivers through CUDA minor-version compatibility; don't let the image refuse them.
ENV NVIDIA_DISABLE_REQUIRE=1
# ffmpeg: decodes the source audio, encodes the opus speech stem, cuts the pieces for the mix.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-distutils ffmpeg libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|aac' | wc -l | grep -qx 2
COPY --from=cudalibs /out/ /usr/local/lib/
COPY --from=pydeps /py /py
RUN ldconfig
# Build gate: the CUDA provider's libraries resolve (libcuda.so.1 comes from the host driver at run time) and the worker's imports work.
RUN missing=$(ldd /py/onnxruntime/capi/libonnxruntime_providers_cuda.so | grep 'not found' | grep -v libcuda.so.1); \
    [ -z "$missing" ] || { echo "$missing"; exit 1; }; \
    python3 -c 'import onnxruntime, numpy, scipy, soundfile, nara_wpe, requests, cryptography, vastai.serverless.server.worker'
# Bandit v2 speech branch (CC BY-SA 4.0, see NOTICE-BANDIT.txt) and emotion2vec+ large; CI downloads them from release assets
# into models/.
COPY NOTICE-BANDIT.txt /app/
COPY models/ /app/models/
ENV STUDIO_MODEL=/app/models/bandit_speech.onnx E2V_ONNX=/app/models/emotion2vec STUDIO_CACHE_DIR=/var/studio-cache STUDIO_CACHE_GB=40
# BLAS / OpenMP would otherwise start one thread per host core (20+ on big hosts) in every process.
ENV OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# The lab code, unchanged (LAB_REF: its commit). Build gate: the mix and the emotion model import.
COPY lab/ /app/lab/
ENV STUDIO_LAB=/app/lab
RUN cd /app/lab && python3 -c 'import mix_clip, dub_mix, channel_match, recording_defects, room_layers, perspective, netflix_channel, emotion_ser' \
 && cat LAB_REF
COPY __init__.py sep.py vast_worker.py /app/studio/
COPY assets/ /app/studio/assets/
ENTRYPOINT ["/bin/sh","-c","cd /app && exec python3 -u -m studio.vast_worker"]
