FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential curl ca-certificates pkg-config libssl-dev python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
ARG ZONOS_REV=96d9e0ec1e39083ed336ac78f24e3dbc93721c0f
RUN git clone https://github.com/Zyphra/ZONOS3-Mini.git /src && cd /src && git checkout "$ZONOS_REV" && git submodule update --init --depth 1 servers/ggml/ggml third_party/zyphra-textnorm
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cd /src/servers/ggml && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='86;89' -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -DCMAKE_BUILD_TYPE=Release && cmake --build build --target zonos3-server -j 3
RUN uv venv --python 3.12 /opt/convert && uv pip install --python /opt/convert/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu && uv pip install --python /opt/convert/bin/python 'huggingface_hub[cli]' gguf==0.19.0 numpy safetensors pyyaml 'transformers>=4.56.0,<=4.57.3' omegaconf soundfile descript-audio-codec tokenizers kernels zyphra-textnorm lmdb fastapi pydantic
ENV PYTHONPATH=/src/model/src
RUN mkdir -p /models && /opt/convert/bin/hf download Zyphra/ZONOS3-Mini --include 'lm/*' --include 'registries/*' --include 'gguf/zac.gguf' --include 'gguf/zac-encoder.gguf' --include 'gguf/ssl-teacher.gguf' --local-dir /release && cp /release/gguf/*.gguf /models/ && cd /src && /opt/convert/bin/python tools/convert_gguf.py lm /release/lm/zonos3_mini --outtype f16 -o /models/backbone-f16.gguf
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libgomp1 libstdc++6 ca-certificates && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir runpod requests
COPY --from=build /src/servers/ggml/build/ /app/build/
COPY --from=build /models/ /app/models/
COPY handler.py /app/handler.py
ENV LD_LIBRARY_PATH=/app/build/bin:/app/build/ggml/src:/app/build/ggml/src/ggml-cuda:/app/build/ggml/src/ggml-cpu:$LD_LIBRARY_PATH
CMD ["python3","-u","/app/handler.py"]
