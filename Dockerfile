# syntax=docker/dockerfile:1
FROM docker.io/astral/uv:0.11.28 AS uv

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TARGETARCH
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCHAUDIO_VERSION=2.9.1

LABEL org.opencontainers.image.title="Reusable CUDA training environment" \
      org.opencontainers.image.description="CUDA 12.8, CPython 3.12, and PyTorch cu128 training environment" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3005,DL3008
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        ninja-build \
        openssh-client \
        rsync \
        tmux \
        wget \
    && rm -rf /opt/nvidia/nsight-compute /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/workspace/.cache/uv \
    test "${TARGETARCH:-amd64}" = "amd64" \
    && uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv \
    && UV_HTTP_TIMEOUT=300 uv pip install --python /opt/venv/bin/python \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
    && UV_HTTP_TIMEOUT=300 uv pip install --python /opt/venv/bin/python \
        "accelerate==1.12.0" \
        "click==8.4.2" \
        "datasets==4.5.0" \
        "hf-transfer==0.1.9" \
        "huggingface-hub==1.23.0" \
        "jaraco-context==6.1.0" \
        "pillow==12.3.0" \
        "safetensors==0.7.0" \
        "setuptools==83.0.0" \
        "wheel==0.46.2" \
    && find /opt -type d -name __pycache__ -prune -exec rm -rf '{}' +

COPY scripts/verify-gpu.py /usr/local/bin/verify-gpu
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint

RUN chmod 0755 /usr/local/bin/verify-gpu /usr/local/bin/entrypoint \
    && mkdir -p /workspace/.cache

WORKDIR /workspace
VOLUME ["/workspace"]
ENTRYPOINT ["/usr/local/bin/entrypoint"]
CMD ["bash"]
