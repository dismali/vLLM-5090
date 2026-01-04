ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04

#################### BASE BUILD IMAGE ####################
# Using standard CUDA devel image since cuda-dl-base may not have 13.0 yet
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo wget gnupg2 \
        python3 python3-dev python3-venv python3-pip tzdata \
        nodejs npm && \
    # Add NVIDIA ML repo for cuDNN/NCCL packages
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update -y && \
    # Install cuDNN and NCCL for deep learning
    apt-get install -y --no-install-recommends \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
        libnccl2 libnccl-dev || true && \
    # Install uv package manager
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv ~/.local/bin/uv /usr/local/bin/ && \
    rm -f /usr/lib/python*/EXTERNALLY-MANAGED && \
    # Cleanup
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 --version && \
    node --version && \
    npm --version

WORKDIR /workspace

# OpenCode + related tools
RUN npm install -g opencode-ai @anthropic-ai/claude-code typescript ts-node

# CUDA arch list for torch / flash-attn
# Blackwell (RTX 50xx) is compute 12.0, keeping 10.0 for Hopper compatibility
ARG torch_cuda_arch_list='10.0+PTX 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV FLASH_ATTN_CUDA_ARCHS=120

# Set CUDA paths explicitly
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

#################### FINAL STABLE IMAGE ####################
FROM base AS final-image

# Install stable vLLM 0.13.0 + LMCache 0.3.11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system vllm==0.13.0 && \
    uv pip install --system lmcache==0.3.11 && \
    uv pip install --system runai tensorizer

WORKDIR /workspace

# Default command (override when running the container)
# ENTRYPOINT ["vllm", "serve"]