# ==============================================================================
# Multi-Stage Dockerfile: Multimodal Vietnamese-to-Hmong Speech-to-Speech Translation
# Base Image: NVIDIA CUDA 12.1 (Ubuntu 22.04 LTS)
# Target Architecture: Production Real-Time Streaming Server (< 1.5s Latency)
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Builder Stage (Compilers, C/C++ Headers & Python Wheel Compilation)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system build tools, Python runtime development libraries, and audio C headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    pkg-config \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment
RUN python3.10 -m venv /opt/venv

WORKDIR /build
COPY requirements.txt .

# Pre-install PyTorch ecosystem targeting CUDA 12.1, then compile remaining dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.2.2+cu121 torchaudio==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# STAGE 2: Runtime Stage (Minimal Security-Hardened Execution Environment)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Install runtime-only dependencies (FFmpeg, eSpeak-ng speech engine, libsndfile1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Enforce security best practice: Unprivileged non-root service account
RUN useradd -m -u 1000 appuser && \
    mkdir -p /workspace/checkpoints /workspace/logs /workspace/data /workspace/configs && \
    chown -R appuser:appuser /workspace

WORKDIR /workspace

# Copy application assets with strict non-root ownership
COPY --chown=appuser:appuser requirements.txt /workspace/
COPY --chown=appuser:appuser configs/ /workspace/configs/
COPY --chown=appuser:appuser src/ /workspace/src/
COPY --chown=appuser:appuser scripts/ /workspace/scripts/
COPY --chown=appuser:appuser docs/ /workspace/docs/

USER appuser

EXPOSE 8000

# Container health monitoring via REST healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production single-worker uvicorn deployment (optimizes CUDA context and VRAM residency)
CMD ["uvicorn", "src.api.app_stream:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
