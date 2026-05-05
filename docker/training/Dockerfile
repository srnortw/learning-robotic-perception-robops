# Training image — amd64 only (workstation + GitHub Actions runner)
# Used for: ONNX export, INT8 quantization, local training experiments
# NOT used for Colab training (Colab manages its own environment)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="robops"
LABEL description="RoboOps DETR training and ONNX export environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY docker/training/requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

# Copy only the pipeline scripts needed for export and training
COPY pipeline/phase_c/ ./pipeline/phase_c/
RUN mkdir -p ./pipeline/phase_b
# drift baseline is optional at build time; generate it at runtime if absent
COPY pipeline/phase_b/ ./pipeline/phase_b/

# Default command runs ONNX export
# Override at runtime: docker run ... python pipeline/phase_c/detr/train.py
CMD ["python", "pipeline/phase_c/detr/export_onnx.py", "--help"]
