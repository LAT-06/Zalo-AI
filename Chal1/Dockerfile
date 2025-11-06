# Dockerfile for NVIDIA Jetson Xavier NX Deployment
# Base image with PyTorch support for Jetson
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    scipy \
    pyyaml \
    tqdm \
    albumentations \
    onnx \
    onnxruntime-gpu \
    tensorrt

# Install additional dependencies for inference
RUN pip3 install --no-cache-dir \
    flask \
    requests

# Copy application files
COPY video_inference.py /app/
COPY stiou_metric.py /app/
COPY generate_submission.py /app/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output

# Copy model weights (these should be mounted at runtime)
# COPY best.pt /app/models/

# Expose port for API (optional)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Default command
CMD ["python3", "video_inference.py", "--help"]
