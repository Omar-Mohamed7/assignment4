# Lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Fix for PyTorch CPU compatibility issues in Docker
ENV DNNL_MAX_CPU_ISA=AVX2
ENV OMP_NUM_THREADS=1

# Efficient layering: copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy matplotlib

# Copy source code
COPY train.py .

# Run training script
CMD ["python", "train.py"]