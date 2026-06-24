# Use python slim base image for a lightweight footprint
FROM python:3.10-slim

# Install system dependencies needed for OpenCV, git, and generic build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables to avoid writing pyc files and to buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    NUMBA_CACHE_DIR=/tmp/numba \
    HOME=/tmp

# Create work directory
WORKDIR /code

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /code/

# Upgrade pip and install dependencies. 
# Force CPU-only version of PyTorch to keep container image small and prevent build timeout/disk exhaustion.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /code/

# Grant ownership and group write permission to the code directory so that running under HF's default UID 1000 has permission to write logs or download cache files without exposing global write permissions
RUN chown -R 1000:1000 /code && chmod -R 775 /code

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run FastAPI app via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
