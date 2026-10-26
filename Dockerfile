# CPU-friendly image: the shipped model (EfficientNet-B0) runs fine on CPU.
# For GPU training, switch the base image to a CUDA-enabled PyTorch image.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries needed by OpenCV (headless) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU build of PyTorch to keep the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/checkpoints /app/results /app/temp_uploads

EXPOSE 8000
# Shell form so $PORT (set by hosts like Render) is honoured; defaults to 8000.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
