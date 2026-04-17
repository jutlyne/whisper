# nvidia/cuda base provides CUDA runtime + cuDNN — no conda, pure pip
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install PyTorch CUDA wheel FIRST — accelerate depends on torch,
# so if we install requirements.txt first, pip pulls CPU torch from PyPI
# and then skips the CUDA reinstall (same version already satisfied).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
