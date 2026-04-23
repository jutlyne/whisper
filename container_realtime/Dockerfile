FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "torch>=2.6" "torchaudio" --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
RUN pip install --no-cache-dir -r requirements.txt

# Ensure faster-whisper/ctranslate2 find bundled cuDNN + cuBLAS shared libs
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# Fetch UFAL whisper_streaming (LocalAgreement-2 streaming ASR wrapper)
# Pin to a known-working commit to avoid upstream breaks.
ADD https://raw.githubusercontent.com/ufal/whisper_streaming/main/whisper_online.py /app/whisper_online.py
RUN pip install --no-cache-dir librosa

COPY . .

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--ws-ping-interval", "20", \
     "--ws-ping-timeout", "60", \
     "--timeout-keep-alive", "120"]
