# Python Recorder

PySide6 desktop application for Windows that records audio only, stores WAV
sessions locally, and runs offline transcription through a standalone
Python-native stack:

- `soundcard` + WASAPI loopback for system audio capture
- `soundfile` for WAV output
- `faster-whisper` for offline transcription on CPU

The Python app no longer depends on the `js/` app runtime.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

## Smoke test

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

## Whisper test

```powershell
.\.venv\Scripts\python.exe scripts\test_whisper.py
```

Or pass a WAV path explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\test_whisper.py path\to\audio.wav
```

## Diarization note

`pyannote.audio` does not have official Windows support, so
`services/diarization_service.py` is currently a guarded spike with a safe
fallback instead of an enabled production feature.
& "c:\Users\kyne\OneDrive\Máy tính\MTT\python\.venv\Scripts\Activate.ps1"

gcloud builds submit --tag gcr.io/{PROJECT_ID}/whisper-transcriber ./container
gcloud run jobs create whisper-transcriber \
  --image gcr.io/{PROJECT_ID}/whisper-transcriber \
  --region asia-east1 --memory 4Gi --cpu 4 --task-timeout 3600
