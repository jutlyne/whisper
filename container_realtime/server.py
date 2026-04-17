"""
FastAPI WebSocket server for realtime transcription using HuggingFace transformers.
Deploy as a Cloud Run Service (always-on).

Protocol:
  - Client connects to ws://{host}/ws?output_lang=vi
  - Client sends binary frames: raw float32 PCM (mono, 16kHz, 100ms = 1600 samples)
  - Client sends text frame "flush" to signal recording stopped
  - Server replies with text frames containing transcribed/translated text
  - Server sends "ready" once model is loaded
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from vad_transcriber import VadTranscriber, SAMPLE_RATE

MODEL_DIR = "/models/whisper-large-v3"

_processor: AutoProcessor | None = None
_model: AutoModelForSpeechSeq2Seq | None = None
_model_ready = threading.Event()
_model_error: str | None = None

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _log(msg: str) -> None:
    print(f"[server] [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_model_worker() -> None:
    """Background thread: load model into GPU."""
    global _processor, _model, _model_error
    t0 = time.time()

    def elapsed() -> str:
        return f"{time.time() - t0:.1f}s"

    # Heartbeat: print every 30s so we know the thread is still alive
    _stop_heartbeat = threading.Event()
    def _heartbeat():
        while not _stop_heartbeat.wait(30):
            _log(f"[heartbeat] still loading... ({elapsed()} elapsed)")
    threading.Thread(target=_heartbeat, daemon=True, name="Heartbeat").start()

    try:
        device = get_device()
        _log(f"device={device} | CUDA available={torch.cuda.is_available()}")
        if device == "cuda:0":
            _log(f"GPU: {torch.cuda.get_device_name(0)} | "
                 f"VRAM total={torch.cuda.get_device_properties(0).total_memory // 1024**2}MB")

        if device != "cuda:0" and not os.environ.get("ALLOW_CPU"):
            raise RuntimeError("GPU không available — STOP (không chạy CPU)")

        # ── 1. Kiểm tra model directory ──────────────────────────────────────
        _log(f"Checking model directory: {MODEL_DIR}")
        config_path = os.path.join(MODEL_DIR, "config.json")
        if not os.path.exists(config_path):
            contents = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else "(directory missing)"
            raise RuntimeError(
                f"config.json not found at {MODEL_DIR}. "
                f"GCS volume mount may have failed. "
                f"Directory contents: {contents}"
            )
        # Print file size to confirm the GCS mount is readable
        model_file = os.path.join(MODEL_DIR, "model.safetensors")
        if os.path.exists(model_file):
            size_gb = os.path.getsize(model_file) / 1024**3
            _log(f"model.safetensors: {size_gb:.2f} GB — mount OK")
        _log(f"Model directory OK ({elapsed()})")

        # ── 2. Load processor ────────────────────────────────────────────────
        _log("Loading processor (tokenizer + feature extractor)...")
        t1 = time.time()
        _processor = AutoProcessor.from_pretrained(MODEL_DIR)
        _log(f"Processor loaded ({time.time() - t1:.1f}s)")

        # ── 3. Load model weights ────────────────────────────────────────────
        _log("Loading model weights into RAM (this takes ~30-120s)...")
        t2 = time.time()
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        _log(f"Model weights loaded into RAM ({time.time() - t2:.1f}s)")

        # ── 4. Move to GPU ───────────────────────────────────────────────────
        _log(f"Moving model to {device}...")
        t3 = time.time()
        _model = _model.to(device)
        _model.eval()
        if device == "cuda:0":
            vram_used = torch.cuda.memory_allocated(0) // 1024**2
            _log(f"Model on GPU — VRAM used: {vram_used}MB ({time.time() - t3:.1f}s)")

        # ── 5. Warmup ────────────────────────────────────────────────────────
        _log("Running warmup inference...")
        t4 = time.time()
        dummy_audio = np.zeros(16_000, dtype=np.float32)
        dummy_features = _processor(dummy_audio, sampling_rate=16_000, return_tensors="pt").input_features
        dummy_features = dummy_features.to(device, dtype=torch.float16)
        with torch.no_grad():
            _ = _model.generate(dummy_features, max_new_tokens=1)
        _log(f"Warmup done ({time.time() - t4:.1f}s)")

        _log(f"Model READY — total load time: {elapsed()}")
        _model_ready.set()

    except Exception:
        _model_error = traceback.format_exc()
        _log(f"LOAD FAILED after {elapsed()}:\n{_model_error}")
    finally:
        _stop_heartbeat.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in background — server listens on port 8080 immediately
    t = threading.Thread(target=_load_model_worker, daemon=True, name="ModelLoader")
    t.start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_DIR,
        "device": get_device(),
        "ready": _model_ready.is_set(),
        "error": _model_error,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, output_lang: str = Query(default="vi")):
    await websocket.accept()
    if not _model_ready.is_set():
        await websocket.send_text("error: model not loaded yet, please retry in a moment")
        await websocket.close()
        return

    device = get_device()

    transcriber = VadTranscriber(_processor, _model, device, output_lang=output_lang)
    await websocket.send_text("ready")
    print(f"[ws] Client connected — output_lang={output_lang}")

    loop = asyncio.get_running_loop()
    try:
        while True:
            message = await websocket.receive()

            if message.get("text") == "flush":
                text = await loop.run_in_executor(None, transcriber.flush)
                if text:
                    await websocket.send_text(text)
                await websocket.send_text("done")
                break

            raw = message.get("bytes")
            if raw is None:
                continue

            chunk = np.frombuffer(raw, dtype=np.float32)
            if len(chunk) == 0:
                continue

            text = await loop.run_in_executor(None, transcriber.feed, chunk)
            if text:
                await websocket.send_text(text)

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print(f"[ws] Error: {e}")
        try:
            await websocket.send_text(f"error: {e}")
        except Exception:
            pass
