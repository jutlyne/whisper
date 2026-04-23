"""
FastAPI WebSocket server for realtime VI↔JA speech translation.

Per-session flow:
  1. StreamingTranscriber.feed(chunk) — Silero/RMS VAD gate.
  2. feed() returns 'step_due' → enqueue 'step' into asr_queue.
  3. feed() returns 'final_due' → drain queue, enqueue 'final'.
  4. Single asr_worker coroutine processes commands sequentially.
  5. Results go into out_queue → sender coroutine → WebSocket.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import ctranslate2
import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer

from whisper_online import FasterWhisperASR

from vad_transcriber import (
    StreamingTranscriber,
    load_silero_vad,
    SILERO_JIT_PATH,
    SAMPLE_RATE,
)

WHISPER_DIR = "/models/whisper-large-v3-turbo-ct2"
WHISPER_V3_DIR = "/models/whisper-large-v3-ct2"
_NLLB_MODEL_SIZE = os.environ.get("NLLB_MODEL", "600M")
NLLB_CT2_DIR = f"/models/nllb-200-distilled-{_NLLB_MODEL_SIZE}-ct2"
SAMPLE_WARMUP_SAMPLES = 16_000

ENABLE_DENOISE = os.environ.get("ENABLE_DENOISE", "1") == "1"
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "1") == "1"
USE_V3 = os.environ.get("WHISPER_MODEL", "auto")
_MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "10"))

_asr: FasterWhisperASR | None = None
_whisper_model_name: str = ""
_nllb_translator: ctranslate2.Translator | None = None
_nllb_tokenizer = None
_silero_model = None
_df_model = None
_df_state = None
_voice_encoder = None
_model_ready = threading.Event()
_model_error: str | None = None

_N_CPU = os.cpu_count() or 4
_asr_executor = ThreadPoolExecutor(max_workers=max(2, _N_CPU // 2), thread_name_prefix="asr")
_mt_executor = ThreadPoolExecutor(max_workers=max(2, _N_CPU // 2), thread_name_prefix="mt")

_session_sem: asyncio.Semaphore | None = None
_active_sessions: int = 0
_tokenizer_lock: asyncio.Lock | None = None

_PUNCT_END = frozenset("。！？.!?")

# ── Global batch MT worker ────────────────────────────────────────────────────
# Collects translation requests from all sessions, batches them every 50ms,
# then calls CT2 translate_batch() once → lower latency under multi-session load.

class _MTRequest:
    __slots__ = ("tokens", "src_lang", "tgt_lang", "future")
    def __init__(self, tokens: list, src_lang: str, tgt_lang: str, future: asyncio.Future):
        self.tokens = tokens
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.future = future

_mt_req_queue: asyncio.Queue | None = None
_MT_BATCH_WINDOW_S = 0.05   # collect requests for 50ms before translating
_MT_MAX_BATCH     = 16      # safety cap


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log(msg: str) -> None:
    print(f"[server] [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _select_whisper_dir() -> tuple[str, str]:
    def has_model(d: str) -> bool:
        return os.path.isdir(d) and any(
            f.endswith(".bin") or f == "model.bin" for f in os.listdir(d)
        )
    if USE_V3 == "v3" and has_model(WHISPER_V3_DIR):
        return WHISPER_V3_DIR, "large-v3"
    if USE_V3 == "turbo" and has_model(WHISPER_DIR):
        return WHISPER_DIR, "large-v3-turbo"
    if has_model(WHISPER_V3_DIR):
        return WHISPER_V3_DIR, "large-v3"
    if has_model(WHISPER_DIR):
        return WHISPER_DIR, "large-v3-turbo"
    raise RuntimeError(f"No Whisper model found at {WHISPER_V3_DIR} or {WHISPER_DIR}")


def _load_model_worker() -> None:
    global _asr, _whisper_model_name, _nllb_translator, _nllb_tokenizer
    global _silero_model, _df_model, _df_state, _voice_encoder, _model_error
    t0 = time.time()

    def elapsed() -> str:
        return f"{time.time() - t0:.1f}s"

    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.wait(30):
            _log(f"[heartbeat] still loading... ({elapsed()} elapsed)")

    threading.Thread(target=heartbeat, daemon=True, name="Heartbeat").start()

    try:
        device = get_device()
        _log(f"device={device}")
        if device != "cuda" and not os.environ.get("ALLOW_CPU"):
            raise RuntimeError("GPU not available — set ALLOW_CPU=1 to run on CPU")

        whisper_dir, model_name = _select_whisper_dir()
        _whisper_model_name = model_name
        _log(f"Loading FasterWhisperASR ({model_name}) from {whisper_dir}...")
        t1 = time.time()
        _asr = FasterWhisperASR(
            lan="vi",
            modelsize=None,
            model_dir=whisper_dir,
            logfile=sys.stderr,
        )
        _log(f"FasterWhisperASR loaded ({time.time() - t1:.1f}s)")

        _log("Loading NLLB tokenizer...")
        t2 = time.time()
        _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_CT2_DIR)
        _log(f"NLLB tokenizer loaded ({time.time() - t2:.1f}s)")

        _log("Loading CTranslate2 NLLB translator...")
        t3 = time.time()
        _nllb_translator = ctranslate2.Translator(
            NLLB_CT2_DIR,
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
        )
        _log(f"CT2 NLLB loaded ({time.time() - t3:.1f}s)")

        _log("Warmup ASR...")
        t4 = time.time()
        try:
            _asr.transcribe(np.zeros(SAMPLE_WARMUP_SAMPLES, dtype=np.float32))
        except Exception as e:
            _log(f"ASR warmup soft-fail: {e}")
        _log(f"ASR warmup done ({time.time() - t4:.1f}s)")

        _log("Warmup NLLB...")
        t5 = time.time()
        _nllb_tokenizer.src_lang = "jpn_Jpan"
        ids = _nllb_tokenizer("テスト", return_tensors=None).input_ids
        src_tokens = _nllb_tokenizer.convert_ids_to_tokens(ids)
        _nllb_translator.translate_batch(
            [src_tokens], target_prefix=[["vie_Latn"]], beam_size=1, max_decoding_length=8
        )
        _log(f"NLLB warmup done ({time.time() - t5:.1f}s)")

        if ENABLE_DENOISE:
            _log("Loading DeepFilterNet...")
            t_df = time.time()
            try:
                from df.enhance import init_df
                _df_model, _df_state, _ = init_df(log_level="WARNING")
                _log(f"DeepFilterNet loaded ({time.time() - t_df:.1f}s, sr={_df_state.sr()})")
            except Exception as e:
                _log(f"DeepFilterNet failed to load: {e}")

        if ENABLE_DIARIZATION:
            _log("Loading resemblyzer VoiceEncoder...")
            t_ve = time.time()
            try:
                from resemblyzer import VoiceEncoder
                _voice_encoder = VoiceEncoder(device="cpu", verbose=False)
                _log(f"VoiceEncoder loaded ({time.time() - t_ve:.1f}s)")
            except Exception as e:
                _log(f"VoiceEncoder failed to load (continuing without diarization): {e}")

        _log("Loading Silero VAD...")
        t6 = time.time()
        try:
            _silero_model = load_silero_vad(SILERO_JIT_PATH)
            _log(f"Silero VAD loaded ({time.time() - t6:.1f}s)")
        except Exception as e:
            _log(f"Silero VAD failed to load: {e}")

        _log(f"ALL MODELS READY — total load: {elapsed()}")
        _model_ready.set()

    except Exception:
        _model_error = traceback.format_exc()
        _log(f"LOAD FAILED after {elapsed()}:\n{_model_error}")
    finally:
        stop_heartbeat.set()


async def _mt_batcher_loop() -> None:
    """Global coroutine: batch-translate requests from all sessions every 50ms."""
    loop = asyncio.get_running_loop()
    while True:
        # Wait for first request
        req = await _mt_req_queue.get()
        batch = [req]
        # Collect more within the window
        deadline = loop.time() + _MT_BATCH_WINDOW_S
        while len(batch) < _MT_MAX_BATCH:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(_mt_req_queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)

        # Group by (src_lang, tgt_lang) — CT2 target_prefix + tokenizer src_lang must be consistent
        from collections import defaultdict
        by_pair: dict[tuple[str, str], list] = defaultdict(list)
        for r in batch:
            by_pair[(r.src_lang, r.tgt_lang)].append(r)

        for (src_lang, tgt_lang), group in by_pair.items():
            try:
                # Thread only does CT2 inference — returns raw hypothesis token strings
                hyp_list = await loop.run_in_executor(
                    _mt_executor,
                    _ct2_batch_translate,
                    [r.tokens for r in group],
                    tgt_lang,
                )
                # Decode back in async context (tokenizer shared state stays in event loop)
                for r, hyp_tokens in zip(group, hyp_list):
                    if r.future.done():
                        continue
                    async with _tokenizer_lock:
                        tgt_ids = _nllb_tokenizer.convert_tokens_to_ids(hyp_tokens)
                        decoded = _nllb_tokenizer.decode(tgt_ids, skip_special_tokens=True).strip()
                    r.future.set_result(decoded)
            except Exception as e:
                for r in group:
                    if not r.future.done():
                        r.future.set_exception(e)


def _ct2_batch_translate(batch_tokens: list[list], tgt_lang: str) -> list[list[str]]:
    """CT2 inference only — returns hypothesis token strings, no tokenizer.decode() here."""
    if _nllb_translator is None:
        return [[] for _ in batch_tokens]
    tgt_nllb = _NLLB_LANG_MAP_SERVER.get(tgt_lang)
    if not tgt_nllb:
        return [[] for _ in batch_tokens]
    results = _nllb_translator.translate_batch(
        batch_tokens,
        target_prefix=[[tgt_nllb]] * len(batch_tokens),
        beam_size=4,
        max_decoding_length=256,
    )
    return [r.hypotheses[0][1:] for r in results]


_NLLB_LANG_MAP_SERVER = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "vi": "vie_Latn",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session_sem, _mt_req_queue, _tokenizer_lock
    _session_sem = asyncio.Semaphore(_MAX_SESSIONS)
    _tokenizer_lock = asyncio.Lock()
    _mt_req_queue = asyncio.Queue(maxsize=256)
    batcher_task = asyncio.create_task(_mt_batcher_loop())
    threading.Thread(target=_load_model_worker, daemon=True, name="ModelLoader").start()
    yield
    batcher_task.cancel()
    try:
        await batcher_task
    except asyncio.CancelledError:
        pass
    _asr_executor.shutdown(wait=False, cancel_futures=True)
    _mt_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": _whisper_model_name,
        "nllb_ct2_dir": NLLB_CT2_DIR,
        "silero_jit": SILERO_JIT_PATH,
        "whisper_ready": _asr is not None,
        "nllb_ct2_ready": _nllb_translator is not None,
        "silero_ready": _silero_model is not None,
        "denoise_enabled": ENABLE_DENOISE,
        "denoise_ready": _df_model is not None,
        "diarization_enabled": ENABLE_DIARIZATION,
        "diarization_ready": _voice_encoder is not None,
        "device": get_device(),
        "ready": _model_ready.is_set(),
        "error": _model_error,
    }


@app.websocket("/ws")
async def ws_endpoint(
    websocket: WebSocket,
    output_lang: str = Query(default="auto"),
    src_langs: str = Query(default="vi,ja"),
    vocab: str = Query(default=""),
    partials: int = Query(default=0),
    idle_flush: float = Query(default=1.5),
):
    global _active_sessions
    if _active_sessions >= _MAX_SESSIONS:
        await websocket.accept()
        await websocket.send_text("error: server busy, max sessions reached")
        await websocket.close()
        return

    await websocket.accept()
    if not _model_ready.is_set():
        await websocket.send_text("error: model not loaded yet, please retry in a moment")
        await websocket.close()
        return

    _active_sessions += 1
    try:
        await _handle_session(
            websocket, output_lang, src_langs, vocab, partials, idle_flush
        )
    finally:
        _active_sessions -= 1


async def _handle_session(
    websocket: WebSocket,
    output_lang: str,
    src_langs: str,
    vocab: str,
    partials: int,
    idle_flush: float,
) -> None:
    transcriber = StreamingTranscriber(
        asr=_asr,
        silero_model=_silero_model,
        src_langs=src_langs,
        df_model=_df_model,
        df_state=_df_state,
        vocab_hint=vocab.replace(",", " "),
        voice_encoder=_voice_encoder,
    )
    allowed = transcriber.allowed_langs
    auto_pair = output_lang == "auto" and len(allowed) == 2

    def target_lang(detected: str) -> str:
        if auto_pair:
            return next((l for l in allowed if l != detected), detected)
        return output_lang

    send_partials = bool(partials)
    loop = asyncio.get_running_loop()

    # ── Outbound queue + sender (backpressure) ───────────────────────────────
    _out_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)
    _sender_dead = False

    async def sender():
        nonlocal _sender_dead
        while True:
            msg = await _out_queue.get()
            if msg is None:
                break
            try:
                await asyncio.wait_for(websocket.send_text(msg), timeout=10.0)
            except Exception:
                _sender_dead = True
                break

    async def send(msg: str, *, is_final: bool = False) -> None:
        if _sender_dead:
            return
        if is_final:
            await _out_queue.put(msg)
        else:
            try:
                _out_queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # drop partial if queue full

    # ── Per-speaker buffer ───────────────────────────────────────────────────
    buf_text = ""
    buf_lang = ""
    buf_speaker: int | None = None
    buf_last_update: float = 0.0
    MAX_BUF_CHARS = 300
    _buf_lock = asyncio.Lock()

    def speaker_tag(sid: int | None) -> str:
        return f"S{sid + 1}" if sid is not None else "?"

    def _should_flush_buf(combined: str) -> bool:
        return any(combined.endswith(p) for p in _PUNCT_END) or len(combined) >= MAX_BUF_CHARS

    async def emit_final(src: str, text: str, sid: int | None) -> None:
        if not text:
            return
        tag = speaker_tag(sid)
        tgt = target_lang(src) if src else output_lang
        if not src or src == tgt or tgt == "auto":
            await send(f"final [{tag} {src}] {text}", is_final=True)
            return
        src_nllb = _NLLB_LANG_MAP_SERVER.get(src)
        tgt_nllb = _NLLB_LANG_MAP_SERVER.get(tgt)
        if not src_nllb or not tgt_nllb or _nllb_tokenizer is None:
            await send(f"final [{tag} {src}] {text}", is_final=True)
            return
        try:
            # Tokenizer src_lang is shared global state — serialize access
            async with _tokenizer_lock:
                _nllb_tokenizer.src_lang = src_nllb
                input_ids = _nllb_tokenizer(text, truncation=True, max_length=512).input_ids
                src_tokens = _nllb_tokenizer.convert_ids_to_tokens(input_ids)
            fut: asyncio.Future[str] = loop.create_future()
            # put_nowait: if queue full, fall back to direct (blocking) translate
            try:
                _mt_req_queue.put_nowait(_MTRequest(src_tokens, src, tgt, fut))
            except asyncio.QueueFull:
                result = await loop.run_in_executor(
                    _mt_executor, _ct2_batch_translate, [src_tokens], tgt
                )
                translated = result[0] if result else text
                await send(f"final [{tag} {src}→{tgt}] {translated}", is_final=True)
                return
            translated = await asyncio.wait_for(fut, timeout=15.0)
        except Exception as e:
            print(f"[mt] error: {e}", flush=True)
            translated = text
        await send(f"final [{tag} {src}→{tgt}] {translated}", is_final=True)

    async def flush_buf() -> None:
        nonlocal buf_text, buf_lang, buf_speaker
        async with _buf_lock:
            text_to_emit = buf_text.strip()
            lang_to_emit = buf_lang
            sid_to_emit = buf_speaker
            buf_text = ""
            buf_lang = ""
            buf_speaker = None
        # emit outside lock so MT translation doesn't hold the lock
        if text_to_emit:
            await emit_final(lang_to_emit, text_to_emit, sid_to_emit)

    async def handle_utterance(text: str, lang: str, sid: int | None) -> None:
        nonlocal buf_text, buf_lang, buf_speaker, buf_last_update
        if not text:
            return
        async with _buf_lock:
            if buf_speaker is not None and sid is not None and sid != buf_speaker:
                text_to_emit = buf_text.strip()
                lang_to_emit = buf_lang
                sid_to_emit = buf_speaker
                buf_text = ""
                buf_lang = ""
                buf_speaker = None
            else:
                text_to_emit = None
                lang_to_emit = None
                sid_to_emit = None
            combined = (buf_text + " " + text).strip() if buf_text else text
            buf_text = combined
            buf_lang = lang or buf_lang
            buf_speaker = sid if sid is not None else buf_speaker
            buf_last_update = time.time()
            should_flush = _should_flush_buf(buf_text)
            if should_flush:
                flush_text = buf_text.strip()
                flush_lang = buf_lang
                flush_sid = buf_speaker
                buf_text = ""
                buf_lang = ""
                buf_speaker = None
            else:
                flush_text = None
                flush_lang = None
                flush_sid = None
        # emit outside lock
        if text_to_emit:
            await emit_final(lang_to_emit, text_to_emit, sid_to_emit)
        if flush_text:
            await emit_final(flush_lang, flush_text, flush_sid)

    async def idle_flusher():
        while True:
            await asyncio.sleep(0.5)
            if buf_text and (time.time() - buf_last_update) >= idle_flush:
                await flush_buf()

    # ── Single ASR worker + bounded queue ───────────────────────────────────
    _asr_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
    _step_pending = False   # true coalescing flag: only 1 step in flight at a time
    _committed_buf = ""     # text committed by steps, prepended to finalize result
    _committed_lang = ""
    _committed_sid: int | None = None

    async def asr_worker():
        nonlocal _step_pending, _committed_buf, _committed_lang, _committed_sid
        while True:
            cmd = await _asr_queue.get()
            try:
                if cmd in ("final", "flush"):
                    text, lang, sid = await loop.run_in_executor(
                        _asr_executor, transcriber.finalize
                    )
                    # Combine committed text from previous steps with finalize tail
                    lang = lang or _committed_lang
                    sid = sid if sid is not None else _committed_sid
                    full_text = (_committed_buf + " " + text).strip() if _committed_buf else text
                    _committed_buf = ""
                    _committed_lang = ""
                    _committed_sid = None
                    await handle_utterance(full_text, lang, sid)
                elif cmd == "step":
                    _step_pending = False
                    text, lang, sid = await loop.run_in_executor(
                        _asr_executor, transcriber.step
                    )
                    if text:
                        _committed_buf = (_committed_buf + " " + text).strip() if _committed_buf else text
                        _committed_lang = lang or _committed_lang
                        if sid is not None:
                            _committed_sid = sid
                        if send_partials:
                            tag = speaker_tag(sid)
                            await send(f"partial [{tag} {lang}] {_committed_buf}")
            except Exception as e:
                print(f"[asr] worker error ({cmd}): {e}", flush=True)
                if cmd == "step":
                    _step_pending = False
            finally:
                _asr_queue.task_done()

    def enqueue_step() -> None:
        nonlocal _step_pending
        if _step_pending:
            return
        _step_pending = True
        try:
            _asr_queue.put_nowait("step")
        except asyncio.QueueFull:
            _step_pending = False

    async def enqueue_final() -> None:
        nonlocal _step_pending
        # Drain pending steps — must call task_done() for each drained item
        drained = 0
        while not _asr_queue.empty():
            try:
                _asr_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        for _ in range(drained):
            _asr_queue.task_done()
        _step_pending = False
        await _asr_queue.put("final")

    sender_task = asyncio.create_task(sender())
    idle_task = asyncio.create_task(idle_flusher())
    worker_task = asyncio.create_task(asr_worker())

    await send("ready", is_final=True)
    print(
        f"[ws] connect — src_langs={src_langs} output={output_lang} "
        f"partials={send_partials} vocab={vocab!r} idle_flush={idle_flush}",
        flush=True,
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("text") == "flush":
                if transcriber.force_final():
                    await enqueue_final()
                    # Wait for worker to finish processing all queued commands
                    await _asr_queue.join()
                await flush_buf()
                await send("done", is_final=True)
                # Drain outbound queue before breaking
                await _out_queue.put(None)
                await sender_task
                break

            raw = message.get("bytes")
            if raw is None:
                continue

            chunk = np.frombuffer(raw, dtype=np.float32)
            if len(chunk) == 0:
                continue

            status = transcriber.feed(chunk)
            if status == "step_due":
                enqueue_step()
            elif status == "final_due":
                await enqueue_final()

    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except Exception as e:
        print(f"[ws] error: {e}")
        try:
            await send(f"error: {e}", is_final=True)
        except Exception:
            pass
    finally:
        idle_task.cancel()
        worker_task.cancel()
        try:
            await idle_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        # Ensure sender shuts down
        try:
            _out_queue.put_nowait(None)
        except asyncio.QueueFull:
            sender_task.cancel()  # queue full → force stop instead of waiting 10s
        try:
            await asyncio.wait_for(sender_task, timeout=2.0)
        except Exception:
            pass
