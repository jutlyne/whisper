"""
Realtime transcription client — streams audio to Cloud Run Service via WebSocket.
Drop-in replacement for RealtimeTranscriptionService but model runs on the cloud.
"""
from __future__ import annotations

import threading
import time
import queue
from collections.abc import Callable

import numpy as np


class RealtimeCloudService:
    """
    Usage:
        svc = RealtimeCloudService(url, on_text=..., on_error=..., on_ready=...)
        svc.start()
        svc.feed(chunk)   # per 100ms audio chunk
        svc.stop()
    """

    MAX_QUEUE_SIZE = 200

    def __init__(
        self,
        url: str,
        *,
        output_language: str = "vi",
        on_text: Callable[[str], None],
        on_error: Callable[[str], None],
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        # Convert http(s) → ws(s), append output_language as query param
        base = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
        self._url = f"{base}?output_lang={output_language}"
        self._on_text = on_text
        self._on_error = on_error
        self._on_ready = on_ready

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="RealtimeCloud")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    def feed(self, chunk: np.ndarray) -> None:
        """Non-blocking. Call from audio capture thread with float32 mono PCM."""
        try:
            self._queue.put_nowait(chunk.flatten().astype(np.float32))
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._on_error("websockets library not installed. Run: pip install websockets")
            return

        max_retries = 10
        for attempt in range(max_retries):
            if self._stop_event.is_set():
                return
            try:
                with connect(self._url, open_timeout=10, ping_interval=None) as ws:
                    # Wait for "ready" from server
                    msg = ws.recv(timeout=30)

                    if "not loaded yet" in str(msg):
                        wait = 15 * (attempt + 1)
                        print(f"[realtime] Model not ready, retrying in {wait}s ({attempt + 1}/{max_retries})")
                        time.sleep(wait)
                        continue

                    if str(msg).startswith("error"):
                        self._on_error(msg)
                        return

                    if msg == "ready" and self._on_ready:
                        self._on_ready()

                    # Receive loop in a separate thread so we can send simultaneously
                    recv_thread = threading.Thread(
                        target=self._recv_loop, args=(ws,), daemon=True
                    )
                    recv_thread.start()

                    # Send loop
                    while not self._stop_event.is_set():
                        try:
                            chunk = self._queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if chunk is None:
                            break
                        ws.send(chunk.tobytes())

                    # Signal server to flush remaining buffer
                    try:
                        ws.send("flush")
                    except Exception:
                        pass

                    recv_thread.join(timeout=10)
                    return  # connected and finished normally

            except Exception as exc:
                self._on_error(f"Realtime connection error: {exc}")
                return

    def _recv_loop(self, ws) -> None:
        try:
            while True:
                msg = ws.recv(timeout=None)
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", errors="replace")
                if msg == "done":
                    break
                if isinstance(msg, str) and msg.startswith("error"):
                    self._on_error(msg)
                    break
                if isinstance(msg, str) and msg:
                    self._on_text(msg)
        except Exception:
            pass  # Connection closed — normal on stop
