import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str


class AudioCaptureService:
    sample_rate = 16_000
    channels = 1
    chunk_frames = 1_600

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._output_path: Path | None = None
        self._error: Exception | None = None
        self._chunk_callback: Callable[[np.ndarray | None, np.ndarray | None], None] | None = None

    def set_chunk_callback(
        self,
        callback: Callable[[np.ndarray | None, np.ndarray | None], None] | None,
    ) -> None:
        """Đăng ký callback nhận (mic_chunk, sys_chunk) mỗi 100ms khi đang ghi.
        Callback chạy trong audio thread — phải non-blocking.
        Truyền None để huỷ đăng ký.
        """
        self._chunk_callback = callback

    def list_audio_devices(self) -> dict[str, list[dict[str, str]]]:
        microphones = [
            AudioDevice(id=microphone.id, name=microphone.name)
            for microphone in sc.all_microphones()
        ]
        system_devices = [
            AudioDevice(id=speaker.id, name=speaker.name)
            for speaker in sc.all_speakers()
        ]

        return {
            "microphones": self._serialize_devices(microphones),
            "systems": self._serialize_devices(system_devices),
        }

    def start_recording(
        self,
        *,
        strategy: str,
        microphone_id: str,
        system_id: str,
        output_path: Path,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("Da co phien ghi dang chay.")

        microphone = self._find_microphone(microphone_id)
        loopback = self._find_loopback_microphone(system_id)
        self._validate_strategy(
            strategy=strategy,
            microphone_id=microphone_id,
            system_id=system_id,
            microphone=microphone,
            loopback=loopback,
        )

        self._output_path = output_path
        self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record_worker,
            kwargs={
                "strategy": strategy,
                "microphone": microphone,
                "loopback": loopback,
                "output_path": output_path,
            },
            daemon=True,
        )
        self._thread.start()

    def stop_recording(self) -> Path:
        if self._thread is None or self._output_path is None:
            raise RuntimeError("Khong co phien ghi dang chay.")

        thread = self._thread
        output_path = self._output_path

        self._stop_event.set()
        thread.join(timeout=10)

        self._thread = None
        self._output_path = None

        if thread.is_alive():
            raise RuntimeError("Khong dung duoc audio recorder trong thoi gian cho phep.")

        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError(str(error)) from error

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Recorder khong tao duoc file WAV hop le khi stop.")

        return output_path

    def is_recording(self) -> bool:
        return self._thread is not None

    @staticmethod
    def _serialize_devices(
        devices: list[AudioDevice],
    ) -> list[dict[str, str]]:
        deduped: list[AudioDevice] = []
        seen_ids: set[str] = set()
        for device in devices:
            if not device.id or device.id in seen_ids:
                continue
            seen_ids.add(device.id)
            deduped.append(device)
        return [
            {
                "id": device.id,
                "name": device.name,
            }
            for device in deduped
        ]

    @staticmethod
    def _find_microphone(microphone_id: str) -> AudioDevice | None:
        if not microphone_id:
            return None
        for microphone in sc.all_microphones():
            if microphone.id == microphone_id:
                return AudioDevice(id=microphone.id, name=microphone.name)
        return None

    @staticmethod
    def _find_loopback_microphone(system_id: str) -> AudioDevice | None:
        if not system_id:
            return None
        loopback = sc.get_microphone(id=system_id, include_loopback=True)
        if loopback is None:
            return None
        return AudioDevice(id=loopback.id, name=loopback.name)

    @staticmethod
    def _validate_strategy(
        *,
        strategy: str,
        microphone_id: str,
        system_id: str,
        microphone: AudioDevice | None,
        loopback: AudioDevice | None,
    ) -> None:
        if strategy == "microphoneOnly":
            if not microphone_id:
                raise RuntimeError("Hay chon microphone de ghi mic-only.")
            if microphone is None:
                raise RuntimeError("Microphone da chon khong con san sang.")
            return

        if strategy == "systemOnly":
            if not system_id:
                raise RuntimeError("Hay chon system audio device de ghi.")
            if loopback is None:
                raise RuntimeError("System audio device da chon khong con san sang.")
            return

        if microphone_id and microphone is None:
            raise RuntimeError("Microphone da chon khong con san sang.")
        if system_id and loopback is None:
            raise RuntimeError("System audio device da chon khong con san sang.")
        if microphone is None and loopback is None:
            raise RuntimeError("Khong co nguon audio nao duoc chon cho phien ghi.")

    def _record_worker(
        self,
        *,
        strategy: str,
        microphone: AudioDevice | None,
        loopback: AudioDevice | None,
        output_path: Path,
    ) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with sf.SoundFile(
                str(output_path),
                mode="w",
                samplerate=self.sample_rate,
                channels=self.channels,
                subtype="PCM_16",
                format="WAV",
            ) as wav_file:
                if strategy == "microphoneOnly" and microphone is not None:
                    self._record_single_source(
                        wav_file=wav_file,
                        device_id=microphone.id,
                        include_loopback=False,
                    )
                    return

                if strategy == "systemOnly" and loopback is not None:
                    self._record_single_source(
                        wav_file=wav_file,
                        device_id=loopback.id,
                        include_loopback=True,
                    )
                    return

                self._record_mixed_sources(
                    wav_file=wav_file,
                    microphone_id=microphone.id if microphone else None,
                    loopback_id=loopback.id if loopback else None,
                )
        except Exception as error:
            self._error = error

    def _record_single_source(
        self,
        *,
        wav_file: sf.SoundFile,
        device_id: str,
        include_loopback: bool,
    ) -> None:
        source = sc.get_microphone(
            id=device_id,
            include_loopback=include_loopback,
        )
        if source is None:
            raise RuntimeError("Khong mo duoc audio source da chon.")

        with source.recorder(
            samplerate=self.sample_rate,
            channels=self.channels,
        ) as recorder:
            while not self._stop_event.is_set():
                chunk = recorder.record(numframes=self.chunk_frames)
                prepared = self._prepare_chunk(chunk)
                wav_file.write(prepared)
                if self._chunk_callback is not None:
                    try:
                        if include_loopback:
                            self._chunk_callback(None, prepared)
                        else:
                            self._chunk_callback(prepared, None)
                    except Exception:
                        pass

    def _record_mixed_sources(
        self,
        *,
        wav_file: sf.SoundFile,
        microphone_id: str | None,
        loopback_id: str | None,
    ) -> None:
        import queue as _q

        if not microphone_id and not loopback_id:
            raise RuntimeError("Khong co nguon audio nao de mix.")

        # Mỗi nguồn chạy trong thread riêng để tránh sequential blocking
        # → không còn WASAPI buffer overflow gây tiếng tạch
        mic_queue: _q.Queue[np.ndarray | None] = _q.Queue(maxsize=100)
        loopback_queue: _q.Queue[np.ndarray | None] = _q.Queue(maxsize=100)

        def _capture(source_id: str, include_loopback: bool, out: _q.Queue) -> None:
            source = sc.get_microphone(id=source_id, include_loopback=include_loopback)
            if source is None:
                out.put(None)
                return
            with source.recorder(samplerate=self.sample_rate, channels=self.channels) as rec:
                while not self._stop_event.is_set():
                    chunk = rec.record(numframes=self.chunk_frames)
                    prepared = self._prepare_chunk(chunk)
                    try:
                        out.put_nowait(prepared)
                    except _q.Full:
                        try:
                            out.get_nowait()   # drop oldest khi queue đầy
                            out.put_nowait(prepared)
                        except (_q.Empty, _q.Full):
                            pass
            out.put(None)  # sentinel

        cap_threads: list[threading.Thread] = []
        if microphone_id:
            t = threading.Thread(
                target=_capture, args=(microphone_id, False, mic_queue), daemon=True
            )
            cap_threads.append(t)
            t.start()
        if loopback_id:
            t = threading.Thread(
                target=_capture, args=(loopback_id, True, loopback_queue), daemon=True
            )
            cap_threads.append(t)
            t.start()

        has_mic = bool(microphone_id)
        has_loopback = bool(loopback_id)

        while not self._stop_event.is_set():
            mic_chunk: np.ndarray | None = None
            loopback_chunk: np.ndarray | None = None

            if has_mic:
                try:
                    mic_chunk = mic_queue.get(timeout=0.5)
                except _q.Empty:
                    continue
                if mic_chunk is None:   # capture thread died
                    break

            if has_loopback:
                try:
                    loopback_chunk = loopback_queue.get(timeout=0.5)
                except _q.Empty:
                    pass   # ghi mic-only chunk nếu loopback chưa sẵn
                if loopback_chunk is None:
                    break

            wav_file.write(self._mix_chunks(mic_chunk, loopback_chunk))
            if self._chunk_callback is not None:
                try:
                    self._chunk_callback(mic_chunk, loopback_chunk)
                except Exception:
                    pass

        for t in cap_threads:
            t.join(timeout=2)

    @staticmethod
    def _prepare_chunk(chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim == 1:
            return chunk.reshape(-1, 1).astype(np.float32)
        return chunk[:, :1].astype(np.float32)

    def _mix_chunks(
        self,
        microphone_chunk: np.ndarray | None,
        loopback_chunk: np.ndarray | None,
    ) -> np.ndarray:
        chunks = [
            self._prepare_chunk(chunk)
            for chunk in (microphone_chunk, loopback_chunk)
            if chunk is not None and chunk.size
        ]
        if not chunks:
            return np.zeros((self.chunk_frames, self.channels), dtype=np.float32)

        max_frames = max(chunk.shape[0] for chunk in chunks)
        mixed = np.zeros((max_frames, self.channels), dtype=np.float32)
        for chunk in chunks:
            mixed[: chunk.shape[0]] += chunk

        mixed /= float(len(chunks))
        return np.clip(mixed, -1.0, 1.0)
