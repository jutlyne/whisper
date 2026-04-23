"""
Streaming transcription backed by UFAL whisper_streaming (LocalAgreement-2).

Our contribution in this wrapper:
  - Silero VAD gate: only feed speech chunks into OnlineASRProcessor.
  - DeepFilterNet denoise on the raw buffer before feed (optional).
  - Hallucination + word-loop filters on committed text.
  - NLLB CT2 translation helper (vi↔ja meeting mode).

Lifecycle:
  1. server.py constructs ONE global FasterWhisperASR + Silero + DF.
  2. Per WebSocket session: new StreamingTranscriber(asr_shared, silero, df).
  3. On each client audio chunk: feed(chunk) → may return committed text.
  4. On 'flush': finish() returns the remaining committed text.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torchaudio.functional as AF

from whisper_online import OnlineASRProcessor

SAMPLE_RATE = 16_000

# VAD
_SILERO_THRESHOLD        = 0.4
_SPEECH_THRESHOLD_MIN    = 0.002
_SPEECH_MULTIPLIER       = 2.5
_NOISE_PERCENTILE        = 20
_HISTORY_SIZE            = 100
_RMS_WARMUP_SAMPLES      = 20

# Streaming cadence
_MIN_AUDIO_FOR_STEP_MS   = 500   # call process_iter() every N ms of ingested speech
_MIN_UTTERANCE_MS        = 400   # skip micro-utterances

# Adaptive silence: short utterance uses tight window, long utterance uses looser window
# (prevents cutting mid-sentence when speaker breathes during a long sentence)
_SILENCE_FINAL_MS_SHORT  = 400   # utterance < _SILENCE_ADAPT_THRESH_MS
_SILENCE_FINAL_MS_LONG   = 650   # utterance >= _SILENCE_ADAPT_THRESH_MS
_SILENCE_ADAPT_THRESH_MS = 4000  # above this → use longer silence window

_RECENT_FINALS_WINDOW    = 6

# Speaker diarization (resemblyzer)
_SPEAKER_SIM_THRESHOLD   = 0.60   # cosine > threshold → same speaker
_SPEAKER_MIN_AUDIO_MS    = 2000   # resemblyzer needs ≥2s for reliable embed
_SPEAKER_CENTROID_ALPHA  = 0.2    # EMA update weight for centroid

_NLLB_LANG_MAP = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "vi": "vie_Latn",
}

_HALLUCINATION_PATTERNS = (
    "ghiền mì gõ",
    "subscribe cho kênh",
    "cảm ơn các bạn đã theo dõi",
    "cảm ơn các bạn đã xem",
    "đừng quên like và subscribe",
    "hẹn gặp lại các bạn trong những video",
    "hãy đăng ký kênh",
    "đăng ký kênh để ủng hộ",
    "nhớ đăng ký",
    "ủng hộ kênh của mình",
    "thanks for watching",
    "subtitles by",
    "transcription by",
    "translated by",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "チャンネル登録",
    "また次の動画で",
    "字幕",
    "제공",
    "请订阅",
)


def _is_hallucination(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _HALLUCINATION_PATTERNS)


def _has_word_loop(text: str) -> bool:
    words = text.split()
    if len(words) < 6:
        return False
    from collections import Counter
    counts = Counter(w.lower() for w in words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    if most_common_count / len(words) > 0.4 and most_common_count >= 4:
        return True
    run = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            run += 1
            if run >= 4:
                return True
        else:
            run = 1
    return False


def _clean_text(text: str) -> str:
    text = text.strip()
    while text.endswith("...") or text.endswith("…"):
        text = text.rstrip(".… ").strip()
    return text


SILERO_JIT_PATH = "/models/silero-vad/silero_vad.jit"


def load_silero_vad(local_path: str = SILERO_JIT_PATH):
    model = torch.jit.load(local_path, map_location="cpu")
    model.eval()
    return model


class StreamingTranscriber:
    """
    One per WebSocket session. Not thread-safe.
    Wraps UFAL OnlineASRProcessor with VAD gating + denoise + filters.
    """

    def __init__(
        self,
        asr,                 # FasterWhisperASR (shared across sessions)
        silero_model,
        src_langs: str = "vi,ja",
        df_model=None,
        df_state=None,
        vocab_hint: str = "",
        voice_encoder=None,  # resemblyzer.VoiceEncoder (shared across sessions)
    ) -> None:
        self._silero = silero_model
        self._df_model = df_model
        self._df_state = df_state
        self._df_sr = df_state.sr() if df_state is not None else None
        self._voice_encoder = voice_encoder

        # Speaker diarization state (per session — each meeting starts fresh)
        self._speaker_centroids: list[np.ndarray] = []   # cosine-normalized
        self._last_speaker_id: int | None = None

        parsed = [l.strip() for l in src_langs.split(",") if l.strip() and l.strip() != "auto"]
        self._allowed_langs = parsed
        self._current_lang: str | None = None

        # OnlineASRProcessor buffers audio + runs LocalAgreement-2 commits.
        self._online = OnlineASRProcessor(
            asr,
            buffer_trimming=("segment", 15),  # trim committed audio every 15s segment boundary
        )
        self._online.init()

        # We do our OWN language detection on first real speech, then fix
        # the ASR lan for the rest of the utterance. Meeting mode resets
        # detection on every finish().
        self._asr = asr

        self._vocab_hint = vocab_hint.strip()

        # Audio ingestion state
        self._buffered_ms = 0
        self._samples_since_step = 0
        self._silence_ms = 0
        self._in_speech = False
        self._utterance_ms = 0
        self._pending_audio: list[np.ndarray] = []   # not yet fed to ASR
        self._utterance_audio: list[np.ndarray] = [] # all audio for current utterance (for speaker ID)

        self._min_step_samples = int(_MIN_AUDIO_FOR_STEP_MS * SAMPLE_RATE / 1000)
        self._silence_final_samples_short = int(_SILENCE_FINAL_MS_SHORT * SAMPLE_RATE / 1000)
        self._silence_final_samples_long  = int(_SILENCE_FINAL_MS_LONG  * SAMPLE_RATE / 1000)
        self._silence_adapt_samples       = int(_SILENCE_ADAPT_THRESH_MS * SAMPLE_RATE / 1000)
        self._min_utterance_samples = int(_MIN_UTTERANCE_MS * SAMPLE_RATE / 1000)

        self._rms_history: list[float] = [_SPEECH_THRESHOLD_MIN] * _RMS_WARMUP_SAMPLES
        self._adaptive_threshold = _SPEECH_THRESHOLD_MIN * 3

        self._recent_finals: deque[str] = deque(maxlen=_RECENT_FINALS_WINDOW)

        if self._silero is not None and hasattr(self._silero, "reset_states"):
            self._silero.reset_states()

    # ── VAD ──────────────────────────────────────────────────────────────────

    def _silero_is_speech(self, chunk: np.ndarray) -> bool:
        if self._silero is None:
            return True
        try:
            tensor = torch.from_numpy(chunk).float()
            with torch.no_grad():
                prob = self._silero(tensor, SAMPLE_RATE).item()
            return prob > _SILERO_THRESHOLD
        except Exception:
            return True

    def _update_threshold(self, energy: float) -> None:
        self._rms_history.append(energy)
        if len(self._rms_history) > _HISTORY_SIZE:
            self._rms_history.pop(0)
        sorted_rms = sorted(self._rms_history)
        noise_floor = sorted_rms[int(len(sorted_rms) * _NOISE_PERCENTILE / 100)]
        self._adaptive_threshold = max(_SPEECH_THRESHOLD_MIN, noise_floor * _SPEECH_MULTIPLIER)

    # ── Denoise ─────────────────────────────────────────────────────────────

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        if self._df_model is None or self._df_state is None:
            return audio
        try:
            from df.enhance import enhance
            tensor = torch.from_numpy(audio).unsqueeze(0)
            if self._df_sr != SAMPLE_RATE:
                tensor = AF.resample(tensor, SAMPLE_RATE, self._df_sr)
            enhanced = enhance(self._df_model, self._df_state, tensor)
            if self._df_sr != SAMPLE_RATE:
                enhanced = AF.resample(enhanced, self._df_sr, SAMPLE_RATE)
            return enhanced.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[denoise] failed: {e}", flush=True)
            return audio

    # ── Speaker diarization ─────────────────────────────────────────────────

    def _identify_speaker(self, audio: np.ndarray) -> int | None:
        """
        Compute speaker embedding and match to existing centroids.
        Returns speaker_id (0-based) or None if audio too short / no encoder.
        Short utterances (< _SPEAKER_MIN_AUDIO_MS) inherit the last speaker
        as a continuation/interjection heuristic.
        """
        if self._voice_encoder is None:
            return None
        if len(audio) < _SPEAKER_MIN_AUDIO_MS * SAMPLE_RATE // 1000:
            return self._last_speaker_id  # short utterance → assume same speaker
        try:
            emb = self._voice_encoder.embed_utterance(audio)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        except Exception as e:
            print(f"[diar] embed error: {e}", flush=True)
            return self._last_speaker_id

        best_id = None
        best_sim = -1.0
        for i, centroid in enumerate(self._speaker_centroids):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_id = i

        if best_id is not None and best_sim >= _SPEAKER_SIM_THRESHOLD:
            # Update centroid with EMA
            c = self._speaker_centroids[best_id]
            new_c = (1 - _SPEAKER_CENTROID_ALPHA) * c + _SPEAKER_CENTROID_ALPHA * emb
            new_c = new_c / (np.linalg.norm(new_c) + 1e-9)
            self._speaker_centroids[best_id] = new_c
            self._last_speaker_id = best_id
            return best_id

        # New speaker — cap at 8 to avoid runaway cluster creation
        if len(self._speaker_centroids) >= 8:
            # Assign to best match even below threshold
            self._last_speaker_id = best_id if best_id is not None else 0
            return self._last_speaker_id

        new_id = len(self._speaker_centroids)
        self._speaker_centroids.append(emb)
        self._last_speaker_id = new_id
        return new_id

    # ── Language detection & ASR lan binding ────────────────────────────────

    def _ensure_language(self, audio: np.ndarray) -> None:
        """
        Determine source language for current utterance (from allowed set) and
        bind it to the ASR. Called once per utterance (after min speech buffered).
        """
        if self._current_lang:
            return
        if not self._allowed_langs:
            return  # let ASR auto-detect
        if len(self._allowed_langs) == 1:
            self._current_lang = self._allowed_langs[0]
        else:
            try:
                # Use transcribe() to get language probabilities (faster-whisper 1.x API)
                segments, info = self._asr.model.transcribe(
                    audio,
                    language=None,
                    beam_size=1,
                    best_of=1,
                    without_timestamps=True,
                    max_new_tokens=1,
                )
                # consume generator to get info populated
                _ = list(segments)
                detected = getattr(info, "language", None)
                if detected and detected in self._allowed_langs:
                    self._current_lang = detected
                else:
                    # pick allowed lang with highest probability
                    all_probs = getattr(info, "all_language_probs", None) or {}
                    if isinstance(all_probs, dict):
                        items = all_probs.items()
                    else:
                        items = all_probs  # list of (lang, prob)
                    best = max(
                        ((lang, prob) for lang, prob in items if lang in self._allowed_langs),
                        key=lambda x: x[1],
                        default=None,
                    )
                    self._current_lang = best[0] if best else self._allowed_langs[0]
            except Exception as e:
                print(f"[asr] detect_language failed: {e}", flush=True)
                self._current_lang = self._allowed_langs[0]
        # Apply to ASR (UFAL wrapper stores language on instance)
        try:
            self._asr.original_language = self._current_lang
        except Exception:
            pass

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, chunk: np.ndarray) -> str:
        """
        Ingest a 100ms chunk. Returns 'continue' | 'step_due' | 'final_due'.
        Caller should then call step() or finalize() in executor.
        """
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        self._update_threshold(energy)

        silero_speech = self._silero_is_speech(chunk)
        rms_speech = energy > self._adaptive_threshold
        is_speech = silero_speech and rms_speech if self._in_speech else (silero_speech or rms_speech)

        n = len(chunk)
        n_ms = int(1000 * n / SAMPLE_RATE)

        if is_speech:
            self._in_speech = True
            self._silence_ms = 0
            self._pending_audio.append(chunk)
            self._utterance_ms += n_ms
            self._samples_since_step += n

            if self._samples_since_step >= self._min_step_samples:
                return "step_due"
            return "continue"

        if self._in_speech:
            self._pending_audio.append(chunk)
            self._utterance_ms += n_ms
            self._silence_ms += n_ms
            utterance_samples = self._utterance_ms * SAMPLE_RATE // 1000
            silence_threshold = (
                self._silence_final_samples_long
                if utterance_samples >= self._silence_adapt_samples
                else self._silence_final_samples_short
            )
            if self._silence_ms * SAMPLE_RATE // 1000 >= silence_threshold:
                if utterance_samples >= self._min_utterance_samples:
                    return "final_due"
                self._reset_utterance()
        return "continue"

    def force_final(self) -> bool:
        return self._in_speech and self._utterance_ms * SAMPLE_RATE // 1000 >= self._min_utterance_samples

    def _flush_pending_to_online(self) -> None:
        """Denoise pending buffer, push into OnlineASRProcessor.
        Also accumulate raw audio into utterance_audio (for speaker ID at finalize)."""
        if not self._pending_audio:
            return
        audio = np.concatenate(self._pending_audio).astype(np.float32)
        self._pending_audio.clear()
        self._samples_since_step = 0
        self._utterance_audio.append(audio)
        if len(audio) >= self._min_utterance_samples:
            self._ensure_language(audio)
        self._online.insert_audio_chunk(self._denoise(audio))

    def step(self) -> tuple[str, str, int | None]:
        """
        Run one streaming commit pass. Returns (newly_committed_text, lang, speaker_id).
        speaker_id carries forward _last_speaker_id (finalized more precisely at end).
        """
        self._flush_pending_to_online()
        try:
            _, _, text = self._online.process_iter()
        except Exception as e:
            print(f"[online] step error: {e}", flush=True)
            return "", self._current_lang or "", self._last_speaker_id

        text = _clean_text(text)
        if not text:
            return "", self._current_lang or "", self._last_speaker_id
        if _is_hallucination(text) or _has_word_loop(text) or self._is_repetition(text):
            print(f"[asr] dropped: {text[:80]!r}", flush=True)
            return "", self._current_lang or "", self._last_speaker_id
        self._recent_finals.append(text)
        return text, self._current_lang or "", self._last_speaker_id

    def finalize(self) -> tuple[str, str, int | None]:
        """End-of-utterance: push remaining pending, finish, compute speaker ID."""
        self._flush_pending_to_online()

        # Compute speaker ID from accumulated utterance audio BEFORE resetting
        speaker_id: int | None = None
        if self._utterance_audio:
            full_audio = np.concatenate(self._utterance_audio).astype(np.float32)
            speaker_id = self._identify_speaker(full_audio)

        try:
            _, _, text = self._online.finish()
        except Exception as e:
            print(f"[online] finish error: {e}", flush=True)
            text = ""

        text = _clean_text(text)
        lang = self._current_lang or ""

        # Reset for next utterance
        self._online.init()
        self._reset_utterance()
        self._current_lang = None
        if self._silero is not None and hasattr(self._silero, "reset_states"):
            self._silero.reset_states()

        if not text:
            return "", lang, speaker_id
        if _is_hallucination(text) or _has_word_loop(text) or self._is_repetition(text):
            print(f"[asr] dropped final: {text[:80]!r}", flush=True)
            return "", lang, speaker_id
        self._recent_finals.append(text)
        return text, lang, speaker_id

    def _reset_utterance(self) -> None:
        self._pending_audio.clear()
        self._utterance_audio.clear()
        self._samples_since_step = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        self._in_speech = False

    def _is_repetition(self, text: str) -> bool:
        normalized = text.strip().lower()
        if len(normalized) < 4:
            return False
        return any(prev.strip().lower() == normalized for prev in self._recent_finals)

    @property
    def allowed_langs(self) -> list[str]:
        return self._allowed_langs

    @property
    def current_lang(self) -> str | None:
        return self._current_lang


# ── NLLB translate helper (unchanged) ───────────────────────────────────────

def nllb_translate(
    nllb_translator, nllb_tokenizer, text: str, src_code: str, tgt_code: str
) -> str:
    src_nllb = _NLLB_LANG_MAP.get(src_code)
    tgt_nllb = _NLLB_LANG_MAP.get(tgt_code)
    if not src_nllb or not tgt_nllb or nllb_translator is None:
        return text
    try:
        nllb_tokenizer.src_lang = src_nllb
        input_ids = nllb_tokenizer(text, truncation=True, max_length=512).input_ids
        src_tokens = nllb_tokenizer.convert_ids_to_tokens(input_ids)
        results = nllb_translator.translate_batch(
            [src_tokens],
            target_prefix=[[tgt_nllb]],
            beam_size=1,
            max_decoding_length=128,
        )
        hyp_tokens = results[0].hypotheses[0][1:]
        tgt_ids = nllb_tokenizer.convert_tokens_to_ids(hyp_tokens)
        return nllb_tokenizer.decode(tgt_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"[nllb] error: {e}", flush=True)
        return text
