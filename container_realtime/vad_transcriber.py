"""
VAD-based realtime transcription using HuggingFace transformers.
Stateful per WebSocket session.
"""
from __future__ import annotations

import numpy as np
import torch

# VAD parameters
_SPEECH_THRESHOLD_MIN   = 0.002
_SPEECH_MULTIPLIER      = 4.0
_NOISE_PERCENTILE       = 20
_HISTORY_SIZE           = 100
_SILENCE_TRIGGER_CHUNKS = 5
_MIN_SPEECH_CHUNKS      = 4
_MAX_SPEECH_CHUNKS      = 150

SAMPLE_RATE = 16_000


class VadTranscriber:
    """
    One instance per WebSocket connection.
    Call feed(pcm_float32_array) for each 100ms chunk.
    Returns text (in output_lang) when a speech segment ends, else None.

    Strategy:
      - Whisper task="transcribe" with auto language detection → source text
      - If detected_lang == output_lang: return directly (no network call)
      - Else: GoogleTranslator detected_lang → output_lang
    """

    def __init__(self, processor, model, device: str, output_lang: str = "vi") -> None:
        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = torch.float16 if "cuda" in device else torch.float32
        self._output_lang = output_lang
        self._speech_buffer: list[np.ndarray] = []
        self._silence_count = 0
        self._in_speech = False
        self._rms_history: list[float] = []
        self._adaptive_threshold = _SPEECH_THRESHOLD_MIN * 3

    def feed(self, chunk: np.ndarray) -> str | None:
        """Process one 100ms PCM chunk. Returns text or None."""
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        self._update_threshold(energy)
        is_speech = energy > self._adaptive_threshold

        if is_speech:
            self._silence_count = 0
            self._in_speech = True
            self._speech_buffer.append(chunk)
            if len(self._speech_buffer) >= _MAX_SPEECH_CHUNKS:
                return self._transcribe_and_reset()
        else:
            if self._in_speech:
                self._speech_buffer.append(chunk)
                self._silence_count += 1
                if self._silence_count >= _SILENCE_TRIGGER_CHUNKS:
                    if len(self._speech_buffer) >= _MIN_SPEECH_CHUNKS:
                        return self._transcribe_and_reset()
                    self._reset()
        return None

    def flush(self) -> str | None:
        """Called when recording stops — transcribe any remaining buffer."""
        if self._in_speech and len(self._speech_buffer) >= _MIN_SPEECH_CHUNKS:
            return self._transcribe_and_reset()
        self._reset()
        return None

    def _update_threshold(self, energy: float) -> None:
        self._rms_history.append(energy)
        if len(self._rms_history) > _HISTORY_SIZE:
            self._rms_history.pop(0)
        if len(self._rms_history) >= 20:
            sorted_rms = sorted(self._rms_history)
            noise_floor = sorted_rms[int(len(sorted_rms) * _NOISE_PERCENTILE / 100)]
            self._adaptive_threshold = max(_SPEECH_THRESHOLD_MIN, noise_floor * _SPEECH_MULTIPLIER)

    def _transcribe_and_reset(self) -> str | None:
        audio = np.concatenate(self._speech_buffer).astype(np.float32)
        self._reset()

        # Step 1: Whisper transcribe — auto-detect source language
        try:
            inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(self._device, dtype=self._dtype)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_features,
                    task="transcribe",
                    return_timestamps=False,
                )

            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if not text:
                return None

            # Detect language from text (more reliable than Whisper token for short audio)
            detected_lang = self._detect_language(text)
        except Exception as e:
            print(f"[transcribe] error: {e}", flush=True)
            return None

        print(f"[transcribe] detected={detected_lang} target={self._output_lang} text={text[:80]!r}", flush=True)

        # Dịch detected_lang → output_lang (do client chọn) nếu khác nhau
        out_text = text
        if detected_lang != self._output_lang:
            out_text = self._translate(text, detected_lang, self._output_lang)

        # Prefix [detected_lang] để client biết ai đang nói
        return f"[{detected_lang}]{out_text}"

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language from text using lingua (high accuracy for short text)."""
        try:
            from lingua import Language, LanguageDetectorBuilder
            _LANG_MAP = {
                Language.VIETNAMESE: "vi",
                Language.JAPANESE: "ja",
                Language.CHINESE: "zh",
                Language.ENGLISH: "en",
                Language.KOREAN: "ko",
                Language.FRENCH: "fr",
                Language.GERMAN: "de",
                Language.SPANISH: "es",
                Language.THAI: "th",
            }
            detector = LanguageDetectorBuilder.from_languages(*_LANG_MAP.keys()).build()
            result = detector.detect_language_of(text)
            lang = _LANG_MAP.get(result, "")
            if lang:
                return lang
        except Exception as e:
            print(f"[lang-detect] lingua failed: {e}", flush=True)

        # Fallback: simple heuristic based on Unicode ranges
        vi_chars = sum(1 for c in text if '\u00C0' <= c <= '\u01B0' or '\u1EA0' <= c <= '\u1EF9')
        ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FFF')
        ko_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7AF')
        zh_chars = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')

        scores = {"vi": vi_chars, "ja": ja_chars, "ko": ko_chars, "zh": zh_chars}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "en"

    @staticmethod
    def _translate(text: str, source: str, target: str) -> str:
        """Translate text using deep_translator, fallback to original on error."""
        try:
            from deep_translator import GoogleTranslator
            result = GoogleTranslator(source=source, target=target).translate(text)
            if result:
                print(f"[translate] {source}→{target}: {result[:80]!r}", flush=True)
                return result
        except Exception as e:
            print(f"[translate] deep_translator failed ({source}→{target}): {e}", flush=True)

        # Fallback: Google Cloud Translation v2 (available on Cloud Run with default SA)
        try:
            from google.cloud import translate_v2 as gtranslate
            client = gtranslate.Client()
            result = client.translate(text, source_language=source, target_language=target)
            translated = result.get("translatedText", "")
            if translated:
                print(f"[translate-gcloud] {source}→{target}: {translated[:80]!r}", flush=True)
                return translated
        except Exception as e:
            print(f"[translate-gcloud] also failed: {e}", flush=True)

        return text

    def _reset(self) -> None:
        self._speech_buffer.clear()
        self._silence_count = 0
        self._in_speech = False
