import os
from dataclasses import dataclass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


def _patch_torchaudio() -> None:
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda x: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def _patch_symlink() -> None:
    import os, shutil
    if not hasattr(os, "_orig_symlink"):
        os._orig_symlink = os.symlink
        def _safe_symlink(src, dst, *args, **kwargs):
            try:
                os._orig_symlink(src, dst, *args, **kwargs)
            except OSError:
                if os.path.isdir(str(src)):
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
        os.symlink = _safe_symlink


@dataclass
class DiarizationProbe:
    available: bool
    reason: str


class DiarizationService:
    def __init__(self) -> None:
        self._encoder = None

    def probe(self, **_) -> DiarizationProbe:
        try:
            _patch_torchaudio()
            _patch_symlink()
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: F401
            from sklearn.cluster import SpectralClustering  # noqa: F401
        except Exception as error:
            return DiarizationProbe(
                available=False,
                reason=f"Khong the nap speechbrain/sklearn: {error}",
            )
        return DiarizationProbe(
            available=True,
            reason="ECAPA-TDNN speaker embeddings san sang.",
        )

    def _get_encoder(self):
        if self._encoder is None:
            _patch_torchaudio()
            _patch_symlink()
            from speechbrain.inference.speaker import EncoderClassifier
            from .runtime_paths import PYTHON_MODELS_DIR
            self._encoder = EncoderClassifier.from_hparams(
                source=_ECAPA_MODEL_ID,
                savedir=str(PYTHON_MODELS_DIR / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
        return self._encoder

    def run_diarization(
        self,
        audio_path: str,
        utterances: list[dict] | None = None,
        n_speakers: int | None = None,
        **_,
    ) -> dict:
        probe = self.probe()
        if not probe.available:
            return {
                "available": False,
                "audioPath": audio_path,
                "reason": probe.reason,
                "segments": [],
            }

        try:
            import torch
            import numpy as np
            import librosa
            from sklearn.preprocessing import normalize
            from sklearn.metrics.pairwise import cosine_similarity

            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            encoder = self._get_encoder()

            if not utterances:
                return {
                    "available": True,
                    "audioPath": audio_path,
                    "reason": "Khong co utterances de diarize.",
                    "segments": [],
                }

            embeddings: list[np.ndarray | None] = []
            for utt in utterances:
                start = int(utt["start"] / 1000 * sr)
                end = int(utt["end"] / 1000 * sr)
                chunk = wav[start:end]
                if len(chunk) < sr * 0.5:
                    embeddings.append(None)
                    continue

                # Tile to minimum 2s
                min_len = sr * 2
                if len(chunk) < min_len:
                    repeats = -(-min_len // len(chunk))
                    chunk = np.tile(chunk, repeats)[:min_len]

                # ECAPA embedding
                waveform = torch.tensor(chunk).unsqueeze(0)
                with torch.no_grad():
                    embed = encoder.encode_batch(waveform)
                ecapa = embed.squeeze().numpy()

                # Pitch statistics (helps separate male/female and voice types)
                f0, _, _ = librosa.pyin(
                    chunk[:sr * 2],
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr,
                )
                valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
                pitch_feat = np.array([
                    np.mean(valid_f0) if len(valid_f0) else 0.0,
                    np.std(valid_f0) if len(valid_f0) else 0.0,
                ])

                # MFCC statistics
                mfcc = librosa.feature.mfcc(y=chunk[:sr * 2], sr=sr, n_mfcc=13)
                mfcc_feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

                # Combine: ECAPA (192-d) + pitch (2-d) + MFCC stats (26-d)
                combined = np.concatenate([
                    normalize(ecapa.reshape(1, -1))[0],
                    pitch_feat / (np.linalg.norm(pitch_feat) + 1e-8),
                    mfcc_feat / (np.linalg.norm(mfcc_feat) + 1e-8),
                ])
                embeddings.append(combined)

            valid_indices = [i for i, e in enumerate(embeddings) if e is not None]

            if len(valid_indices) < 2:
                return {
                    "available": True,
                    "audioPath": audio_path,
                    "reason": "Qua it doan audio de phan biet nguoi noi.",
                    "segments": [
                        {"id": utt["id"], "start": utt["start"], "end": utt["end"], "speaker": "Speaker 1"}
                        for utt in utterances
                    ],
                }

            X = np.array([embeddings[i] for i in valid_indices])

            if n_speakers is None:
                n_speakers = self._estimate_n_speakers(X)

            if n_speakers == 1:
                labels = np.zeros(len(X), dtype=int)
            else:
                # Spectral clustering with cosine affinity
                affinity = cosine_similarity(X)
                affinity = np.clip(affinity, 0, 1)
                from sklearn.cluster import SpectralClustering
                labels = SpectralClustering(
                    n_clusters=n_speakers,
                    affinity="precomputed",
                    random_state=0,
                    n_init=10,
                ).fit_predict(affinity)

            label_map = {idx: int(lbl) for idx, lbl in zip(valid_indices, labels)}
            segments = []
            for i, utt in enumerate(utterances):
                if i in label_map:
                    speaker_idx = label_map[i]
                else:
                    nearest = min(valid_indices, key=lambda x: abs(x - i))
                    speaker_idx = label_map[nearest]
                segments.append({
                    "id": utt["id"],
                    "start": utt["start"],
                    "end": utt["end"],
                    "speaker": f"Speaker {speaker_idx + 1}",
                })

            n_found = len(set(labels))
            return {
                "available": True,
                "audioPath": audio_path,
                "reason": f"Diarization thanh cong ({n_found} nguoi noi).",
                "segments": segments,
            }

        except Exception as error:
            return {
                "available": False,
                "audioPath": audio_path,
                "reason": str(error),
                "segments": [],
            }

    @staticmethod
    def _estimate_n_speakers(X: "np.ndarray", max_speakers: int = 6) -> int:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        SILHOUETTE_MIN = 0.05
        best_n = 1
        best_score = SILHOUETTE_MIN

        affinity = np.clip(cosine_similarity(X), 0, 1)

        for n in range(2, min(max_speakers + 1, len(X))):
            try:
                labels = SpectralClustering(
                    n_clusters=n,
                    affinity="precomputed",
                    random_state=0,
                    n_init=10,
                ).fit_predict(affinity)
                score = silhouette_score(X, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception:
                continue

        return best_n
