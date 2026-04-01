from pathlib import Path
import sys

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.whisper_service import WhisperService


def main() -> None:
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1]).resolve()
    else:
        recordings_dir = (
            Path(__file__).resolve().parents[1]
            / "storage"
            / "recordings"
        )
        audio_files = sorted(
            recordings_dir.glob("*.wav"),
            key=lambda target: target.stat().st_mtime,
            reverse=True,
        )
        if not audio_files:
            raise RuntimeError(
                "Khong tim thay WAV trong python/storage/recordings. "
                "Hay truyen duong dan file vao script."
            )
        audio_path = audio_files[0]

    transcript = WhisperService().generate_transcript(
        audio_path=str(audio_path),
        language="auto",
    )
    print(f"Audio: {audio_path}")
    print(f"Language: {transcript['language']}")
    print(f"Segments: {len(transcript['utterances'])}")
    print(f"Preview: {transcript['text'][:240]}")


if __name__ == "__main__":
    main()
