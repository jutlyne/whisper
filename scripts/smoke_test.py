import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from services.audio_capture import AudioCaptureService
from services.diarization_service import DiarizationService
from services.runtime_paths import PYTHON_MODELS_DIR
from services.session_store import SessionStore
from services.whisper_service import WhisperService


def main() -> None:
    store = SessionStore()
    devices = AudioCaptureService().list_audio_devices()
    diarization = DiarizationService().probe()
    whisper = WhisperService()

    whisper.ensure_runtime()

    print(f"Python storage: {store.build_final_audio_path('smoke').parent.parent}")
    print(f"Whisper cache: {PYTHON_MODELS_DIR}")
    print(f"Detected microphones: {len(devices.get('microphones', []))}")
    print(f"Detected system devices: {len(devices.get('systems', []))}")
    print(f"Diarization available: {diarization.available}")
    print("Python smoke test passed.")


if __name__ == "__main__":
    main()
