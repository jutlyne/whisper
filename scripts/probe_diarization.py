import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from services.diarization_service import DiarizationService


def main() -> None:
    probe = DiarizationService().probe()
    print(f"Available: {probe.available}")
    print(f"Reason: {probe.reason}")


if __name__ == "__main__":
    main()
