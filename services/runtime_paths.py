import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    PYTHON_DIR = Path(sys.executable).resolve().parent
else:
    PYTHON_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PYTHON_DIR.parent

PYTHON_STORAGE_DIR = PYTHON_DIR / "storage"
PYTHON_RECORDINGS_DIR = PYTHON_STORAGE_DIR / "recordings"
PYTHON_METADATA_DIR = PYTHON_STORAGE_DIR / "metadata"
PYTHON_TRANSCRIPTS_DIR = PYTHON_STORAGE_DIR / "transcripts"
PYTHON_TEMP_DIR = PYTHON_STORAGE_DIR / "temp"
PYTHON_MODELS_DIR = PYTHON_DIR / "models"
