import shutil
from pathlib import Path


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(input_path), str(output_path))
    return output_path
