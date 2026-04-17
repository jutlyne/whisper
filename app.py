import ctypes
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# PyInstaller: sys._MEIPASS is the temp extract dir, but .env lives next to the exe
if getattr(sys, "frozen", False):
    _app_dir = Path(sys.executable).parent
else:
    _app_dir = Path(__file__).parent
load_dotenv(_app_dir / ".env")

# Must run before any library (soundcard/WASAPI) can initialize COM as MTA.
# Qt needs STA (COINIT_APARTMENTTHREADED) for OLE clipboard & shell operations.
# CoInitialize(None) == CoInitializeEx(None, COINIT_APARTMENTTHREADED).
ctypes.windll.ole32.CoInitialize(None)

# WASAPI loopback thỉnh thoảng báo discontinuity khi CPU bận — không ảnh hưởng ghi âm.
warnings.filterwarnings("ignore", message="data discontinuity", category=RuntimeWarning)

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402  (imports soundcard here)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
