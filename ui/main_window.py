import ctypes
import os
import shutil
import threading
from pathlib import Path

import numpy as np

from PySide6.QtCore import Property, Qt, QEvent, QPropertyAnimation, QSize, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.audio_capture import AudioCaptureService
from services.audio_processing import normalize_audio
from services.cloud_run_service import CloudRunService, JobConfig
from services.realtime_cloud_service import RealtimeCloudService
from services.runtime_paths import PYTHON_RECORDINGS_DIR
from services.session_store import SessionStore
from ui.job_status_widget import JobStatusWidget
from ui.live_captions_widget import LiveCaptionsWidget
from ui.vu_meter_widget import VUMeterWidget

# Custom roles for transcript items
_ACTIVE_ROLE = Qt.ItemDataRole.UserRole + 100
_COLOR_ROLE = Qt.ItemDataRole.UserRole + 101

_HIGHLIGHT_BG = QColor("#f59e0b")
_HIGHLIGHT_FG = QColor("#0f1117")

_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_kernel32 = ctypes.windll.kernel32
_user32 = ctypes.windll.user32
_kernel32.GlobalAlloc.restype = ctypes.c_void_p
_kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
_kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
_user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]


def _win32_copy(text: str) -> None:
    """Copy text to clipboard via Win32 API, bypassing Qt's OLE clipboard."""
    data = text.encode("utf-16-le") + b"\x00\x00"
    h = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
    if h:
        p = _kernel32.GlobalLock(h)
        if p:
            ctypes.memmove(p, data, len(data))
            _kernel32.GlobalUnlock(h)
        if _user32.OpenClipboard(0):
            _user32.EmptyClipboard()
            _user32.SetClipboardData(_CF_UNICODETEXT, h)
            _user32.CloseClipboard()


class _TranscriptDelegate(QStyledItemDelegate):
    """Custom delegate — bypasses Qt stylesheet so highlight always works."""

    edit_committed = Signal(int, str)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setStyleSheet(
            "QLineEdit { background: #1e2130; color: #e2e8f0; border: 1px solid #5865f2;"
            " border-radius: 4px; padding: 4px 8px; font-size: 13px; }"
        )
        editor.installEventFilter(self)
        return editor

    def eventFilter(self, obj, event):
        if isinstance(obj, QLineEdit) and event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Copy):
                selected = obj.selectedText()
                if selected:
                    _win32_copy(selected)
                return True
        return super().eventFilter(obj, event)

    def setEditorData(self, editor, index):
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        # Display format: "[ts]  Speaker:  text" — extract the text part after ":  "
        if ":  " in text:
            text = text.split(":  ", 1)[1]
        editor.setText(text)

    def setModelData(self, editor, model, index):
        new_text = editor.text().strip()
        if not new_text:
            return
        display = index.data(Qt.ItemDataRole.DisplayRole) or ""
        # Rebuild: keep prefix (everything up to and including ":  "), replace text part
        if ":  " in display:
            prefix = display.split(":  ", 1)[0] + ":  "
            model.setData(index, prefix + new_text, Qt.ItemDataRole.DisplayRole)
        else:
            model.setData(index, new_text, Qt.ItemDataRole.DisplayRole)
        self.edit_committed.emit(index.row(), new_text)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def paint(self, painter: QPainter, option, index) -> None:
        is_active: bool = bool(index.data(_ACTIVE_ROLE))
        color_hex: str = index.data(_COLOR_ROLE) or "#cbd5e1"
        text: str = index.data(Qt.ItemDataRole.DisplayRole) or ""

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = option.rect.adjusted(4, 2, -4, -2)
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()), float(r.width()), float(r.height()), 6.0, 6.0)

        if is_active:
            painter.fillPath(path, _HIGHLIGHT_BG)
            painter.setPen(_HIGHLIGHT_FG)
        elif selected:
            painter.fillPath(path, QColor("#5865f2"))
            painter.setPen(QColor("#ffffff"))
        elif hovered:
            painter.fillPath(path, QColor("#1e2130"))
            painter.setPen(QColor(color_hex))
        else:
            painter.setPen(QColor(color_hex))

        painter.setFont(option.font)
        painter.drawText(
            r.adjusted(10, 4, -10, -4),
            Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
            text,
        )
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        w = option.rect.width() if option.rect.width() > 0 else 700
        bound = option.fontMetrics.boundingRect(
            0, 0, w - 24, 0, Qt.TextFlag.TextWordWrap, text
        )
        return QSize(w, bound.height() + 20)

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #2d3044;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

QLabel {
    color: #cbd5e1;
    font-size: 12px;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #1e2130;
    border: 1px solid #2d3044;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e2e8f0;
    selection-background-color: #5865f2;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #5865f2;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #94a3b8;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #1e2130;
    border: 1px solid #2d3044;
    selection-background-color: #5865f2;
    outline: none;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: transparent;
    border: none;
    width: 16px;
}

QPushButton {
    background-color: #2d3044;
    color: #e2e8f0;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #3a3f5c;
}
QPushButton:pressed {
    background-color: #252840;
}
QPushButton:disabled {
    background-color: #1a1d2e;
    color: #4a5066;
}

QPushButton#btn_start {
    background-color: #23a55a;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#btn_start:hover  { background-color: #1e8f4d; }
QPushButton#btn_start:disabled { background-color: #1a3d28; color: #2e6644; }

QPushButton#btn_stop {
    background-color: #f23f43;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#btn_stop:hover  { background-color: #d93538; }
QPushButton#btn_stop:disabled { background-color: #3d1a1b; color: #7a3436; }

QPushButton#btn_play {
    background-color: #5865f2;
    color: #ffffff;
    font-weight: 600;
    min-width: 90px;
}
QPushButton#btn_play:hover  { background-color: #4752c4; }
QPushButton#btn_play:disabled { background-color: #1e2254; color: #3d4580; }

QPushButton#btn_delete {
    background-color: #7f1d1d;
    color: #fca5a5;
    font-weight: 600;
}
QPushButton#btn_delete:hover  { background-color: #991b1b; }
QPushButton#btn_delete:disabled { background-color: #1a0a0a; color: #4a2020; }

QPushButton#btn_transcribe {
    background-color: #0ea5e9;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#btn_transcribe:hover  { background-color: #0284c7; }
QPushButton#btn_transcribe:disabled { background-color: #0c2e3e; color: #1a5a78; }

QPushButton#btn_import {
    background-color: #7c3aed;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#btn_import:hover  { background-color: #6d28d9; }
QPushButton#btn_import:disabled { background-color: #2e1a4a; color: #5b3a8a; }

QPushButton#btn_export {
    background-color: #059669;
    color: #ffffff;
    font-weight: 600;
    padding: 4px 10px;
    font-size: 12px;
}
QPushButton#btn_export:hover  { background-color: #047857; }
QPushButton#btn_export:disabled { background-color: #052e1e; color: #1a5a3a; }

QProgressBar {
    background: #1e2130;
    border: 1px solid #2d3044;
    border-radius: 6px;
    color: #e2e8f0;
    text-align: center;
    font-size: 12px;
    max-height: 18px;
}
QProgressBar::chunk {
    background: #0ea5e9;
    border-radius: 5px;
}

QListWidget {
    background-color: #13161f;
    border: 1px solid #2d3044;
    border-radius: 8px;
    outline: none;
    padding: 4px;
}
/* QListWidget items are painted entirely by _TranscriptDelegate — no item rules here */

QTextEdit {
    background-color: #13161f;
    border: 1px solid #2d3044;
    border-radius: 8px;
    padding: 8px;
    color: #94a3b8;
    font-size: 12px;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #2d3044;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #5865f2;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #ffffff;
    border-radius: 7px;
}
QSlider::handle:horizontal:disabled {
    background: #3a3f5c;
}

QSplitter::handle {
    background-color: #2d3044;
    width: 1px;
}

QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #13161f;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2d3044;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""

# Speaker colors (cycling)
_SPEAKER_COLORS = [
    QColor("#60a5fa"),  # blue
    QColor("#f87171"),  # red
    QColor("#34d399"),  # emerald
    QColor("#fbbf24"),  # amber
    QColor("#a78bfa"),  # violet
    QColor("#fb923c"),  # orange
    QColor("#38bdf8"),  # sky
    QColor("#f472b6"),  # pink
]


class MainWindow(QMainWindow):
    _stop_success = Signal(str)
    _stop_failed = Signal(str)
    _uploading = Signal()               # GCS finalize in progress
    _live_caption = Signal(str)
    _live_caption_error = Signal(str)
    _live_caption_ready = Signal()
    _import_success = Signal(str, str)
    _import_failed = Signal(str)
    _import_upload_progress = Signal(int, int)  # bytes_uploaded, total_bytes
    _audio_levels = Signal(float, float)
    # Cloud Run Job signals
    _job_submitted = Signal(str)        # session_id
    _job_submit_failed = Signal(str)    # error message
    _job_done = Signal(str)             # session_id
    _job_poll_failed = Signal(str, str) # session_id, error message

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Meeting Recorder")
        self.resize(1280, 860)
        self.setStyleSheet(_STYLESHEET)

        self.store = SessionStore()
        self.audio_capture = AudioCaptureService()
        self.cloud_run = CloudRunService()
        self._gcs_upload = None   # GCSStreamingUpload, active during recording
        self._realtime_svc: RealtimeCloudService | None = None
        self._poll_timers: dict[str, QTimer] = {}
        self._poll_fail_counts: dict[str, int] = {}
        self.active_session_id: str | None = None

        # Audio player
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(1.0)
        self._utterances: list[dict] = []
        self._highlighted_row: int = -1
        self._row_speaker_colors: dict[int, QColor] = {}
        self._speaker_index: dict[str, int] = {}

        self.__progress_value = 0

        # Animation timers
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._pulse_state = False
        self._dots_timer = QTimer()
        self._dots_timer.timeout.connect(self._dots_tick)
        self._dots_count = 0
        self._dots_label = ""

        self._init_widgets()
        self._connect_signals()
        self._build_layout()

        self._progress_anim = QPropertyAnimation(self, b"_progress_value")
        self._progress_anim.setDuration(300)

        self.refresh_devices()
        self.load_sessions()
        # Resume polling for any jobs that were in-flight when app was last closed
        for session in self.store.list_pending_jobs():
            self._start_polling(session["id"])

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _init_widgets(self) -> None:
        # Recording
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Meeting title...")

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["mixed", "microphoneOnly", "systemOnly"])

        self.microphone_combo = QComboBox()
        self.system_combo = QComboBox()

        # Cloud Transcription settings
        self.output_language_combo = QComboBox()
        self.output_language_combo.addItem("Tiếng Việt", "vi")
        self.output_language_combo.addItem("日本語", "ja")

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText(
            "e.g. .NET, React, junior, senior, CV, part-time"
        )

        self.job_status_widget = JobStatusWidget()

        # Action buttons
        self.refresh_button = QPushButton("⟳  Refresh")
        self.start_button = QPushButton("⏺  Record")
        self.start_button.setObjectName("btn_start")
        self.stop_button = QPushButton("⏹  Stop")
        self.stop_button.setObjectName("btn_stop")
        self.stop_button.setEnabled(False)
        self.transcribe_button = QPushButton("✦  Transcribe")
        self.transcribe_button.setObjectName("btn_transcribe")

        self.import_button = QPushButton("📂  Import")
        self.import_button.setObjectName("btn_import")

        self.rename_button = QPushButton("✎  Rename")
        self.rename_button.setEnabled(False)
        self.delete_button = QPushButton("🗑  Delete")
        self.delete_button.setObjectName("btn_delete")
        self.delete_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.export_button = QPushButton("⎘  Copy")
        self.export_button.setObjectName("btn_export")
        self.export_button.setEnabled(False)

        # Status / notes
        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")
        self.note_label = QLabel(
            "Select a capture strategy and press Record to begin."
        )
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: #64748b; font-size: 12px;")

        # Sessions list
        self.sessions_list = QListWidget()
        self.sessions_list.setFont(QFont("Segoe UI", 12))

        # Playback controls
        self.play_pause_button = QPushButton("▶  Play")
        self.play_pause_button.setObjectName("btn_play")
        self.play_pause_button.setEnabled(False)

        self.stop_playback_button = QPushButton("⏹")
        self.stop_playback_button.setEnabled(False)
        self.stop_playback_button.setFixedWidth(36)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("color: #64748b; font-size: 12px; min-width: 90px;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Transcript — delegate handles all painting so stylesheet can't interfere
        self.transcript_list = QListWidget()
        self.transcript_list.setWordWrap(True)
        self.transcript_list.setSpacing(1)
        self.transcript_list.setMouseTracking(True)
        self.transcript_list.viewport().setMouseTracking(True)
        self.transcript_list.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self._delegate = _TranscriptDelegate(self.transcript_list)
        self.transcript_list.setItemDelegate(self._delegate)

        # Errors
        self.error_output = QTextEdit()
        self.error_output.setReadOnly(True)
        self.error_output.setMaximumHeight(90)
        self.error_output.setPlaceholderText("No errors.")

        # Audio level meter + live captions
        self.vu_meter = VUMeterWidget()
        self.live_captions_check = QCheckBox("Live captions (Cloud)")
        self.live_captions = LiveCaptionsWidget()

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.import_button.clicked.connect(self.import_audio)
        self.transcribe_button.clicked.connect(self.submit_transcription_job)
        self.export_button.clicked.connect(self._export_clipboard)
        self.transcript_list.installEventFilter(self)
        self.rename_button.clicked.connect(self.rename_session)
        self.delete_button.clicked.connect(self.delete_session)
        self.sessions_list.currentItemChanged.connect(self.render_selected_session)
        self._stop_success.connect(self._on_stop_success)
        self._stop_failed.connect(self._on_stop_failed)
        self._uploading.connect(self._on_uploading)
        self._import_success.connect(self._on_import_success)
        self._import_failed.connect(self._on_import_failed)
        self._import_upload_progress.connect(self._on_import_upload_progress)
        self._job_submitted.connect(self._on_job_submitted)
        self._job_submit_failed.connect(self._on_job_submit_failed)
        self._job_done.connect(self._on_job_done)
        self._job_poll_failed.connect(self._on_job_poll_failed)
        self._delegate.edit_committed.connect(self._on_transcript_edit)

        self.play_pause_button.clicked.connect(self._toggle_play_pause)
        self.stop_playback_button.clicked.connect(self._stop_playback)
        self.seek_slider.sliderMoved.connect(self._player.setPosition)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._audio_levels.connect(self.vu_meter.update_levels)
        self._live_caption.connect(self.live_captions.append_text)
        self._live_caption_error.connect(lambda msg: self._set_error(f"[Live] {msg}"))
        self._live_caption_ready.connect(self._on_live_caption_ready)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # ---- Left panel ----
        left = QWidget()
        left.setMaximumWidth(340)
        left.setMinimumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(12)

        # App title
        title_lbl = QLabel("Meeting Recorder")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #e2e8f0;")
        left_layout.addWidget(title_lbl)

        # Recording group
        rec_group = QGroupBox("Recording")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setSpacing(6)
        rec_layout.addWidget(QLabel("Title"))
        rec_layout.addWidget(self.title_input)
        rec_layout.addWidget(QLabel("Capture strategy"))
        rec_layout.addWidget(self.strategy_combo)
        rec_layout.addWidget(QLabel("Microphone"))
        rec_layout.addWidget(self.microphone_combo)
        rec_layout.addWidget(QLabel("System audio"))
        rec_layout.addWidget(self.system_combo)
        rec_layout.addWidget(QLabel("Audio levels"))
        rec_layout.addWidget(self.vu_meter)
        rec_layout.addWidget(self.live_captions_check)
        left_layout.addWidget(rec_group)

        # Cloud Transcription group
        trans_group = QGroupBox("Cloud Transcription")
        trans_layout = QVBoxLayout(trans_group)
        trans_layout.setSpacing(6)

        out_lang_row = QHBoxLayout()
        out_lang_row.addWidget(QLabel("Ngôn ngữ output"))
        out_lang_row.addWidget(self.output_language_combo)
        trans_layout.addLayout(out_lang_row)

        trans_layout.addWidget(QLabel("Vocabulary hint"))
        trans_layout.addWidget(self.prompt_input)
        trans_layout.addWidget(self.job_status_widget)
        left_layout.addWidget(trans_group)

        # Action buttons
        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.start_button)
        btn_row1.addWidget(self.stop_button)
        left_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.refresh_button)
        btn_row2.addWidget(self.import_button)
        left_layout.addLayout(btn_row2)

        left_layout.addWidget(self.transcribe_button)
        left_layout.addWidget(self.progress_bar)

        btn_row3 = QHBoxLayout()
        btn_row3.addWidget(self.rename_button)
        btn_row3.addWidget(self.delete_button)
        left_layout.addLayout(btn_row3)

        # Status
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.note_label)
        left_layout.addStretch()

        # ---- Right panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 16, 16, 16)
        right_layout.setSpacing(10)

        # Sessions
        sessions_lbl = QLabel("Sessions")
        sessions_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #94a3b8;")
        right_layout.addWidget(sessions_lbl)
        self.sessions_list.setMaximumHeight(160)
        right_layout.addWidget(self.sessions_list)

        # Playback bar
        playback_frame = QFrame()
        playback_frame.setStyleSheet(
            "QFrame { background-color: #1a1d27; border-radius: 8px; }"
        )
        pb_layout = QHBoxLayout(playback_frame)
        pb_layout.setContentsMargins(10, 8, 10, 8)
        pb_layout.setSpacing(8)
        pb_layout.addWidget(self.play_pause_button)
        pb_layout.addWidget(self.stop_playback_button)
        pb_layout.addWidget(self.seek_slider, stretch=1)
        pb_layout.addWidget(self.time_label)
        right_layout.addWidget(playback_frame)
        right_layout.addWidget(self.live_captions)

        # Transcript header row
        transcript_header = QHBoxLayout()
        transcript_lbl = QLabel("Transcript")
        transcript_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #94a3b8;")
        transcript_header.addWidget(transcript_lbl)
        transcript_header.addStretch()
        transcript_header.addWidget(self.export_button)
        right_layout.addLayout(transcript_header)
        right_layout.addWidget(self.transcript_list, stretch=1)

        # Errors
        right_layout.addWidget(self.error_output)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    # Qt property required by QPropertyAnimation
    def _get_progress_value(self) -> int:
        return self.__progress_value

    def _set_progress_value(self, value: int) -> None:
        self.__progress_value = value
        self.progress_bar.setValue(value)

    _progress_value = Property(int, _get_progress_value, _set_progress_value)

    def _animate_progress(self, target: int) -> None:
        self._progress_anim.stop()
        self._progress_anim.setStartValue(self.__progress_value)
        self._progress_anim.setEndValue(target)
        self._progress_anim.start()

    def _flash_button(self, btn: QPushButton) -> None:
        """Brief opacity dip on click to confirm the action was received."""
        effect = QGraphicsOpacityEffect(btn)
        btn.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", btn)
        anim.setDuration(350)
        anim.setKeyValueAt(0.0, 1.0)
        anim.setKeyValueAt(0.25, 0.35)
        anim.setKeyValueAt(1.0, 1.0)
        anim.finished.connect(lambda: btn.setGraphicsEffect(None))
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _start_pulse(self, label_text: str, color_a: str, color_b: str) -> None:
        """Pulse the status label between two colours (e.g. recording blink)."""
        self._pulse_color_a = color_a
        self._pulse_color_b = color_b
        self._pulse_label = label_text
        self._pulse_state = False
        self._pulse_timer.start(600)

    def _stop_pulse(self) -> None:
        self._pulse_timer.stop()

    def _pulse_tick(self) -> None:
        self._pulse_state = not self._pulse_state
        color = self._pulse_color_a if self._pulse_state else self._pulse_color_b
        self.status_label.setText(f"● {self._pulse_label}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _start_dots(self, label_text: str, color: str) -> None:
        """Animated '...' suffix on the status label while a task is running."""
        self._dots_label = label_text
        self._dots_color = color
        self._dots_count = 0
        self._dots_timer.start(400)

    def _stop_dots(self) -> None:
        self._dots_timer.stop()

    def _dots_tick(self) -> None:
        self._dots_count = (self._dots_count + 1) % 4
        dots = "." * self._dots_count
        self.status_label.setText(f"● {self._dots_label}{dots}")
        self.status_label.setStyleSheet(
            f"color: {self._dots_color}; font-weight: 600;"
        )

    def _set_actions_locked(self, locked: bool) -> None:
        """Freeze/unfreeze all action buttons to prevent spam during async ops."""
        for btn in (
            self.start_button, self.import_button,
            self.transcribe_button, self.rename_button, self.delete_button,
        ):
            btn.setEnabled(not locked)
        # stop_button has its own separate state — don't touch it here

    def _set_error(self, message: str) -> None:
        self.error_output.setPlainText(message)

    def _set_note(self, message: str) -> None:
        self.note_label.setText(message)

    def _set_status(self, text: str, color: str = "#64748b") -> None:
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _selected_name(self, combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data.get("name", "")) if data else ""

    def _selected_id(self, combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data.get("id", "")) if data else ""

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        total_sec = max(ms, 0) // 1000
        minutes, secs = divmod(total_sec, 60)
        return f"{minutes}:{secs:02d}"

    def _speaker_color(self, speaker: str) -> QColor:
        if speaker not in self._speaker_index:
            self._speaker_index[speaker] = len(self._speaker_index)
        return _SPEAKER_COLORS[self._speaker_index[speaker] % len(_SPEAKER_COLORS)]

    # ------------------------------------------------------------------
    # Device refresh
    # ------------------------------------------------------------------

    def refresh_devices(self) -> None:
        self._set_error("")
        devices = self.audio_capture.list_audio_devices()
        microphones = devices.get("microphones", [])
        systems = devices.get("systems", [])

        self.microphone_combo.clear()
        self.system_combo.clear()

        if not microphones:
            self.microphone_combo.addItem("No microphone detected", None)
        if not systems:
            self.system_combo.addItem("No system audio device detected", None)

        if not microphones and not systems:
            self._set_note("No audio devices found.")
            return

        for device in microphones:
            self.microphone_combo.addItem(device["name"], device)
        for device in systems:
            self.system_combo.addItem(device["name"], device)

        self._set_note("Devices refreshed.")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def load_sessions(self) -> None:
        self.sessions_list.clear()
        for session in self.store.list_sessions():
            status_text = session["status"]
            job_status = session.get("jobStatus")
            if job_status and job_status not in ("done", "cancelled"):
                status_text += f" | job: {job_status}"
            item = QListWidgetItem(f"{session['title']}  [{status_text}]")
            item.setData(32, session["id"])
            self.sessions_list.addItem(item)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        self._flash_button(self.start_button)
        try:
            session = self.store.create_session(
                title=self.title_input.text().strip() or "Meeting",
                strategy=self.strategy_combo.currentText(),
                microphone_name=self._selected_name(self.microphone_combo),
                system_name=self._selected_name(self.system_combo),
            )
            session_id = session["id"]

            # Start GCS streaming upload if cloud config is present
            if self.cloud_run.gcs_bucket and self.cloud_run.project_id:
                try:
                    self._gcs_upload = self.cloud_run.start_streaming_upload(session_id)
                    self.store.update_session(
                        session_id,
                        gcsFolderUri=self.cloud_run.session_folder_uri(session_id),
                        jobStatus="uploading",
                    )
                except Exception as gcs_err:
                    self._set_error(f"GCS streaming upload failed to start: {gcs_err}")
                    self._gcs_upload = None

            # Start realtime cloud service if checkbox is on and URL is configured
            realtime_url = os.environ.get("REALTIME_SERVICE_URL", "").strip()
            if self.live_captions_check.isChecked() and realtime_url:
                self._realtime_svc = RealtimeCloudService(
                    realtime_url,
                    output_language=self.output_language_combo.currentData(),
                    on_text=lambda t: self._live_caption.emit(t),
                    on_error=lambda e: self._live_caption_error.emit(e),
                    on_ready=lambda: self._live_caption_ready.emit(),
                )
                self._realtime_svc.start()
                self.live_captions.start_session()

            def _chunk_callback(
                mic: "np.ndarray | None",
                sys_ch: "np.ndarray | None",
            ) -> None:
                mic_rms = float(np.sqrt(np.mean(mic ** 2))) if mic is not None else 0.0
                sys_rms = float(np.sqrt(np.mean(sys_ch ** 2))) if sys_ch is not None else 0.0
                self._audio_levels.emit(mic_rms, sys_rms)

                mixed_chunks = [c.flatten() for c in (mic, sys_ch) if c is not None]
                if not mixed_chunks:
                    return
                mixed = np.mean(np.stack(mixed_chunks), axis=0) if len(mixed_chunks) > 1 else mixed_chunks[0]

                # GCS streaming upload
                if self._gcs_upload is not None:
                    pcm = (mixed * 32767).clip(-32768, 32767).astype(np.int16)
                    self._gcs_upload.write(pcm.tobytes())

                # Realtime transcription
                if self._realtime_svc is not None:
                    self._realtime_svc.feed(mixed)

            self.audio_capture.set_chunk_callback(_chunk_callback)
            self.audio_capture.start_recording(
                strategy=self.strategy_combo.currentText(),
                microphone_id=self._selected_id(self.microphone_combo),
                system_id=self._selected_id(self.system_combo),
                output_path=Path(session["tempAudioPath"]),
            )
            self.active_session_id = session["id"]
            self._start_pulse("Recording", "#f23f43", "#7f1d1d")
            self._set_actions_locked(True)
            self.stop_button.setEnabled(True)
            self._set_note("Recording in progress...")
            self.load_sessions()
        except Exception as error:
            self._set_error(str(error))

    def stop_recording(self) -> None:
        if not self.active_session_id:
            return
        self._flash_button(self.stop_button)
        self._stop_pulse()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self._start_dots("Stopping", "#f59e0b")
        session_id = self.active_session_id
        self.active_session_id = None

        # Detach callback trước khi audio thread dừng
        self.audio_capture.set_chunk_callback(None)

        gcs_upload = self._gcs_upload
        self._gcs_upload = None

        def _do_stop() -> None:
            try:
                temp_path = self.audio_capture.stop_recording()
                final_path = self.store.build_final_audio_path(session_id)
                normalize_audio(temp_path, final_path)
                temp_path.unlink(missing_ok=True)
                # Finalize GCS streaming upload (flush remaining buffer)
                if gcs_upload is not None:
                    self._uploading.emit()
                    gcs_upload.finalize()
                self.store.update_session(
                    session_id,
                    status="processed",
                    finishedAt=self.store._timestamp(),
                    finalAudioPath=str(final_path),
                    jobStatus="uploaded" if gcs_upload is not None else None,
                )
                self._stop_success.emit(session_id)
            except Exception as error:
                self._stop_failed.emit(str(error))

        threading.Thread(target=_do_stop, daemon=True).start()

    def _on_uploading(self) -> None:
        self._stop_dots()
        self._start_dots("Uploading to GCS", "#22c55e")
        self.job_status_widget.set_status("uploading")

    def _stop_realtime_svc(self) -> None:
        if self._realtime_svc is not None:
            self._realtime_svc.stop()
            self._realtime_svc = None
        self.live_captions.stop_session()

    def _on_stop_success(self, session_id: str) -> None:
        self._stop_realtime_svc()
        self.vu_meter.reset()
        self._stop_dots()
        self._set_status("Idle", "#64748b")
        self._set_note("Recording saved. Ready to transcribe.")
        self.stop_button.setEnabled(False)
        self._set_actions_locked(False)
        self.job_status_widget.set_status(None)
        self.load_sessions()
        self._select_session(session_id)

    def _on_stop_failed(self, message: str) -> None:
        self._stop_realtime_svc()
        self.vu_meter.reset()
        self._stop_dots()
        self._set_error(message)
        self.stop_button.setEnabled(False)
        self._set_actions_locked(False)
        self._set_status("Idle", "#64748b")

    def _on_live_caption_ready(self) -> None:
        self._set_note("Live captions connected.")

    # ------------------------------------------------------------------
    # Cloud transcription
    # ------------------------------------------------------------------

    def _apply_cloud_config(self) -> bool:
        """Validate that env-based cloud config is present."""
        if not self.cloud_run.project_id or not self.cloud_run.gcs_bucket:
            self._set_error(
                "GCP config missing. Set GCP_PROJECT_ID, GCS_BUCKET, "
                "GCR_JOB_NAME and GOOGLE_APPLICATION_CREDENTIALS in .env"
            )
            return False
        return True

    def submit_transcription_job(self) -> None:
        item = self.sessions_list.currentItem()
        if not item:
            self._set_error("Select a session first.")
            return

        session_id = item.data(32)
        session = self.store.load_session(session_id)

        if not self._apply_cloud_config():
            return

        gcs_folder_uri = session.get("gcsFolderUri") or self.cloud_run.session_folder_uri(session_id)
        config = JobConfig(
            session_id=session_id,
            session_folder_uri=gcs_folder_uri,
            language="auto",
            output_language=self.output_language_combo.currentData(),

            initial_prompt=self.prompt_input.text().strip(),
        )

        self._flash_button(self.transcribe_button)
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self._start_dots("Submitting", "#0ea5e9")
        self.job_status_widget.set_status("submitted")

        def _do_submit() -> None:
            try:
                execution_name = self.cloud_run.submit_job(config)
                self.store.save_cloud_job(
                    session_id,
                    gcs_folder_uri=gcs_folder_uri,
                    execution_name=execution_name,
                )
                self._job_submitted.emit(session_id)
            except Exception as error:
                self._job_submit_failed.emit(str(error))

        threading.Thread(target=_do_submit, daemon=True).start()

    def _on_job_submitted(self, session_id: str) -> None:
        self._stop_dots()
        self.progress_bar.setRange(0, 0)   # keep indeterminate while job runs
        self._set_status("Transcribing", "#8b5cf6")
        self._set_note("Job submitted — waiting for Cloud Run to finish...")
        self.job_status_widget.set_status("running")
        self._set_actions_locked(True)
        self.load_sessions()
        self._start_polling(session_id)

    def _on_job_submit_failed(self, message: str) -> None:
        self._stop_dots()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self._set_error(f"Job submission failed: {message}")
        self._set_status("Idle", "#64748b")
        self.job_status_widget.set_status("error")
        self._set_actions_locked(False)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _start_polling(self, session_id: str) -> None:
        if session_id in self._poll_timers:
            return
        self._poll_fail_counts[session_id] = 0
        timer = QTimer(self)
        timer.setInterval(15_000)
        timer.timeout.connect(lambda: self._poll_once(session_id))
        self._poll_timers[session_id] = timer
        timer.start()
        # Poll immediately on first tick
        self._poll_once(session_id)

    def _stop_polling(self, session_id: str) -> None:
        timer = self._poll_timers.pop(session_id, None)
        if timer:
            timer.stop()
        self._poll_fail_counts.pop(session_id, None)

    def _poll_once(self, session_id: str) -> None:
        session = self.store.load_session(session_id)
        execution_name = session.get("executionName")
        if not execution_name:
            self._stop_polling(session_id)
            return
        threading.Thread(
            target=self._do_poll, args=(session_id, execution_name), daemon=True
        ).start()

    def _do_poll(self, session_id: str, execution_name: str) -> None:
        print(f"[poll] {session_id[:8]} execution={execution_name}")
        try:
            status = self.cloud_run.poll_execution(execution_name)
            self._poll_fail_counts[session_id] = 0

            if not status.done:
                self.store.update_job_status(session_id, job_status="running")
                return

            if status.succeeded:
                self._finalize_job(session_id)
            else:
                self.store.update_job_status(
                    session_id, job_status="error", error_message=status.error
                )
                self._job_poll_failed.emit(session_id, status.error or "Job failed")
        except Exception as exc:
            fail_count = self._poll_fail_counts.get(session_id, 0) + 1
            self._poll_fail_counts[session_id] = fail_count
            print(f"[poll] error #{fail_count}: {exc}")
            if fail_count >= 20:
                self._job_poll_failed.emit(
                    session_id,
                    f"Polling stopped after {fail_count} consecutive errors: {exc}",
                )
                self._stop_polling(session_id)

    def _finalize_job(self, session_id: str) -> None:
        """Download transcript and save to disk."""
        try:
            transcript = self.cloud_run.download_transcript(session_id)
            self.store.save_transcript(session_id, transcript)
            self.store.update_job_status(session_id, job_status="done")
            self._job_done.emit(session_id)
        except Exception as exc:
            self.store.update_job_status(
                session_id, job_status="error", error_message=str(exc)
            )
            self._job_poll_failed.emit(session_id, str(exc))

    def _on_job_done(self, session_id: str) -> None:
        self._stop_polling(session_id)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self._set_status("Idle", "#64748b")
        self._set_note("Transcription complete.")
        self.job_status_widget.set_status("done")
        self._set_actions_locked(False)
        self.load_sessions()
        self._select_session(session_id)

    def _on_job_poll_failed(self, session_id: str, message: str) -> None:
        self._stop_polling(session_id)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self._set_error(f"[{session_id[:8]}] {message}")
        self._set_status("Idle", "#64748b")
        self.job_status_widget.set_status("error")
        self._set_actions_locked(False)
        self.load_sessions()

    # ------------------------------------------------------------------
    # Render session
    # ------------------------------------------------------------------

    def render_selected_session(self) -> None:
        self._stop_playback()
        item = self.sessions_list.currentItem()
        has_selection = item is not None
        self.rename_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        if not item:
            self._clear_transcript()
            self.export_button.setEnabled(False)
            return

        session = self.store.load_session(item.data(32))
        self.job_status_widget.set_status(session.get("jobStatus"))

        audio_path = session.get("finalAudioPath")
        has_audio = audio_path and Path(audio_path).exists()
        self._player.setSource(
            QUrl.fromLocalFile(audio_path) if has_audio else QUrl()
        )
        for w in (self.play_pause_button, self.stop_playback_button, self.seek_slider):
            w.setEnabled(bool(has_audio))

        transcript_path = session.get("transcriptPath")
        if not transcript_path:
            self._clear_transcript()
            self.export_button.setEnabled(False)
            item = QListWidgetItem("No transcript yet.")
            item.setForeground(QColor("#4a5066"))
            self.transcript_list.addItem(item)
            return

        self.export_button.setEnabled(True)

        transcript = self.store.read_transcript(transcript_path)
        self._utterances = transcript.get("utterances", [])
        self._highlighted_row = -1
        self._speaker_index = {}
        self._row_speaker_colors = {}

        self.transcript_list.clear()
        for row, utt in enumerate(self._utterances):
            speaker = utt.get("speaker") or "Unknown"
            color = self._speaker_color(speaker)
            self._row_speaker_colors[row] = color

            ts = ""
            if utt.get("timestampFrom"):
                ts = f"[{utt['timestampFrom']}]  "

            list_item = QListWidgetItem(f"{ts}{speaker}:  {utt['text']}")
            list_item.setData(_ACTIVE_ROLE, False)
            list_item.setData(_COLOR_ROLE, color.name())
            list_item.setFlags(
                list_item.flags() | Qt.ItemFlag.ItemIsEditable
            )
            self.transcript_list.addItem(list_item)

    def _clear_transcript(self) -> None:
        self.transcript_list.clear()
        self._utterances = []
        self._highlighted_row = -1
        self._row_speaker_colors = {}

    def _select_session(self, session_id: str) -> None:
        for index in range(self.sessions_list.count()):
            item = self.sessions_list.item(index)
            if item.data(32) == session_id:
                self.sessions_list.setCurrentItem(item)
                return

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _stop_playback(self) -> None:
        self._player.stop()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_pause_button.setText("⏸  Pause" if playing else "▶  Play")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.seek_slider.setMaximum(duration_ms)
        self.time_label.setText(f"0:00 / {self._fmt_ms(duration_ms)}")

    def _on_position_changed(self, position_ms: int) -> None:
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position_ms)

        self.time_label.setText(
            f"{self._fmt_ms(position_ms)} / {self._fmt_ms(self._player.duration())}"
        )

        # Find active utterance
        active_row = -1
        for i, utt in enumerate(self._utterances):
            if utt.get("start", 0) <= position_ms < utt.get("end", 0):
                active_row = i
                break

        if active_row == self._highlighted_row:
            return

        # Remove old highlight
        if self._highlighted_row >= 0:
            old_item = self.transcript_list.item(self._highlighted_row)
            if old_item:
                old_item.setData(_ACTIVE_ROLE, False)

        if active_row >= 0:
            new_item = self.transcript_list.item(active_row)
            if new_item:
                new_item.setData(_ACTIVE_ROLE, True)
                self.transcript_list.scrollToItem(
                    new_item, QListWidget.ScrollHint.PositionAtCenter
                )

        self._highlighted_row = active_row

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def rename_session(self) -> None:
        item = self.sessions_list.currentItem()
        if not item:
            return
        session_id = item.data(32)
        session = self.store.load_session(session_id)
        new_title, ok = QInputDialog.getText(
            self,
            "Rename session",
            "New title:",
            text=session.get("title", ""),
        )
        if ok and new_title.strip():
            self.store.rename_session(session_id, new_title.strip())
            self.load_sessions()
            self._select_session(session_id)

    def delete_session(self) -> None:
        item = self.sessions_list.currentItem()
        if not item:
            return
        session_id = item.data(32)
        session = self.store.load_session(session_id)
        reply = QMessageBox.question(
            self,
            "Delete session",
            f'Delete "{session.get("title", session_id)}"?\n\nAudio and transcript files will also be removed.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._stop_playback()
            self._player.setSource(QUrl())  # release file handle
            self._stop_polling(session_id)
            # Clean up GCS folder if configured
            gcs_folder = session.get("gcsFolderUri")
            if gcs_folder:
                try:
                    self.cloud_run.delete_session_folder(session_id)
                except Exception:
                    pass  # best-effort cleanup
            self.store.delete_session(session_id)
            self.load_sessions()
            self._clear_transcript()
            self.job_status_widget.set_status(None)
            self._set_note("Session deleted.")

    # ------------------------------------------------------------------
    # Import audio
    # ------------------------------------------------------------------

    def import_audio(self) -> None:
        self._flash_button(self.import_button)
        dialog = QFileDialog(self, "Import Audio File")
        dialog.setNameFilter("Audio Files (*.wav *.mp3 *.m4a *.ogg *.flac)")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if not dialog.exec():
            return
        files = dialog.selectedFiles()
        if not files:
            return
        src_path = files[0]
        if not src_path:
            return
        src = Path(src_path)
        session_id_tmp = self.store.create_session(
            title=src.stem,
            strategy="imported",
            microphone_name="",
            system_name="",
        )["id"]
        dest = PYTHON_RECORDINGS_DIR / f"{session_id_tmp}{src.suffix}"

        self._set_actions_locked(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._start_dots("Importing", "#7c3aed")
        self._set_note(f"Importing {src.name}...")

        def _do_import() -> None:
            try:
                shutil.copy2(src, dest)
                updates = {
                    "status": "processed",
                    "finishedAt": self.store._timestamp(),
                    "finalAudioPath": str(dest),
                }
                # Upload to GCS if configured
                if self.cloud_run.gcs_bucket and self.cloud_run.project_id:
                    gcs_uri = self.cloud_run.upload_audio_file(
                        dest, session_id_tmp,
                        progress_callback=lambda u, t: self._import_upload_progress.emit(u, t),
                    )
                    updates["gcsFolderUri"] = self.cloud_run.session_folder_uri(session_id_tmp)
                    updates["jobStatus"] = "uploaded"
                self.store.update_session(session_id_tmp, **updates)
                self._import_success.emit(session_id_tmp, src.name)
            except Exception as e:
                self._import_failed.emit(str(e))

        threading.Thread(target=_do_import, daemon=True).start()

    def _on_import_upload_progress(self, uploaded: int, total: int) -> None:
        if total > 0:
            pct = min(int(uploaded / total * 100), 99)
            self.progress_bar.setValue(pct)
            self._set_note(f"Uploading to GCS... {pct}%")

    def _on_import_success(self, session_id: str, filename: str) -> None:
        self._stop_dots()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self._set_actions_locked(False)
        self.job_status_widget.set_status(None)
        self.load_sessions()
        self._select_session(session_id)
        self._set_note(f"Imported: {filename}")

    def _on_import_failed(self, message: str) -> None:
        self._stop_dots()
        self.progress_bar.setVisible(False)
        self._set_actions_locked(False)
        self._set_error(f"Import failed: {message}")

    # ------------------------------------------------------------------
    # Export transcript
    # ------------------------------------------------------------------

    def _show_export_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1e2130; border: 1px solid #2d3044; color: #e2e8f0; }"
            "QMenu::item:selected { background: #5865f2; }"
        )
        menu.addAction("Copy to clipboard", self._export_clipboard)
        menu.exec(self.export_button.mapToGlobal(self.export_button.rect().bottomLeft()))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.transcript_list and event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Copy):
                self._export_clipboard()
                return True  # consume event, prevent QListWidget's OLE copy
        return super().eventFilter(obj, event)

    def _export_clipboard(self) -> None:
        self._flash_button(self.export_button)
        txt = self._build_txt_content()
        if not txt:
            return
        try:
            _win32_copy(txt)
            self._set_note("Transcript copied to clipboard.")
        except Exception as e:
            self._set_error(f"Clipboard error: {e}")

    def _export_txt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", "", "Text Files (*.txt)"
        )
        if path:
            Path(path).write_text(self._build_txt_content(), encoding="utf8")
            self._set_note(f"Saved: {Path(path).name}")

    def _export_srt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", "", "SRT Files (*.srt)"
        )
        if path:
            Path(path).write_text(self._build_srt_content(), encoding="utf8")
            self._set_note(f"Saved: {Path(path).name}")

    def _build_txt_content(self) -> str:
        lines = []
        for utt in self._utterances:
            speaker = utt.get("speaker") or "Unknown"
            lines.append(f"[{speaker}] {utt['text']}")
        return "\n".join(lines)

    def _build_srt_content(self) -> str:
        blocks = []
        for i, utt in enumerate(self._utterances, start=1):
            start = self._ms_to_srt_time(utt.get("start", 0))
            end = self._ms_to_srt_time(utt.get("end", 0))
            speaker = utt.get("speaker") or "Unknown"
            blocks.append(f"{i}\n{start} --> {end}\n[{speaker}] {utt['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        ms = max(ms, 0)
        hours, ms = divmod(ms, 3_600_000)
        minutes, ms = divmod(ms, 60_000)
        secs, millis = divmod(ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    # ------------------------------------------------------------------
    # Inline transcript edit
    # ------------------------------------------------------------------

    def _on_transcript_edit(self, row: int, new_text: str) -> None:
        if row >= len(self._utterances):
            return
        self._utterances[row]["text"] = new_text

        item = self.sessions_list.currentItem()
        if not item:
            return
        session_id = item.data(32)
        session = self.store.load_session(session_id)
        transcript_path = session.get("transcriptPath")
        if not transcript_path:
            return

        transcript = self.store.read_transcript(transcript_path)
        utt_id = self._utterances[row].get("id")
        for utt in transcript.get("utterances", []):
            if utt.get("id") == utt_id:
                utt["text"] = new_text
                break
        transcript["text"] = " ".join(
            u["text"] for u in transcript.get("utterances", [])
        ).strip()
        self.store.save_transcript(session_id, transcript)

