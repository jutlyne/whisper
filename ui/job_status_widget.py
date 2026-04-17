"""
Small status badge widget showing the current Cloud Run Job state for a session.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

# status → (label text, badge colour)
_STATUS_STYLES: dict[str, tuple[str, str]] = {
    "uploading":  ("Uploading...",  "#f59e0b"),
    "submitted":  ("Submitted",     "#3b82f6"),
    "running":    ("Transcribing",  "#8b5cf6"),
    "done":       ("Done",          "#22c55e"),
    "error":      ("Error",         "#ef4444"),
    "cancelled":  ("Cancelled",     "#6b7280"),
}

_DEFAULT_STYLE = ("Idle", "#475569")


class JobStatusWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedHeight(22)
        self._badge.setStyleSheet(
            "border-radius: 4px; padding: 0 8px; font-size: 11px; font-weight: 600;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._badge)
        layout.addStretch()

        self.set_status(None)

    def set_status(self, status: str | None) -> None:
        label, color = _STATUS_STYLES.get(status or "", _DEFAULT_STYLE)
        self._badge.setText(label)
        self._badge.setStyleSheet(
            f"border-radius: 4px; padding: 0 8px; font-size: 11px; font-weight: 600;"
            f"background: {color}22; color: {color}; border: 1px solid {color}66;"
        )
