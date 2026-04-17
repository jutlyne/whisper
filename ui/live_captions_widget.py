from __future__ import annotations

import re
from html import escape

from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout, QWidget

# Format server gửi: "[vi]Bạn khỏe không?" hoặc "[ja]Xin chào"
_LANG_RE = re.compile(r"^\[([a-z]{2,3})\](.+)$", re.DOTALL)

# Màu theo ngôn ngữ nguồn — dùng để phân biệt người nói
_LANG_COLORS: dict[str, str] = {
    "vi": "#38bdf8",   # xanh dương — người Việt
    "ja": "#fb923c",   # cam         — người Nhật
    "en": "#34d399",   # xanh lá     — người nói tiếng Anh
    "zh": "#f472b6",   # hồng        — người Trung
    "ko": "#a78bfa",   # tím         — người Hàn
}
_DEFAULT_LANG_COLOR = "#94a3b8"  # xám — ngôn ngữ khác


class LiveCaptionsWidget(QWidget):
    """Hiển thị live captions trong lúc ghi âm, phân biệt người nói bằng màu."""

    MAX_LINES = 80

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)

        frame = QFrame()
        frame.setStyleSheet(
            "QFrame {"
            "  background-color: #0f1e2a;"
            "  border: 1px solid #1e4060;"
            "  border-radius: 8px;"
            "}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(10, 7, 10, 7)
        inner.setSpacing(4)

        header = QLabel("Live Captions")
        header.setStyleSheet(
            "color: #38bdf8;"
            "font-size: 11px;"
            "font-weight: 600;"
            "background: transparent;"
            "border: none;"
        )

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setMaximumHeight(140)
        self._text_edit.setStyleSheet(
            "QTextEdit {"
            "  background: transparent;"
            "  border: none;"
            "  font-size: 13px;"
            "}"
        )

        inner.addWidget(header)
        inner.addWidget(self._text_edit)
        outer.addWidget(frame)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start_session(self) -> None:
        self._text_edit.clear()
        self.setVisible(True)

    def stop_session(self) -> None:
        self.setVisible(False)

    def append_text(self, text: str) -> None:
        """Nhận text từ server, parse tag người nói và render HTML có màu."""
        text = text.strip()
        if not text:
            return

        m = _LANG_RE.match(text)
        if m:
            lang = m.group(1)               # "vi", "ja", ...
            content = m.group(2).strip()    # phần text thực sự
            color = _LANG_COLORS.get(lang, _DEFAULT_LANG_COLOR)
            html = (
                f'<span style="color:{color};font-weight:700;">'
                f"[{lang.upper()}]"
                f"</span>"
                f' <span style="color:#e2e8f0;">{escape(content)}</span>'
            )
        else:
            # Fallback: server gửi plain text (backward compat)
            html = f'<span style="color:#e2e8f0;">{escape(text)}</span>'

        self._text_edit.append(html)
        self._trim_lines()
        self._scroll_to_bottom()

    def get_full_text(self) -> str:
        return self._text_edit.toPlainText().strip()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _trim_lines(self) -> None:
        doc = self._text_edit.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self._text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.movePosition(
                cursor.MoveOperation.NextBlock,
                cursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()

    def _scroll_to_bottom(self) -> None:
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())
