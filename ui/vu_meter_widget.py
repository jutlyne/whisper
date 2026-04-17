from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget


class VUMeterWidget(QWidget):
    """Dual-channel VU meter: một thanh cho Mic, một cho System audio."""

    _BAR_HEIGHT = 10
    _SPACING = 8
    _LABEL_WIDTH = 36

    # Mức độ giảm mỗi tick (~100ms)
    _DECAY_FACTOR = 0.75
    # Số tick giữ peak (~800ms ở 100ms/tick)
    _PEAK_HOLD_TICKS = 8

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mic_level: float = 0.0
        self._sys_level: float = 0.0
        self._mic_peak: float = 0.0
        self._sys_peak: float = 0.0
        self._mic_peak_ticks: int = 0
        self._sys_peak_ticks: int = 0
        self.setMinimumHeight(48)
        self.setMaximumHeight(60)

    def update_levels(self, mic_rms: float, sys_rms: float) -> None:
        """Cập nhật mức âm lượng. Gọi trong main thread qua Qt signal."""
        self._mic_level = max(mic_rms, self._mic_level * self._DECAY_FACTOR)
        self._sys_level = max(sys_rms, self._sys_level * self._DECAY_FACTOR)

        self._mic_peak, self._mic_peak_ticks = self._update_peak(
            mic_rms, self._mic_peak, self._mic_peak_ticks
        )
        self._sys_peak, self._sys_peak_ticks = self._update_peak(
            sys_rms, self._sys_peak, self._sys_peak_ticks
        )
        self.update()

    def reset(self) -> None:
        self._mic_level = self._sys_level = 0.0
        self._mic_peak = self._sys_peak = 0.0
        self._mic_peak_ticks = self._sys_peak_ticks = 0
        self.update()

    def _update_peak(
        self, rms: float, peak: float, ticks: int
    ) -> tuple[float, int]:
        if rms >= peak:
            return rms, self._PEAK_HOLD_TICKS
        ticks -= 1
        if ticks <= 0:
            peak = max(peak - 0.05, 0.0)
        return peak, max(ticks, 0)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_w = self.width() - self._LABEL_WIDTH - 4
        self._draw_bar(p, y=4, label="Mic", level=self._mic_level, peak=self._mic_peak, bar_w=bar_w)
        self._draw_bar(
            p,
            y=4 + self._BAR_HEIGHT + self._SPACING,
            label="Sys",
            level=self._sys_level,
            peak=self._sys_peak,
            bar_w=bar_w,
        )
        p.end()

    def _draw_bar(
        self,
        p: QPainter,
        *,
        y: int,
        label: str,
        level: float,
        peak: float,
        bar_w: int,
    ) -> None:
        x0 = self._LABEL_WIDTH

        # Nền track
        p.fillRect(x0, y, bar_w, self._BAR_HEIGHT, QColor("#1e2130"))

        # Thanh gradient
        if level > 0 and bar_w > 0:
            fill_w = max(1, int(bar_w * min(level, 1.0)))
            grad = QLinearGradient(x0, 0, x0 + bar_w, 0)
            grad.setColorAt(0.0, QColor("#23a55a"))   # xanh lá
            grad.setColorAt(0.7, QColor("#f59e0b"))   # vàng
            grad.setColorAt(1.0, QColor("#f23f43"))   # đỏ
            p.fillRect(x0, y, fill_w, self._BAR_HEIGHT, grad)

        # Peak marker
        if peak > 0 and bar_w > 0:
            px = x0 + int(bar_w * min(peak, 1.0)) - 2
            p.fillRect(px, y, 2, self._BAR_HEIGHT, QColor("#ffffff"))

        # Nhãn
        p.setPen(QColor("#64748b"))
        p.drawText(
            0, y, self._LABEL_WIDTH - 4, self._BAR_HEIGHT,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            label,
        )
