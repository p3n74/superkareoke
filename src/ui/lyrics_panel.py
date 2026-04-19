"""Lyrics strip synced to playback: phrase-level RTL scroll like the pitch lane.

Each LRC line is one bar. The bar's horizontal extent is the literal
`[line_start, next_line_start]` mapped through the SAME time→x function as
PitchWidget, so the dashed playhead lines up with the note bars exactly.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPainter,
    QFont,
    QPen,
    QColor,
    QBrush,
    QLinearGradient,
    QFontMetrics,
)

from src.lyrics.lrc import parse_lrc, LyricLine

# Match PitchWidget exactly: future on the right, scrolls left toward playhead
_VISIBLE_SECONDS = 6.0
_PLAYHEAD_X_FRAC = 0.25

_COLOR_BG_TOP = QColor(15, 15, 35)
_COLOR_BG_BOT = QColor(25, 25, 50)
_COLOR_BAR_FUTURE = QColor(0, 150, 255, 140)
_COLOR_BAR_ACTIVE = QColor(0, 255, 100, 200)
_COLOR_BAR_PAST = QColor(100, 100, 140, 90)
_COLOR_TEXT = QColor(240, 240, 242)
_COLOR_TEXT_DIM = QColor(140, 145, 170)
_COLOR_TEXT_FUTURE = QColor(210, 215, 235)
_COLOR_PLAYHEAD = QColor(255, 255, 255, 200)


class LyricsPanel(QWidget):
    """Per-phrase lyric bars; same scroll & time mapping as PitchWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines: list[LyricLine] = []
        self._line_end_s: list[float] = []
        self._current_time: float = 0.0
        self._empty_message: str = ""
        self._lyrics_offset_s = 0.0  # positive = delay lyrics vs audio

        self.setMinimumHeight(72)
        self.setMaximumHeight(110)
        self.setStyleSheet("LyricsPanel { background: #141428; border-radius: 8px; }")

    def clear(self) -> None:
        self._lines = []
        self._line_end_s = []
        self._current_time = 0.0
        self._empty_message = (
            "No synced lyrics yet. Run Process on the song again to fetch from LRCLIB."
        )
        self.update()

    def set_lyrics_offset(self, seconds: float) -> None:
        """Positive value plays each lyric *later* (delays it vs the audio)."""
        self._lyrics_offset_s = float(seconds)
        self.update()

    def load_lrc_file(self, path: Path) -> None:
        if not path.is_file():
            self.clear()
            return
        raw = path.read_text(encoding="utf-8", errors="replace")
        self._lines = parse_lrc(raw)
        if not self._lines:
            self._line_end_s = []
            self._empty_message = "Lyrics file found but could not be parsed."
            self.update()
            return
        self._empty_message = ""
        self._rebuild_line_ends()
        self._current_time = 0.0
        self.update()

    def _rebuild_line_ends(self) -> None:
        ends: list[float] = []
        n = len(self._lines)
        for i in range(n):
            if i + 1 < n:
                ends.append(self._lines[i + 1].time_s)
            else:
                ends.append(self._lines[i].time_s + 4.0)
        self._line_end_s = ends

    def set_time(self, t: float) -> None:
        self._current_time = t
        self.update()

    def _time_to_x(self, t: float, area: QRectF) -> float:
        playhead_x = area.left() + area.width() * _PLAYHEAD_X_FRAC
        pps = area.width() / _VISIBLE_SECONDS
        return playhead_x + (t - self._current_time) * pps

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, _COLOR_BG_TOP)
        grad.setColorAt(1, _COLOR_BG_BOT)
        painter.fillRect(0, 0, w, h, grad)

        if self._empty_message and not self._lines:
            painter.setPen(QPen(_COLOR_TEXT_DIM))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                QRectF(12, 8, w - 24, h - 16),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._empty_message,
            )
            return

        if not self._lines:
            return

        margin_x = 8.0
        margin_y = 6.0
        area = QRectF(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
        playhead_x = area.left() + area.width() * _PLAYHEAD_X_FRAC
        offset = self._lyrics_offset_s
        t_cur = self._current_time
        t_vis0 = t_cur - _PLAYHEAD_X_FRAC * _VISIBLE_SECONDS
        t_vis1 = t_cur + (1.0 - _PLAYHEAD_X_FRAC) * _VISIBLE_SECONDS

        bar_top = area.top()
        bar_h = area.height()

        # Render past first, then future, then active so active sits on top
        def layer(idx: int) -> tuple[int, float]:
            ln = self._lines[idx]
            t_s = ln.time_s + offset
            t_e = (self._line_end_s[idx] if idx < len(self._line_end_s) else t_s + 4.0) + offset
            if t_s <= t_cur < t_e:
                return (2, t_s)
            if t_e <= t_cur:
                return (0, t_s)
            return (1, t_s)

        order = sorted(range(len(self._lines)), key=layer)

        for i in order:
            ln = self._lines[i]
            t_s = ln.time_s + offset
            t_e = (self._line_end_s[i] if i < len(self._line_end_s) else t_s + 4.0) + offset
            if t_e < t_vis0 or t_s > t_vis1:
                continue

            x1 = self._time_to_x(t_s, area)
            x2 = self._time_to_x(t_e, area)
            left = min(x1, x2)
            right = max(x1, x2)

            # Strict bar extent: NO minimum-width clamp. The bar's right edge
            # is exactly when the lyric ends, so it lines up with notes.
            draw_left = max(area.left(), left)
            draw_right = min(area.right(), right)
            if draw_right <= draw_left + 1:
                continue

            active = t_s <= t_cur < t_e
            past = t_e <= t_cur

            if past:
                bar_color = _COLOR_BAR_PAST
                text_color = _COLOR_TEXT_DIM
                weight = QFont.Weight.Normal
                size = 11
            elif active:
                bar_color = _COLOR_BAR_ACTIVE
                text_color = _COLOR_TEXT
                weight = QFont.Weight.Bold
                size = 13
            else:
                bar_color = _COLOR_BAR_FUTURE
                text_color = _COLOR_TEXT_FUTURE
                weight = QFont.Weight.Normal
                size = 12

            rect = QRectF(draw_left, bar_top, draw_right - draw_left, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bar_color))
            painter.drawRoundedRect(rect, 4, 4)

            font = QFont("Segoe UI", size, weight)
            painter.setFont(font)
            painter.setPen(QPen(text_color))
            fm = QFontMetrics(font)
            pad = 6
            text_rect = rect.adjusted(pad, 0, -pad, 0)
            elided = fm.elidedText(
                ln.text,
                Qt.TextElideMode.ElideRight,
                int(max(16, text_rect.width())),
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                elided,
            )

        # Dashed playhead — exactly the same fraction as PitchWidget's
        pen = QPen(_COLOR_PLAYHEAD, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(playhead_x), int(area.top()), int(playhead_x), int(area.bottom()))
