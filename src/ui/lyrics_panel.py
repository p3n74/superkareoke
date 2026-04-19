"""Line-synced lyrics display driven by playback time."""

from __future__ import annotations

import bisect
from pathlib import Path

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.lyrics.lrc import parse_lrc, LyricLine


class LyricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines: list[LyricLine] = []
        self._starts: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(4)

        self._prev = QLabel("")
        self._curr = QLabel("")
        self._next = QLabel("")

        for w in (self._prev, self._curr, self._next):
            w.setWordWrap(True)
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)

        prev_font = QFont()
        prev_font.setPointSize(11)
        self._prev.setFont(prev_font)
        self._prev.setStyleSheet("color: #666;")

        curr_font = QFont()
        curr_font.setPointSize(16)
        curr_font.setBold(True)
        self._curr.setFont(curr_font)
        self._curr.setStyleSheet("color: #f8f8f2;")

        next_font = QFont()
        next_font.setPointSize(11)
        self._next.setFont(next_font)
        self._next.setStyleSheet("color: #888;")

        layout.addWidget(self._prev)
        layout.addWidget(self._curr)
        layout.addWidget(self._next)

        self.setStyleSheet("LyricsPanel { background: #12192e; border-radius: 8px; }")
        self.clear()

    def clear(self) -> None:
        self._lines = []
        self._starts = []
        self._prev.setText("")
        self._curr.setText("No synced lyrics yet. Run Process on the song again to fetch from LRCLIB.")
        self._next.setText("")

    def load_lrc_file(self, path: Path) -> None:
        if not path.is_file():
            self.clear()
            return
        raw = path.read_text(encoding="utf-8", errors="replace")
        self._lines = parse_lrc(raw)
        self._starts = [ln.time_s for ln in self._lines]
        if not self._lines:
            self._prev.setText("")
            self._curr.setText("Lyrics file found but could not be parsed.")
            self._next.setText("")
            return
        self._prev.setText("")
        self._curr.setText("")
        self._next.setText("")
        self.set_time(0.0)

    def set_time(self, t: float) -> None:
        if not self._starts:
            return
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            self._prev.setText("")
            self._curr.setText(self._lines[0].text if self._lines else "")
            self._next.setText(self._lines[1].text if len(self._lines) > 1 else "")
            return

        self._prev.setText(self._lines[i - 1].text if i > 0 else "")
        self._curr.setText(self._lines[i].text)
        self._next.setText(self._lines[i + 1].text if i + 1 < len(self._lines) else "")
