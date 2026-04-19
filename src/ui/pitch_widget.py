import math
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QFont, QPainterPath,
)

from src.database.models import PitchMap, PitchEvent
from src.config import PITCH_FMIN, PITCH_FMAX, display_lane_half_semitones


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Colors
COLOR_BG = QColor(20, 20, 40)
COLOR_GRID_LINE = QColor(40, 40, 70)
COLOR_GRID_TEXT = QColor(80, 80, 120)
COLOR_NOTE_BAR = QColor(0, 150, 255, 160)
COLOR_NOTE_BAR_HIT = QColor(0, 255, 100, 200)
COLOR_NOTE_BAR_CLOSE = QColor(255, 255, 0, 180)
COLOR_NOTE_BAR_MISS = QColor(100, 100, 140, 100)
COLOR_USER_PITCH = QColor(255, 80, 80)
COLOR_USER_TRAIL = QColor(255, 80, 80, 100)
COLOR_PLAYHEAD = QColor(255, 255, 255, 180)
COLOR_LYRICS = QColor(220, 220, 240)


def freq_to_semitone(freq: float) -> float:
    """Convert frequency to a continuous semitone value (MIDI-like)."""
    if freq <= 0:
        return -1
    return 69 + 12 * math.log2(freq / 440.0)


def semitone_to_freq(semitone: float) -> float:
    return 440.0 * (2.0 ** ((semitone - 69) / 12.0))


def semitone_to_note_name(semitone: int) -> str:
    octave = (semitone // 12) - 1
    note_idx = semitone % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


class PitchWidget(QWidget):
    """Rock Band 4-style scrolling pitch visualization.

    Shows target notes as horizontal bars and the user's pitch as a
    moving indicator with a trailing tail.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 300)

        # Pitch range in semitones
        self._semitone_min = freq_to_semitone(PITCH_FMIN)  # ~C2
        self._semitone_max = freq_to_semitone(PITCH_FMAX)  # ~C6

        # Time window: how many seconds visible on screen
        self._visible_seconds = 6.0
        # Playhead position (fraction of width from left)
        self._playhead_x_frac = 0.25

        # Pitch map data (target notes)
        self._pitch_map: Optional[PitchMap] = None
        self._note_segments: list[dict] = []

        # Current state
        self._current_time: float = 0.0
        self._user_freq: float = 0.0
        self._user_confidence: float = 0.0
        self._user_trail: list[tuple[float, float]] = []  # (time, semitone)
        self._max_trail_points = 200

        # Hit state for each note segment
        self._segment_hits: dict[int, str] = {}  # index -> "perfect"/"great"/"good"/"ok"/"miss"

        # Score overlay
        self._score_percent: float = 0.0
        self._combo: int = 0

        # Lyrics
        self._lyrics_line: str = ""

        # Target lane vertical span: 0 = one row; casual widens to +/- N semitones
        self._lane_half_semitones: float = 0.0

        # Render timer (60 fps)
        self._timer = QTimer()
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)

    def set_display_difficulty(self, difficulty_id: str) -> None:
        """Widen target note bars on casual (±1 semitone) to match scoring tolerance."""
        self._lane_half_semitones = display_lane_half_semitones(difficulty_id)
        self.update()

    def set_pitch_map(self, pitch_map: PitchMap):
        self._pitch_map = pitch_map
        self._build_note_segments()
        self._segment_hits.clear()

    def _build_note_segments(self):
        """Convert raw pitch events into drawable note segments.

        Groups consecutive pitch events at similar pitch into bars.
        """
        if not self._pitch_map or not self._pitch_map.events:
            self._note_segments = []
            return

        segments = []
        events = self._pitch_map.events
        hop_s = self._pitch_map.hop_ms / 1000.0

        current_seg = None
        for i, ev in enumerate(events):
            if ev.frequency <= 0 or ev.confidence < 0.4:
                if current_seg is not None:
                    segments.append(current_seg)
                    current_seg = None
                continue

            semitone = freq_to_semitone(ev.frequency)
            rounded = round(semitone)

            if current_seg is None:
                current_seg = {
                    "start_time": ev.time,
                    "end_time": ev.time + hop_s,
                    "semitone": rounded,
                    "freq": ev.frequency,
                }
            elif abs(rounded - current_seg["semitone"]) <= 0.5:
                current_seg["end_time"] = ev.time + hop_s
            else:
                segments.append(current_seg)
                current_seg = {
                    "start_time": ev.time,
                    "end_time": ev.time + hop_s,
                    "semitone": rounded,
                    "freq": ev.frequency,
                }

        if current_seg is not None:
            segments.append(current_seg)

        # Filter out very short segments (< 50ms)
        self._note_segments = [s for s in segments if s["end_time"] - s["start_time"] >= 0.05]

    def set_time(self, time_s: float):
        self._current_time = time_s

    def set_user_pitch(self, frequency: float, confidence: float):
        self._user_freq = frequency
        self._user_confidence = confidence

        if frequency > 0 and confidence > 0.3:
            semitone = freq_to_semitone(frequency)
            self._user_trail.append((self._current_time, semitone))
            if len(self._user_trail) > self._max_trail_points:
                self._user_trail = self._user_trail[-self._max_trail_points:]

    def set_segment_hit(self, segment_idx: int, hit_type: str):
        self._segment_hits[segment_idx] = hit_type

    def set_score(self, percent: float, combo: int):
        self._score_percent = percent
        self._combo = combo

    def set_lyrics(self, text: str):
        self._lyrics_line = text

    def start_rendering(self):
        self._timer.start()

    def stop_rendering(self):
        self._timer.stop()

    def reset(self):
        self._current_time = 0.0
        self._user_freq = 0.0
        self._user_confidence = 0.0
        self._user_trail.clear()
        self._segment_hits.clear()
        self._score_percent = 0.0
        self._combo = 0
        self._lyrics_line = ""

    # ── Painting ──

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Reserve space for lyrics at top and score
        lyrics_h = 40
        score_h = 0
        pitch_area = QRectF(0, lyrics_h, w, h - lyrics_h - score_h)

        self._draw_background(painter, w, h)
        self._draw_grid(painter, pitch_area)
        self._draw_note_segments(painter, pitch_area)
        self._draw_user_trail(painter, pitch_area)
        self._draw_user_indicator(painter, pitch_area)
        self._draw_playhead(painter, pitch_area)
        self._draw_lyrics(painter, QRectF(0, 0, w, lyrics_h))
        self._draw_score(painter, QRectF(w - 200, lyrics_h + 4, 190, 60))

        painter.end()

    def _time_to_x(self, t: float, area: QRectF) -> float:
        playhead_x = area.left() + area.width() * self._playhead_x_frac
        time_offset = t - self._current_time
        pixels_per_second = area.width() / self._visible_seconds
        return playhead_x + time_offset * pixels_per_second

    def _semitone_to_y(self, semitone: float, area: QRectF) -> float:
        frac = (semitone - self._semitone_min) / (self._semitone_max - self._semitone_min)
        frac = max(0, min(1, frac))
        return area.bottom() - frac * area.height()

    def _draw_background(self, painter: QPainter, w: int, h: int):
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0, QColor(15, 15, 35))
        gradient.setColorAt(1, QColor(25, 25, 50))
        painter.fillRect(0, 0, w, h, gradient)

    def _draw_grid(self, painter: QPainter, area: QRectF):
        pen = QPen(COLOR_GRID_LINE, 1)
        painter.setPen(pen)

        font = QFont("Consolas", 8)
        painter.setFont(font)

        semi_min = int(self._semitone_min)
        semi_max = int(self._semitone_max)

        for semi in range(semi_min, semi_max + 1):
            y = self._semitone_to_y(semi, area)
            if y < area.top() or y > area.bottom():
                continue

            note_idx = semi % 12
            is_c = note_idx == 0

            if is_c:
                painter.setPen(QPen(QColor(60, 60, 100), 1))
            else:
                painter.setPen(QPen(COLOR_GRID_LINE, 0.5))

            painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))

            # Label every C and every natural note
            if is_c or note_idx in (2, 4, 5, 7, 9, 11):
                name = semitone_to_note_name(semi)
                painter.setPen(QPen(COLOR_GRID_TEXT))
                painter.drawText(int(area.left() + 4), int(y - 2), name)

    def _draw_note_segments(self, painter: QPainter, area: QRectF):
        time_start = self._current_time - self._playhead_x_frac * self._visible_seconds
        time_end = self._current_time + (1 - self._playhead_x_frac) * self._visible_seconds

        row_px = area.height() / max(1e-6, (self._semitone_max - self._semitone_min))
        bar_height = max(6, row_px * 0.8)

        for i, seg in enumerate(self._note_segments):
            if seg["end_time"] < time_start or seg["start_time"] > time_end:
                continue

            x1 = self._time_to_x(seg["start_time"], area)
            x2 = self._time_to_x(seg["end_time"], area)
            center = float(seg["semitone"])
            half = self._lane_half_semitones
            if half > 0:
                y_hi = self._semitone_to_y(center + half, area)
                y_lo = self._semitone_to_y(center - half, area)
                top_px = min(y_hi, y_lo)
                h_px = max(abs(y_lo - y_hi), bar_height)
                rect = QRectF(x1, top_px, x2 - x1, h_px)
            else:
                y = self._semitone_to_y(center, area)
                rect = QRectF(x1, y - bar_height / 2, x2 - x1, bar_height)

            hit = self._segment_hits.get(i, "")
            if hit == "perfect":
                color = COLOR_NOTE_BAR_HIT
            elif hit in ("great", "good"):
                color = COLOR_NOTE_BAR_CLOSE
            elif hit in ("ok", "miss"):
                color = COLOR_NOTE_BAR_MISS
            else:
                color = COLOR_NOTE_BAR

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 3, 3)

            # Glow effect for hit notes
            if hit in ("perfect", "great"):
                glow = QColor(color)
                glow.setAlpha(40)
                glow_rect = rect.adjusted(-2, -2, 2, 2)
                painter.setBrush(QBrush(glow))
                painter.drawRoundedRect(glow_rect, 5, 5)

    def _draw_user_trail(self, painter: QPainter, area: QRectF):
        if len(self._user_trail) < 2:
            return

        time_start = self._current_time - self._playhead_x_frac * self._visible_seconds

        path = QPainterPath()
        started = False

        for t, semi in self._user_trail:
            if t < time_start:
                continue
            x = self._time_to_x(t, area)
            y = self._semitone_to_y(semi, area)
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)

        if started:
            pen = QPen(COLOR_USER_TRAIL, 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

    def _draw_user_indicator(self, painter: QPainter, area: QRectF):
        if self._user_freq <= 0 or self._user_confidence < 0.3:
            return

        semitone = freq_to_semitone(self._user_freq)
        playhead_x = area.left() + area.width() * self._playhead_x_frac
        y = self._semitone_to_y(semitone, area)

        # Outer glow
        glow_color = QColor(COLOR_USER_PITCH)
        glow_color.setAlpha(60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(int(playhead_x - 12), int(y - 12), 24, 24)

        # Inner circle
        painter.setBrush(QBrush(COLOR_USER_PITCH))
        painter.drawEllipse(int(playhead_x - 6), int(y - 6), 12, 12)

        # White center
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.drawEllipse(int(playhead_x - 3), int(y - 3), 6, 6)

    def _draw_playhead(self, painter: QPainter, area: QRectF):
        x = area.left() + area.width() * self._playhead_x_frac
        pen = QPen(COLOR_PLAYHEAD, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))

    def _draw_lyrics(self, painter: QPainter, area: QRectF):
        if not self._lyrics_line:
            return
        font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(COLOR_LYRICS))
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self._lyrics_line)

    def _draw_score(self, painter: QPainter, area: QRectF):
        font = QFont("Consolas", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255, 200)))
        painter.drawText(
            area, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{self._score_percent:.0f}%",
        )

        if self._combo > 1:
            combo_font = QFont("Consolas", 10)
            painter.setFont(combo_font)
            painter.setPen(QPen(QColor(255, 200, 50, 200)))
            combo_area = area.adjusted(0, 22, 0, 22)
            painter.drawText(
                combo_area, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                f"x{self._combo} combo",
            )

    # ── Public helpers ──

    @property
    def note_segments(self) -> list[dict]:
        return self._note_segments

    @property
    def visible_time_range(self) -> tuple[float, float]:
        t_start = self._current_time - self._playhead_x_frac * self._visible_seconds
        t_end = self._current_time + (1 - self._playhead_x_frac) * self._visible_seconds
        return t_start, t_end
