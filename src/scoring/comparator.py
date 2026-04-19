import math
from typing import Optional

from src.config import scoring_thresholds_cents


def cents_between(freq_a: float, freq_b: float) -> float:
    """Calculate the difference in cents between two frequencies."""
    if freq_a <= 0 or freq_b <= 0:
        return float("inf")
    return abs(1200 * math.log2(freq_a / freq_b))


def classify_hit(cents_off: float, thresholds: tuple[int, int, int, int]) -> str:
    perfect, great, good, ok = thresholds
    if cents_off <= perfect:
        return "perfect"
    if cents_off <= great:
        return "great"
    if cents_off <= good:
        return "good"
    if cents_off <= ok:
        return "ok"
    return "miss"


def hit_points(hit_type: str) -> int:
    return {
        "perfect": 100,
        "great": 75,
        "good": 50,
        "ok": 25,
        "miss": 0,
    }.get(hit_type, 0)


class PitchComparator:
    """Compares user pitch against a target note segment in real-time.

    Tracks which segments have been hit and accumulates scores.
    """

    def __init__(self, note_segments: list[dict], difficulty_id: str = "strict"):
        self._segments = note_segments
        self._thresholds = scoring_thresholds_cents(difficulty_id)
        self._segment_samples: dict[int, list[float]] = {}
        self._segment_results: dict[int, str] = {}
        self._last_checked_idx = 0

        # Running score
        self._total_points = 0
        self._max_points = 0
        self._combo = 0
        self._max_combo = 0

        # Per-hit counters
        self.perfect_count = 0
        self.great_count = 0
        self.good_count = 0
        self.ok_count = 0
        self.miss_count = 0

    def update(self, current_time: float, user_freq: float, user_confidence: float):
        """Called each frame with the current user pitch.

        Returns (segment_idx, hit_type) if a segment was just scored, else None.
        """
        if user_confidence < 0.3:
            user_freq = 0.0

        # Find which segment the current time falls in
        active_idx = self._find_active_segment(current_time)

        if active_idx is not None and active_idx not in self._segment_results:
            if active_idx not in self._segment_samples:
                self._segment_samples[active_idx] = []
            if user_freq > 0:
                seg = self._segments[active_idx]
                c = cents_between(user_freq, seg["freq"])
                self._segment_samples[active_idx].append(c)

        # Finalize segments that we've passed
        results = []
        for idx in list(self._segment_samples.keys()):
            seg = self._segments[idx]
            if current_time > seg["end_time"] + 0.05 and idx not in self._segment_results:
                result = self._finalize_segment(idx)
                results.append((idx, result))

        return results

    def finalize_remaining(self):
        """Score any segments not yet finalized (call at end of song)."""
        results = []
        for idx in range(len(self._segments)):
            if idx not in self._segment_results:
                result = self._finalize_segment(idx)
                results.append((idx, result))
        return results

    def _find_active_segment(self, t: float) -> Optional[int]:
        for i in range(self._last_checked_idx, len(self._segments)):
            seg = self._segments[i]
            if seg["start_time"] <= t <= seg["end_time"]:
                self._last_checked_idx = max(0, i - 1)
                return i
            if seg["start_time"] > t:
                break
        return None

    def _finalize_segment(self, idx: int) -> str:
        samples = self._segment_samples.get(idx, [])
        if not samples:
            hit_type = "miss"
        else:
            avg_cents = sum(samples) / len(samples)
            hit_type = classify_hit(avg_cents, self._thresholds)

        self._segment_results[idx] = hit_type
        points = hit_points(hit_type)
        self._total_points += points
        self._max_points += 100

        if hit_type != "miss":
            self._combo += 1
            self._max_combo = max(self._max_combo, self._combo)
        else:
            self._combo = 0

        # Update counters
        counter_map = {
            "perfect": "perfect_count",
            "great": "great_count",
            "good": "good_count",
            "ok": "ok_count",
            "miss": "miss_count",
        }
        attr = counter_map.get(hit_type)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)

        return hit_type

    @property
    def score_percent(self) -> float:
        if self._max_points == 0:
            return 0.0
        return (self._total_points / self._max_points) * 100

    @property
    def combo(self) -> int:
        return self._combo

    @property
    def max_combo(self) -> int:
        return self._max_combo

    @property
    def total_segments(self) -> int:
        return len(self._segments)

    @property
    def scored_segments(self) -> int:
        return len(self._segment_results)
