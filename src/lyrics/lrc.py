"""Parse LRC (line-based) synced lyrics into timed lines."""

from __future__ import annotations

import re
from dataclasses import dataclass

# [mm:ss.xx] or [mm:ss.xxx] or [mm:ss] — line may contain multiple tags
_TS = re.compile(r"\[(\d+):(\d+)(?:\.(\d{1,6}))?\]")
_META = re.compile(r"^\[[a-zA-Z]+:")


@dataclass(frozen=True)
class LyricLine:
    time_s: float
    text: str


def _fraction_to_seconds(frac: str | None) -> float:
    if not frac:
        return 0.0
    n = len(frac)
    v = int(frac)
    if n == 1:
        return v / 10.0
    if n == 2:
        return v / 100.0
    return int(frac[:3]) / 1000.0


def _parse_timestamp(m: re.Match[str]) -> float:
    mm = int(m.group(1))
    ss = int(m.group(2))
    return mm * 60 + ss + _fraction_to_seconds(m.group(3))


def _strip_word_level_tags(text: str) -> str:
    """Remove enhanced-LRC inline timing like <00:12.34>."""
    return re.sub(r"<\d{1,2}:\d{2}(?:\.\d{1,3})?>", "", text).strip()


# Enhanced LRC: <mm:ss.xx>word (absolute times, same as bracket LRC)
_WORD_TAG = re.compile(r"<(\d+):(\d+)(?:\.(\d{1,6}))?>\s*([^<]*)")


def _try_expand_enhanced_words(chunk: str) -> list[LyricLine] | None:
    """If chunk uses inline <time> tags, return one LyricLine per word; else None."""
    if "<" not in chunk or not re.search(r"<\d{1,2}:\d{2}", chunk):
        return None
    out: list[LyricLine] = []
    for m in _WORD_TAG.finditer(chunk):
        mm, ss, frac = int(m.group(1)), int(m.group(2)), m.group(3)
        ts = mm * 60 + ss + _fraction_to_seconds(frac)
        tx = (m.group(4) or "").strip()
        if tx:
            out.append(LyricLine(time_s=ts, text=tx))
    return out if out else None


def parse_lrc(raw: str) -> list[LyricLine]:
    """Return sorted lyric lines (empty lines and metadata skipped)."""
    entries: list[LyricLine] = []
    for raw_line in raw.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or _META.match(line):
            continue
        matches = list(_TS.finditer(line))
        if not matches:
            continue
        for j, m in enumerate(matches):
            t = _parse_timestamp(m)
            end = matches[j + 1].start() if j + 1 < len(matches) else len(line)
            chunk = line[m.end() : end].strip()
            expanded = _try_expand_enhanced_words(chunk)
            if expanded:
                entries.extend(expanded)
            else:
                text = _strip_word_level_tags(chunk)
                if text:
                    entries.append(LyricLine(time_s=t, text=text))

    entries.sort(key=lambda e: e.time_s)
    # Drop duplicate (time, text) pairs
    out: list[LyricLine] = []
    seen: set[tuple[float, str]] = set()
    for e in entries:
        key = (round(e.time_s, 3), e.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out
