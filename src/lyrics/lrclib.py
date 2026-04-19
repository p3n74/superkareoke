"""Fetch synchronized LRC text from LRCLIB (https://lrclib.net)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from src.config import APP_NAME, APP_VERSION

logger = logging.getLogger(__name__)

LRCLIB_BASE = "https://lrclib.net/api"
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (desktop; respects lrclib.net)"


def _http_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_synced_lrc(title: str, artist: str, duration_sec: float | None) -> str | None:
    """
    Return LRC string or None. Uses /api/search then picks closest duration match.
    """
    title = (title or "").strip()
    artist = (artist or "").strip()
    if not title:
        return None

    def run_search(params: dict[str, str]) -> list[dict]:
        q = urllib.parse.urlencode(params)
        url = f"{LRCLIB_BASE}/search?{q}"
        try:
            data = _http_json(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            logger.warning("LRCLIB HTTP error %s for %s", e.code, url)
            return []
        except Exception as e:
            logger.warning("LRCLIB request failed: %s", e)
            return []
        if isinstance(data, dict) and data.get("code") == 404:
            return []
        return data if isinstance(data, list) else []

    # Prefer structured search
    data = run_search({"track_name": title, "artist_name": artist})
    if not data and artist:
        data = run_search({"q": f"{title} {artist}"})
    elif not data:
        data = run_search({"q": title})

    if not data:
        logger.info("LRCLIB: no search results for %r / %r", title, artist)
        return None

    candidates = [
        r
        for r in data
        if not r.get("instrumental")
        and (r.get("syncedLyrics") or "").strip()
    ]
    if not candidates:
        logger.info("LRCLIB: results but none with syncedLyrics for %r / %r", title, artist)
        return None

    if duration_sec and duration_sec > 1.0:
        candidates.sort(
            key=lambda r: abs(float(r.get("duration") or 0) - float(duration_sec)),
        )

    lrc = (candidates[0].get("syncedLyrics") or "").strip()
    if not lrc:
        return None
    logger.info(
        "LRCLIB: matched %r by %r (duration record=%s)",
        candidates[0].get("trackName"),
        candidates[0].get("artistName"),
        candidates[0].get("duration"),
    )
    return lrc
