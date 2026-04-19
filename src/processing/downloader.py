import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _yt_dlp_command_prefix() -> list[str]:
    """Resolve how to run yt-dlp (Windows often lacks Scripts/ on PATH)."""
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    for name in ("yt-dlp", "yt-dlp.exe"):
        path = shutil.which(name)
        if path:
            return [path]
    raise RuntimeError(
        "yt-dlp is not available. Install it in this Python environment: pip install yt-dlp"
    )


class YouTubeDownloader:
    """Downloads audio from YouTube using yt-dlp."""

    def search_and_download(
        self, title: str, artist: str, output_dir: Path,
        progress_callback=None,
    ) -> Optional[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "original.wav"

        query = f"{title} {artist} audio"
        logger.info("yt-dlp search: %s", query)

        cmd = [
            *_yt_dlp_command_prefix(),
            f"ytsearch1:{query}",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--no-playlist",
            "--output", str(output_dir / "original.%(ext)s"),
            "--quiet",
            "--no-warnings",
            "--print", "after_move:filepath",
        ]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "Could not run yt-dlp (or Python). If audio still fails, install "
                "FFmpeg and ensure it is on your PATH — yt-dlp needs it for WAV export."
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp download timed out after 5 minutes")

        if result.returncode != 0:
            logger.error("yt-dlp stderr: %s", result.stderr.strip()[:2000])
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

        logger.info("yt-dlp finished in %.1fs", time.perf_counter() - t0)
        downloaded_path = result.stdout.strip().split("\n")[-1]
        downloaded = Path(downloaded_path)
        if downloaded.exists():
            if downloaded != output_path:
                downloaded.rename(output_path)
            return output_path

        if output_path.exists():
            return output_path

        raise FileNotFoundError(
            f"Download completed but file not found at {output_path}"
        )

    def get_video_info(self, title: str, artist: str) -> Optional[dict]:
        query = f"{title} {artist} audio"
        cmd = [
            *_yt_dlp_command_prefix(),
            f"ytsearch1:{query}",
            "--dump-json",
            "--no-download",
            "--quiet",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        return None
