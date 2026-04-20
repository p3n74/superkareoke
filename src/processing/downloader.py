import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ffmpeg_bin_dir() -> Optional[Path]:
    """Return a directory that contains both ``ffmpeg`` and ``ffprobe`` executables.

    GUI-launched apps on macOS often lack Homebrew's ``/opt/homebrew/bin`` on
    ``PATH``, so we probe common locations in addition to ``shutil.which``.
    """
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(p: Optional[str]) -> None:
        if not p:
            return
        d = Path(p).resolve().parent
        key = str(d)
        if key not in seen:
            seen.add(key)
            candidates.append(d)

    _add(shutil.which("ffmpeg"))
    _add(shutil.which("ffprobe"))

    for raw in (
        "/opt/homebrew/bin",  # Apple Silicon Homebrew
        "/usr/local/bin",  # Intel Mac Homebrew / many Linux installs
    ):
        d = Path(raw)
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            candidates.append(d)

    if sys.platform == "win32":
        for raw in (
            r"C:\ffmpeg\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
            r"C:\Program Files\ffmpeg\bin",
        ):
            d = Path(raw)
            if not d.exists():
                continue
            key = str(d.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(d)

    ext = ".exe" if sys.platform == "win32" else ""
    for d in candidates:
        if (d / f"ffmpeg{ext}").is_file() and (d / f"ffprobe{ext}").is_file():
            return d
    return None


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


def _yt_dlp_ffmpeg_args() -> list[str]:
    """Extra yt-dlp arguments so postprocessing can find ffmpeg/ffprobe."""
    d = _ffmpeg_bin_dir()
    if d is None:
        return []
    loc = str(d)
    logger.info("yt-dlp: using --ffmpeg-location %s", loc)
    return ["--ffmpeg-location", loc]


def _subprocess_env_with_ffmpeg() -> dict[str, str]:
    """Prepend the ffmpeg directory to PATH for subprocesses."""
    env = os.environ.copy()
    d = _ffmpeg_bin_dir()
    if d is None:
        return env
    prefix = str(d)
    path = env.get("PATH", "")
    if prefix not in path.split(os.pathsep):
        env["PATH"] = prefix + os.pathsep + path
    return env


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
            *_yt_dlp_ffmpeg_args(),
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
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=_subprocess_env_with_ffmpeg(),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "Could not run yt-dlp (or Python). If audio still fails, install "
                "FFmpeg and ensure it is on your PATH — yt-dlp needs it for WAV export."
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp download timed out after 5 minutes")

        if result.returncode != 0:
            err = result.stderr.strip()
            logger.error("yt-dlp stderr: %s", err[:2000])
            if "ffmpeg" in err.lower() or "ffprobe" in err.lower():
                raise RuntimeError(
                    "yt-dlp needs FFmpeg (ffmpeg and ffprobe). Install it — on macOS "
                    "with Homebrew: `brew install ffmpeg` — then restart the app so "
                    f"PATH is picked up, or ensure both binaries are on PATH.\n{err}"
                )
            raise RuntimeError(f"yt-dlp failed: {err}")

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
