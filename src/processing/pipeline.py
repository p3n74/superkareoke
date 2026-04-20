import logging
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

import soundfile as sf

from src.config import CATALOG_DIR, get_processing_preset
from src.processing.memory import release_torch_memory
from src.processing.torch_device import resolve_processing_device_string
from src.database.manager import DatabaseManager
from src.database.models import Song, SongStatus
from src.processing.downloader import YouTubeDownloader

logger = logging.getLogger(__name__)


class PipelineWorker(QObject):
    """Runs the full processing pipeline for a song in a background thread."""

    progress = pyqtSignal(int, str, float)  # song_id, stage_name, 0.0-1.0
    finished = pyqtSignal(int, bool, str)   # song_id, success, error_msg

    def __init__(
        self, song: Song, db: DatabaseManager,
        downloader: YouTubeDownloader,
        separator, extractor,
    ):
        super().__init__()
        self.song = song
        self.db = db
        self.downloader = downloader
        self.separator = separator
        self.extractor = extractor
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        song = self.song
        song_dir = CATALOG_DIR / str(song.id)
        song_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        logger.info(
            "Pipeline start song_id=%s title=%r artist=%r",
            song.id, song.title, song.artist,
        )
        try:
            # Step 1: Download
            if self._cancelled:
                return
            self.db.update_song_status(song.id, SongStatus.DOWNLOADING)
            self.progress.emit(song.id, "Downloading", 0.0)

            t_dl = time.perf_counter()
            audio_path = self.downloader.search_and_download(
                song.title, song.artist, song_dir,
            )
            if audio_path is None:
                raise RuntimeError("Download returned no file")
            logger.info("Download done in %.1fs path=%s", time.perf_counter() - t_dl, audio_path)

            self.db.update_song_paths(song.id, original_path=str(audio_path))
            self.progress.emit(song.id, "Downloading", 1.0)

            # Step 2: Vocal separation
            if self._cancelled:
                return
            self.db.update_song_status(song.id, SongStatus.SEPARATING)
            self.progress.emit(song.id, "Separating vocals", 0.0)

            t_sep = time.perf_counter()
            stems = self.separator.separate(
                audio_path, song_dir,
                progress_callback=lambda p: self.progress.emit(
                    song.id, "Separating vocals", p
                ),
            )
            logger.info("Separation done in %.1fs", time.perf_counter() - t_sep)
            release_torch_memory()

            self.db.update_song_paths(
                song.id,
                vocals_path=str(stems["vocals"]),
                instrumental_path=str(stems["instrumental"]),
            )
            self.progress.emit(song.id, "Separating vocals", 1.0)

            # Step 3: Pitch extraction
            if self._cancelled:
                return
            self.db.update_song_status(song.id, SongStatus.EXTRACTING_PITCH)
            self.progress.emit(song.id, "Extracting pitch", 0.0)

            t_pitch = time.perf_counter()
            pitch_map = self.extractor.extract(
                stems["vocals"], song.id,
                progress_callback=lambda p: self.progress.emit(
                    song.id, "Extracting pitch", p
                ),
            )
            logger.info("Pitch extraction done in %.1fs", time.perf_counter() - t_pitch)

            pitch_path = song_dir / "pitch_map.json"
            pitch_path.write_text(pitch_map.to_json(), encoding="utf-8")
            self.db.update_song_paths(song.id, pitch_map_path=str(pitch_path))

            # Synced lyrics (LRCLIB) — best-effort, never fails the pipeline
            try:
                from src.lyrics.lrclib import fetch_synced_lrc

                inst_path = stems["instrumental"]
                dur = float(sf.info(str(inst_path)).duration)
                lrc_text = fetch_synced_lrc(song.title, song.artist, dur)
                if lrc_text:
                    lrc_path = song_dir / "lyrics.lrc"
                    lrc_path.write_text(lrc_text, encoding="utf-8")
                    logger.info("Saved synced lyrics to %s", lrc_path)
                else:
                    logger.info("No synced lyrics from LRCLIB for this track.")
            except Exception as lyrics_err:
                logger.warning("Lyrics fetch skipped: %s", lyrics_err)

            del stems
            release_torch_memory()

            # Done
            self.db.update_song_status(song.id, SongStatus.READY)
            self.progress.emit(song.id, "Ready", 1.0)
            logger.info(
                "Pipeline success song_id=%s total %.1fs",
                song.id, time.perf_counter() - t0,
            )
            self.finished.emit(song.id, True, "")

        except Exception as e:
            error_msg = str(e)
            logger.exception(
                "Pipeline failed song_id=%s after %.1fs: %s",
                song.id, time.perf_counter() - t0, error_msg,
            )
            self.db.update_song_status(song.id, SongStatus.ERROR, error_msg)
            self.finished.emit(song.id, False, error_msg)


class ProcessingPipeline(QObject):
    """Manages the queue of songs being processed."""

    song_progress = pyqtSignal(int, str, float)
    song_finished = pyqtSignal(int, bool, str)

    def __init__(
        self,
        db: DatabaseManager,
        use_gpu_fn: Optional[Callable[[], bool]] = None,
        processing_preset_fn: Optional[Callable[[], str]] = None,
    ):
        super().__init__()
        self.db = db
        self.downloader = YouTubeDownloader()
        self._use_gpu_fn = use_gpu_fn
        self._processing_preset_fn = processing_preset_fn
        self._separator = None
        self._extractor = None
        self._active_workers: dict[int, tuple[QThread, PipelineWorker]] = {}

    def _resolve_processing_device(self) -> str:
        """Return a ``torch.device`` string (``cpu``, ``cuda``, ``cuda:N``, ``mps``, …)."""
        try:
            want_gpu = self._use_gpu_fn() if self._use_gpu_fn else True
        except Exception:
            want_gpu = True
        return resolve_processing_device_string(want_gpu)

    def reset_processors(self) -> None:
        """Drop cached Demucs / CREPE models (e.g. after GPU or quality preset changes)."""
        self._separator = None
        self._extractor = None
        release_torch_memory()
        logger.info("Processing models reset for next job.")

    def _current_processing_preset(self) -> dict:
        try:
            pid = self._processing_preset_fn() if self._processing_preset_fn else "fast"
        except Exception:
            pid = "fast"
        return get_processing_preset(str(pid))

    @property
    def separator(self):
        if self._separator is None:
            from src.processing.separator import VocalSeparator
            import torch
            dev = self._resolve_processing_device()
            preset = self._current_processing_preset()
            self._separator = VocalSeparator(device=dev, preset=preset)
            logger.info(
                "VocalSeparator device=%s preset=%s bundle=%s seg=%s torch.cuda=%s",
                self._separator.device,
                self._processing_preset_fn() if self._processing_preset_fn else "fast",
                preset.get("demucs_bundle"),
                preset.get("demucs_segment_s"),
                torch.cuda.is_available(),
            )
        return self._separator

    @property
    def extractor(self):
        if self._extractor is None:
            from src.processing.pitch_extractor import PitchExtractor
            import torch
            dev = self._resolve_processing_device()
            preset = self._current_processing_preset()
            self._extractor = PitchExtractor(device=dev, preset=preset)
            logger.info(
                "PitchExtractor device=%s preset=%s crepe=%s hop=%sms batch=%s torch.cuda=%s",
                self._extractor.device,
                self._processing_preset_fn() if self._processing_preset_fn else "fast",
                preset.get("crepe_model"),
                preset.get("pitch_hop_ms"),
                preset.get("crepe_batch_size"),
                torch.cuda.is_available(),
            )
        return self._extractor

    def process_song(self, song: Song):
        if song.id in self._active_workers:
            return

        thread = QThread()
        worker = PipelineWorker(
            song, self.db, self.downloader, self.separator, self.extractor,
        )
        worker.moveToThread(thread)

        worker.progress.connect(self.song_progress.emit)
        worker.finished.connect(self.song_finished.emit)
        worker.finished.connect(lambda *_: self._cleanup_worker(song.id))

        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)

        self._active_workers[song.id] = (thread, worker)
        thread.start()

    def cancel_song(self, song_id: int):
        if song_id in self._active_workers:
            _, worker = self._active_workers[song_id]
            worker.cancel()

    def _cleanup_worker(self, song_id: int):
        if song_id in self._active_workers:
            thread, worker = self._active_workers.pop(song_id)
            thread.quit()
            thread.wait(5000)

    def shutdown(self):
        for song_id in list(self._active_workers):
            self.cancel_song(song_id)
            self._cleanup_worker(song_id)
