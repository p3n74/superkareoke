"""Realtime vocal pitch using torchcrepe (tiny).

Inference runs on a dedicated ``QThread`` so the GUI thread never blocks on
``torchcrepe.predict`` (which was freezing Sing mode). Mic chunks are copied
and queued; predictions are throttled and run on a bounded tail window.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

from src.config import PITCH_FMIN, PITCH_FMAX

log = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _freq_to_note(freq: float) -> str:
    if freq <= 0:
        return ""
    semitones = 12 * math.log2(freq / 440.0)
    midi = round(69 + semitones)
    octave = (midi // 12) - 1
    note_idx = midi % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


class _PitchWorker(QObject):
    """Runs torchcrepe on a background thread."""

    result = pyqtSignal(float, float, str)

    def __init__(self, device_str: str):
        super().__init__()
        self._device = torch.device(device_str)
        self._model_ready = False
        self._tail = np.zeros(0, dtype=np.float32)
        self._max_tail = 24000  # ~0.55 s @ 44.1 kHz cap for concat growth
        self._analysis = 12288  # samples fed to crepe (bounded work)
        self._hop = 320
        self._min_period_s = 0.045  # max ~22 crepe passes / second
        self._last_infer = 0.0
        if device_str.startswith("cuda"):
            self._batch = 128
        elif device_str == "mps":
            self._batch = 48
        else:
            self._batch = 64

    @pyqtSlot()
    def prepare(self) -> None:
        if not HAS_TORCH:
            return
        try:
            import torchcrepe
            torchcrepe.load.model(self._device, "tiny")
            self._model_ready = True
            # One cheap graph warm-up on the inference device
            n = 4096
            dummy = torch.zeros(1, n, device=self._device, dtype=torch.float32)
            hop = min(self._hop, n // 2)
            if hop >= 64:
                with torch.no_grad():
                    torchcrepe.predict(
                        dummy, 44100, hop,
                        fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                        model="tiny", device=self._device,
                        return_periodicity=True,
                        batch_size=self._batch,
                    )
            log.info("RealtimePitchDetector: CREPE tiny ready on %s", self._device)
        except Exception:
            log.exception("RealtimePitchDetector: failed to load CREPE model")

    @pyqtSlot()
    def clear_buffer(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        self._last_infer = 0.0

    @pyqtSlot(object, int)
    def ingest(self, chunk: np.ndarray, sr: int) -> None:
        if not HAS_TORCH or not self._model_ready or chunk is None or chunk.size == 0 or sr <= 0:
            return
        try:
            import torchcrepe
        except ImportError:
            return

        c = np.ascontiguousarray(chunk, dtype=np.float32).ravel()
        if self._tail.size == 0:
            self._tail = c.copy()
        else:
            self._tail = np.concatenate((self._tail, c))
        if self._tail.size > self._max_tail:
            self._tail = self._tail[-self._max_tail :].copy()

        now = time.perf_counter()
        if now - self._last_infer < self._min_period_s:
            return
        self._last_infer = now

        if self._tail.size < 2048:
            return

        x = self._tail[-self._analysis :]
        hop = self._hop
        if x.size < hop * 2:
            hop = max(64, x.size // 4)

        t = torch.from_numpy(x).unsqueeze(0).to(self._device)
        try:
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    t, sr, hop,
                    fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                    model="tiny", device=self._device,
                    return_periodicity=True,
                    batch_size=self._batch,
                )
            freq = float(pitch[0, -1].detach().cpu())
            conf = float(periodicity[0, -1].detach().cpu())
        except Exception:
            log.debug("pitch worker infer failed", exc_info=True)
            return
        finally:
            del t

        if conf < 0.3:
            freq = 0.0
        note = _freq_to_note(freq) if freq > 0 else ""
        self.result.emit(freq, conf, note)


class RealtimePitchDetector(QObject):
    """Realtime pitch: queue mic audio to a worker thread, emit results on the GUI thread."""

    pitch_detected = pyqtSignal(float, float, str)
    _chunk_queued = pyqtSignal(object, int)
    _clear_requested = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QObject] = None,
        sample_rate: int = 44100,
        device: Optional[str] = None,
        use_gpu_fn: Optional[Callable[[], bool]] = None,
        buffer_duration_ms: int = 80,
    ):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self._use_gpu_fn = use_gpu_fn

        if HAS_TORCH:
            if device is None:
                from src.processing.torch_device import resolve_processing_device_string

                want = True
                if use_gpu_fn is not None:
                    try:
                        want = bool(use_gpu_fn())
                    except Exception:
                        want = True
                device = resolve_processing_device_string(want)
            dev_str = str(device)
        else:
            dev_str = "cpu"

        self._thread = QThread(self)
        self._worker = _PitchWorker(dev_str)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.prepare)
        self._chunk_queued.connect(self._worker.ingest, Qt.ConnectionType.QueuedConnection)
        self._clear_requested.connect(self._worker.clear_buffer, Qt.ConnectionType.QueuedConnection)
        self._worker.result.connect(self._emit_pitch, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    @pyqtSlot(float, float, str)
    def _emit_pitch(self, freq: float, conf: float, note: str) -> None:
        self.pitch_detected.emit(freq, conf, note)

    def process_chunk(self, audio: np.ndarray, sr: int) -> None:
        """Enqueue a mic chunk for background analysis (non-blocking)."""
        if not HAS_TORCH:
            return
        # NumPy < 2: ascontiguousarray has no ``copy=`` kwarg; np.array(..., copy=True) is portable.
        chunk_copy = np.array(audio, dtype=np.float32, copy=True, order="C")
        self._chunk_queued.emit(chunk_copy, int(sr))

    def reset_buffer(self) -> None:
        """Clear queued tail (call when stopping performance)."""
        if HAS_TORCH and self._thread.isRunning():
            self._clear_requested.emit()

    def shutdown(self) -> None:
        """Stop the worker thread (e.g. app teardown)."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
