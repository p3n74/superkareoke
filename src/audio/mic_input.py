import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)


class MicInput(QObject):
    """Captures audio from the microphone in real-time.

    In addition to the ``audio_chunk`` Qt signal (queued to the GUI thread for
    pitch detection), an optional ``set_audio_processor`` callback is invoked
    *directly from the PortAudio thread* so latency-sensitive consumers (the
    vocal monitor) aren't subject to GUI-thread jitter.
    """

    audio_chunk = pyqtSignal(np.ndarray, int)  # mono float32 chunk, sample_rate

    def __init__(
        self, sample_rate: int = 44100, chunk_size: int = 512,
        device: Optional[int] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._processor: Optional[Callable[[np.ndarray, int], None]] = None

    def set_audio_processor(self, fn: Optional[Callable[[np.ndarray, int], None]]) -> None:
        """Register a callback fired on the audio thread before the Qt signal emit.

        The callback receives ``(chunk, sample_rate)``. Exceptions are caught
        so a misbehaving consumer can never stall the PortAudio callback.
        """
        self._processor = fn

    def set_chunk_size(self, chunk_size: int) -> None:
        """Change capture block size; takes effect on next start()."""
        chunk_size = max(64, int(chunk_size))
        if chunk_size == self.chunk_size:
            return
        was_running = self._running
        if was_running:
            self.stop()
        self.chunk_size = chunk_size
        if was_running:
            self.start()

    @staticmethod
    def list_devices() -> list[dict]:
        devices = sd.query_devices()
        mic_devices = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                mic_devices.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sample_rate": d["default_samplerate"],
                })
        return mic_devices

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            latency="low",
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._running:
            return
        chunk = indata[:, 0].copy()
        # Audio-thread fast path (low jitter — used by vocal monitor)
        proc = self._processor
        if proc is not None:
            try:
                proc(chunk, self.sample_rate)
            except Exception:
                log.debug("MicInput audio-thread processor raised", exc_info=True)
        self.audio_chunk.emit(chunk, self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._running
