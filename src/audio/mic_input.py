import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from typing import Optional


class MicInput(QObject):
    """Captures audio from the microphone in real-time."""

    audio_chunk = pyqtSignal(np.ndarray, int)  # mono float32 chunk, sample_rate

    def __init__(
        self, sample_rate: int = 44100, chunk_size: int = 2048,
        device: Optional[int] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._running = False

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
        self.audio_chunk.emit(chunk, self.sample_rate)

    @property
    def is_running(self) -> bool:
        return self._running
