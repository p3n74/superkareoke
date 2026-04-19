import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Optional


class AudioPlayback(QObject):
    """Plays audio files (instrumental/backing tracks) with position tracking."""

    position_changed = pyqtSignal(float)  # current position in seconds
    playback_finished = pyqtSignal()

    @staticmethod
    def list_output_devices() -> list[dict]:
        """Devices with at least one output channel (PortAudio indices)."""
        devices = sd.query_devices()
        out: list[dict] = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                out.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_output_channels"],
                    "sample_rate": d["default_samplerate"],
                })
        return out

    def __init__(self, device: Optional[int] = None):
        super().__init__()
        self.device = device
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._stream: Optional[sd.OutputStream] = None
        self._position: int = 0  # current sample position
        self._playing = False
        self._paused = False
        self._duration: float = 0.0

        self._timer = QTimer()
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._emit_position)

    def _make_output_stream(self) -> sd.OutputStream:
        if self._audio_data is None:
            raise RuntimeError("AudioPlayback: load a file before opening an output stream")
        return sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._audio_data.shape[1],
            device=self.device,
            dtype="float32",
            callback=self._playback_callback,
            blocksize=2048,
        )

    def set_output_device(self, device: Optional[int]) -> None:
        """Change PortAudio output (``None`` = host default). Reopens the stream if playback is active or paused."""
        if device == self.device:
            return
        self.device = device
        if self._stream is None:
            return
        pos = self._position
        was_paused = self._paused
        was_active = self._playing and not self._paused
        self._playing = False
        self._timer.stop()
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._position = pos
        if was_active:
            self._playing = True
            self._paused = False
            self._stream = self._make_output_stream()
            self._stream.start()
            self._timer.start()
        elif was_paused:
            self._playing = False
            self._paused = True

    def load(self, audio_path: Path):
        self.stop()
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._audio_data = data
        self._sample_rate = sr
        self._duration = len(data) / sr
        self._position = 0

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def current_time(self) -> float:
        if self._audio_data is None:
            return 0.0
        return self._position / self._sample_rate

    def play(self):
        if self._audio_data is None:
            return

        if self._paused and self._stream is not None:
            self._paused = False
            self._playing = True
            self._timer.start()
            return

        # Paused after output device change (stream closed, position unchanged)
        if self._paused and self._stream is None:
            self._paused = False
            self._playing = True
            self._stream = self._make_output_stream()
            self._stream.start()
            self._timer.start()
            return

        self.stop()
        self._playing = True
        self._stream = self._make_output_stream()
        self._stream.start()
        self._timer.start()

    def pause(self):
        self._paused = True
        self._playing = False
        self._timer.stop()

    def stop(self):
        self._playing = False
        self._paused = False
        self._timer.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._position = 0

    def seek(self, time_s: float):
        if self._audio_data is None:
            return
        self._position = int(time_s * self._sample_rate)
        self._position = max(0, min(self._position, len(self._audio_data)))

    def _playback_callback(self, outdata, frames, time_info, status):
        if not self._playing or self._paused or self._audio_data is None:
            outdata[:] = 0
            return

        end = self._position + frames
        if end > len(self._audio_data):
            remaining = len(self._audio_data) - self._position
            outdata[:remaining] = self._audio_data[self._position:len(self._audio_data)]
            outdata[remaining:] = 0
            self._position = len(self._audio_data)
            self._playing = False
            self.playback_finished.emit()
        else:
            outdata[:] = self._audio_data[self._position:end]
            self._position = end

    def _emit_position(self):
        if self._playing:
            self.position_changed.emit(self.current_time)

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused
