import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Optional


class SystemAudioCapture(QObject):
    """Captures system audio via WASAPI loopback on Windows."""

    audio_chunk = pyqtSignal(np.ndarray, int)  # mono float32 chunk, sample_rate

    def __init__(self, chunk_size: int = 2048):
        super().__init__()
        self.chunk_size = chunk_size
        self._stream = None
        self._running = False
        self._pyaudio = None
        self._sample_rate = 44100

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self):
        if self._running:
            return
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise RuntimeError(
                "PyAudioWPatch is required for system audio capture. "
                "Install it with: pip install PyAudioWPatch"
            )

        self._pyaudio = pyaudio.PyAudio()

        try:
            wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI not available on this system")

        default_speakers = self._pyaudio.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in self._pyaudio.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                raise RuntimeError("No loopback device found for default speakers")

        self._sample_rate = int(default_speakers["defaultSampleRate"])
        channels = default_speakers["maxInputChannels"]

        self._running = True
        self._channels = channels

        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=default_speakers["index"],
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        if not self._running:
            return (None, pyaudio.paComplete)

        audio = np.frombuffer(in_data, dtype=np.float32)
        if self._channels > 1:
            audio = audio.reshape(-1, self._channels).mean(axis=1)

        self.audio_chunk.emit(audio.copy(), self._sample_rate)
        return (None, pyaudio.paContinue)

    @property
    def is_running(self) -> bool:
        return self._running
