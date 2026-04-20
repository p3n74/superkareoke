import logging
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)

# Virtual loopback drivers often used on macOS (and occasionally elsewhere).
_LOOPBACK_NAME_HINTS = (
    "blackhole",
    "soundflower",
    "loopback",
    "wave link",
    "wavelink",
    "groundcontrol",
)


def system_audio_uses_wasapi_loopback() -> bool:
    """True when the host provides WASAPI loopback (Windows + PyAudioWPatch)."""
    return sys.platform == "win32"


def list_system_audio_input_devices() -> list[dict]:
    """Input devices (PortAudio indices) that can carry a system / speaker mix."""
    import sounddevice as sd

    devices = sd.query_devices()
    out: list[dict] = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name_lower = str(d["name"]).lower()
        priority = 0
        for hint in _LOOPBACK_NAME_HINTS:
            if hint in name_lower:
                priority = 1
                break
        out.append({
            "index": i,
            "name": d["name"],
            "channels": d["max_input_channels"],
            "sample_rate": d["default_samplerate"],
            "priority": priority,
        })
    out.sort(key=lambda x: (-x["priority"], str(x["name"]).lower()))
    return out


def guess_default_loopback_device() -> Optional[int]:
    """Prefer a virtual loopback device if one is present."""
    for d in list_system_audio_input_devices():
        if d["priority"] > 0:
            return int(d["index"])
    return None


class SystemAudioCapture(QObject):
    """Captures the backing track / system mix for Live Mode.

    On Windows this uses WASAPI loopback via PyAudioWPatch (no extra drivers).
    On macOS and other platforms it records from a normal input device — typically
    BlackHole (or similar) configured as a Multi-Output Device with your speakers.
    """

    audio_chunk = pyqtSignal(np.ndarray, int)  # mono float32 chunk, sample_rate

    def __init__(self, chunk_size: int = 2048):
        super().__init__()
        self.chunk_size = chunk_size
        self._stream = None
        self._running = False
        self._pyaudio = None
        self._sample_rate = 44100
        self._channels = 1
        self._backend: Optional[str] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self, input_device: Optional[int] = None):
        """Begin capture.

        ``input_device`` is the PortAudio input index (macOS / Linux). Ignored on
        Windows where WASAPI loopback follows the default output device.
        """
        if self._running:
            return
        if sys.platform == "win32":
            self._start_wasapi_loopback()
        else:
            self._start_portaudio_input(input_device)

    def _start_wasapi_loopback(self) -> None:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError(
                "PyAudioWPatch is required for WASAPI loopback on Windows. "
                "Install it with: pip install PyAudioWPatch"
            ) from e

        self._pyaudio = pyaudio.PyAudio()
        self._backend = "wasapi"

        try:
            wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as e:
            self._pyaudio.terminate()
            self._pyaudio = None
            self._backend = None
            raise RuntimeError("WASAPI not available on this system") from e

        default_speakers = self._pyaudio.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in self._pyaudio.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                self._pyaudio.terminate()
                self._pyaudio = None
                self._backend = None
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
            stream_callback=self._pyaudio_callback,
        )
        self._stream.start_stream()

    def _start_portaudio_input(self, input_device: Optional[int]) -> None:
        import sounddevice as sd

        self._backend = "portaudio"
        idx = input_device
        if idx is None:
            idx = guess_default_loopback_device()
        if idx is None:
            try:
                default = sd.default.device
                if hasattr(default, "input"):
                    def_in = default.input
                elif isinstance(default, dict):
                    def_in = default.get("input")
                elif isinstance(default, (list, tuple)) and len(default) >= 1:
                    def_in = default[0]
                else:
                    def_in = None
                if isinstance(def_in, int) and def_in >= 0:
                    idx = def_in
            except Exception:
                idx = None
        if idx is None:
            raise RuntimeError(
                "No input device selected for system audio. On macOS, install "
                "BlackHole (https://github.com/ExistentialAudio/BlackHole), add a "
                "Multi-Output Device in Audio MIDI Setup that includes BlackHole and "
                "your speakers, set it as the system output, then choose the "
                "BlackHole input here."
            )

        info = sd.query_devices(idx)
        if info["max_input_channels"] <= 0:
            raise RuntimeError(f"Device «{info['name']}» has no input channels")

        self._sample_rate = int(info.get("default_samplerate") or 48000)
        self._channels = min(int(info["max_input_channels"]), 2)
        self._running = True

        log.info(
            "System audio capture (PortAudio): device=%s sr=%d ch=%d",
            info.get("name"),
            self._sample_rate,
            self._channels,
        )

        self._stream = sd.InputStream(
            device=idx,
            channels=self._channels,
            samplerate=self._sample_rate,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=self._sd_audio_callback,
            latency="low",
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
        self._backend = None

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio
        if not self._running:
            return (None, pyaudio.paComplete)

        audio = np.frombuffer(in_data, dtype=np.float32)
        if self._channels > 1:
            audio = audio.reshape(-1, self._channels).mean(axis=1)

        self.audio_chunk.emit(audio.copy(), self._sample_rate)
        return (None, pyaudio.paContinue)

    def _sd_audio_callback(self, indata, frames, time_info, status):
        if not self._running:
            return
        if self._channels == 1:
            audio = indata[:, 0].copy()
        else:
            audio = indata.mean(axis=1).astype(np.float32, copy=True)
        self.audio_chunk.emit(audio, self._sample_rate)

    @property
    def is_running(self) -> bool:
        return self._running
