import logging
import sys
from math import gcd

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from typing import Optional

from src.audio.device_hints import (
    is_bluetooth_output_device,
    max_output_channels,
    preferred_blocksize_for_output,
    preferred_latency_for_output,
)

log = logging.getLogger(__name__)


def _wasapi_extras_for(device: Optional[int]):
    """Return ``sd.WasapiSettings()`` if ``device`` is on the WASAPI host API.

    Using WASAPI shared mode dramatically reduces output latency on Windows.
    Returns ``None`` for non-WASAPI devices (or any error) so the caller can
    safely fall through to the host's default backend.
    """
    if sys.platform != "win32":
        return None
    WasapiSettings = getattr(sd, "WasapiSettings", None)
    if WasapiSettings is None:
        return None
    if device is None:
        return None
    try:
        info = sd.query_devices(device)
        host = sd.query_hostapis(info["hostapi"])
        name = str(host.get("name", "")).lower()
        if "wasapi" in name:
            return WasapiSettings()
    except Exception:
        pass
    return None


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
                name = str(d["name"])
                out.append({
                    "index": i,
                    "name": name,
                    "channels": d["max_output_channels"],
                    "sample_rate": d["default_samplerate"],
                    "bluetooth_hint": is_bluetooth_output_device(i),
                })
        return out

    def __init__(self, device: Optional[int] = None, blocksize: int = 512):
        super().__init__()
        self.device = device
        self.blocksize = int(blocksize)
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._stream: Optional[sd.OutputStream] = None
        self._position: int = 0  # current sample position
        self._playing = False
        self._paused = False
        self._duration: float = 0.0
        self._file_channels: int = 1
        self._stream_out_channels: int = 1  # PortAudio stream channel count (may differ from file)

        self._timer = QTimer()
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._emit_position)

    def _make_output_stream(self) -> sd.OutputStream:
        if self._audio_data is None:
            raise RuntimeError("AudioPlayback: load a file before opening an output stream")

        extras = _wasapi_extras_for(self.device)
        dev_max = max_output_channels(self.device)
        self._stream_out_channels = max(1, min(self._file_channels, dev_max))
        if self._stream_out_channels < self._file_channels:
            log.info(
                "Output device allows %d ch; backing track has %d — mixing/downmixing for playback.",
                dev_max,
                self._file_channels,
            )

        base_bs = preferred_blocksize_for_output(self.device, self.blocksize)
        lat_primary = preferred_latency_for_output(self.device)
        attempts: list[tuple[str, int]] = [(lat_primary, base_bs)]
        if not is_bluetooth_output_device(self.device):
            attempts.append(("high", max(base_bs, 1024)))
        attempts.append(("high", max(base_bs, 2048)))

        seen: set[tuple[str, int]] = set()
        deduped: list[tuple[str, int]] = []
        for pair in attempts:
            if pair not in seen:
                seen.add(pair)
                deduped.append(pair)

        def _try_open(use_extras: bool, latency: str, bs: int) -> sd.OutputStream:
            kwargs = dict(
                samplerate=self._sample_rate,
                channels=self._stream_out_channels,
                device=self.device,
                dtype="float32",
                callback=self._playback_callback,
                blocksize=max(64, int(bs)),
                latency=latency,
            )
            if use_extras and extras is not None:
                kwargs["extra_settings"] = extras
            return sd.OutputStream(**kwargs)

        last_err: Optional[Exception] = None
        for latency, bs in deduped:
            for use_extras in (True, False):
                if use_extras and extras is None:
                    continue
                try:
                    return _try_open(use_extras, latency, bs)
                except sd.PortAudioError as e:
                    last_err = e
                    log.debug(
                        "OutputStream open failed latency=%s blocksize=%d wasapi=%s: %s",
                        latency,
                        bs,
                        use_extras,
                        e,
                    )

        target_sr = self._device_default_samplerate()
        if target_sr is not None and target_sr != self._sample_rate:
            log.info(
                "Output device rejected %d Hz; resampling backing track to %d Hz",
                self._sample_rate,
                target_sr,
            )
            self._resample_to(target_sr)
            try:
                return _try_open(False, "high", max(base_bs, 1024))
            except sd.PortAudioError as e:
                last_err = e

        if last_err is not None:
            raise last_err
        raise RuntimeError("AudioPlayback: failed to open output stream")

    def _device_default_samplerate(self) -> Optional[int]:
        try:
            if self.device is None:
                info = sd.query_devices(kind="output")
            else:
                info = sd.query_devices(self.device)
            sr = int(info.get("default_samplerate", 0) or 0)
            return sr if sr > 0 else None
        except Exception:
            return None

    def _resample_to(self, target_sr: int) -> None:
        """Resample the loaded buffer in-place to ``target_sr`` (mono / multi-ch)."""
        if self._audio_data is None or target_sr <= 0 or target_sr == self._sample_rate:
            return
        from scipy.signal import resample_poly  # local import to avoid startup cost

        old_sr = self._sample_rate
        g = gcd(int(target_sr), int(old_sr))
        up = int(target_sr // g)
        down = int(old_sr // g)
        n_ch = self._audio_data.shape[1]
        # resample_poly works on 1-D; do per-channel and stack
        cols = []
        for ch in range(n_ch):
            cols.append(
                resample_poly(self._audio_data[:, ch], up, down).astype(np.float32, copy=False)
            )
        new_len = min(len(c) for c in cols)
        new_data = np.empty((new_len, n_ch), dtype=np.float32)
        for ch, c in enumerate(cols):
            new_data[:, ch] = c[:new_len]
        # Preserve current playback position in seconds
        cur_time = self._position / max(1, old_sr)
        self._audio_data = new_data
        self._sample_rate = int(target_sr)
        self._duration = new_len / float(target_sr)
        self._position = max(0, min(int(cur_time * target_sr), new_len))

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
        self._file_channels = int(data.shape[1])
        self._sample_rate = sr
        self._duration = len(data) / sr
        self._position = 0

    def set_blocksize(self, blocksize: int) -> None:
        """Update preferred device block size; reopens the stream if needed."""
        blocksize = max(64, int(blocksize))
        if blocksize == self.blocksize:
            return
        self.blocksize = blocksize
        if self._stream is None:
            return
        was_paused = self._paused
        was_active = self._playing and not self._paused
        self._playing = False
        self._timer.stop()
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if was_active:
            self._playing = True
            self._paused = False
            self._stream = self._make_output_stream()
            self._stream.start()
            self._timer.start()
        elif was_paused:
            self._paused = True
            self._playing = False

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
            sl = self._audio_data[self._position : len(self._audio_data)]
            fc = self._file_channels
            oc = self._stream_out_channels
            if fc == oc:
                outdata[:remaining] = sl
            elif fc == 2 and oc == 1:
                outdata[:remaining, 0] = 0.5 * (sl[:, 0] + sl[:, 1])
            elif fc == 1 and oc >= 2:
                outdata[:remaining, 0] = sl[:, 0]
                for c in range(1, oc):
                    outdata[:remaining, c] = sl[:, 0]
            else:
                outdata[:remaining] = sl[:, :oc]
            outdata[remaining:] = 0
            self._position = len(self._audio_data)
            self._playing = False
            self.playback_finished.emit()
        else:
            sl = self._audio_data[self._position:end]
            fc = self._file_channels
            oc = self._stream_out_channels
            if fc == oc:
                outdata[:] = sl
            elif fc == 2 and oc == 1:
                outdata[:, 0] = 0.5 * (sl[:, 0] + sl[:, 1])
            elif fc == 1 and oc >= 2:
                outdata[:, 0] = sl[:, 0]
                for c in range(1, oc):
                    outdata[:, c] = sl[:, 0]
            else:
                outdata[:] = sl[:, :oc]
            self._position = end

    def _emit_position(self):
        if self._playing:
            self.position_changed.emit(self.current_time)

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused
