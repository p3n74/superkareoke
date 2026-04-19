"""Live vocal monitor output stream.

Pulls processed mic audio from a thread-safe ring buffer and writes it to a
``sounddevice.OutputStream``. The output device can be changed on the fly
without restarting the singer.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from src.audio.playback import _wasapi_extras_for

log = logging.getLogger(__name__)


class VocalMonitor:
    def __init__(self, sample_rate: int = 44100, blocksize: int = 512):
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.device: Optional[int] = None
        self.enabled: bool = False
        self.gain_lin: float = 1.0
        self._lock = threading.Lock()
        # Generous ring (~16 blocks ≈ 186 ms @ 512/44.1 k). With the writer
        # now running on the PortAudio thread (no Qt-loop jitter) we don't
        # need this much, but giving the buffer headroom kills the residual
        # underruns that show up as "kztzt".
        self._cap = self._compute_capacity()
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._w = 0
        self._r = 0
        self._underruns = 0
        self._dropped_overflow = 0
        self._prime_silence()
        self._stream: Optional[sd.OutputStream] = None

    def _compute_capacity(self) -> int:
        return max(8192, self.blocksize * 16)

    def _prime_silence(self) -> None:
        # Pre-fill ~2 blocks of silence so the very first reader callbacks
        # always find data. This also gives the writer a comfortable steady
        # state of "~2 blocks ahead" rather than racing the reader.
        self._w = self.blocksize * 2
        self._r = 0

    # ── Lifecycle ───────────────────────────────────────────────────────
    def set_device(self, device: Optional[int]) -> None:
        if device == self.device:
            return
        self.device = device
        was_running = self._stream is not None
        if was_running:
            self._close_stream()
        if was_running and self.enabled:
            self._open_stream()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            if self._stream is None:
                self._open_stream()
        else:
            self._close_stream()

    def set_gain_db(self, db: float) -> None:
        self.gain_lin = float(10.0 ** (float(db) / 20.0))

    def set_sample_rate(self, sr: int) -> None:
        sr = int(sr)
        if sr == self.sample_rate:
            return
        self.sample_rate = sr
        self._cap = self._compute_capacity()
        with self._lock:
            self._buf = np.zeros(self._cap, dtype=np.float32)
            self._prime_silence()
        if self.enabled and self._stream is not None:
            self._close_stream()
            self._open_stream()

    def write(self, mono: np.ndarray, source_sr: Optional[int] = None) -> None:
        """Push processed mic audio.

        ``source_sr`` is the sample rate of ``mono``. When the monitor stream
        is running at a different rate (e.g. WASAPI shared mode forced 48 kHz
        for a 44.1 kHz mic) the input is resampled in-place to the monitor's
        rate. This is what prevents the "kztztzt" buffer-rate mismatch.
        """
        if not self.enabled or mono.size == 0:
            return
        if source_sr is not None and int(source_sr) != int(self.sample_rate):
            try:
                from math import gcd
                from scipy.signal import resample_poly
                src = int(source_sr)
                dst = int(self.sample_rate)
                g = gcd(src, dst) or 1
                up = max(1, dst // g)
                down = max(1, src // g)
                mono = resample_poly(
                    mono.astype(np.float32, copy=False), up, down
                ).astype(np.float32, copy=False)
            except Exception:
                log.debug("VocalMonitor: resample failed, dropping chunk", exc_info=True)
                return
        chunk = (mono.astype(np.float32, copy=False) * np.float32(self.gain_lin))
        n = chunk.shape[0]
        # Cap queued audio at ~6 blocks (~70 ms @ 512/44.1 k). Above that,
        # latency starts to feel sluggish — but below ~4 blocks we underrun
        # on PortAudio's high-latency hosts (MME / DirectSound on Windows).
        target_max = self.blocksize * 6
        with self._lock:
            cap = self._cap
            start = self._w % cap
            end = start + n
            if end <= cap:
                self._buf[start:end] = chunk
            else:
                first = cap - start
                self._buf[start:] = chunk[:first]
                self._buf[: n - first] = chunk[first:]
            self._w += n
            queued = self._w - self._r
            if queued > target_max:
                self._dropped_overflow += queued - target_max
                self._r = self._w - target_max

    def shutdown(self) -> None:
        self._close_stream()

    # ── Stream plumbing ─────────────────────────────────────────────────
    def _open_stream(self) -> None:
        try:
            with self._lock:
                self._prime_silence()
            extras = _wasapi_extras_for(self.device)

            def _try_open(sr: int, use_extras: bool) -> sd.OutputStream:
                kwargs = dict(
                    samplerate=sr,
                    channels=1,
                    device=self.device,
                    dtype="float32",
                    callback=self._callback,
                    blocksize=self.blocksize,
                    latency="low",
                )
                if use_extras and extras is not None:
                    kwargs["extra_settings"] = extras
                return sd.OutputStream(**kwargs)

            try:
                self._stream = _try_open(self.sample_rate, use_extras=True)
            except sd.PortAudioError:
                try:
                    self._stream = _try_open(self.sample_rate, use_extras=False)
                except sd.PortAudioError:
                    target_sr = self._device_default_samplerate()
                    if target_sr and target_sr != self.sample_rate:
                        log.info(
                            "VocalMonitor device rejected %d Hz; switching to device default %d Hz",
                            self.sample_rate, target_sr,
                        )
                        self.set_sample_rate(target_sr)
                        self._stream = _try_open(self.sample_rate, use_extras=False)
                    else:
                        raise
            self._stream.start()
            log.info(
                "VocalMonitor stream open (device=%s, sr=%d, blocksize=%d, latency=%s, wasapi=%s)",
                self.device,
                self.sample_rate,
                self.blocksize,
                self._stream.latency,
                extras is not None,
            )
        except Exception:
            log.warning("VocalMonitor failed to open output stream", exc_info=True)
            self._stream = None
            self.enabled = False

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

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            log.debug("VocalMonitor close error", exc_info=True)
        finally:
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        with self._lock:
            available = self._w - self._r
            n = min(frames, available)
            if n > 0:
                cap = self._cap
                start = self._r % cap
                end = start + n
                if end <= cap:
                    out[:n] = self._buf[start:end]
                else:
                    first = cap - start
                    out[:first] = self._buf[start:]
                    out[first:n] = self._buf[: n - first]
                self._r += n
            if n < frames:
                self._underruns += (frames - n)
                out[n:] = 0.0

    def diagnostics(self) -> dict:
        """Snapshot underrun / overflow counters (samples since stream open)."""
        with self._lock:
            return {
                "queued_samples": self._w - self._r,
                "underrun_samples": int(self._underruns),
                "dropped_overflow_samples": int(self._dropped_overflow),
                "capacity": int(self._cap),
                "sample_rate": int(self.sample_rate),
            }
