"""Microphone cleanup chain: input gain + high-pass + noise gate + optional VAD.

The pipeline produces two signals:

* ``clean``   — fed to the pitch detector (must stay neutral / wide-band).
* ``monitor`` — same as clean, optionally soft-masked by a voice-activity
  detector so the singer's headphones stay quiet during silences.

Speaker-bleed cancellation has been removed — it added artefacts singers
described as "static / watery" — so this stage is intentionally simple and
fast. The only reference-aware processing left is the optional VAD.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import lfilter

from src.config import MIC_PROCESSOR_DEFAULTS

log = logging.getLogger(__name__)


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


@dataclass
class MicProcessorSettings:
    input_gain_db: float = 0.0
    gate_db: float = -42.0
    use_vad: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "MicProcessorSettings":
        base = dict(MIC_PROCESSOR_DEFAULTS)
        if d:
            base.update(d)
        return cls(
            input_gain_db=float(base.get("input_gain_db", 0.0)),
            gate_db=float(base.get("gate_db", -42.0)),
            use_vad=bool(base.get("use_vad", False)),
        )


class _NoiseGate:
    """RMS noise gate with attack/release smoothing and hysteresis.

    The output gain is **linearly ramped across each block** instead of being
    a single scalar applied to the whole 512-sample block. That's what removes
    the audible click/zip you'd otherwise hear at every block boundary when
    the gate opens or closes.
    """

    def __init__(
        self,
        sr: int = 44100,
        threshold_db: float = -42.0,
        attack_ms: float = 5.0,
        release_ms: float = 180.0,
        hysteresis_db: float = 8.0,
    ):
        self.sr = int(sr)
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.hysteresis_db = float(hysteresis_db)
        self._gain_state = 0.0
        self._open = False

    def set_threshold_db(self, db: float) -> None:
        self.threshold_db = float(db)

    def process(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        n = x.shape[0]
        rms_db = 20.0 * math.log10(_safe_rms(x) + 1e-12)
        open_thresh = self.threshold_db
        close_thresh = self.threshold_db - self.hysteresis_db
        if not self._open and rms_db > open_thresh:
            self._open = True
        elif self._open and rms_db < close_thresh:
            self._open = False
        target = 1.0 if self._open else 0.0
        ms = self.attack_ms if target > self._gain_state else self.release_ms
        block_ms = 1000.0 * n / max(1, self.sr)
        coeff = math.exp(-block_ms / max(0.5, ms))
        prev_gain = float(self._gain_state)
        new_gain = coeff * prev_gain + (1.0 - coeff) * target
        self._gain_state = new_gain
        if abs(new_gain - prev_gain) < 1e-6:
            return (x * np.float32(new_gain)), rms_db
        ramp = np.linspace(prev_gain, new_gain, n, dtype=np.float32, endpoint=True)
        return (x * ramp), rms_db


class MicProcessor:
    """Cleanup orchestrator. Call ``process(mic_chunk)`` per audio block.

    Returns ``(mic_clean, mic_for_monitor)`` — both float32, same length.
    The two paths only diverge when VAD is enabled (it soft-masks the monitor
    output).
    """

    def __init__(self, sample_rate: int = 44100, settings: MicProcessorSettings | None = None):
        self.sample_rate = int(sample_rate)
        self.settings = settings or MicProcessorSettings()
        self._gate = _NoiseGate(sr=self.sample_rate, threshold_db=self.settings.gate_db)
        self._vad_model = None
        self._vad_attempted = False
        self._vad_smooth = 0.0
        self._hpf_b, self._hpf_a = self._hpf_coeffs(80.0)
        self._hpf_zi = np.zeros(2, dtype=np.float64)
        self._last_rms_db = -120.0

    def apply_settings(self, settings: MicProcessorSettings) -> None:
        self.settings = settings
        self._gate.set_threshold_db(settings.gate_db)
        if not settings.use_vad:
            self._vad_model = None
            self._vad_attempted = False
            self._vad_smooth = 0.0

    @property
    def last_rms_db(self) -> float:
        return float(self._last_rms_db)

    # Kept for backward compat with PerformanceView's existing call site.
    @property
    def aec_delay_samples(self) -> int:
        return 0

    def _hpf_coeffs(self, fc: float) -> tuple[np.ndarray, np.ndarray]:
        sr = self.sample_rate
        if fc <= 0.0 or fc >= 0.5 * sr:
            return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        w0 = 2.0 * math.pi * fc / sr
        q = 0.707
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        return (
            np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
            np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
        )

    def _hpf(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y, self._hpf_zi = lfilter(
            self._hpf_b, self._hpf_a, x.astype(np.float64, copy=False), zi=self._hpf_zi,
        )
        return y.astype(np.float32, copy=False)

    def _try_init_vad(self) -> bool:
        if self._vad_attempted:
            return self._vad_model is not None
        self._vad_attempted = True
        try:
            import torch  # noqa: F401
            model, _utils = __import__("torch").hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._vad_model = model
            log.info("Silero VAD loaded")
            return True
        except Exception:
            log.info("Silero VAD unavailable; skipping voice gate", exc_info=True)
            self._vad_model = None
            return False

    def _vad_score(self, x: np.ndarray) -> float:
        if not self._try_init_vad() or self._vad_model is None:
            return 1.0
        try:
            import torch
            sr = 16000 if self.sample_rate >= 16000 else self.sample_rate
            if self.sample_rate != sr:
                step = self.sample_rate // sr
                if step < 1:
                    step = 1
                xs = x[::step]
            else:
                xs = x
            if xs.shape[0] < 512:
                pad = 512 - xs.shape[0]
                xs = np.concatenate([xs, np.zeros(pad, dtype=np.float32)])
            t = torch.from_numpy(xs.astype(np.float32))
            with torch.no_grad():
                score = float(self._vad_model(t, sr).item())
            return max(0.0, min(1.0, score))
        except Exception:
            log.debug("VAD inference failed", exc_info=True)
            return 1.0

    def process(
        self, mic: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if mic.size == 0:
            return mic, mic
        s = self.settings

        x = mic.astype(np.float32, copy=False)
        if s.input_gain_db != 0.0:
            x = x * np.float32(_db_to_linear(s.input_gain_db))

        # Absolute hard floor: if the raw mic block is below ~-65 dBFS RMS the
        # user simply isn't producing sound. Don't run any further processing
        # on it — sending bit-noise into the FX chain (especially reverb) is
        # what produces the audible static when the gain slider is at zero.
        raw_rms_db = 20.0 * math.log10(_safe_rms(x) + 1e-12)
        if raw_rms_db < -65.0:
            self._last_rms_db = raw_rms_db
            self._gate._gain_state *= 0.5  # bleed off any residual gain
            self._hpf_zi *= 0.0
            zero = np.zeros_like(x, dtype=np.float32)
            return zero, zero

        x = self._hpf(x)
        gated, rms_db = self._gate.process(x)
        self._last_rms_db = rms_db
        clean = gated

        monitor = clean
        if s.use_vad:
            score = self._vad_score(clean)
            self._vad_smooth = 0.7 * self._vad_smooth + 0.3 * score
            mask = float(np.clip(self._vad_smooth, 0.0, 1.0))
            monitor = clean * np.float32(mask)

        return clean, monitor

    def reset(self) -> None:
        self._hpf_zi = np.zeros(2, dtype=np.float64)
        self._gate._gain_state = 0.0
        self._gate._open = False
        self._vad_smooth = 0.0
        self._last_rms_db = -120.0
