"""Real-time vocal effects chain: HPF → 3-band EQ → compressor → reverb → gain.

All stages are stateful so audio can be processed in arbitrary block sizes
without click artifacts at block boundaries. Designed for mono float32 input
at the microphone sample rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from src.config import voice_fx_preset


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# ── Biquad helpers (Audio EQ Cookbook) ────────────────────────────────────


def _biquad_highpass(fc: float, sr: int, q: float = 0.707) -> tuple[np.ndarray, np.ndarray]:
    if fc <= 0.0:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    w0 = 2.0 * math.pi * fc / sr
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


def _biquad_low_shelf(fc: float, sr: int, gain_db: float, slope: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + beta)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - beta)
    a0 = (A + 1) + (A - 1) * cos_w0 + beta
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - beta
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


def _biquad_high_shelf(fc: float, sr: int, gain_db: float, slope: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + beta)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - beta)
    a0 = (A + 1) - (A - 1) * cos_w0 + beta
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - beta
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


def _biquad_peaking(fc: float, sr: int, gain_db: float, q: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


class _Biquad:
    """Vectorized biquad filter (uses scipy.signal.lfilter for the inner loop)."""

    __slots__ = ("b", "a", "_zi")

    def __init__(self) -> None:
        self.b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._zi = np.zeros(2, dtype=np.float64)

    def set(self, b: np.ndarray, a: np.ndarray) -> None:
        self.b = b.astype(np.float64, copy=False)
        self.a = a.astype(np.float64, copy=False)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y, self._zi = lfilter(self.b, self.a, x.astype(np.float64, copy=False), zi=self._zi)
        return y.astype(np.float32, copy=False)


# ── Compressor ────────────────────────────────────────────────────────────


@dataclass
class _CompressorState:
    """Vectorized compressor.

    Uses a single envelope coefficient (geometric mean of attack/release) so the
    detector becomes a one-pole IIR that we can run with ``scipy.signal.lfilter``
    in a single vectorized call instead of a per-sample Python loop.
    """

    env: float = 1e-6
    sr: int = 44100
    threshold_db: float = -18.0
    ratio: float = 2.0
    attack_ms: float = 8.0
    release_ms: float = 120.0
    makeup_db: float = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.ratio <= 1.001 or x.size == 0:
            return x
        atk_t = max(1.0, self.attack_ms * 1e-3 * self.sr)
        rel_t = max(1.0, self.release_ms * 1e-3 * self.sr)
        # Single coefficient, balanced between attack and release
        time_const = math.sqrt(atk_t * rel_t)
        a = math.exp(-1.0 / time_const)
        b_coef = 1.0 - a
        ax = np.abs(x.astype(np.float64, copy=False))
        # Direct-Form II Transposed initial state for a 1st-order filter
        # b=[b_coef], a=[1, -a]: zi[0] = a * y[-1]
        zi = np.array([a * self.env], dtype=np.float64)
        env, zf = lfilter([b_coef], [1.0, -a], ax, zi=zi)
        self.env = float(zf[0]) / max(a, 1e-12)
        thresh_lin = _db_to_linear(self.threshold_db)
        slope = 1.0 - 1.0 / self.ratio
        makeup = _db_to_linear(self.makeup_db)
        # Vectorized gain computation: above threshold → (env / thresh)^(-slope)
        gain = np.ones_like(env, dtype=np.float64)
        mask = env > thresh_lin
        if np.any(mask):
            gain[mask] = (env[mask] / thresh_lin) ** (-slope)
        out = x.astype(np.float64, copy=False) * gain * makeup
        return out.astype(np.float32, copy=False)


# ── Freeverb-style reverb ─────────────────────────────────────────────────

_FREEVERB_COMB_LENGTHS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
_FREEVERB_ALLPASS_LENGTHS = [556, 441, 341, 225]


class _CombFilter:
    """Lowpass-feedback comb (Freeverb style), vectorized.

    The output is just the delayed buffer slice (vectorized read), and the
    one-pole damping IIR on the read-back is computed with lfilter so there is
    no per-sample Python loop.
    """

    __slots__ = ("buf", "idx", "feedback", "damp", "store")

    def __init__(self, length: int) -> None:
        self.buf = np.zeros(max(1, int(length)), dtype=np.float32)
        self.idx = 0
        self.feedback = 0.84
        self.damp = 0.5
        self.store = 0.0  # one-pole filter state (last damped sample)

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        buf = self.buf
        L = buf.shape[0]
        idx = self.idx
        damp1 = float(self.damp)
        damp2 = 1.0 - damp1
        feedback = float(self.feedback)
        out = np.empty(n, dtype=np.float32)

        i = 0
        while i < n:
            chunk = min(n - i, L - idx)
            # Output is the buffered (delayed) audio
            y_chunk = buf[idx : idx + chunk].astype(np.float64, copy=True)
            out[i : i + chunk] = y_chunk.astype(np.float32, copy=False)
            # store[k] = damp2 * y[k] + damp1 * store[k-1]
            # One-pole IIR with b=[damp2], a=[1, -damp1]; zi seeded from prior store.
            zi = np.array([damp1 * self.store], dtype=np.float64)
            store_chunk, zf = lfilter([damp2], [1.0, -damp1], y_chunk, zi=zi)
            self.store = float(zf[0]) / max(damp1, 1e-12) if damp1 > 0 else float(store_chunk[-1])
            new_buf = (
                x[i : i + chunk].astype(np.float64, copy=False)
                + feedback * store_chunk
            ).astype(np.float32, copy=False)
            buf[idx : idx + chunk] = new_buf
            idx += chunk
            if idx >= L:
                idx = 0
            i += chunk

        self.idx = idx
        return out


class _AllPass:
    """Schroeder all-pass, fully vectorized (no recursive scalar state)."""

    __slots__ = ("buf", "idx", "feedback")

    def __init__(self, length: int) -> None:
        self.buf = np.zeros(max(1, int(length)), dtype=np.float32)
        self.idx = 0
        self.feedback = 0.5

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        buf = self.buf
        L = buf.shape[0]
        idx = self.idx
        fb = float(self.feedback)
        out = np.empty(n, dtype=np.float32)

        i = 0
        while i < n:
            chunk = min(n - i, L - idx)
            b_chunk = buf[idx : idx + chunk].copy()
            x_chunk = x[i : i + chunk]
            out[i : i + chunk] = (-x_chunk + b_chunk).astype(np.float32, copy=False)
            buf[idx : idx + chunk] = (x_chunk + b_chunk * fb).astype(np.float32, copy=False)
            idx += chunk
            if idx >= L:
                idx = 0
            i += chunk

        self.idx = idx
        return out


class _Reverb:
    """Subset of Freeverb (4 combs + 2 allpass) — small, cheap, mono."""

    def __init__(self, sr: int = 44100) -> None:
        scale = sr / 44100.0
        comb_lens = [int(L * scale) for L in _FREEVERB_COMB_LENGTHS[:4]]
        ap_lens = [int(L * scale) for L in _FREEVERB_ALLPASS_LENGTHS[:2]]
        self.combs = [_CombFilter(L) for L in comb_lens]
        self.allpasses = [_AllPass(L) for L in ap_lens]
        self.set_params(room=0.5, damp=0.5)

    def set_params(self, room: float, damp: float) -> None:
        # room (0..1) → feedback; damp (0..1) → low-pass amount
        feedback = 0.7 + 0.28 * float(np.clip(room, 0.0, 1.0))
        damp_v = float(np.clip(damp, 0.0, 1.0))
        for c in self.combs:
            c.feedback = feedback
            c.damp = damp_v

    def process(self, x: np.ndarray) -> np.ndarray:
        wet = np.zeros_like(x, dtype=np.float32)
        for c in self.combs:
            wet = wet + c.process(x)
        wet = wet / max(1, len(self.combs))
        for ap in self.allpasses:
            wet = ap.process(wet)
        return wet


# ── Top-level chain ───────────────────────────────────────────────────────


@dataclass
class VoiceFxParams:
    hpf_hz: float = 0.0
    eq_low_db: float = 0.0
    eq_mid_db: float = 0.0
    eq_high_db: float = 0.0
    comp_threshold_db: float = 0.0
    comp_ratio: float = 1.0
    reverb_mix: float = 0.0
    reverb_room: float = 0.5
    reverb_damp: float = 0.5
    output_gain_db: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


class VoiceFx:
    """Stateful effects chain.

    Use ``apply_preset`` or ``apply_params`` to update settings; ``process``
    one mono float32 block at a time.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._hpf = _Biquad()
        self._eq_low = _Biquad()
        self._eq_mid = _Biquad()
        self._eq_high = _Biquad()
        self._comp = _CompressorState(sr=sample_rate)
        self._reverb = _Reverb(sample_rate)
        self.params = VoiceFxParams()
        self._coeffs_dirty = True
        self.apply_preset("none")

    def apply_preset(self, name: str) -> None:
        d = voice_fx_preset(name)
        self.apply_params(d)

    def apply_params(self, p: dict) -> None:
        self.params = VoiceFxParams(
            hpf_hz=float(p.get("hpf_hz", 0.0)),
            eq_low_db=float(p.get("eq_low_db", 0.0)),
            eq_mid_db=float(p.get("eq_mid_db", 0.0)),
            eq_high_db=float(p.get("eq_high_db", 0.0)),
            comp_threshold_db=float(p.get("comp_threshold_db", 0.0)),
            comp_ratio=float(p.get("comp_ratio", 1.0)),
            reverb_mix=float(p.get("reverb_mix", 0.0)),
            reverb_room=float(p.get("reverb_room", 0.5)),
            reverb_damp=float(p.get("reverb_damp", 0.5)),
            output_gain_db=float(p.get("output_gain_db", 0.0)),
        )
        self._coeffs_dirty = True

    def set_sample_rate(self, sr: int) -> None:
        if sr == self.sample_rate:
            return
        self.sample_rate = int(sr)
        self._comp = _CompressorState(sr=self.sample_rate)
        self._reverb = _Reverb(self.sample_rate)
        self._coeffs_dirty = True

    def _refresh_coeffs(self) -> None:
        sr = self.sample_rate
        p = self.params
        b, a = _biquad_highpass(p.hpf_hz, sr) if p.hpf_hz > 0.0 else (
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        )
        self._hpf.set(b, a)
        self._eq_low.set(*_biquad_low_shelf(180.0, sr, p.eq_low_db))
        self._eq_mid.set(*_biquad_peaking(1200.0, sr, p.eq_mid_db, q=0.9))
        self._eq_high.set(*_biquad_high_shelf(6000.0, sr, p.eq_high_db))
        self._comp.threshold_db = p.comp_threshold_db
        self._comp.ratio = max(1.0, p.comp_ratio)
        self._reverb.set_params(p.reverb_room, p.reverb_damp)
        self._coeffs_dirty = False

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self._coeffs_dirty:
            self._refresh_coeffs()
        y = x.astype(np.float32, copy=False)
        if self.params.hpf_hz > 0.0:
            y = self._hpf.process(y)
        if self.params.eq_low_db != 0.0:
            y = self._eq_low.process(y)
        if self.params.eq_mid_db != 0.0:
            y = self._eq_mid.process(y)
        if self.params.eq_high_db != 0.0:
            y = self._eq_high.process(y)
        if self.params.comp_ratio > 1.001:
            y = self._comp.process(y)
        if self.params.reverb_mix > 1e-4:
            wet = self._reverb.process(y)
            mix = float(np.clip(self.params.reverb_mix, 0.0, 1.0))
            y = (1.0 - mix) * y + mix * wet
        if self.params.output_gain_db != 0.0:
            y = y * _db_to_linear(self.params.output_gain_db)
        # Soft clip to avoid hard digital clipping when boosted
        return np.tanh(y).astype(np.float32, copy=False)
