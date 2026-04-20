"""Resolve which ``torch.device`` string to use for Demucs, CREPE, and live pitch.

Configure via environment (``.env`` is loaded from ``src.config`` before imports):

- **``TORCH_DEVICE``** — explicit override when GPU mode is on, e.g. ``cuda``,
  ``cuda:0``, ``mps`` (Apple GPU), ``cpu``. Invalid or unavailable devices fall back safely.
- **``TORCH_CUDA_DEVICE_INDEX``** or **``CUDA_DEVICE_INDEX``** — integer GPU index
  when ``TORCH_DEVICE`` is unset (``0`` = first NVIDIA GPU). Ignored if CUDA is
  unavailable.
- **``CUDA_VISIBLE_DEVICES``** — standard NVIDIA / driver variable (restricts
  which physical GPUs are visible); set in ``.env`` before the first ``import torch``.

When the user turns off GPU in Settings, processing always uses ``cpu`` regardless
of these variables.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def _mps_available() -> bool:
    try:
        import torch

        return bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _clamp_cuda_index(idx: int, n: int) -> int:
    if n <= 0:
        return 0
    if idx < 0 or idx >= n:
        log.warning(
            "CUDA device index %s is out of range (torch sees %d GPU(s)); using 0",
            idx,
            n,
        )
        return 0
    return idx


def resolve_processing_device_string(user_wants_gpu: bool) -> str:
    """Return a string suitable for ``torch.device(...)`` for batch processing / live CREPE."""
    import torch

    if not user_wants_gpu:
        return "cpu"

    override = os.getenv("TORCH_DEVICE", "").strip()
    if override:
        try:
            d = torch.device(override)
        except (RuntimeError, ValueError) as e:
            log.warning("Invalid TORCH_DEVICE=%r (%s); using auto selection", override, e)
        else:
            if d.type == "cuda":
                if not torch.cuda.is_available():
                    log.warning("TORCH_DEVICE=%r but CUDA is not available; using cpu", override)
                    return "cpu"
                n = torch.cuda.device_count()
                if d.index is not None:
                    idx = _clamp_cuda_index(int(d.index), n)
                    return f"cuda:{idx}"
                return "cuda"
            if d.type == "mps":
                if not _mps_available():
                    log.warning("TORCH_DEVICE=%r but MPS is not available; using cpu", override)
                    return "cpu"
                return "mps"
            if d.type == "cpu":
                return "cpu"
            log.warning("Unsupported TORCH_DEVICE=%r; using auto selection", override)

    idx_raw = os.getenv("TORCH_CUDA_DEVICE_INDEX", os.getenv("CUDA_DEVICE_INDEX", "")).strip()
    if idx_raw.isdigit() and torch.cuda.is_available():
        idx = _clamp_cuda_index(int(idx_raw), torch.cuda.device_count())
        return f"cuda:{idx}"

    if torch.cuda.is_available():
        return "cuda"

    if _mps_available():
        return "mps"

    return "cpu"


def processing_device_summary(user_wants_gpu: bool) -> str:
    """Short human-readable summary for logs / settings (after resolution)."""
    s = resolve_processing_device_string(user_wants_gpu)
    try:
        import torch

        d = torch.device(s)
        if d.type == "cuda" and torch.cuda.is_available():
            idx = d.index if d.index is not None else torch.cuda.current_device()
            idx = _clamp_cuda_index(int(idx), torch.cuda.device_count())
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
        if d.type == "mps":
            return "mps (Apple GPU)"
    except Exception:
        pass
    return s
