"""Heuristics for PortAudio / sounddevice output devices (Bluetooth, headsets, …)."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Substrings in PortAudio device names (macOS / Windows) that usually mean Bluetooth
# or telephony / narrow-band profiles — these need higher latency and often mono.
_BLUETOOTH_OR_HEADSET_MARKERS = (
    "bluetooth",
    "bt ",
    "bt-",
    "airpods",
    "beats ",
    "bose ",
    "sony wh-",
    "sony wf-",
    "galaxy buds",
    "jbl ",
    "hands-free",
    "headset",
    "hfp",
    "sco ",
    "speakerphone",
    "poly ",
    "plantronics",
    "jabra",
    "sennheiser",
    "shokz",
    "aftershokz",
)


def looks_like_bluetooth_or_headset_name(name: str) -> bool:
    n = (name or "").lower()
    return any(m in n for m in _BLUETOOTH_OR_HEADSET_MARKERS)


def output_device_name(device: Optional[int]) -> str:
    """Return lowercased PortAudio output device name, or ``\"\"``."""
    try:
        import sounddevice as sd

        if device is None:
            info = sd.query_devices(kind="output")
        else:
            info = sd.query_devices(int(device))
        return str(info.get("name", "")).lower()
    except Exception:
        return ""


def is_bluetooth_output_device(device: Optional[int]) -> bool:
    """True when the chosen output (or system default) looks like BT / headset audio."""
    return looks_like_bluetooth_or_headset_name(output_device_name(device))


def max_output_channels(device: Optional[int]) -> int:
    try:
        import sounddevice as sd

        if device is None:
            info = sd.query_devices(kind="output")
        else:
            info = sd.query_devices(int(device))
        return max(1, int(info.get("max_output_channels", 1)))
    except Exception as e:
        log.debug("max_output_channels: %s", e)
        return 2


def preferred_blocksize_for_output(device: Optional[int], base: int = 512) -> int:
    """Larger blocks reduce underruns on Bluetooth A2DP / HFP."""
    b = max(64, int(base))
    if is_bluetooth_output_device(device):
        return max(b, 1024)
    return b


def preferred_latency_for_output(device: Optional[int]) -> str:
    """``\"high\"`` for Bluetooth-friendly scheduling; ``\"low\"`` otherwise."""
    return "high" if is_bluetooth_output_device(device) else "low"
