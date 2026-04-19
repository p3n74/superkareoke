import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CATALOG_DIR = DATA_DIR / "catalog"
DB_PATH = DATA_DIR / "songs.db"

CATALOG_DIR.mkdir(parents=True, exist_ok=True)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_SCOPE = "user-library-read playlist-read-private playlist-read-collaborative"

SAMPLE_RATE = 44100
PITCH_HOP_MS = 10
PITCH_FMIN = 65.0   # C2
PITCH_FMAX = 1047.0  # C6

# Scoring thresholds (in cents; 100 cents = 1 semitone / one piano key)
# Legacy names = "strict" preset (tightest).
SCORE_PERFECT_CENTS = 25
SCORE_GREAT_CENTS = 50
SCORE_GOOD_CENTS = 100
SCORE_OK_CENTS = 150

# (perfect, great, good, ok) max cents from target for each tier; above ok => miss
SCORING_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "strict": (SCORE_PERFECT_CENTS, SCORE_GREAT_CENTS, SCORE_GOOD_CENTS, SCORE_OK_CENTS),
    # ~half to ~2.5 semitone windows — friendlier but still rewards accuracy
    "normal": (45, 95, 190, 260),
    # ~2–3+ semitones can still land OK before a miss — party / casual sing
    "casual": (75, 150, 260, 320),
}


def scoring_thresholds_cents(difficulty: str) -> tuple[int, int, int, int]:
    """Return (perfect, great, good, ok) cent limits for a difficulty preset id."""
    return SCORING_PRESETS.get(difficulty, SCORING_PRESETS["strict"])


# How many semitones above/below the target center the lane bar spans (visual only).
DISPLAY_LANE_HALF_SEMITONES: dict[str, float] = {
    "strict": 0.0,
    "normal": 0.0,
    "casual": 1.0,  # one piano key up and down (three rows total)
}


def display_lane_half_semitones(difficulty: str) -> float:
    """Half-width of target lanes on the pitch chart in semitones (0 = single row)."""
    return float(DISPLAY_LANE_HALF_SEMITONES.get(difficulty, 0.0))

# ── Vocal effects presets ──
# Parameters used by VoiceFx. All gains in dB, ratios linear, frequencies in Hz,
# reverb mix is wet/dry (0..1). "Custom" exists only as a UI label.
VOICE_FX_PRESETS: dict[str, dict[str, float]] = {
    "none": {
        "hpf_hz": 0.0,
        "eq_low_db": 0.0, "eq_mid_db": 0.0, "eq_high_db": 0.0,
        "comp_threshold_db": 0.0, "comp_ratio": 1.0,
        "reverb_mix": 0.0, "reverb_room": 0.5, "reverb_damp": 0.5,
        "output_gain_db": 0.0,
    },
    "natural": {
        "hpf_hz": 90.0,
        "eq_low_db": 0.0, "eq_mid_db": 1.0, "eq_high_db": 1.5,
        "comp_threshold_db": -18.0, "comp_ratio": 2.0,
        "reverb_mix": 0.08, "reverb_room": 0.45, "reverb_damp": 0.55,
        "output_gain_db": 1.0,
    },
    "warm": {
        "hpf_hz": 80.0,
        "eq_low_db": 2.5, "eq_mid_db": -1.0, "eq_high_db": 0.5,
        "comp_threshold_db": -16.0, "comp_ratio": 2.5,
        "reverb_mix": 0.12, "reverb_room": 0.55, "reverb_damp": 0.65,
        "output_gain_db": 1.5,
    },
    "bright": {
        "hpf_hz": 100.0,
        "eq_low_db": -1.0, "eq_mid_db": 1.5, "eq_high_db": 4.0,
        "comp_threshold_db": -18.0, "comp_ratio": 2.5,
        "reverb_mix": 0.10, "reverb_room": 0.50, "reverb_damp": 0.40,
        "output_gain_db": 1.0,
    },
    "stage_reverb": {
        "hpf_hz": 90.0,
        "eq_low_db": 0.5, "eq_mid_db": 1.0, "eq_high_db": 2.5,
        "comp_threshold_db": -18.0, "comp_ratio": 2.5,
        "reverb_mix": 0.30, "reverb_room": 0.80, "reverb_damp": 0.45,
        "output_gain_db": 1.0,
    },
    "studio_polish": {
        "hpf_hz": 100.0,
        "eq_low_db": 1.0, "eq_mid_db": 2.0, "eq_high_db": 3.0,
        "comp_threshold_db": -20.0, "comp_ratio": 3.5,
        "reverb_mix": 0.18, "reverb_room": 0.60, "reverb_damp": 0.50,
        "output_gain_db": 2.0,
    },
}

VOICE_FX_PRESET_LABELS: list[tuple[str, str]] = [
    ("none", "None (dry)"),
    ("natural", "Natural"),
    ("warm", "Warm"),
    ("bright", "Bright"),
    ("stage_reverb", "Stage Reverb"),
    ("studio_polish", "Studio Polish"),
]


def voice_fx_preset(name: str) -> dict[str, float]:
    """Return a copy of the named preset, falling back to 'none'."""
    base = VOICE_FX_PRESETS.get(name) or VOICE_FX_PRESETS["none"]
    return dict(base)


# ── Microphone enhancement defaults ──
MIC_PROCESSOR_DEFAULTS: dict[str, float | bool] = {
    "input_gain_db": 0.0,
    # Most consumer mics have ~ -45 dBFS self-noise. Set the gate above that so
    # idle hiss never reaches the FX chain (and never gets fed into reverb).
    "gate_db": -42.0,
    "use_vad": False,
}


APP_NAME = "Karaoke Pitch Tracker"
APP_VERSION = "0.1.0"
