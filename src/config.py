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

APP_NAME = "Karaoke Pitch Tracker"
APP_VERSION = "0.1.0"
