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

# Scoring thresholds (in cents; 100 cents = 1 semitone)
SCORE_PERFECT_CENTS = 25
SCORE_GREAT_CENTS = 50
SCORE_GOOD_CENTS = 100
SCORE_OK_CENTS = 150

APP_NAME = "Karaoke Pitch Tracker"
APP_VERSION = "0.1.0"
