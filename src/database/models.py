from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class SongStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    EXTRACTING_PITCH = "extracting_pitch"
    READY = "ready"
    ERROR = "error"


@dataclass
class Song:
    id: Optional[int] = None
    title: str = ""
    artist: str = ""
    spotify_id: str = ""
    duration_s: float = 0.0
    status: SongStatus = SongStatus.PENDING
    original_path: str = ""
    vocals_path: str = ""
    instrumental_path: str = ""
    pitch_map_path: str = ""
    error_message: str = ""


@dataclass
class PitchEvent:
    time: float
    frequency: float
    note: str
    confidence: float


@dataclass
class PitchMap:
    song_id: int
    events: list[PitchEvent] = field(default_factory=list)
    sample_rate: int = 44100
    hop_ms: int = 10

    def to_json(self) -> str:
        return json.dumps({
            "song_id": self.song_id,
            "sample_rate": self.sample_rate,
            "hop_ms": self.hop_ms,
            "events": [
                {"time": e.time, "frequency": e.frequency,
                 "note": e.note, "confidence": e.confidence}
                for e in self.events
            ],
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "PitchMap":
        d = json.loads(data)
        events = [PitchEvent(**e) for e in d["events"]]
        return cls(
            song_id=d["song_id"],
            events=events,
            sample_rate=d.get("sample_rate", 44100),
            hop_ms=d.get("hop_ms", 10),
        )


@dataclass
class Performance:
    id: Optional[int] = None
    song_id: int = 0
    score_percent: float = 0.0
    star_rating: int = 0
    perfect_count: int = 0
    great_count: int = 0
    good_count: int = 0
    ok_count: int = 0
    miss_count: int = 0
    timestamp: str = ""
