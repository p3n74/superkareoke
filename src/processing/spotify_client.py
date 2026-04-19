from dataclasses import dataclass
from typing import Optional

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIPY = True
except ImportError:
    HAS_SPOTIPY = False

from src.config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI, SPOTIFY_SCOPE,
)


@dataclass
class SpotifyTrack:
    spotify_id: str
    title: str
    artist: str
    album: str
    duration_ms: int
    preview_url: Optional[str] = None


class SpotifyClient:
    def __init__(self):
        self._sp = None

    @property
    def is_configured(self) -> bool:
        return HAS_SPOTIPY and bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

    def authenticate(self) -> bool:
        if not self.is_configured:
            return False
        try:
            auth = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=SPOTIFY_SCOPE,
                open_browser=True,
            )
            self._sp = spotipy.Spotify(auth_manager=auth)
            self._sp.current_user()
            return True
        except Exception:
            self._sp = None
            return False

    @property
    def sp(self) -> spotipy.Spotify:
        if self._sp is None:
            raise RuntimeError("Spotify not authenticated. Call authenticate() first.")
        return self._sp

    def get_playlists(self) -> list[dict]:
        results = self.sp.current_user_playlists(limit=50)
        playlists = []
        for item in results.get("items", []):
            playlists.append({
                "id": item["id"],
                "name": item["name"],
                "track_count": item["tracks"]["total"],
                "image_url": item["images"][0]["url"] if item.get("images") else "",
            })
        return playlists

    def get_playlist_tracks(self, playlist_id: str) -> list[SpotifyTrack]:
        tracks: list[SpotifyTrack] = []
        results = self.sp.playlist_items(
            playlist_id, fields="items(track(id,name,artists,album,duration_ms,preview_url)),next",
        )
        while results:
            for item in results.get("items", []):
                t = item.get("track")
                if not t or not t.get("id"):
                    continue
                artists = ", ".join(a["name"] for a in t.get("artists", []))
                tracks.append(SpotifyTrack(
                    spotify_id=t["id"],
                    title=t["name"],
                    artist=artists,
                    album=t.get("album", {}).get("name", ""),
                    duration_ms=t.get("duration_ms", 0),
                    preview_url=t.get("preview_url"),
                ))
            results = self.sp.next(results) if results.get("next") else None
        return tracks

    def search_track(self, query: str, limit: int = 10) -> list[SpotifyTrack]:
        results = self.sp.search(q=query, type="track", limit=limit)
        tracks = []
        for t in results.get("tracks", {}).get("items", []):
            artists = ", ".join(a["name"] for a in t.get("artists", []))
            tracks.append(SpotifyTrack(
                spotify_id=t["id"],
                title=t["name"],
                artist=artists,
                album=t.get("album", {}).get("name", ""),
                duration_ms=t.get("duration_ms", 0),
                preview_url=t.get("preview_url"),
            ))
        return tracks
