import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.database.models import Song, SongStatus, Performance


class DatabaseManager:
    """SQLite access from multiple threads (e.g. Qt main thread + QThread workers).

    Each thread gets its own connection; WAL mode keeps them coherent on one file.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._init_db()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        with self._all_conns_lock:
            self._all_conns.append(conn)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._thread_local, "sqlite_conn", None)
        if c is None:
            c = self._open_connection()
            self._thread_local.sqlite_conn = c
        return c

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                artist TEXT NOT NULL DEFAULT '',
                spotify_id TEXT DEFAULT '',
                duration_s REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                original_path TEXT DEFAULT '',
                vocals_path TEXT DEFAULT '',
                instrumental_path TEXT DEFAULT '',
                pitch_map_path TEXT DEFAULT '',
                error_message TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                score_percent REAL DEFAULT 0,
                star_rating INTEGER DEFAULT 0,
                perfect_count INTEGER DEFAULT 0,
                great_count INTEGER DEFAULT 0,
                good_count INTEGER DEFAULT 0,
                ok_count INTEGER DEFAULT 0,
                miss_count INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status);
            CREATE INDEX IF NOT EXISTS idx_performances_song ON performances(song_id);
        """)
        self.conn.commit()

    # ── Song CRUD ──

    def add_song(self, song: Song) -> int:
        cur = self.conn.execute(
            """INSERT INTO songs (title, artist, spotify_id, duration_s, status,
               original_path, vocals_path, instrumental_path, pitch_map_path, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (song.title, song.artist, song.spotify_id, song.duration_s,
             song.status.value, song.original_path, song.vocals_path,
             song.instrumental_path, song.pitch_map_path, song.error_message),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_song(self, song_id: int) -> Optional[Song]:
        row = self.conn.execute("SELECT * FROM songs WHERE id = ?", (song_id,)).fetchone()
        return self._row_to_song(row) if row else None

    def get_all_songs(self) -> list[Song]:
        rows = self.conn.execute("SELECT * FROM songs ORDER BY id DESC").fetchall()
        return [self._row_to_song(r) for r in rows]

    def get_ready_songs(self) -> list[Song]:
        rows = self.conn.execute(
            "SELECT * FROM songs WHERE status = ? ORDER BY title",
            (SongStatus.READY.value,),
        ).fetchall()
        return [self._row_to_song(r) for r in rows]

    def update_song_status(self, song_id: int, status: SongStatus, error: str = ""):
        self.conn.execute(
            "UPDATE songs SET status = ?, error_message = ? WHERE id = ?",
            (status.value, error, song_id),
        )
        self.conn.commit()

    def update_song_paths(self, song_id: int, **paths: str):
        allowed = {"original_path", "vocals_path", "instrumental_path", "pitch_map_path"}
        filtered = {k: v for k, v in paths.items() if k in allowed}
        if not filtered:
            return
        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [song_id]
        self.conn.execute(f"UPDATE songs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def delete_song(self, song_id: int):
        self.conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        self.conn.commit()

    def song_exists_by_spotify_id(self, spotify_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM songs WHERE spotify_id = ?", (spotify_id,)
        ).fetchone()
        return row is not None

    def find_duplicate_song(self, title: str, artist: str) -> Optional[Song]:
        """Same title + artist (case-insensitive, trimmed), ignoring empty vs missing artist."""
        t = (title or "").strip()
        a = (artist or "").strip()
        row = self.conn.execute(
            """
            SELECT * FROM songs
            WHERE lower(trim(title)) = lower(?)
              AND lower(trim(COALESCE(artist, ''))) = lower(?)
            LIMIT 1
            """,
            (t, a),
        ).fetchone()
        return self._row_to_song(row) if row else None

    # ── Performance CRUD ──

    def add_performance(self, perf: Performance) -> int:
        cur = self.conn.execute(
            """INSERT INTO performances (song_id, score_percent, star_rating,
               perfect_count, great_count, good_count, ok_count, miss_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (perf.song_id, perf.score_percent, perf.star_rating,
             perf.perfect_count, perf.great_count, perf.good_count,
             perf.ok_count, perf.miss_count),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_performances_for_song(self, song_id: int) -> list[Performance]:
        rows = self.conn.execute(
            "SELECT * FROM performances WHERE song_id = ? ORDER BY timestamp DESC",
            (song_id,),
        ).fetchall()
        return [self._row_to_performance(r) for r in rows]

    def get_best_performance(self, song_id: int) -> Optional[Performance]:
        row = self.conn.execute(
            "SELECT * FROM performances WHERE song_id = ? ORDER BY score_percent DESC LIMIT 1",
            (song_id,),
        ).fetchone()
        return self._row_to_performance(row) if row else None

    # ── Helpers ──

    @staticmethod
    def _row_to_song(row: sqlite3.Row) -> Song:
        return Song(
            id=row["id"],
            title=row["title"],
            artist=row["artist"],
            spotify_id=row["spotify_id"],
            duration_s=row["duration_s"],
            status=SongStatus(row["status"]),
            original_path=row["original_path"],
            vocals_path=row["vocals_path"],
            instrumental_path=row["instrumental_path"],
            pitch_map_path=row["pitch_map_path"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _row_to_performance(row: sqlite3.Row) -> Performance:
        return Performance(
            id=row["id"],
            song_id=row["song_id"],
            score_percent=row["score_percent"],
            star_rating=row["star_rating"],
            perfect_count=row["perfect_count"],
            great_count=row["great_count"],
            good_count=row["good_count"],
            ok_count=row["ok_count"],
            miss_count=row["miss_count"],
            timestamp=row["timestamp"],
        )

    def close(self):
        with self._all_conns_lock:
            conns = self._all_conns[:]
            self._all_conns.clear()
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass
        self._thread_local = threading.local()
