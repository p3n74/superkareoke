from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QGridLayout, QCheckBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.config import CATALOG_DIR
from src.database.manager import DatabaseManager
from src.database.models import Song, PitchMap
from src.audio.mic_input import MicInput
from src.audio.playback import AudioPlayback
from src.audio.pitch_detector import RealtimePitchDetector
from src.scoring.comparator import PitchComparator
from src.scoring.scorer import build_performance
from src.ui.pitch_widget import PitchWidget
from src.ui.lyrics_panel import LyricsPanel


class ResultsDialog(QDialog):
    def __init__(self, perf, song: Song, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance Results")
        self.setMinimumSize(400, 350)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 32, 32, 32)

        title = QLabel(f"{song.title}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #e94560; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        artist = QLabel(f"by {song.artist}")
        artist.setAlignment(Qt.AlignmentFlag.AlignCenter)
        artist.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(artist)

        stars = "★" * perf.star_rating + "☆" * (5 - perf.star_rating)
        star_label = QLabel(stars)
        star_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        star_label.setStyleSheet("color: #f0ad4e; font-size: 28px;")
        layout.addWidget(star_label)

        score_label = QLabel(f"{perf.score_percent:.1f}%")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_label.setStyleSheet("color: #fff; font-size: 36px; font-weight: bold;")
        layout.addWidget(score_label)

        grid = QGridLayout()
        grid.setSpacing(8)
        stats = [
            ("Perfect", perf.perfect_count, "#5cb85c"),
            ("Great", perf.great_count, "#5bc0de"),
            ("Good", perf.good_count, "#f0ad4e"),
            ("OK", perf.ok_count, "#e0e0e0"),
            ("Miss", perf.miss_count, "#d9534f"),
        ]
        for i, (name, count, color) in enumerate(stats):
            lbl = QLabel(f"{name}: {count}")
            lbl.setStyleSheet(f"color: {color}; font-size: 13px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, i // 3, i % 3)
        layout.addLayout(grid)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background: #0f3460; color: #ddd; border: none;
                border-radius: 6px; padding: 10px 24px; font-size: 13px;
            }
            QPushButton:hover { background: #1a4a7a; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet("QDialog { background: #1a1a2e; }")


class PerformanceView(QWidget):
    def __init__(
        self,
        db: DatabaseManager,
        mic_device_fn: Optional[Callable[[], Optional[int]]] = None,
        output_device_fn: Optional[Callable[[], Optional[int]]] = None,
        scoring_difficulty_fn: Optional[Callable[[], str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.db = db
        self._mic_device_fn = mic_device_fn or (lambda: None)
        self._output_device_fn = output_device_fn or (lambda: None)
        self._scoring_difficulty_fn = scoring_difficulty_fn or (lambda: "strict")
        self._song: Song | None = None
        self._pitch_map: PitchMap | None = None
        self._comparator: PitchComparator | None = None

        self._mic = MicInput()
        self._playback = AudioPlayback()
        self._detector = RealtimePitchDetector()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(16, 8, 16, 8)

        self._song_label = QLabel("No song loaded")
        self._song_label.setStyleSheet("color: #e0e0e0; font-size: 14px; font-weight: bold;")
        top_bar.addWidget(self._song_label)
        top_bar.addStretch()

        self._play_btn = QPushButton("▶ Start")
        self._play_btn.setStyleSheet("""
            QPushButton {
                background: #5cb85c; color: white; border: none;
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #6dd06d; }
        """)
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setEnabled(False)
        top_bar.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background: #d9534f; color: white; border: none;
                border-radius: 6px; padding: 8px 24px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #e96b67; }
        """)
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        top_bar.addWidget(self._stop_btn)

        layout.addLayout(top_bar)

        backing_row = QHBoxLayout()
        backing_row.setContentsMargins(16, 0, 16, 8)
        self._guide_vocal_check = QCheckBox("Include guide vocals (original mix with lead vocal)")
        self._guide_vocal_check.setStyleSheet("color: #c8c8d0; font-size: 12px;")
        self._guide_vocal_check.setToolTip(
            "When checked, the backing track is the full downloaded mix so you can hear the "
            "original singer for reference. When unchecked, only the isolated instrumental plays."
        )
        self._guide_vocal_check.stateChanged.connect(self._on_guide_vocal_toggled)
        backing_row.addWidget(self._guide_vocal_check)
        backing_row.addStretch()
        layout.addLayout(backing_row)

        self._lyrics = LyricsPanel()
        layout.addWidget(self._lyrics, stretch=0)

        # Pitch visualizer
        self._pitch_widget = PitchWidget()
        layout.addWidget(self._pitch_widget, stretch=1)

        # Bottom info bar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 4, 16, 8)
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setStyleSheet("color: #888; font-size: 11px;")
        bottom.addWidget(self._time_label)
        bottom.addStretch()
        self._note_label = QLabel("")
        self._note_label.setStyleSheet("color: #5bc0de; font-size: 14px; font-weight: bold;")
        bottom.addWidget(self._note_label)
        layout.addLayout(bottom)

    def _connect_signals(self):
        self._mic.audio_chunk.connect(self._on_mic_chunk)
        self._detector.pitch_detected.connect(self._on_pitch_detected)
        self._playback.position_changed.connect(self._on_position)
        self._playback.playback_finished.connect(self._on_song_end)

    def _original_mix_path(self) -> Path | None:
        if not self._song or not self._song.original_path:
            return None
        p = Path(self._song.original_path)
        return p if p.is_file() else None

    def _backing_track_path(self) -> Path | None:
        """File used for speakers: full mix if guide vocals on, else instrumental."""
        if not self._song or not self._song.instrumental_path:
            return None
        inst = Path(self._song.instrumental_path)
        if not inst.is_file():
            return None
        if self._guide_vocal_check.isChecked():
            orig = self._original_mix_path()
            if orig is not None:
                return orig
        return inst

    def _on_guide_vocal_toggled(self, _state: int) -> None:
        if self._song is None or self._playback.is_playing:
            return
        path = self._backing_track_path()
        if path is None:
            return
        t_save = self._playback.current_time
        self._playback.load(path)
        if t_save > 0.01:
            self._playback.seek(t_save)
        self._update_time_label(self._playback.current_time, self._playback.duration)

    def _sync_guide_checkbox(self) -> None:
        """Enable guide option only when original exists; reflect availability."""
        self._guide_vocal_check.blockSignals(True)
        has_orig = self._original_mix_path() is not None
        self._guide_vocal_check.setEnabled(has_orig and not self._playback.is_playing)
        if not has_orig:
            self._guide_vocal_check.setChecked(False)
        self._guide_vocal_check.blockSignals(False)

    def apply_audio_devices(self) -> None:
        """Apply microphone and backing-track output from Settings."""
        out = self._output_device_fn()
        mic = self._mic_device_fn()
        self._playback.set_output_device(out)
        if self._mic.is_running and self._mic.device != mic:
            self._mic.stop()
            self._mic.device = mic
            if self._playback.is_playing:
                self._mic.start()
        else:
            self._mic.device = mic

    def load_song(self, song: Song):
        self._stop()
        self._lyrics.clear()
        self._song = song
        self._song_label.setText(f"{song.title} — {song.artist}")

        # Load pitch map
        if song.pitch_map_path and Path(song.pitch_map_path).exists():
            data = Path(song.pitch_map_path).read_text(encoding="utf-8")
            self._pitch_map = PitchMap.from_json(data)
            self._pitch_widget.set_pitch_map(self._pitch_map)
            self._pitch_widget.set_display_difficulty(self._scoring_difficulty_fn())
        else:
            self._pitch_map = None

        # Load backing track (instrumental or original mix per checkbox)
        self._sync_guide_checkbox()
        backing = self._backing_track_path()
        if backing is not None:
            self._playback.load(backing)
            self._play_btn.setEnabled(True)
            self._update_time_label(0.0, self._playback.duration)
            self.apply_audio_devices()
        else:
            self._play_btn.setEnabled(False)

        lrc_path = CATALOG_DIR / str(song.id) / "lyrics.lrc"
        if lrc_path.is_file():
            self._lyrics.load_lrc_file(lrc_path)
        else:
            self._lyrics.clear()

    def _toggle_play(self):
        if self._playback.is_playing:
            self._playback.pause()
            self._mic.stop()
            self._pitch_widget.stop_rendering()
            self._play_btn.setText("▶ Resume")
            self._sync_guide_checkbox()
        else:
            if self._comparator is None and self._pitch_map:
                diff = self._scoring_difficulty_fn()
                self._pitch_widget.set_display_difficulty(diff)
                self._comparator = PitchComparator(
                    self._pitch_widget.note_segments,
                    difficulty_id=diff,
                )
            self.apply_audio_devices()
            self._playback.play()
            self._mic.start()
            self._sync_guide_checkbox()
            self._pitch_widget.start_rendering()
            self._play_btn.setText("⏸ Pause")
            self._stop_btn.setEnabled(True)

    def _stop(self):
        self._playback.stop()
        self._mic.stop()
        self._pitch_widget.stop_rendering()
        self._pitch_widget.reset()
        self._comparator = None
        self._play_btn.setText("▶ Start")
        self._play_btn.setEnabled(self._song is not None and bool(self._song.instrumental_path))
        self._stop_btn.setEnabled(False)
        self._note_label.setText("")
        self._lyrics.set_time(0.0)
        self._sync_guide_checkbox()

    def _on_mic_chunk(self, chunk, sr):
        self._detector.process_chunk(chunk, sr)

    def _on_pitch_detected(self, freq: float, conf: float, note: str):
        self._pitch_widget.set_user_pitch(freq, conf)
        self._note_label.setText(note if note else "—")

        if self._comparator:
            results = self._comparator.update(
                self._pitch_widget._current_time, freq, conf,
            )
            for seg_idx, hit_type in results:
                self._pitch_widget.set_segment_hit(seg_idx, hit_type)
            self._pitch_widget.set_score(
                self._comparator.score_percent, self._comparator.combo,
            )

    def _on_position(self, time_s: float):
        self._pitch_widget.set_time(time_s)
        self._lyrics.set_time(time_s)
        self._update_time_label(time_s, self._playback.duration)

    def _update_time_label(self, current: float, total: float):
        cur_m, cur_s = divmod(int(current), 60)
        tot_m, tot_s = divmod(int(total), 60)
        self._time_label.setText(f"{cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}")

    def _on_song_end(self):
        self._mic.stop()
        self._pitch_widget.stop_rendering()
        self._play_btn.setText("▶ Start")
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._sync_guide_checkbox()

        if self._comparator and self._song:
            self._comparator.finalize_remaining()
            perf = build_performance(
                song_id=self._song.id,
                score_percent=self._comparator.score_percent,
                perfect_count=self._comparator.perfect_count,
                great_count=self._comparator.great_count,
                good_count=self._comparator.good_count,
                ok_count=self._comparator.ok_count,
                miss_count=self._comparator.miss_count,
            )
            self.db.add_performance(perf)

            dlg = ResultsDialog(perf, self._song, self)
            dlg.exec()

        self._comparator = None
