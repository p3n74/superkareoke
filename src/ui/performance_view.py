import logging
from pathlib import Path
from typing import Callable, Optional

from src.scoring.comparator import PitchComparator

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QGridLayout, QCheckBox, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

log = logging.getLogger(__name__)

from src.config import CATALOG_DIR
from src.database.manager import DatabaseManager
from src.database.models import Song, PitchMap
from src.audio.mic_input import MicInput
from src.audio.playback import AudioPlayback
from src.audio.pitch_detector import RealtimePitchDetector
from src.audio.mic_processor import MicProcessor, MicProcessorSettings
from src.audio.voice_fx import VoiceFx
from src.audio.device_hints import preferred_blocksize_for_output
from src.audio.vocal_monitor import VocalMonitor
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
        vocal_chain_settings_fn: Optional[Callable[[], dict]] = None,
        use_gpu_fn: Optional[Callable[[], bool]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.db = db
        self._mic_device_fn = mic_device_fn or (lambda: None)
        self._output_device_fn = output_device_fn or (lambda: None)
        self._scoring_difficulty_fn = scoring_difficulty_fn or (lambda: "strict")
        self._vocal_chain_settings_fn = vocal_chain_settings_fn or (lambda: {})
        self._song: Song | None = None
        self._pitch_map: PitchMap | None = None
        self._comparator: PitchComparator | None = None

        self._mic = MicInput(chunk_size=512)
        self._playback = AudioPlayback(blocksize=512)
        self._detector = RealtimePitchDetector(
            parent=self,
            use_gpu_fn=use_gpu_fn,
        )
        self._mic_proc = MicProcessor(self._mic.sample_rate)
        self._voice_fx = VoiceFx(self._mic.sample_rate)
        self._monitor = VocalMonitor(sample_rate=self._mic.sample_rate, blocksize=512)
        self._last_monitor_status: str = ""
        self._bypass_effects: bool = False

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
        self._play_btn.setToolTip("")
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

        self._sing_hint = QLabel(
            "Use Sing! in the Catalog on a song with green (Ready) status, "
            "or open Sing in the sidebar after a song is loaded."
        )
        self._sing_hint.setWordWrap(True)
        self._sing_hint.setStyleSheet("color: #888; font-size: 11px; padding: 0 16px 4px 16px;")
        layout.addWidget(self._sing_hint)

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

        # Pitch visualizer (slightly lower repaint rate than default to keep UI responsive)
        self._pitch_widget = PitchWidget()
        self._pitch_widget.set_render_interval_ms(33)
        layout.addWidget(self._pitch_widget, stretch=1)

        # Bottom info bar
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 4, 16, 8)
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setStyleSheet("color: #888; font-size: 11px;")
        bottom.addWidget(self._time_label)
        self._monitor_status_label = QLabel("")
        self._monitor_status_label.setStyleSheet("color: #f0ad4e; font-size: 11px; padding-left: 12px;")
        bottom.addWidget(self._monitor_status_label)
        bottom.addStretch()
        self._note_label = QLabel("")
        self._note_label.setStyleSheet("color: #5bc0de; font-size: 14px; font-weight: bold;")
        bottom.addWidget(self._note_label)
        layout.addLayout(bottom)

    def _connect_signals(self):
        self._mic.audio_chunk.connect(self._on_mic_chunk)
        # Low-jitter monitor pipeline runs on the audio thread, not via the
        # Qt event loop. This keeps the monitor immune to GUI repaints / busy
        # event-loop ticks (which were causing the "kztzt" buffer drops).
        self._mic.set_audio_processor(self._audio_thread_pipeline)
        self._detector.pitch_detected.connect(self._on_pitch_detected)
        self._playback.position_changed.connect(self._on_position)
        self._playback.playback_finished.connect(self._on_song_end)

    def _audio_thread_pipeline(self, chunk, sr):
        """Runs in the PortAudio capture thread. Keep this CHEAP and SAFE."""
        if not self._monitor.enabled:
            return
        try:
            if self._bypass_effects:
                self._monitor.write(chunk.astype("float32", copy=False), source_sr=sr)
                return
            _clean, monitor_in = self._mic_proc.process(chunk)
            wet = self._voice_fx.process(monitor_in)
            self._monitor.write(wet, source_sr=sr)
        except Exception:
            log.debug("audio-thread pipeline error", exc_info=True)

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
        """Apply microphone, backing-track output, monitor device and vocal chain settings."""
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

        self._apply_vocal_chain_settings()

    def _compose_monitor_status(
        self,
        requested: bool,
        applied: bool,
        feedback_blocked: bool,
        same_device: bool,
    ) -> str:
        if not requested:
            return ""
        if feedback_blocked:
            return (
                "Vocal monitor muted: same output as backing track. "
                "Pick a separate Monitor output (e.g. headphones) in Settings, "
                "or disable Feedback protection."
            )
        if applied and not self._playback.is_playing:
            return "Vocal monitor will start when you press Start."
        if applied and same_device:
            return "Vocal monitor on (mixed into backing-track output — use headphones to avoid feedback)."
        if applied:
            return "Vocal monitor on."
        return ""

    def _apply_vocal_chain_settings(self) -> None:
        cfg = self._vocal_chain_settings_fn() or {}
        mic_proc = cfg.get("mic_processor") or {}
        fx_params = cfg.get("voice_fx") or {}
        monitor_enabled = bool(cfg.get("monitor_enabled", False))
        monitor_device = cfg.get("monitor_device")  # None = same as backing
        monitor_gain_db = float(cfg.get("monitor_gain_db", 0.0))
        feedback_protection = bool(cfg.get("feedback_protection", True))
        self._bypass_effects = bool(cfg.get("bypass_effects", False))

        self._mic_proc.apply_settings(MicProcessorSettings.from_dict(mic_proc))
        if fx_params:
            self._voice_fx.apply_params(fx_params)

        backing_dev = self._output_device_fn()
        # If "Same as backing", use the backing-track output device
        target_dev = monitor_device if monitor_device is not None else backing_dev

        # Feedback protection: if monitor would share the backing-track device,
        # mute the monitor unless the user has explicitly turned protection off.
        feedback_blocked = False
        if monitor_enabled and feedback_protection and target_dev == backing_dev:
            feedback_blocked = True
            monitor_enabled = False
        status = self._compose_monitor_status(
            requested=bool(cfg.get("monitor_enabled", False)),
            applied=monitor_enabled,
            feedback_blocked=feedback_blocked,
            same_device=target_dev == backing_dev,
        )
        if hasattr(self, "_monitor_status_label"):
            self._monitor_status_label.setText(status)
        if status != self._last_monitor_status:
            if status:
                log.info("Vocal monitor status: %s", status)
            self._last_monitor_status = status

        # Larger blocks on Bluetooth reduce underruns; WASAPI/low-latency paths
        # still use the base size from preferred_blocksize_for_output.
        bs_out = preferred_blocksize_for_output(backing_dev, 512)
        bs_mon = preferred_blocksize_for_output(target_dev, 512)
        bs = max(bs_out, bs_mon)
        self._mic.set_chunk_size(bs)
        self._playback.set_blocksize(bs)
        self._monitor.set_blocksize(bs)

        self._monitor.set_gain_db(monitor_gain_db)
        self._monitor.set_device(target_dev)
        self._monitor.set_enabled(monitor_enabled and self._playback.is_playing)

    def unload_song_if_removed(self, song_id: int) -> None:
        """If the loaded song was deleted from the library, stop audio and reset the Sing UI."""
        if self._song is None or self._song.id != song_id:
            return
        self._stop()
        self._song = None
        self._pitch_map = None
        self._comparator = None
        self._song_label.setText("No song loaded")
        self._pitch_widget.set_pitch_map(PitchMap(song_id=0, events=[]))
        self._lyrics.clear()
        self._play_btn.setEnabled(False)
        self._play_btn.setToolTip("")
        self._sing_hint.setText(
            "That song was removed from the catalog. Pick another song with Sing! "
            "from the Catalog tab."
        )
        self._sing_hint.setStyleSheet("color: #f0ad4e; font-size: 11px; padding: 0 16px 4px 16px;")

    def load_song(self, song: Song):
        self._stop()
        self._lyrics.clear()
        if song.id is not None:
            fresh = self.db.get_song(song.id)
            if fresh is not None:
                song = fresh
        self._song = song
        self._song_label.setText(f"{song.title} — {song.artist}")

        # Load pitch map
        if song.pitch_map_path and Path(song.pitch_map_path).exists():
            data = Path(song.pitch_map_path).read_text(encoding="utf-8")
            self._pitch_map = PitchMap.from_json(data)
            self._pitch_widget.set_pitch_map(self._pitch_map)
        else:
            self._pitch_map = None
            self._pitch_widget.set_pitch_map(
                PitchMap(song_id=int(song.id or 0), events=[]),
            )

        diff = self._scoring_difficulty_fn()
        self._pitch_widget.set_display_difficulty(diff)
        if self._pitch_map:
            self._comparator = PitchComparator(
                self._pitch_widget.note_segments,
                difficulty_id=diff,
            )
        else:
            self._comparator = None

        # Load backing track (instrumental or original mix per checkbox)
        self._sync_guide_checkbox()
        backing = self._backing_track_path()
        if backing is not None:
            try:
                self._playback.load(backing)
            except Exception as e:
                log.exception("Failed to load backing track %s", backing)
                self._play_btn.setEnabled(False)
                self._play_btn.setToolTip(f"Could not open backing file: {e}")
                self._sing_hint.setText(
                    f"Could not load backing audio ({backing.name}). "
                    f"If the file was moved or deleted, re-process the song. ({e})"
                )
                self._sing_hint.setStyleSheet("color: #f0ad4e; font-size: 11px; padding: 0 16px 4px 16px;")
            else:
                self._play_btn.setEnabled(True)
                self._play_btn.setToolTip("Start or resume: backing track + microphone + scoring.")
                self._sing_hint.setText("")
                self._sing_hint.setStyleSheet("color: #888; font-size: 11px; padding: 0 16px 4px 16px;")
                self._update_time_label(0.0, self._playback.duration)
                self.apply_audio_devices()
        else:
            self._play_btn.setEnabled(False)
            inst = song.instrumental_path or "(not set)"
            self._play_btn.setToolTip(
                "No instrumental.wav to play. Finish Processing for this song, "
                "or check that files still exist under your data/catalog folder."
            )
            self._sing_hint.setText(
                f"No backing track on disk. Expected instrumental at:\n{inst}"
            )
            self._sing_hint.setStyleSheet("color: #f0ad4e; font-size: 11px; padding: 0 16px 4px 16px;")

        if song.id is not None:
            lrc_path = CATALOG_DIR / str(song.id) / "lyrics.lrc"
            if lrc_path.is_file():
                self._lyrics.load_lrc_file(lrc_path)
            else:
                self._lyrics.clear()
        else:
            self._lyrics.clear()

    def _toggle_play(self):
        if self._playback.is_playing:
            try:
                self._playback.pause()
                self._mic.stop()
            except Exception:
                log.debug("pause/stop mic", exc_info=True)
            self._monitor.set_enabled(False)
            self._pitch_widget.stop_rendering()
            self._play_btn.setText("▶ Resume")
            self._sync_guide_checkbox()
        else:
            try:
                if self._comparator is None and self._pitch_map:
                    diff = self._scoring_difficulty_fn()
                    self._pitch_widget.set_display_difficulty(diff)
                    self._comparator = PitchComparator(
                        self._pitch_widget.note_segments,
                        difficulty_id=diff,
                    )
                self.apply_audio_devices()
                self._mic_proc.reset()
                self._playback.play()
                self._mic.start()
            except Exception as e:
                log.exception("Could not start performance audio")
                try:
                    self._playback.pause()
                except Exception:
                    pass
                try:
                    self._mic.stop()
                except Exception:
                    pass
                self._monitor.set_enabled(False)
                self._pitch_widget.stop_rendering()
                QMessageBox.warning(
                    self,
                    "Could not start audio",
                    f"The backing track or microphone could not be opened.\n\n{e}\n\n"
                    "Check Settings → microphone and output device, "
                    "then try again. On macOS, grant microphone access if prompted.",
                )
                return
            self._sync_guide_checkbox()
            self._pitch_widget.start_rendering()
            # apply_audio_devices() ran before play(); re-evaluate monitor now that
            # playback is actually running so the stream opens.
            self._apply_vocal_chain_settings()
            self._play_btn.setText("⏸ Pause")
            self._stop_btn.setEnabled(True)

    def _stop(self):
        self._playback.stop()
        self._mic.stop()
        self._detector.reset_buffer()
        self._monitor.set_enabled(False)
        self._mic_proc.reset()
        self._pitch_widget.stop_rendering()
        self._pitch_widget.reset()
        self._comparator = None
        self._play_btn.setText("▶ Start")
        inst_path = (
            Path(self._song.instrumental_path)
            if self._song and self._song.instrumental_path
            else None
        )
        can_play = inst_path is not None and inst_path.is_file()
        self._play_btn.setEnabled(bool(can_play))
        if not can_play and self._song:
            self._play_btn.setToolTip("Missing instrumental file for this song.")
        self._stop_btn.setEnabled(False)
        self._note_label.setText("")
        self._lyrics.set_time(0.0)
        self._sync_guide_checkbox()

    def _on_mic_chunk(self, chunk, sr):
        """Runs on the Qt thread. Pitch detection only — the monitor pipeline
        runs on the audio thread via ``_audio_thread_pipeline``.
        """
        if sr != self._mic_proc.sample_rate:
            self._mic_proc.sample_rate = sr
        if sr != self._voice_fx.sample_rate:
            self._voice_fx.set_sample_rate(sr)
        # Pitch can use the raw chunk; torchcrepe is robust to a little noise
        # and we don't want to fight the audio-thread's mic_proc state.
        self._detector.process_chunk(chunk.astype("float32", copy=False), sr)

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
        # Every ~1 s log monitor diagnostics so we can spot underruns/drops.
        if int(time_s) != getattr(self, "_last_diag_sec", -1) and self._monitor.enabled:
            self._last_diag_sec = int(time_s)
            d = self._monitor.diagnostics()
            log.debug(
                "monitor diag: queued=%d underrun=%d dropped=%d cap=%d sr=%d",
                d["queued_samples"], d["underrun_samples"],
                d["dropped_overflow_samples"], d["capacity"], d["sample_rate"],
            )

    def _update_time_label(self, current: float, total: float):
        cur_m, cur_s = divmod(int(current), 60)
        tot_m, tot_s = divmod(int(total), 60)
        self._time_label.setText(f"{cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}")

    def _on_song_end(self):
        self._mic.stop()
        self._detector.reset_buffer()
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
