from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.audio.mic_input import MicInput
from src.audio.system_capture import (
    SystemAudioCapture,
    list_system_audio_input_devices,
    system_audio_uses_wasapi_loopback,
)
from src.audio.pitch_detector import RealtimePitchDetector
from src.ui.pitch_widget import PitchWidget


class LiveView(QWidget):
    """Live/impromptu mode: captures system audio and mic simultaneously,
    shows pitch comparison without a pre-processed pitch map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mic = MicInput()
        self._system_capture = SystemAudioCapture()
        self._user_detector = RealtimePitchDetector(
            parent=self, buffer_duration_ms=80,
        )
        self._system_detector = RealtimePitchDetector(
            parent=self, buffer_duration_ms=80,
        )
        self._running = False

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Live Mode")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #e94560;")
        layout.addWidget(title)

        if system_audio_uses_wasapi_loopback():
            desc_text = (
                "Play any song on your computer (Spotify, YouTube, etc.) and sing along.\n"
                "The app captures system audio (WASAPI loopback) and your microphone "
                "to compare pitch in real-time."
            )
        else:
            desc_text = (
                "Play any song on your computer (Spotify, YouTube, etc.) and sing along.\n"
                "Route playback through a virtual loopback device (e.g. BlackHole on macOS) "
                "so the «System audio input» below receives the mix, while the app still "
                "captures your microphone separately."
            )
        desc = QLabel(desc_text)
        desc.setStyleSheet("color: #888; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        if not system_audio_uses_wasapi_loopback():
            dev_row = QHBoxLayout()
            dev_row.addWidget(QLabel("System audio input:"))
            self._system_input_combo = QComboBox()
            self._system_input_combo.setMinimumWidth(320)
            self._refresh_system_input_devices()
            dev_row.addWidget(self._system_input_combo, 1)
            layout.addLayout(dev_row)

        # Controls
        controls = QHBoxLayout()

        self._start_btn = QPushButton("▶ Start Live Mode")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: #5cb85c; color: white; border: none;
                border-radius: 6px; padding: 10px 28px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background: #6dd06d; }
        """)
        self._start_btn.clicked.connect(self._toggle)
        controls.addWidget(self._start_btn)

        controls.addStretch()

        # Note display
        self._note_label = QLabel("")
        self._note_label.setStyleSheet("color: #5bc0de; font-size: 20px; font-weight: bold;")
        controls.addWidget(self._note_label)

        self._system_note_label = QLabel("")
        self._system_note_label.setStyleSheet("color: #f0ad4e; font-size: 14px;")
        controls.addWidget(self._system_note_label)

        layout.addLayout(controls)

        # Pitch visualization
        self._pitch_widget = PitchWidget()
        self._pitch_widget.set_render_interval_ms(33)
        layout.addWidget(self._pitch_widget, stretch=1)

        # Info
        info_layout = QHBoxLayout()
        self._your_pitch_label = QLabel("Your pitch: —")
        self._your_pitch_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        info_layout.addWidget(self._your_pitch_label)
        info_layout.addStretch()
        self._song_pitch_label = QLabel("Song pitch: —")
        self._song_pitch_label.setStyleSheet("color: #f0ad4e; font-size: 12px;")
        info_layout.addWidget(self._song_pitch_label)
        layout.addLayout(info_layout)

    def _refresh_system_input_devices(self) -> None:
        if system_audio_uses_wasapi_loopback():
            return
        self._system_input_combo.blockSignals(True)
        self._system_input_combo.clear()
        for d in list_system_audio_input_devices():
            label = d["name"]
            if d.get("priority"):
                label = f"★ {label}"
            self._system_input_combo.addItem(label, userData=d["index"])
        if self._system_input_combo.count() == 0:
            self._system_input_combo.addItem("(no input devices found)", userData=None)
        self._system_input_combo.blockSignals(False)

    def _connect_signals(self):
        self._mic.audio_chunk.connect(self._on_mic_chunk)
        self._system_capture.audio_chunk.connect(self._on_system_chunk)
        self._user_detector.pitch_detected.connect(self._on_user_pitch)
        self._system_detector.pitch_detected.connect(self._on_system_pitch)

    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        input_dev = None
        if not system_audio_uses_wasapi_loopback():
            input_dev = self._system_input_combo.currentData()
            if input_dev is None:
                self._note_label.setText("Error: choose a system audio input device.")
                return
        try:
            self._system_capture.start(input_device=input_dev)
        except RuntimeError as e:
            self._note_label.setText(f"Error: {e}")
            return

        self._mic.start()
        self._pitch_widget.start_rendering()
        self._running = True
        self._start_btn.setText("■ Stop")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: #d9534f; color: white; border: none;
                border-radius: 6px; padding: 10px 28px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background: #e96b67; }
        """)

        # Use a timer to advance time for the pitch widget
        from PyQt6.QtCore import QTimer
        self._live_timer = QTimer()
        self._live_timer.setInterval(50)
        self._time_counter = 0.0
        self._live_timer.timeout.connect(self._tick)
        self._live_timer.start()

    def _stop(self):
        self._mic.stop()
        self._system_capture.stop()
        self._user_detector.reset_buffer()
        self._system_detector.reset_buffer()
        self._pitch_widget.stop_rendering()
        self._pitch_widget.reset()
        self._running = False
        if hasattr(self, "_live_timer"):
            self._live_timer.stop()
        self._start_btn.setText("▶ Start Live Mode")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: #5cb85c; color: white; border: none;
                border-radius: 6px; padding: 10px 28px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background: #6dd06d; }
        """)
        self._note_label.setText("")
        self._system_note_label.setText("")

    def _tick(self):
        self._time_counter += 0.05
        self._pitch_widget.set_time(self._time_counter)

    def _on_mic_chunk(self, chunk, sr):
        self._user_detector.process_chunk(chunk, sr)

    def _on_system_chunk(self, chunk, sr):
        self._system_detector.process_chunk(chunk, sr)

    def _on_user_pitch(self, freq: float, conf: float, note: str):
        self._pitch_widget.set_user_pitch(freq, conf)
        self._note_label.setText(note if note else "—")
        self._your_pitch_label.setText(
            f"Your pitch: {note} ({freq:.0f} Hz)" if note else "Your pitch: —"
        )

    def _on_system_pitch(self, freq: float, conf: float, note: str):
        self._system_note_label.setText(f"Song: {note}" if note else "")
        self._song_pitch_label.setText(
            f"Song pitch: {note} ({freq:.0f} Hz)" if note else "Song pitch: —"
        )
