from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QCheckBox, QLineEdit, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import sounddevice as sd

from src.audio.mic_input import MicInput
from src.audio.playback import AudioPlayback
from src.config import SPOTIFY_CLIENT_ID


class SettingsView(QWidget):
    """GPU preference changes are emitted so the pipeline can drop cached torch models."""

    gpu_preference_changed = pyqtSignal()
    audio_devices_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #e94560;")
        layout.addWidget(title)

        # Audio devices group
        audio_group = QGroupBox("Audio Devices")
        audio_group.setStyleSheet("""
            QGroupBox {
                color: #e0e0e0; border: 1px solid #0f3460;
                border-radius: 8px; padding: 16px; margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        audio_form = QFormLayout(audio_group)

        self._mic_combo = QComboBox()
        self._mic_combo.setStyleSheet("""
            QComboBox {
                background: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
                border-radius: 4px; padding: 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #16213e; color: #e0e0e0;
                selection-background-color: #0f3460;
            }
        """)
        self._populate_mic_devices()
        self._mic_combo.currentIndexChanged.connect(self._emit_audio_devices_changed)
        audio_form.addRow("Microphone:", self._mic_combo)

        self._output_combo = QComboBox()
        self._output_combo.setStyleSheet(self._mic_combo.styleSheet())
        self._populate_output_devices()
        self._output_combo.currentIndexChanged.connect(self._emit_audio_devices_changed)
        self._output_combo.setToolTip(
            "PortAudio output for the instrumental. "
            '"System default" uses the OS default playback device. '
            "Changes apply while you sing (backing track may briefly glitch when switching)."
        )
        audio_form.addRow("Output (backing track):", self._output_combo)

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #0f3460; color: #ddd; border: none;
                border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover { background: #1a4a7a; }
        """)
        refresh_btn.clicked.connect(self._refresh_all_audio_devices)
        audio_form.addRow("", refresh_btn)

        layout.addWidget(audio_group)

        # Spotify group
        spotify_group = QGroupBox("Spotify Integration")
        spotify_group.setStyleSheet(audio_group.styleSheet())
        spotify_form = QFormLayout(spotify_group)

        status_text = "Configured" if SPOTIFY_CLIENT_ID else "Not configured"
        status_color = "#5cb85c" if SPOTIFY_CLIENT_ID else "#d9534f"
        spotify_status = QLabel(status_text)
        spotify_status.setStyleSheet(f"color: {status_color};")
        spotify_form.addRow("Status:", spotify_status)

        spotify_help = QLabel(
            "To use Spotify integration, create a .env file with:\n"
            "SPOTIFY_CLIENT_ID=your_id\n"
            "SPOTIFY_CLIENT_SECRET=your_secret\n\n"
            "Get credentials at developer.spotify.com"
        )
        spotify_help.setStyleSheet("color: #888; font-size: 11px;")
        spotify_help.setWordWrap(True)
        spotify_form.addRow(spotify_help)

        layout.addWidget(spotify_group)

        # Scoring difficulty
        score_group = QGroupBox("Scoring difficulty")
        score_group.setStyleSheet(audio_group.styleSheet())
        score_form = QFormLayout(score_group)
        self._difficulty_combo = QComboBox()
        self._difficulty_combo.setStyleSheet(self._mic_combo.styleSheet())
        self._difficulty_combo.addItem("Strict (tight pitch)", "strict")
        self._difficulty_combo.addItem("Normal (balanced)", "normal")
        self._difficulty_combo.addItem("Casual (2-3 keys forgiving)", "casual")
        self._difficulty_combo.setCurrentIndex(1)
        self._difficulty_combo.setToolTip(
            "How far from each target note (in cents; 100 = one piano key) still counts as "
            "Perfect … OK before a Miss. Casual is best for sing-alongs; Strict matches chart pro games."
        )
        score_form.addRow("Pitch tolerance:", self._difficulty_combo)
        score_help = QLabel(
            "Easier presets widen each tier so you can be a couple of semitones off and still score. "
            "On Casual, the green target lanes also span one key above and below the note on the chart. "
            "Applies the next time you press Start on the Sing screen (or when you load a song)."
        )
        score_help.setWordWrap(True)
        score_help.setStyleSheet("color: #888; font-size: 11px;")
        score_form.addRow(score_help)
        layout.addWidget(score_group)

        # Processing group
        proc_group = QGroupBox("Processing")
        proc_group.setStyleSheet(audio_group.styleSheet())
        proc_form = QFormLayout(proc_group)

        self._gpu_check = QCheckBox("Use GPU (CUDA) when processing songs")
        self._gpu_check.setStyleSheet("color: #e0e0e0;")
        proc_form.addRow(self._gpu_check)

        torch_installed = True
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            torch_built_cuda = torch.version.cuda is not None
        except ImportError:
            torch_installed = False
            cuda_available = False
            torch_built_cuda = False

        if cuda_available:
            self._gpu_check.setChecked(True)
            self._gpu_check.setEnabled(True)
            self._gpu_check.stateChanged.connect(self._emit_gpu_preference_changed)
        else:
            self._gpu_check.setChecked(False)
            self._gpu_check.setEnabled(False)
            self._gpu_check.setToolTip(
                "This Python install does not expose CUDA. Typical fix: reinstall "
                "torch with a CUDA wheel from pytorch.org, then restart the app."
            )

        gpu_status = "Available" if cuda_available else "Not available (CPU only)"
        gpu_color = "#5cb85c" if cuda_available else "#f0ad4e"
        gpu_label = QLabel(gpu_status)
        gpu_label.setStyleSheet(f"color: {gpu_color}; font-size: 11px;")
        proc_form.addRow("CUDA:", gpu_label)

        if not cuda_available:
            if not torch_installed:
                hint = "Install PyTorch first (see project README / requirements.txt)."
            else:
                hint = (
                    "Your install has no working CUDA (CPU-only wheel, missing driver, "
                    "or no NVIDIA GPU). Install a CUDA-enabled torch that matches your "
                    "driver, for example:\n"
                    "  pip install --upgrade torch torchaudio --index-url "
                    "https://download.pytorch.org/whl/cu124\n"
                    "Pick the exact command from https://pytorch.org/get-started/locally/ "
                    "then restart this app."
                )
                if not torch_built_cuda:
                    hint = (
                        "This PyTorch wheel is CPU-only (no CUDA build). "
                        "Reinstall using a CUDA index URL from pytorch.org, then restart.\n\n"
                    ) + hint
            gpu_hint = QLabel(hint)
            gpu_hint.setWordWrap(True)
            gpu_hint.setStyleSheet("color: #888; font-size: 11px;")
            proc_form.addRow(gpu_hint)

        layout.addWidget(proc_group)
        layout.addStretch()

    def _refresh_all_audio_devices(self):
        cur_mic = self._mic_combo.currentData()
        cur_out = self._output_combo.currentData()
        self._populate_mic_devices()
        self._populate_output_devices()
        self._restore_combo_by_data(self._mic_combo, cur_mic)
        self._restore_combo_by_data(self._output_combo, cur_out)
        self.audio_devices_changed.emit()

    @staticmethod
    def _restore_combo_by_data(combo: QComboBox, data) -> None:
        if data is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _populate_mic_devices(self):
        self._mic_combo.blockSignals(True)
        self._mic_combo.clear()
        devices = MicInput.list_devices()
        for d in devices:
            self._mic_combo.addItem(d["name"], d["index"])
        if not devices:
            self._mic_combo.addItem("No microphone found", -1)
        self._mic_combo.blockSignals(False)

    def _populate_output_devices(self):
        self._output_combo.blockSignals(True)
        self._output_combo.clear()
        self._output_combo.addItem("System default", None)
        for d in AudioPlayback.list_output_devices():
            self._output_combo.addItem(d["name"], d["index"])
        self._output_combo.blockSignals(False)

    @property
    def selected_mic_index(self) -> int | None:
        idx = self._mic_combo.currentData()
        if idx is None:
            return None
        if isinstance(idx, int) and idx >= 0:
            return idx
        return None

    @property
    def selected_output_device(self) -> int | None:
        """PortAudio output device index, or ``None`` for host default."""
        return self._output_combo.currentData()

    def _emit_audio_devices_changed(self, *_args) -> None:
        self.audio_devices_changed.emit()

    def _emit_gpu_preference_changed(self, _state: int) -> None:
        if self._gpu_check.isEnabled():
            self.gpu_preference_changed.emit()

    @property
    def use_gpu(self) -> bool:
        return self._gpu_check.isChecked() and self._gpu_check.isEnabled()

    @property
    def scoring_difficulty_id(self) -> str:
        data = self._difficulty_combo.currentData()
        if isinstance(data, str) and data in ("strict", "normal", "casual"):
            return data
        return "strict"
