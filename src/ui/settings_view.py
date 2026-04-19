from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QCheckBox, QLineEdit, QMessageBox,
    QSlider, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import sounddevice as sd

from src.audio.mic_input import MicInput
from src.audio.playback import AudioPlayback
from src.config import (
    SPOTIFY_CLIENT_ID,
    VOICE_FX_PRESET_LABELS,
    voice_fx_preset,
    MIC_PROCESSOR_DEFAULTS,
)


class SettingsView(QWidget):
    """GPU preference changes are emitted so the pipeline can drop cached torch models."""

    gpu_preference_changed = pyqtSignal()
    audio_devices_changed = pyqtSignal()
    vocal_chain_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Loaded with the "natural" preset values to give pleasant defaults
        self._fx_params: dict[str, float] = voice_fx_preset("natural")
        self._fx_preset_id: str = "natural"
        self._suppress_fx_signals: bool = False
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        layout = QVBoxLayout(content)
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

        # ── Microphone enhancement ──
        mic_enh = QGroupBox("Microphone enhancement")
        mic_enh.setStyleSheet(audio_group.styleSheet())
        mic_form = QFormLayout(mic_enh)

        self._mic_gain_slider, mic_gain_row = self._make_slider(
            -12, 24, int(MIC_PROCESSOR_DEFAULTS["input_gain_db"]), unit="dB"
        )
        mic_form.addRow("Input gain:", mic_gain_row)

        self._mic_gate_slider, mic_gate_row = self._make_slider(
            -70, -20, int(MIC_PROCESSOR_DEFAULTS["gate_db"]), unit="dB"
        )
        self._mic_gate_slider.setToolTip(
            "Below this RMS the mic is muted (with hysteresis). Raise it if you "
            "still hear background noise / static when not singing."
        )
        mic_form.addRow("Noise gate:", mic_gate_row)

        self._mic_vad_check = QCheckBox("Voice activity gate (Silero VAD)")
        self._mic_vad_check.setStyleSheet("color: #e0e0e0;")
        self._mic_vad_check.setChecked(bool(MIC_PROCESSOR_DEFAULTS["use_vad"]))
        self._mic_vad_check.setToolTip(
            "Mutes the monitor signal when no human voice is detected. Pitch detection "
            "is unaffected."
        )
        mic_form.addRow(self._mic_vad_check)

        for w in (self._mic_gain_slider, self._mic_gate_slider):
            w.valueChanged.connect(self._emit_vocal_chain_changed)
        self._mic_vad_check.stateChanged.connect(self._emit_vocal_chain_changed)

        layout.addWidget(mic_enh)

        # ── Vocal monitor ──
        monitor_group = QGroupBox("Vocal monitor (hear yourself sing)")
        monitor_group.setStyleSheet(audio_group.styleSheet())
        monitor_form = QFormLayout(monitor_group)

        self._monitor_check = QCheckBox("Enable vocal monitor while singing")
        self._monitor_check.setStyleSheet("color: #e0e0e0;")
        self._monitor_check.setChecked(False)
        monitor_form.addRow(self._monitor_check)

        self._monitor_combo = QComboBox()
        self._monitor_combo.setStyleSheet(self._mic_combo.styleSheet())
        self._populate_monitor_devices()
        self._monitor_combo.setToolTip(
            "Where to play the singer's voice. Use headphones (different from "
            "the backing-track output) to avoid feedback."
        )
        monitor_form.addRow("Monitor output:", self._monitor_combo)

        self._monitor_gain_slider, monitor_gain_row = self._make_slider(
            -24, 12, -3, unit="dB"
        )
        monitor_form.addRow("Vocal gain:", monitor_gain_row)

        self._monitor_protect_check = QCheckBox(
            "Feedback protection: auto-mute monitor when it shares the backing-track output"
        )
        self._monitor_protect_check.setStyleSheet("color: #e0e0e0;")
        self._monitor_protect_check.setChecked(True)
        monitor_form.addRow(self._monitor_protect_check)

        self._monitor_bypass_check = QCheckBox(
            "Bypass effects (raw mic monitor — skip gate, EQ, compressor, reverb)"
        )
        self._monitor_bypass_check.setStyleSheet("color: #e0e0e0;")
        self._monitor_bypass_check.setChecked(True)
        self._monitor_bypass_check.setToolTip(
            "Sends the microphone straight to the monitor output with no DSP. "
            "Useful when troubleshooting artefacts in the FX chain."
        )
        monitor_form.addRow(self._monitor_bypass_check)

        self._monitor_warn = QLabel("")
        self._monitor_warn.setStyleSheet("color: #f0ad4e; font-size: 11px;")
        self._monitor_warn.setWordWrap(True)
        monitor_form.addRow(self._monitor_warn)

        self._monitor_check.stateChanged.connect(self._emit_vocal_chain_changed)
        self._monitor_combo.currentIndexChanged.connect(self._emit_vocal_chain_changed)
        self._monitor_gain_slider.valueChanged.connect(self._emit_vocal_chain_changed)
        self._monitor_protect_check.stateChanged.connect(self._emit_vocal_chain_changed)
        self._monitor_bypass_check.stateChanged.connect(self._emit_vocal_chain_changed)

        layout.addWidget(monitor_group)

        # ── Vocal effects ──
        fx_group = QGroupBox("Vocal effects")
        fx_group.setStyleSheet(audio_group.styleSheet())
        fx_form = QFormLayout(fx_group)

        self._fx_preset_combo = QComboBox()
        self._fx_preset_combo.setStyleSheet(self._mic_combo.styleSheet())
        for pid, label in VOICE_FX_PRESET_LABELS:
            self._fx_preset_combo.addItem(label, pid)
        # plus a Custom marker
        self._fx_preset_combo.addItem("Custom (manual sliders)", "custom")
        # Default to "Natural"
        self._restore_combo_by_data(self._fx_preset_combo, self._fx_preset_id)
        self._fx_preset_combo.currentIndexChanged.connect(self._on_fx_preset_changed)
        fx_form.addRow("Preset:", self._fx_preset_combo)

        # Manual sliders. They reflect (and override) the active preset.
        self._fx_hpf_slider, fx_hpf_row = self._make_slider(0, 250, int(self._fx_params["hpf_hz"]), unit="Hz")
        self._fx_low_slider, fx_low_row = self._make_slider(-12, 12, int(self._fx_params["eq_low_db"]), unit="dB")
        self._fx_mid_slider, fx_mid_row = self._make_slider(-12, 12, int(self._fx_params["eq_mid_db"]), unit="dB")
        self._fx_high_slider, fx_high_row = self._make_slider(-12, 12, int(self._fx_params["eq_high_db"]), unit="dB")
        self._fx_comp_thr_slider, fx_comp_thr_row = self._make_slider(-40, 0, int(self._fx_params["comp_threshold_db"]), unit="dB")
        self._fx_comp_ratio_slider, fx_comp_ratio_row = self._make_slider(10, 80, int(self._fx_params["comp_ratio"] * 10), unit="x10")
        self._fx_reverb_slider, fx_reverb_row = self._make_slider(0, 80, int(self._fx_params["reverb_mix"] * 100), unit="%")
        self._fx_gain_slider, fx_gain_row = self._make_slider(-12, 18, int(self._fx_params["output_gain_db"]), unit="dB")

        fx_form.addRow("High-pass:", fx_hpf_row)
        fx_form.addRow("Low EQ:", fx_low_row)
        fx_form.addRow("Mid EQ:", fx_mid_row)
        fx_form.addRow("High EQ:", fx_high_row)
        fx_form.addRow("Compressor threshold:", fx_comp_thr_row)
        fx_form.addRow("Compressor ratio (×10):", fx_comp_ratio_row)
        fx_form.addRow("Reverb mix:", fx_reverb_row)
        fx_form.addRow("Output gain:", fx_gain_row)

        for slider in (
            self._fx_hpf_slider, self._fx_low_slider, self._fx_mid_slider, self._fx_high_slider,
            self._fx_comp_thr_slider, self._fx_comp_ratio_slider, self._fx_reverb_slider,
            self._fx_gain_slider,
        ):
            slider.valueChanged.connect(self._on_fx_slider_changed)

        layout.addWidget(fx_group)

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
        cur_mon = self._monitor_combo.currentData() if hasattr(self, "_monitor_combo") else None
        self._populate_mic_devices()
        self._populate_output_devices()
        if hasattr(self, "_monitor_combo"):
            self._populate_monitor_devices()
            self._restore_combo_by_data(self._monitor_combo, cur_mon)
        self._restore_combo_by_data(self._mic_combo, cur_mic)
        self._restore_combo_by_data(self._output_combo, cur_out)
        self.audio_devices_changed.emit()
        self._refresh_monitor_warning()

    # ── Helpers for the new sections ───────────────────────────────────
    def _make_slider(self, lo: int, hi: int, val: int, unit: str = "") -> tuple[QSlider, QWidget]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(val)
        slider.setMinimumWidth(180)
        label = QLabel(self._fmt_slider(val, unit))
        label.setStyleSheet("color: #c8c8d0; min-width: 56px;")

        def on_change(v):
            label.setText(self._fmt_slider(v, unit))

        slider.valueChanged.connect(on_change)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(slider, stretch=1)
        rl.addWidget(label, stretch=0)
        return slider, row

    @staticmethod
    def _fmt_slider(v: int, unit: str) -> str:
        if unit == "x10":
            return f"{v / 10:.1f}x"
        if unit:
            return f"{v} {unit}"
        return str(v)

    def _populate_monitor_devices(self):
        self._monitor_combo.blockSignals(True)
        self._monitor_combo.clear()
        self._monitor_combo.addItem("Same as backing track", "__same__")
        for d in AudioPlayback.list_output_devices():
            self._monitor_combo.addItem(d["name"], d["index"])
        self._monitor_combo.blockSignals(False)

    def _emit_vocal_chain_changed(self, *_a):
        if self._suppress_fx_signals:
            return
        self._refresh_monitor_warning()
        self.vocal_chain_changed.emit()

    def _refresh_monitor_warning(self) -> None:
        if not hasattr(self, "_monitor_warn"):
            return
        if not self._monitor_check.isChecked():
            self._monitor_warn.setText("")
            return
        mon = self._monitor_combo.currentData()
        same_device = mon == "__same__" or mon == self._output_combo.currentData()
        if same_device and self._monitor_protect_check.isChecked():
            self._monitor_warn.setText(
                "Monitor and backing track share the same output — feedback protection "
                "will keep the monitor muted. Pick a separate output (e.g. headphones) "
                "or uncheck feedback protection to override."
            )
        elif same_device:
            self._monitor_warn.setText(
                "Monitor and backing track share the same output. Use headphones or "
                "you'll hear feedback."
            )
        else:
            self._monitor_warn.setText("")

    def _on_fx_preset_changed(self, _idx: int) -> None:
        if self._suppress_fx_signals:
            return
        pid = self._fx_preset_combo.currentData()
        if pid == "custom":
            # Don't overwrite slider values; just publish current params.
            self._fx_preset_id = "custom"
            self._fx_params = self._read_fx_params_from_sliders()
            self._emit_vocal_chain_changed()
            return
        self._fx_preset_id = str(pid)
        self._fx_params = voice_fx_preset(self._fx_preset_id)
        self._suppress_fx_signals = True
        try:
            self._fx_hpf_slider.setValue(int(self._fx_params["hpf_hz"]))
            self._fx_low_slider.setValue(int(self._fx_params["eq_low_db"]))
            self._fx_mid_slider.setValue(int(self._fx_params["eq_mid_db"]))
            self._fx_high_slider.setValue(int(self._fx_params["eq_high_db"]))
            self._fx_comp_thr_slider.setValue(int(self._fx_params["comp_threshold_db"]))
            self._fx_comp_ratio_slider.setValue(int(self._fx_params["comp_ratio"] * 10))
            self._fx_reverb_slider.setValue(int(self._fx_params["reverb_mix"] * 100))
            self._fx_gain_slider.setValue(int(self._fx_params["output_gain_db"]))
        finally:
            self._suppress_fx_signals = False
        self._emit_vocal_chain_changed()

    def _on_fx_slider_changed(self, _v: int) -> None:
        if self._suppress_fx_signals:
            return
        self._fx_params = self._read_fx_params_from_sliders()
        self._fx_preset_id = "custom"
        # Reflect "Custom" without retriggering this slot
        self._suppress_fx_signals = True
        try:
            self._restore_combo_by_data(self._fx_preset_combo, "custom")
        finally:
            self._suppress_fx_signals = False
        self._emit_vocal_chain_changed()

    def _read_fx_params_from_sliders(self) -> dict[str, float]:
        return {
            "hpf_hz": float(self._fx_hpf_slider.value()),
            "eq_low_db": float(self._fx_low_slider.value()),
            "eq_mid_db": float(self._fx_mid_slider.value()),
            "eq_high_db": float(self._fx_high_slider.value()),
            "comp_threshold_db": float(self._fx_comp_thr_slider.value()),
            "comp_ratio": float(self._fx_comp_ratio_slider.value()) / 10.0,
            "reverb_mix": float(self._fx_reverb_slider.value()) / 100.0,
            "reverb_room": float(self._fx_params.get("reverb_room", 0.5)),
            "reverb_damp": float(self._fx_params.get("reverb_damp", 0.5)),
            "output_gain_db": float(self._fx_gain_slider.value()),
        }

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

    # ── Vocal chain getters used by PerformanceView ────────────────────
    @property
    def selected_monitor_device(self) -> int | None:
        data = self._monitor_combo.currentData()
        if data == "__same__":
            return None
        if isinstance(data, int):
            return data
        return None

    @property
    def mic_processor_settings(self) -> dict:
        return {
            "input_gain_db": float(self._mic_gain_slider.value()),
            "gate_db": float(self._mic_gate_slider.value()),
            "use_vad": bool(self._mic_vad_check.isChecked()),
        }

    @property
    def voice_fx_settings(self) -> dict:
        return self._read_fx_params_from_sliders()

    @property
    def vocal_chain_settings(self) -> dict:
        return {
            "mic_processor": self.mic_processor_settings,
            "voice_fx": self.voice_fx_settings,
            "monitor_enabled": bool(self._monitor_check.isChecked()),
            "monitor_device": self.selected_monitor_device,
            "monitor_gain_db": float(self._monitor_gain_slider.value()),
            "feedback_protection": bool(self._monitor_protect_check.isChecked()),
            "bypass_effects": bool(self._monitor_bypass_check.isChecked()),
        }
