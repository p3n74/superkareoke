import math
import numpy as np
from typing import Optional
from collections import deque

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from PyQt6.QtCore import QObject, pyqtSignal

from src.config import PITCH_FMIN, PITCH_FMAX

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _freq_to_note(freq: float) -> str:
    if freq <= 0:
        return ""
    semitones = 12 * math.log2(freq / 440.0)
    midi = round(69 + semitones)
    octave = (midi // 12) - 1
    note_idx = midi % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


class RealtimePitchDetector(QObject):
    """Real-time pitch detection using torchcrepe tiny model.

    Receives audio chunks and emits detected pitch.
    """

    pitch_detected = pyqtSignal(float, float, str)  # frequency, confidence, note_name

    def __init__(
        self, sample_rate: int = 44100,
        device: Optional[str] = None,
        buffer_duration_ms: int = 80,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self._model_loaded = False
        self.device = None
        if HAS_TORCH:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)

        # Buffer enough audio for reliable pitch detection
        buffer_samples = int(sample_rate * buffer_duration_ms / 1000)
        self._buffer = deque(maxlen=buffer_samples)
        self._buffer_size = buffer_samples
        self._min_samples = int(sample_rate * 0.04)  # 40ms minimum

    def _ensure_model(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed")
        if not self._model_loaded:
            import torchcrepe
            torchcrepe.load.model(self.device, "tiny")
            self._model_loaded = True

    def process_chunk(self, audio: np.ndarray, sr: int):
        """Feed an audio chunk; emits pitch_detected when enough data is buffered."""
        if not HAS_TORCH:
            return
        try:
            self._ensure_model()
        except Exception:
            return

        self._buffer.extend(audio.tolist())

        if len(self._buffer) < self._min_samples:
            return

        try:
            import torchcrepe

            buf = np.array(self._buffer, dtype=np.float32)
            tensor = torch.from_numpy(buf).unsqueeze(0).to(self.device)

            hop_length = len(buf)

            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    tensor, sr, hop_length,
                    fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                    model="tiny", device=self.device,
                    return_periodicity=True,
                    batch_size=1,
                )

            freq = float(pitch[0, -1].cpu())
            conf = float(periodicity[0, -1].cpu())

            if conf < 0.3:
                freq = 0.0

            note = _freq_to_note(freq) if freq > 0 else ""
            self.pitch_detected.emit(freq, conf, note)

        except Exception:
            pass
        finally:
            keep = self._buffer_size // 2
            buf_list = list(self._buffer)
            self._buffer.clear()
            self._buffer.extend(buf_list[-keep:])
