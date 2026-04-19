import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from src.config import PITCH_FMIN, PITCH_FMAX, PITCH_HOP_MS, SAMPLE_RATE
from src.processing.wav_io import load_wave
from src.database.models import PitchEvent, PitchMap

logger = logging.getLogger(__name__)


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq: float) -> str:
    if freq <= 0:
        return ""
    semitones = 12 * math.log2(freq / 440.0)
    midi = round(69 + semitones)
    octave = (midi // 12) - 1
    note_idx = midi % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


def freq_to_cents(freq: float, ref_freq: float) -> float:
    if freq <= 0 or ref_freq <= 0:
        return float("inf")
    return 1200 * math.log2(freq / ref_freq)


class PitchExtractor:
    """Extracts pitch from audio using torchcrepe."""

    def __init__(self, model_size: str = "full", device: Optional[str] = None):
        self.model_size = model_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def extract(
        self, audio_path: Path, song_id: int,
        confidence_threshold: float = 0.5,
        progress_callback=None,
    ) -> PitchMap:
        import torchcrepe

        t0 = time.perf_counter()
        audio, sr = load_wave(audio_path)
        n_sec = audio.shape[-1] / float(sr)
        logger.info(
            "Pitch: loaded %s (%.2fs audio, %d ch @ %d Hz) device=%s model=%s",
            audio_path, n_sec, audio.shape[0], sr, self.device, self.model_size,
        )

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            t_rs = time.perf_counter()
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            logger.debug("Pitch: resampled %d Hz -> %d in %.2fs", sr, SAMPLE_RATE, time.perf_counter() - t_rs)

        audio = audio.to(self.device)
        hop_length = int(SAMPLE_RATE * PITCH_HOP_MS / 1000)
        n_frames = 1 + (audio.shape[-1] - 1) // hop_length
        if self.device.type == "cuda":
            backend = f"GPU (CUDA, {self.device})"
        elif self.device.type == "mps":
            backend = "GPU (Apple MPS)"
        else:
            backend = "CPU"
        logger.info(
            "Pitch: running torchcrepe.predict on %s (~%d frames, hop=%d ms). "
            "Expect this step to be slow on CPU (often many minutes per song); "
            "on CUDA it is usually much faster (roughly ~0.5-3x realtime is typical).",
            backend,
            n_frames,
            PITCH_HOP_MS,
        )

        t_pred = time.perf_counter()
        pitch, periodicity = torchcrepe.predict(
            audio, SAMPLE_RATE, hop_length,
            fmin=PITCH_FMIN, fmax=PITCH_FMAX,
            model=self.model_size, device=self.device,
            return_periodicity=True,
            batch_size=2048,
        )
        logger.info("Pitch: torchcrepe.predict finished in %.1fs", time.perf_counter() - t_pred)

        pitch = pitch.squeeze().cpu().numpy()
        periodicity = periodicity.squeeze().cpu().numpy()

        # Apply Viterbi smoothing
        pitch = np.where(periodicity > confidence_threshold, pitch, 0.0)

        events: list[PitchEvent] = []
        n_pitch = len(pitch)
        last_emit = -1

        for i, (f, c) in enumerate(zip(pitch, periodicity)):
            t = i * PITCH_HOP_MS / 1000.0
            freq = float(f) if c > confidence_threshold else 0.0
            note = freq_to_note(freq) if freq > 0 else ""
            events.append(PitchEvent(
                time=round(t, 4),
                frequency=round(freq, 2),
                note=note,
                confidence=round(float(c), 4),
            ))

            frac = (i + 1) / n_pitch
            decile = int(frac * 10)
            if decile > last_emit:
                last_emit = decile
                logger.info("Pitch: building events ~%d%% (%d/%d)", int(frac * 100), i + 1, n_pitch)
                if progress_callback:
                    progress_callback(frac)

        if progress_callback:
            progress_callback(1.0)
        logger.info("Pitch: total extract time %.1fs (events=%d)", time.perf_counter() - t0, len(events))

        return PitchMap(
            song_id=song_id,
            events=events,
            sample_rate=SAMPLE_RATE,
            hop_ms=PITCH_HOP_MS,
        )
