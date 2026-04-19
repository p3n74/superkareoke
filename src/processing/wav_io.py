"""WAV load/save without torchaudio I/O (avoids torchCodec on recent torchaudio)."""

from pathlib import Path

import soundfile as sf
import torch


def load_wave(path: Path) -> tuple[torch.Tensor, int]:
    """Load WAV as float32 tensor shaped [channels, samples]."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, int(sr)


def save_wave(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save tensor [channels, samples] as float WAV."""
    w = waveform.detach().cpu().contiguous()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    arr = w.numpy().T.astype("float32", copy=False)
    sf.write(str(path), arr, sample_rate, format="WAV", subtype="FLOAT")
