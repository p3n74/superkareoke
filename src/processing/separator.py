import logging
from pathlib import Path
from typing import Any, Optional
import time
import torch
import torchaudio

from src.processing.memory import release_torch_memory
from src.processing.wav_io import load_wave, save_wave

logger = logging.getLogger(__name__)


class VocalSeparator:
    """Separates vocals from a song using Hybrid Demucs via torchaudio."""

    def __init__(
        self,
        device: Optional[str] = None,
        preset: Optional[dict[str, Any]] = None,
    ):
        if device is None:
            from src.processing.torch_device import resolve_processing_device_string

            device = resolve_processing_device_string(True)
        self.device = torch.device(device)
        self._preset = dict(preset) if preset else {}
        self._model = None
        self._bundle = None

    def _load_model(self):
        if self._model is not None:
            return
        import torchaudio.pipelines as tap

        t0 = time.perf_counter()
        name = str(self._preset.get("demucs_bundle", "HDEMUCS_HIGH_MUSDB"))
        bundle_obj = getattr(tap, name, None)
        if bundle_obj is None:
            logger.warning(
                "Demucs bundle %r not found in torchaudio; using HDEMUCS_HIGH_MUSDB_PLUS",
                name,
            )
            bundle_obj = tap.HDEMUCS_HIGH_MUSDB_PLUS
        self._bundle = bundle_obj
        self._model = self._bundle.get_model().to(self.device).eval()
        logger.info(
            "Demucs loaded in %.1fs bundle=%s device=%s segment=%ss overlap=%s",
            time.perf_counter() - t0,
            name,
            self.device,
            self._preset.get("demucs_segment_s", "?"),
            self._preset.get("demucs_overlap", "?"),
        )

    def separate(
        self, audio_path: Path, output_dir: Path,
        segment: Optional[float] = None,
        overlap: Optional[float] = None,
        progress_callback=None,
    ) -> dict[str, Path]:
        self._load_model()
        output_dir.mkdir(parents=True, exist_ok=True)

        seg = float(segment if segment is not None else self._preset.get("demucs_segment_s", 6.0))
        ovl = float(overlap if overlap is not None else self._preset.get("demucs_overlap", 0.08))

        waveform, sr = load_wave(audio_path)
        dur_s = waveform.shape[-1] / float(sr)
        logger.info(
            "Separating: %s (%.2fs @ %d Hz) -> %s [segment=%.1fs overlap=%.2f]",
            audio_path, dur_s, sr, output_dir, seg, ovl,
        )

        target_sr = self._bundle.sample_rate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        last_decile = [-1]

        def sep_progress(p: float) -> None:
            if progress_callback:
                progress_callback(p)
            d = int(p * 10)
            if d > last_decile[0]:
                last_decile[0] = d
                logger.info("Demucs segment pass ~%d%%", int(p * 100))

        sources = self._separate_sources(
            waveform, segment=seg, overlap=ovl,
            progress_callback=sep_progress,
        )
        del waveform
        release_torch_memory()

        # Demucs outputs: drums, bass, other, vocals (index order)
        source_names = ["drums", "bass", "other", "vocals"]
        output_paths: dict[str, Path] = {}

        for i, name in enumerate(source_names):
            out_path = output_dir / f"{name}.wav"
            save_wave(out_path, sources[i].cpu(), target_sr)
            output_paths[name] = out_path

        instrumental = sources[0] + sources[1] + sources[2]
        instrumental_path = output_dir / "instrumental.wav"
        save_wave(instrumental_path, instrumental.cpu(), target_sr)
        output_paths["instrumental"] = instrumental_path

        del instrumental, sources
        release_torch_memory()

        return output_paths

    def _separate_sources(
        self, mix: torch.Tensor, segment: float, overlap: float,
        progress_callback=None,
    ) -> torch.Tensor:
        """Process audio in overlapping segments to manage memory."""
        mix = mix.to(self.device)
        batch = mix.unsqueeze(0)
        sr = self._bundle.sample_rate

        if segment is None or segment <= 0:
            with torch.no_grad():
                return self._model(batch)[0]

        n_samples = mix.shape[-1]
        seg_len = int(segment * sr)
        overlap_len = int(overlap * seg_len)
        stride = seg_len - overlap_len

        n_segments = max(1, (n_samples - overlap_len + stride - 1) // stride)
        n_sources = 4
        output = torch.zeros(n_sources, mix.shape[0], n_samples, device=self.device)
        weight = torch.zeros(n_samples, device=self.device)

        for i in range(n_segments):
            start = i * stride
            end = min(start + seg_len, n_samples)
            chunk = batch[..., start:end]

            with torch.no_grad():
                separated = self._model(chunk)[0]

            chunk_len = end - start
            fade = torch.ones(chunk_len, device=self.device)
            if i > 0 and overlap_len > 0:
                fade_in = torch.linspace(0, 1, min(overlap_len, chunk_len), device=self.device)
                fade[:len(fade_in)] = fade_in
            if i < n_segments - 1 and overlap_len > 0:
                fade_out = torch.linspace(1, 0, min(overlap_len, chunk_len), device=self.device)
                fade[-len(fade_out):] = fade_out

            for s in range(n_sources):
                output[s, :, start:end] += separated[s] * fade
            weight[start:end] += fade

            del separated, chunk, fade

            if progress_callback:
                progress_callback((i + 1) / n_segments)

        weight = weight.clamp(min=1e-8)
        output /= weight
        del batch, mix, weight
        return output
