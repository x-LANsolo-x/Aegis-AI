"""Dataset loader for ASVspoof manifests.

Reads JSONL manifest and provides audio loading utilities.
Supports torchaudio (preferred) with soundfile fallback.

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class AudioSample:
    """A single audio sample from the manifest."""

    path: str
    label: str
    duration_sec: float
    sample_rate: int
    split: str


def read_manifest(manifest_path: Path) -> Iterator[AudioSample]:
    """Read manifest JSONL and yield AudioSample records."""

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield AudioSample(
                path=rec["path"],
                label=rec["label"],
                duration_sec=rec["duration_sec"],
                sample_rate=rec["sample_rate"],
                split=rec["split"],
            )


def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio file and return (waveform, sample_rate).

    Tries torchaudio first, falls back to soundfile if not available.
    Returns waveform as numpy array (channels, samples) or (samples,) for mono.

    Args:
        path: audio file path
        target_sr: if provided, resample to this rate (requires torchaudio)

    Returns:
        (waveform, sample_rate)
    """

    # Try torchaudio
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        waveform = waveform.numpy()

        if target_sr and target_sr != sr:
            import torchaudio.functional as F

            waveform_torch = torchaudio.torch.from_numpy(waveform)
            waveform_torch = F.resample(waveform_torch, sr, target_sr)
            waveform = waveform_torch.numpy()
            sr = target_sr

        return waveform, sr

    except ImportError:
        pass

    # Fallback to soundfile
    try:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32")
        # soundfile returns (samples,) or (samples, channels); transpose if needed
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # (1, samples)
        else:
            waveform = waveform.T  # (channels, samples)

        if target_sr and target_sr != sr:
            sys.stderr.write(
                f"Warning: target_sr={target_sr} requested but resampling requires torchaudio; using native sr={sr}\n"
            )

        return waveform, sr

    except ImportError:
        raise RuntimeError("Neither torchaudio nor soundfile is available. Install one to load audio.")


def load_sample(sample: AudioSample, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int, str]:
    """Load audio from an AudioSample and return (waveform, sample_rate, label)."""

    waveform, sr = load_audio(sample.path, target_sr=target_sr)
    return waveform, sr, sample.label
