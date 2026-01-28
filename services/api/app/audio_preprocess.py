"""Audio preprocessing utilities for inference.

Provides helpers to:
- load audio (soundfile)
- convert to mono
- resample (scipy.signal.resample or basic)
- normalize amplitude
- pad/crop to fixed length

"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return (waveform, sample_rate).

    Returns waveform as 1D float32 array (mono).
    """

    waveform, sr = sf.read(path, dtype="float32")
    return waveform, sr


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging."""

    if waveform.ndim > 1:
        return waveform.mean(axis=1).astype(np.float32)
    return waveform


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Uses scipy.signal.resample (free, stdlib-adjacent).
    Falls back to simple decimation/interpolation if scipy unavailable.
    """

    if orig_sr == target_sr:
        return waveform

    try:
        from scipy.signal import resample as scipy_resample

        target_length = int(len(waveform) * target_sr / orig_sr)
        return scipy_resample(waveform, target_length).astype(np.float32)

    except ImportError:
        # Basic fallback: linear interpolation
        target_length = int(len(waveform) * target_sr / orig_sr)
        indices = np.linspace(0, len(waveform) - 1, target_length)
        return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


def normalize_amplitude(waveform: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """Normalize waveform to [-target_max, target_max]."""

    max_val = max(abs(waveform.max()), abs(waveform.min()))
    if max_val > 0:
        return (waveform / max_val * target_max).astype(np.float32)
    return waveform


def pad_or_crop(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or crop to target_length."""

    if len(waveform) < target_length:
        # Pad
        pad_width = target_length - len(waveform)
        return np.pad(waveform, (0, pad_width), mode="constant").astype(np.float32)
    elif len(waveform) > target_length:
        # Crop (center crop)
        start = (len(waveform) - target_length) // 2
        return waveform[start : start + target_length].astype(np.float32)
    return waveform


def preprocess_audio_for_model(
    path: str,
    *,
    target_sr: int = 16000,
    target_length: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for model input.

    Returns a 1D float32 numpy array ready for inference.
    """

    waveform, sr = load_audio(path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sr, target_sr)

    if normalize:
        waveform = normalize_amplitude(waveform)

    if target_length:
        waveform = pad_or_crop(waveform, target_length)

    return waveform
