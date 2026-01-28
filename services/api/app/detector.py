from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from pydantic import BaseModel, Field


class DetectorInput(BaseModel):
    file_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectorOutput(BaseModel):
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanations: List[str] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)
    model_version: str


class Detector(Protocol):
    def run(self, data: DetectorInput) -> DetectorOutput:  # pragma: no cover
        ...


@dataclass
class OnnxAudioDetector:
    """Production ONNX-based audio deepfake detector.

    Loads an ONNX model at startup and runs inference on audio files.
    """

    model_path: str
    model_version: str = "onnx-0.1"
    target_sample_rate: int = 16000
    threshold_suspicious: float = 0.3  # p_fake >= 0.3 -> SUSPICIOUS
    threshold_deepfake: float = 0.7  # p_fake >= 0.7 -> DEEPFAKE

    def __post_init__(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(self.model_path)
        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self, data: DetectorInput) -> DetectorOutput:
        import numpy as np
        import soundfile as sf

        # 1. Load audio
        try:
            waveform, sr = sf.read(data.file_path, dtype="float32")
        except Exception as e:
            return DetectorOutput(
                verdict="FAILED",
                confidence=0.0,
                explanations=[f"Audio load error: {e}"],
                signals={"error": "load_failed"},
                model_version=self.model_version,
            )

        # 2. Resample / normalize
        # (Basic: assume model expects mono 16kHz; production should add resampling)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # stereo -> mono

        # Normalize to [-1, 1]
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            waveform = waveform / max(abs(waveform.max()), abs(waveform.min()))

        # 3. Convert to model input (example: expecting (1, samples) float32)
        # Adjust shape based on actual model input requirements.
        waveform = waveform.astype(np.float32)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # (1, samples)

        # 4. Run ONNX session
        try:
            outputs = self.session.run([self.output_name], {self.input_name: waveform})
            logits = outputs[0]  # shape depends on model; assume (1, num_classes) or (1,)
        except Exception as e:
            return DetectorOutput(
                verdict="FAILED",
                confidence=0.0,
                explanations=[f"Inference error: {e}"],
                signals={"error": "inference_failed"},
                model_version=self.model_version,
            )

        # 5. Build DetectorOutput
        # Assume binary classification: logits is (1,) or (1, 2) -> extract p_fake
        if logits.ndim == 2 and logits.shape[1] == 2:
            # Two-class output: [p_real, p_fake]
            p_fake = float(logits[0, 1])
        else:
            # Single output: assume sigmoid already applied or raw score -> clip to [0,1]
            p_fake = float(np.clip(logits.flatten()[0], 0.0, 1.0))

        # Verdict logic
        if p_fake >= self.threshold_deepfake:
            verdict = "DEEPFAKE"
        elif p_fake >= self.threshold_suspicious:
            verdict = "SUSPICIOUS"
        else:
            verdict = "AUTHENTIC"

        # Explanations (stub; enhance with feature analysis later)
        explanations = [
            f"Model output probability: {p_fake:.3f}",
            "ONNX inference completed",
        ]

        return DetectorOutput(
            verdict=verdict,
            confidence=float(p_fake),
            explanations=explanations,
            signals={"p_fake": p_fake, "logits": logits.tolist()},
            model_version=self.model_version,
        )


@dataclass
class DummyDetector:
    """A deterministic detector for development/testing.

    Returns a fixed response. Optional sleep_seconds simulates slow inference.
    """

    sleep_seconds: float = 0.0

    def run(self, data: DetectorInput) -> DetectorOutput:
        if self.sleep_seconds and self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        return DetectorOutput(
            verdict="SAFE",
            confidence=0.1,
            explanations=["dummy detector"],
            signals={"dummy": True},
            model_version="dummy-0.1",
        )
