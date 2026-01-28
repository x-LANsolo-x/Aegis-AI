from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisRecordOut(BaseModel):
    id: str
    created_at: datetime

    media_type: str
    filename: str
    sha256: str
    duration_s: Optional[float] = None

    verdict: str
    confidence: float

    explanations: List[str] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)

    model_version: Optional[str] = None
    processing_ms: Optional[int] = None


class EvidenceOut(BaseModel):
    filename: str
    sha256: str
    media_type: str
    size_bytes: Optional[int] = None


class ChainOfCustodyOut(BaseModel):
    created_at: datetime
    device_id: Optional[str] = None


class VideoMetadataOut(BaseModel):
    """Video-specific metadata."""
    duration_sec: float
    fps: float
    resolution: str
    codec: Optional[str] = None
    num_frames_analyzed: int
    faces_detected: int


class FrameAnalysisOut(BaseModel):
    """Per-frame analysis result."""
    frame_number: int
    timestamp_sec: float
    verdict: str
    confidence: float
    face_bbox: Optional[Dict[str, int]] = None
    artifacts_detected: List[str] = Field(default_factory=list)


class AnalysisReportOut(BaseModel):
    report_version: str = "1.0"
    analysis_id: str

    evidence: EvidenceOut

    verdict: str
    confidence: float

    key_findings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

    chain_of_custody: ChainOfCustodyOut
    
    # Video-specific (optional)
    video_metadata: Optional[VideoMetadataOut] = None
    frame_analyses: Optional[List[FrameAnalysisOut]] = None
