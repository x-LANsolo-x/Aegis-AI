import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class AnalysisRecord(SQLModel, table=True):
    __tablename__ = "analysis_records"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)

    # "audio", "video", "image"
    media_type: str = Field(index=True)
    filename: str
    # NOTE: do NOT enforce uniqueness. The product may store multiple analysis
    # records for the same media hash (e.g., repeated uploads, different operators).
    sha256: str = Field(index=True)
    duration_s: Optional[float] = None

    verdict: str = Field(default="PENDING", index=True)
    confidence: float = Field(default=0.0)

    # NOTE: use default_factory to avoid mutable default pitfalls
    explanations: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    signals: Dict = Field(sa_column=Column(JSON), default_factory=dict)

    model_version: Optional[str] = None
    processing_ms: Optional[int] = None


class VideoMetadata(SQLModel, table=True):
    """Video-specific metadata for video analysis."""
    __tablename__ = "video_metadata"
    
    analysis_id: uuid.UUID = Field(foreign_key="analysis_records.id", primary_key=True)
    
    duration_sec: float = Field(nullable=False)
    fps: float = Field(nullable=False)
    resolution: str = Field(nullable=False)  # e.g., "1920x1080"
    codec: Optional[str] = None
    num_frames_analyzed: int = Field(nullable=False)
    faces_detected: int = Field(nullable=False)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)


class FrameAnalysis(SQLModel, table=True):
    """Per-frame analysis results for video."""
    __tablename__ = "frame_analysis"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    analysis_id: uuid.UUID = Field(foreign_key="analysis_records.id", nullable=False, index=True)
    
    frame_number: int = Field(nullable=False)
    timestamp_sec: float = Field(nullable=False)
    verdict: str = Field(nullable=False)
    confidence: float = Field(nullable=False)
    
    face_bbox: Optional[Dict] = Field(sa_column=Column(JSON), default=None)  # {x, y, w, h}
    artifacts_detected: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
