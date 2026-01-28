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
    sha256: str = Field(index=True, unique=True)
    duration_s: Optional[float] = None

    # PENDING, AUTHENTIC, SUSPICIOUS, DEEPFAKE
    verdict: str = Field(default="PENDING", index=True)
    confidence: float = Field(default=0.0)

    # NOTE: use default_factory to avoid mutable default pitfalls
    explanations: List[str] = Field(sa_column=Column(JSON), default_factory=list)
    signals: Dict = Field(sa_column=Column(JSON), default_factory=dict)

    model_version: Optional[str] = None
    processing_ms: Optional[int] = None
