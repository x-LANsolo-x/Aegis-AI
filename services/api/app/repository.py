from __future__ import annotations

import uuid
from typing import Optional, Union

from sqlmodel import Session, select

from services.api.app.models import AnalysisRecord


def create_analysis_record(
    session: Session,
    *,
    media_type: str,
    filename: str,
    sha256: str,
    duration_s: Optional[float] = None,
    verdict: str = "PENDING",
    confidence: float = 0.0,
) -> AnalysisRecord:
    rec = AnalysisRecord(
        media_type=media_type,
        filename=filename,
        sha256=sha256,
        duration_s=duration_s,
        verdict=verdict,
        confidence=confidence,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def get_analysis_record_by_id(
    session: Session, record_id: Union[str, uuid.UUID]
) -> Optional[AnalysisRecord]:
    try:
        rid = uuid.UUID(record_id) if isinstance(record_id, str) else record_id
    except (ValueError, TypeError):
        return None

    return session.get(AnalysisRecord, rid)


def get_analysis_record_by_sha256(session: Session, sha256: str) -> Optional[AnalysisRecord]:
    return session.exec(select(AnalysisRecord).where(AnalysisRecord.sha256 == sha256)).first()


# Step 3 additions (API-friendly naming)

def get_analysis_by_id(session: Session, analysis_id: Union[str, uuid.UUID]) -> Optional[AnalysisRecord]:
    return get_analysis_record_by_id(session, analysis_id)


def list_analyses(session: Session, *, limit: int, offset: int):
    return session.exec(
        select(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()


def count_analyses(session: Session) -> int:
    from sqlalchemy import func

    return int(session.exec(select(func.count()).select_from(AnalysisRecord)).one())


def update_analysis_record(session: Session, rec: AnalysisRecord) -> AnalysisRecord:
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec
