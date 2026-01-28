from __future__ import annotations

from typing import Optional

from sqlmodel import Session, select

from services.api.app.models import AnalysisRecord


def create_analysis_record(
    session: Session,
    *,
    media_type: str,
    filename: str,
    sha256: str,
    duration_s: Optional[float] = None,
) -> AnalysisRecord:
    rec = AnalysisRecord(
        media_type=media_type,
        filename=filename,
        sha256=sha256,
        duration_s=duration_s,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def get_analysis_record_by_id(session: Session, record_id) -> Optional[AnalysisRecord]:
    return session.get(AnalysisRecord, record_id)


def get_analysis_record_by_sha256(session: Session, sha256: str) -> Optional[AnalysisRecord]:
    return session.exec(select(AnalysisRecord).where(AnalysisRecord.sha256 == sha256)).first()


def update_analysis_record(session: Session, rec: AnalysisRecord) -> AnalysisRecord:
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec
