from __future__ import annotations

from pathlib import Path

from services.api.app.config import ensure_upload_dir_exists
from services.api.app.schemas import AnalysisReportOut


REPORT_VERSION = "1.0"


def build_report(record) -> AnalysisReportOut:
    """Build an AnalysisReportOut from an AnalysisRecord.

    Centralizes report generation to keep output stable.
    """

    # Best-effort size_bytes from file on disk.
    try:
        size_bytes = (ensure_upload_dir_exists() / record.filename).stat().st_size
    except Exception:
        size_bytes = None

    # Stub recommended actions for now; later make policy-driven.
    recommended_actions = []
    if getattr(record, "verdict", None) in {"SUSPICIOUS", "DEEPFAKE", "FAILED"}:
        recommended_actions = ["Verify via secondary trusted channel"]

    return AnalysisReportOut(
        report_version=REPORT_VERSION,
        analysis_id=str(record.id),
        evidence={
            "filename": record.filename,
            "sha256": record.sha256,
            "media_type": record.media_type,
            "size_bytes": size_bytes,
        },
        verdict=record.verdict,
        confidence=record.confidence,
        key_findings=list(getattr(record, "explanations", None) or []),
        recommended_actions=recommended_actions,
        chain_of_custody={
            "created_at": record.created_at,
            "device_id": None,
        },
    )
