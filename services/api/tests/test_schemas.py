from datetime import datetime, timezone

from services.api.app.schemas import AnalysisRecordOut, AnalysisReportOut


def test_analysis_record_out_serialization():
    now = datetime.now(timezone.utc)
    out = AnalysisRecordOut(
        id="123",
        created_at=now,
        media_type="audio",
        filename="a.wav",
        sha256="h",
        verdict="SUSPICIOUS",
        confidence=0.0,
        explanations=["x"],
        signals={"s": 1},
        model_version=None,
        processing_ms=None,
    )

    data = out.model_dump()
    assert data["id"] == "123"
    assert data["media_type"] == "audio"
    assert data["explanations"] == ["x"]


def test_analysis_report_out_serialization():
    now = datetime.now(timezone.utc)
    report = AnalysisReportOut(
        analysis_id="123",
        evidence={"filename": "a.wav", "sha256": "h", "media_type": "audio", "size_bytes": 11},
        verdict="SAFE",
        confidence=0.1,
        key_findings=["k1"],
        recommended_actions=["Verify via secondary channel"],
        chain_of_custody={"created_at": now, "device_id": None},
    )

    data = report.model_dump()
    assert data["report_version"] == "1.0"
    assert data["evidence"]["size_bytes"] == 11
    assert data["chain_of_custody"]["created_at"]
