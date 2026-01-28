from datetime import datetime, timezone
from pathlib import Path

from services.api.app.reporting import build_report


class _Rec:
    def __init__(self):
        self.id = "123"
        self.filename = "a.wav"
        self.sha256 = "h"
        self.media_type = "audio"
        self.created_at = datetime.now(timezone.utc)
        self.verdict = "SUSPICIOUS"
        self.confidence = 0.7
        self.explanations = ["x"]


def test_build_report_basic(tmp_path, monkeypatch):
    # Ensure uploads dir resolves to tmp and contains file
    monkeypatch.chdir(tmp_path)
    up = Path(tmp_path) / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    (up / "a.wav").write_bytes(b"abc")

    rec = _Rec()
    report = build_report(rec)

    assert report.report_version == "1.0"
    assert report.analysis_id == "123"
    assert report.evidence.filename == "a.wav"
    assert report.evidence.size_bytes == 3
    assert report.key_findings == ["x"]
    assert report.recommended_actions
