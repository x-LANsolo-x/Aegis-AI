import importlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from services.api.app.models import AnalysisRecord


def _build_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    return main.app


def _engine(tmp_path: Path):
    return create_engine(
        f"sqlite:///{(tmp_path / 'test_api.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )


def _insert_record(tmp_path: Path, rec: AnalysisRecord):
    engine = _engine(tmp_path)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(rec)
        s.commit()


def test_single_record_get(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    rid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _insert_record(
        tmp_path,
        AnalysisRecord(
            id=rid,
            created_at=created_at,
            media_type="audio",
            filename="a.wav",
            sha256="h",
            verdict="SUSPICIOUS",
            confidence=0.7,
            explanations=["x"],
            signals={"s": 1},
            model_version="dummy-0.1",
        ),
    )

    with TestClient(app) as client:
        r = client.get(f"/v1/analysis/{rid}")
        assert r.status_code == 200
        data = r.json()

    assert data["id"] == str(rid)
    assert data["filename"] == "a.wav"
    assert data["sha256"] == "h"
    assert data["media_type"] == "audio"
    assert data["verdict"] == "SUSPICIOUS"
    assert data["confidence"] == 0.7
    assert data["explanations"] == ["x"]
    assert data["signals"] == {"s": 1}


def test_list_with_pagination_order(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Insert three records with increasing created_at; API sorts DESC.
    recs = [
        AnalysisRecord(
            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            created_at=base,
            media_type="audio",
            filename="0.wav",
            sha256="h0",
            verdict="SUSPICIOUS",
            confidence=0.0,
        ),
        AnalysisRecord(
            id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
            created_at=base.replace(minute=1),
            media_type="audio",
            filename="1.wav",
            sha256="h1",
            verdict="SUSPICIOUS",
            confidence=0.0,
        ),
        AnalysisRecord(
            id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
            created_at=base.replace(minute=2),
            media_type="audio",
            filename="2.wav",
            sha256="h2",
            verdict="SUSPICIOUS",
            confidence=0.0,
        ),
    ]

    for r in recs:
        _insert_record(tmp_path, r)

    with TestClient(app) as client:
        r = client.get("/v1/analysis?limit=2&offset=1")
        assert r.status_code == 200
        data = r.json()

    assert data["total"] == 3
    assert len(data["items"]) == 2

    # Expected order is created_at DESC, so items are [2.wav, 1.wav, 0.wav]
    # offset=1 => [1.wav, 0.wav]
    assert data["items"][0]["filename"] == "1.wav"
    assert data["items"][1]["filename"] == "0.wav"


def test_report_endpoint_contract_and_snapshot(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    rid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Ensure uploads file exists so size_bytes is deterministic
    uploads = Path(tmp_path) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "a.wav").write_bytes(b"abc")

    _insert_record(
        tmp_path,
        AnalysisRecord(
            id=rid,
            created_at=created_at,
            media_type="audio",
            filename="a.wav",
            sha256="h",
            verdict="SUSPICIOUS",
            confidence=0.7,
            explanations=["x"],
            signals={"s": 1},
        ),
    )

    with TestClient(app) as client:
        r = client.get(f"/v1/analysis/{rid}/report")
        assert r.status_code == 200
        data = r.json()

    # Contract keys
    for key in [
        "report_version",
        "analysis_id",
        "evidence",
        "verdict",
        "confidence",
        "key_findings",
        "recommended_actions",
        "chain_of_custody",
    ]:
        assert key in data

    # Snapshot compare to fixture
    fixture_path = Path(__file__).parent / "fixtures" / "report_snapshot.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert data == expected
