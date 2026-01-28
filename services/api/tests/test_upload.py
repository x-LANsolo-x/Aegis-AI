import hashlib
import importlib
import uuid
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

from services.api.app.models import AnalysisRecord


def _build_app(tmp_path, monkeypatch, *, max_bytes: int = 20 * 1024 * 1024):
    """Build an app instance isolated to a temp directory.

    Ensures:
    - uploads go into tmp_path/uploads
    - sqlite DB file goes into tmp_path/test_api.db
    - config/database modules pick up env vars via reload
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(max_bytes))
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


def _db_engine_for_tmp(tmp_path: Path):
    # Match DATABASE_URL used in _build_app
    db_file = tmp_path / "test_api.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_upload_happy_path_and_db_row(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    payload = b"aegis-upload-fixture"
    expected_sha = hashlib.sha256(payload).hexdigest()

    with TestClient(app) as client:
        r = client.post(
            "/v1/analysis/upload",
            files={"file": ("fixture.wav", BytesIO(payload), "audio/wav")},
        )

    assert r.status_code == 200
    data = r.json()

    # analysis_id is valid UUID
    analysis_id = uuid.UUID(data["analysis_id"])
    assert data["sha256"] == expected_sha

    # verify DB row exists
    engine = _db_engine_for_tmp(tmp_path)
    with Session(engine) as s:
        rec = s.exec(select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)).one()
        assert rec.sha256 == expected_sha
        assert rec.filename == "fixture.wav"
        assert rec.verdict == "SUSPICIOUS"
        assert rec.confidence == 0.0


def test_upload_missing_file_400(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.post("/v1/analysis/upload", files={})

    assert r.status_code == 400


def test_upload_unsupported_extension_415(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.post(
            "/v1/analysis/upload",
            files={"file": ("evil.exe", BytesIO(b"x"), "application/octet-stream")},
        )

    assert r.status_code == 415


def test_upload_huge_file_limit_413(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch, max_bytes=1024)  # 1 KB

    payload = b"x" * 2048
    with TestClient(app) as client:
        r = client.post(
            "/v1/analysis/upload",
            files={"file": ("big.wav", BytesIO(payload), "audio/wav")},
        )

    assert r.status_code == 413


def test_upload_traversal_filename_sanitized_in_db(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    payload = b"abc"
    with TestClient(app) as client:
        r = client.post(
            "/v1/analysis/upload",
            files={"file": ("../../evil.wav", BytesIO(payload), "audio/wav")},
        )

    assert r.status_code == 200
    analysis_id = uuid.UUID(r.json()["analysis_id"])

    engine = _db_engine_for_tmp(tmp_path)
    with Session(engine) as s:
        rec = s.get(AnalysisRecord, analysis_id)
        assert rec is not None
        assert ".." not in rec.filename
        assert "/" not in rec.filename
        assert "\\" not in rec.filename


def test_repeat_upload_same_hash_different_analysis_id(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    payload = b"repeatable"
    expected_sha = hashlib.sha256(payload).hexdigest()

    with TestClient(app) as client:
        r1 = client.post(
            "/v1/analysis/upload",
            files={"file": ("same.wav", BytesIO(payload), "audio/wav")},
        )
        r2 = client.post(
            "/v1/analysis/upload",
            files={"file": ("same.wav", BytesIO(payload), "audio/wav")},
        )

    assert r1.status_code == 200
    assert r2.status_code == 200

    id1 = uuid.UUID(r1.json()["analysis_id"])
    id2 = uuid.UUID(r2.json()["analysis_id"])
    assert id1 != id2

    assert r1.json()["sha256"] == expected_sha
    assert r2.json()["sha256"] == expected_sha

    engine = _db_engine_for_tmp(tmp_path)
    with Session(engine) as s:
        rows = s.exec(select(AnalysisRecord).where(AnalysisRecord.sha256 == expected_sha)).all()
        assert len(rows) == 2
