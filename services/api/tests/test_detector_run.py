import importlib
import uuid
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


def _build_app(tmp_path, monkeypatch, *, detector_timeout: float = 0.5, sleep_seconds: float = 0.0):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")
    monkeypatch.setenv("DETECTOR_TIMEOUT_SECONDS", str(detector_timeout))

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.detector as detector
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(detector)
    importlib.reload(main)

    # Override dependency to inject a DummyDetector with sleep
    main.app.dependency_overrides[main.get_detector] = lambda: detector.DummyDetector(sleep_seconds=sleep_seconds)

    return main.app


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test_api.db').as_posix()}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_run_happy_path_updates_db(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch, detector_timeout=1.0, sleep_seconds=0.0)

    with TestClient(app) as client:
        up = client.post(
            "/v1/analysis/upload",
            files={"file": ("a.wav", BytesIO(b"abc"), "audio/wav")},
        )
        analysis_id = up.json()["analysis_id"]

        r = client.post(f"/v1/analysis/{analysis_id}/run")
        assert r.status_code == 200

    from services.api.app.models import AnalysisRecord

    with Session(_engine(tmp_path)) as s:
        rec = s.get(AnalysisRecord, uuid.UUID(analysis_id))
        assert rec is not None
        assert rec.verdict == "SAFE"
        assert rec.confidence == 0.1
        assert rec.explanations == ["dummy detector"]
        assert rec.signals == {"dummy": True}
        assert rec.model_version == "dummy-0.1"


def test_run_not_found_404(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.post(f"/v1/analysis/{uuid.uuid4()}/run")
        assert r.status_code == 404


def test_run_timeout_sets_failed_verdict(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch, detector_timeout=0.1, sleep_seconds=2.0)

    with TestClient(app) as client:
        up = client.post(
            "/v1/analysis/upload",
            files={"file": ("a.wav", BytesIO(b"abc"), "audio/wav")},
        )
        analysis_id = up.json()["analysis_id"]

        r = client.post(f"/v1/analysis/{analysis_id}/run")
        assert r.status_code == 504

    from services.api.app.models import AnalysisRecord

    with Session(_engine(tmp_path)) as s:
        rec = s.get(AnalysisRecord, uuid.UUID(analysis_id))
        assert rec is not None
        assert rec.verdict == "FAILED"
        assert rec.confidence == 0.0
        assert rec.explanations == ["Detector timeout"]
        assert rec.model_version == "timeout"
        assert rec.signals.get("error") == "timeout"
