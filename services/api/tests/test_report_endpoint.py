import importlib
import uuid
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient


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


def test_report_404(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get(f"/v1/analysis/{uuid.uuid4()}/report")
        assert r.status_code == 404


def test_report_happy_path_contains_version_and_evidence(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    payload = b"abc" * 10
    with TestClient(app) as client:
        up = client.post(
            "/v1/analysis/upload",
            files={"file": ("a.wav", BytesIO(payload), "audio/wav")},
        )
        analysis_id = up.json()["analysis_id"]

        r = client.get(f"/v1/analysis/{analysis_id}/report")
        assert r.status_code == 200
        data = r.json()

        assert data["report_version"] == "1.0"
        assert data["analysis_id"] == analysis_id
        assert data["evidence"]["filename"] == "a.wav"
        assert data["evidence"]["sha256"] == up.json()["sha256"]
        assert data["evidence"]["media_type"] == "audio"
        assert data["evidence"]["size_bytes"] == len(payload)
        assert "chain_of_custody" in data
        assert "created_at" in data["chain_of_custody"]
