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


def test_get_analysis_404(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get(f"/v1/analysis/{uuid.uuid4()}")
        assert r.status_code == 404


def test_get_analysis_happy_path(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        up = client.post(
            "/v1/analysis/upload",
            files={"file": ("a.wav", BytesIO(b"abc"), "audio/wav")},
        )
        analysis_id = up.json()["analysis_id"]

        r = client.get(f"/v1/analysis/{analysis_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == analysis_id
        assert data["filename"] == "a.wav"
        assert data["verdict"] == "SUSPICIOUS"


def test_list_analysis_pagination(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        # create 3 records
        for i in range(3):
            client.post(
                "/v1/analysis/upload",
                files={"file": (f"{i}.wav", BytesIO(b"abc" + bytes([i])), "audio/wav")},
            )

        r = client.get("/v1/analysis?limit=2&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["total"] == 3
        assert len(data["items"]) == 2

        r2 = client.get("/v1/analysis?limit=2&offset=2")
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["total"] == 3
        assert len(data2["items"]) == 1


def test_list_analysis_invalid_params(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        assert client.get("/v1/analysis?limit=0").status_code == 400
        assert client.get("/v1/analysis?offset=-1").status_code == 400
        assert client.get("/v1/analysis?limit=9999").status_code == 400
