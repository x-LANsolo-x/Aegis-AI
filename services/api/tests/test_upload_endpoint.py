import importlib
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient


def _build_test_client(tmp_path, monkeypatch, *, max_bytes: int = 20 * 1024 * 1024):
    """Create an app client isolated to a tmp working directory.

    We chdir + set env vars BEFORE importing modules so config constants bind
    to the test environment.
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", str(max_bytes))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")

    # Reload modules so they pick up the env vars for constants.
    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    return main.app


def test_upload_success_creates_record_and_returns_sha(tmp_path, monkeypatch):
    app = _build_test_client(tmp_path, monkeypatch, max_bytes=5_000_000)

    payload = b"hello world"
    files = {"file": ("test.wav", BytesIO(payload), "audio/wav")}

    with TestClient(app) as client:
        r = client.post("/v1/analysis/upload", files=files)
    assert r.status_code == 200

    data = r.json()
    assert "analysis_id" in data
    assert "sha256" in data
    assert isinstance(data["analysis_id"], str)
    assert len(data["sha256"]) == 64

    # Ensure file was saved under uploads
    uploads_dir = Path(tmp_path) / "uploads"
    assert uploads_dir.exists()
    assert (uploads_dir / "test.wav").exists()


def test_upload_missing_file_returns_400(tmp_path, monkeypatch):
    app = _build_test_client(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.post("/v1/analysis/upload", files={})
    assert r.status_code == 400


def test_upload_unsupported_extension_returns_415(tmp_path, monkeypatch):
    app = _build_test_client(tmp_path, monkeypatch)

    files = {"file": ("evil.exe", BytesIO(b"x"), "application/octet-stream")}
    with TestClient(app) as client:
        r = client.post("/v1/analysis/upload", files=files)
    assert r.status_code == 415


def test_upload_oversize_returns_413_and_removes_file(tmp_path, monkeypatch):
    # Set a tiny max size
    app = _build_test_client(tmp_path, monkeypatch, max_bytes=10)

    payload = b"0123456789ABCDEF"  # 16 bytes
    files = {"file": ("big.wav", BytesIO(payload), "audio/wav")}

    with TestClient(app) as client:
        r = client.post("/v1/analysis/upload", files=files)
    assert r.status_code == 413

    # Ensure oversized file is removed
    uploads_dir = Path(tmp_path) / "uploads"
    assert not (uploads_dir / "big.wav").exists()
