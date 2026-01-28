from pathlib import Path


def test_is_allowed_extension():
    from services.api.app.config import is_allowed_extension

    assert is_allowed_extension("a.wav") is True
    assert is_allowed_extension("a.WAV") is True
    assert is_allowed_extension("a.mp4") is True
    assert is_allowed_extension("a.jpeg") is True
    assert is_allowed_extension("a.exe") is False
    assert is_allowed_extension("noext") is False


def test_upload_dir_created_on_startup(tmp_path, monkeypatch):
    # Change cwd so default UPLOAD_DIR (./uploads) is inside tmp_path.
    monkeypatch.chdir(tmp_path)

    from services.api.app.main import app

    uploads_dir = Path(tmp_path) / "uploads"
    assert uploads_dir.exists() is False

    from fastapi.testclient import TestClient

    with TestClient(app):
        pass

    assert uploads_dir.exists() is True
    assert uploads_dir.is_dir() is True
