import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def _build_app(tmp_path, monkeypatch, *, model_path=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")
    if model_path:
        monkeypatch.setenv("ONNX_MODEL_PATH", str(model_path))

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    return main.app


def test_models_endpoint_without_model(tmp_path, monkeypatch):
    app = _build_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()

        assert data["audio"]["current"] == "DummyDetector"
        assert data["audio"]["path"] is None


def test_models_endpoint_with_model(tmp_path, monkeypatch):
    # Create a fake model file
    model_file = tmp_path / "V1.0.0.onnx"
    model_file.write_bytes(b"fake onnx")

    # Note: startup will fail if file doesn't exist, so we create it.
    # But OnnxAudioDetector __post_init__ will fail loading; we skip that by not actually calling run.
    # For this test, we just verify the endpoint returns correct metadata.

    # Since OnnxAudioDetector loads at get_detector() call, not startup, we can test endpoint safely.
    app = _build_app(tmp_path, monkeypatch, model_path=str(model_file))

    with TestClient(app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()

        assert data["audio"]["current"] == "V1.0.0"
        assert str(model_file) in data["audio"]["path"]
