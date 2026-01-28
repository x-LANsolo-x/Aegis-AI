import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_startup_fails_with_invalid_model_path(tmp_path, monkeypatch):
    """Startup should fail with clear error if model path is invalid and REQUIRE_MODEL=true."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")
    monkeypatch.setenv("ONNX_MODEL_PATH", "/nonexistent/model.onnx")
    monkeypatch.setenv("REQUIRE_MODEL", "false")  # optional: test both true and false

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    # Attempt to enter lifespan (startup checks happen here)
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        with TestClient(main.app):
            pass


def test_startup_fails_when_require_model_but_no_path(tmp_path, monkeypatch):
    """Startup should fail if REQUIRE_MODEL=true but ONNX_MODEL_PATH is not set."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")
    monkeypatch.setenv("REQUIRE_MODEL", "true")
    # Do not set ONNX_MODEL_PATH

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    with pytest.raises(RuntimeError, match="REQUIRE_MODEL=true but ONNX_MODEL_PATH is not set"):
        with TestClient(main.app):
            pass


def test_startup_succeeds_with_valid_model(tmp_path, monkeypatch):
    """Startup should succeed if model file exists."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")

    # Create a fake model file
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"fake onnx")

    monkeypatch.setenv("ONNX_MODEL_PATH", str(model_file))

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    # Should not raise
    with TestClient(main.app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_hot_reload_model_version_changes(tmp_path, monkeypatch):
    """Verify that swapping the model file and restarting changes model_version."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_DIR", str(Path(tmp_path) / "uploads"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(Path(tmp_path) / 'test_api.db').as_posix()}")

    # Start with V1.0.0
    model_v1 = tmp_path / "V1.0.0.onnx"
    model_v1.write_bytes(b"fake onnx v1")

    monkeypatch.setenv("ONNX_MODEL_PATH", str(model_v1))

    import services.api.app.config as config
    import services.api.app.database as database
    import services.api.app.db as db
    import services.api.app.main as main

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    with TestClient(main.app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json()["audio"]["current"] == "V1.0.0"

    # Now swap to V2.0.0 and restart
    model_v2 = tmp_path / "V2.0.0.onnx"
    model_v2.write_bytes(b"fake onnx v2")

    monkeypatch.setenv("ONNX_MODEL_PATH", str(model_v2))

    importlib.reload(config)
    importlib.reload(database)
    importlib.reload(db)
    importlib.reload(main)

    with TestClient(main.app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json()["audio"]["current"] == "V2.0.0"
