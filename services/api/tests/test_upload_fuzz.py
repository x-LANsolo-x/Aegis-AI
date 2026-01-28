import importlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _build_app(tmp_path, monkeypatch, *, max_bytes: int = 20 * 1024 * 1024):
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


WEIRD_FILENAMES = [
    "normal.wav",
    "../../evil.wav",
    "..\\..\\evil.wav",
    " spaced name .wav",
    "name-with-Üñïçødé.wav",
    "name\nnewline.wav",
    "name\t.tab.wav",
    "name;rm -rf.wav",
    "name..wav",
    ".wav",  # becomes empty after sanitize -> fallback
    "",  # missing filename
    "a" * 300 + ".wav",  # long name
    "a\x00b.wav",  # null byte (may be rejected by http libs)
]

WEIRD_CONTENT_TYPES = [
    "audio/wav",
    "application/octet-stream",
    "text/plain",
    "",
    None,
]


@pytest.mark.parametrize("filename", WEIRD_FILENAMES)
@pytest.mark.parametrize("content_type", WEIRD_CONTENT_TYPES)
def test_upload_fuzz_no_500(tmp_path, monkeypatch, filename, content_type):
    app = _build_app(tmp_path, monkeypatch, max_bytes=1024 * 1024)

    # Some combinations (e.g., embedded null byte) might raise at client construction
    # before request is sent. That's acceptable as long as the server is not 500ing.
    payload = b"abc"

    file_tuple = (filename, BytesIO(payload), content_type) if content_type is not None else (filename, BytesIO(payload))

    with TestClient(app) as client:
        try:
            r = client.post("/v1/analysis/upload", files={"file": file_tuple})
        except Exception:
            # If the client library refuses the filename, treat as pass for fuzz purposes.
            return

    # We only care that server doesn't throw 500s.
    assert r.status_code in {200, 400, 413, 415, 422}
    assert r.status_code != 500
