import importlib
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.api.app.auth import ROLE_ADMIN, ROLE_ANALYST, ROLE_FIELD, User, require_role


def _build_real_app(tmp_path, monkeypatch):
    """Build the real API app with isolated sqlite + uploads."""
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


def test_role_constants():
    assert ROLE_FIELD == "FIELD"
    assert ROLE_ANALYST == "ANALYST"
    assert ROLE_ADMIN == "ADMIN"


def test_middleware_default_user(tmp_path, monkeypatch):
    # Call a simple endpoint without X-User; ensure request succeeds (no breakage).
    app = _build_real_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_header_parsing_not_500(tmp_path, monkeypatch):
    # Send X-User: alice:field,admin (lowercase roles); ensure request succeeds.
    app = _build_real_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        r = client.get("/health", headers={"X-User": "alice:field,admin"})
        assert r.status_code == 200


def test_role_stub_admin_403_and_200():
    # Dummy endpoint in a test app requiring ADMIN.
    app = FastAPI()

    @app.middleware("http")
    async def attach_user(request, call_next):
        # Default user is non-admin; can be overridden per-request via header.
        header = request.headers.get("X-User")
        if header == "admin":
            request.state.user = User(id="admin", roles=["ADMIN"])
        else:
            request.state.user = User(id="analyst", roles=["ANALYST"])
        return await call_next(request)

    @app.get("/admin")
    def admin(_user: User = Depends(require_role("ADMIN"))):
        return {"ok": True}

    with TestClient(app) as client:
        # Non-admin
        r = client.get("/admin")
        assert r.status_code == 403

        # Admin
        r2 = client.get("/admin", headers={"X-User": "admin"})
        assert r2.status_code == 200
        assert r2.json() == {"ok": True}
