import importlib
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.api.app.auth import User, get_current_user


def test_get_current_user_returns_state_user(tmp_path, monkeypatch):
    # Minimal app to test dependency resolution
    app = FastAPI()

    @app.middleware("http")
    async def attach_user(request, call_next):
        request.state.user = User(id="u1", roles=["ANALYST"])
        return await call_next(request)

    @app.get("/me")
    def me(user: User = Depends(get_current_user)):
        return user.model_dump()

    with TestClient(app) as client:
        r = client.get("/me")
        assert r.status_code == 200
        assert r.json() == {"id": "u1", "roles": ["ANALYST"]}
