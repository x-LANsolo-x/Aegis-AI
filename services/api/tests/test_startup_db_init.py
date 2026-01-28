from sqlalchemy import inspect


def test_startup_creates_tables(tmp_path, monkeypatch):
    # We use cwd because database.py uses a relative sqlite path (./aegis_api.db)
    monkeypatch.chdir(tmp_path)

    # Import after changing cwd so the DB file lands in tmp_path.
    from services.api.app.main import app
    from services.api.app import database

    # Exercise the app lifespan; this should call create_db_and_tables().
    with database.engine.begin():
        pass

    # FastAPI's TestClient enters lifespan automatically.
    from fastapi.testclient import TestClient

    with TestClient(app):
        pass

    insp = inspect(database.engine)
    assert "analysis_records" in insp.get_table_names()
