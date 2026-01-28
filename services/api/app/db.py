from __future__ import annotations

from contextlib import contextmanager

from sqlmodel import Session

# Keep `db.py` as the internal session helper module.
# The core engine/table-init logic lives in `database.py` per project guide.
from services.api.app.database import DATABASE_URL, create_db_and_tables, engine


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
