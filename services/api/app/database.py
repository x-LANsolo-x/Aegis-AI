import os

from sqlmodel import SQLModel, create_engine

# Use a local SQLite file for the database (overridable via DATABASE_URL)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./aegis_api.db")

# The connect_args are needed only for SQLite to allow multi-threaded access
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)


def create_db_and_tables():
    """Initializes the database and creates tables if they don't exist."""
    SQLModel.metadata.create_all(engine)
