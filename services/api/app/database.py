from sqlmodel import SQLModel, create_engine

# Use a local SQLite file for the database
DATABASE_URL = "sqlite:///./aegis_api.db"

# The connect_args are needed only for SQLite to allow multi-threaded access
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def create_db_and_tables():
    """Initializes the database and creates tables if they don't exist."""
    SQLModel.metadata.create_all(engine)
