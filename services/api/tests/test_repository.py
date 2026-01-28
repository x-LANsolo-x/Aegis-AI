from sqlmodel import Session, SQLModel, create_engine

from services.api.app.repository import (
    create_analysis_record,
    get_analysis_record_by_id,
    get_analysis_record_by_sha256,
)


def test_repository_create_and_get_roundtrip():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        rec = create_analysis_record(
            s,
            media_type="audio",
            filename="x.wav",
            sha256="hash1",
            duration_s=1.23,
        )

        got_by_id = get_analysis_record_by_id(s, rec.id)
        assert got_by_id is not None
        assert got_by_id.sha256 == "hash1"

        got_by_hash = get_analysis_record_by_sha256(s, "hash1")
        assert got_by_hash is not None
        assert got_by_hash.id == rec.id
