import uuid

from sqlmodel import Session, SQLModel, create_engine, select

from services.api.app.models import AnalysisRecord


def test_analysis_record_roundtrip_sqlite_in_memory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)

    rec = AnalysisRecord(
        media_type="audio",
        filename="sample.wav",
        sha256="abc123",
        explanations=["x"],
        signals={"k": 1},
    )

    with Session(engine) as s:
        s.add(rec)
        s.commit()
        s.refresh(rec)

        assert isinstance(rec.id, uuid.UUID)

        got = s.exec(select(AnalysisRecord).where(AnalysisRecord.sha256 == "abc123")).one()
        assert got.filename == "sample.wav"
        assert got.media_type == "audio"
        assert got.verdict == "PENDING"
        assert got.confidence == 0.0
        assert got.explanations == ["x"]
        assert got.signals == {"k": 1}


def test_mutable_defaults_are_isolated():
    # Ensure default_factory prevents shared state between instances
    a = AnalysisRecord(media_type="audio", filename="a.wav", sha256="h1")
    b = AnalysisRecord(media_type="audio", filename="b.wav", sha256="h2")

    a.explanations.append("reason")
    a.signals["s"] = 1

    assert b.explanations == []
    assert b.signals == {}
