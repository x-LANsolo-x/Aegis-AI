from sqlmodel import Session, SQLModel, create_engine

from services.api.app.repository import count_analyses, create_analysis_record, list_analyses


def test_count_and_list():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as s:
        assert count_analyses(s) == 0

        # Create 3 records
        for i in range(3):
            create_analysis_record(
                s,
                media_type="audio",
                filename=f"{i}.wav",
                sha256=f"h{i}",
            )

        assert count_analyses(s) == 3

        first_two = list_analyses(s, limit=2, offset=0)
        assert len(first_two) == 2

        last_one = list_analyses(s, limit=2, offset=2)
        assert len(last_one) == 1
