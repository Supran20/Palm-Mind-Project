import uuid

from app.services.vector_store import build_point_id


def test_point_id_is_deterministic() -> None:
    doc = uuid.uuid4()
    assert build_point_id(doc, 3) == build_point_id(doc, 3)


def test_point_id_differs_per_chunk_and_document() -> None:
    doc = uuid.uuid4()
    assert build_point_id(doc, 0) != build_point_id(doc, 1)
    assert build_point_id(doc, 0) != build_point_id(uuid.uuid4(), 0)