import pytest

from app.services.chunking import (
    ChunkingStrategy,
    FixedSizeChunker,
    ParagraphChunker,
    get_chunker,
)


def test_fixed_chunks_respect_size_and_overlap() -> None:
    text = "a" * 250
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=20).chunk(text)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) == 3


def test_fixed_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)


def test_paragraph_packs_small_paragraphs_together() -> None:
    text = "First para.\n\nSecond para.\n\nThird para."
    chunks = ParagraphChunker(chunk_size=100).chunk(text)
    assert chunks == ["First para.\n\nSecond para.\n\nThird para."]


def test_paragraph_splits_when_over_limit() -> None:
    text = "A" * 60 + "\n\n" + "B" * 60
    chunks = ParagraphChunker(chunk_size=100).chunk(text)
    assert chunks == ["A" * 60, "B" * 60]


def test_paragraph_falls_back_to_sentences() -> None:
    text = "One two three. Four five six. Seven eight nine."
    chunks = ParagraphChunker(chunk_size=30).chunk(text)
    assert all(len(c) <= 30 for c in chunks)
    assert len(chunks) >= 2


def test_factory_returns_selected_strategy() -> None:
    assert isinstance(get_chunker(ChunkingStrategy.FIXED), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkingStrategy.PARAGRAPH), ParagraphChunker)