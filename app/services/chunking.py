from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Protocol

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    PARAGRAPH = "paragraph"


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """Cuts text into windows of `chunk_size` characters, overlapping by `chunk_overlap`."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    def chunk(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return chunks


@dataclass(frozen=True, slots=True)
class ParagraphChunker:
    """Packs whole paragraphs into chunks up to `chunk_size` characters.

    A paragraph longer than `chunk_size` is split by sentences instead, and a
    single sentence longer than `chunk_size` is hard-split as a last resort.
    """

    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        buffer = ""
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split_long_paragraph(paragraph))
                continue
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = paragraph
        if buffer:
            chunks.append(buffer)
        return chunks

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        chunks: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(FixedSizeChunker(self.chunk_size, 0).chunk(sentence))
                continue
            candidate = f"{buffer} {sentence}" if buffer else sentence
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = sentence
        if buffer:
            chunks.append(buffer)
        return chunks


def get_chunker(
    strategy: ChunkingStrategy,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Chunker:
    """Return the chunker for the caller's chosen strategy."""
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker(chunk_size, chunk_overlap)
    return ParagraphChunker(chunk_size)