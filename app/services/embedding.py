import asyncio
from collections.abc import Sequence
from typing import Protocol

from sentence_transformers import SentenceTransformer

Vector = list[float]


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...


class SentenceTransformerEmbedder:
    """Local embedding model. Vectors are normalized, so cosine similarity works well."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding size for '{model_name}'")
        self._dimension: int = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> Vector:
        vectors = await asyncio.to_thread(self._encode, [text])
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[Vector]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: list[Vector] = embeddings.tolist()
        return vectors