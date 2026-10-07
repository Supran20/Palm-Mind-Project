import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.services.chunking import ChunkingStrategy


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int | None
    num_chunks: int
    created_at: datetime