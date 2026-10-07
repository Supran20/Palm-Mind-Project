from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.dependencies import get_ingestion_service
from app.schemas.document import DocumentIngestResponse
from app.services.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingStrategy,
)
from app.services.extraction import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFileTypeError,
)
from app.services.ingestion import IngestionService, InvalidChunkingConfigError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .pdf or .txt file, chunk it, embed it, and store it",
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    chunking_strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.FIXED,
    chunk_size: Annotated[int, Form(ge=100, le=4000)] = DEFAULT_CHUNK_SIZE,
    chunk_overlap: Annotated[int, Form(ge=0, le=1000)] = DEFAULT_CHUNK_OVERLAP,
) -> DocumentIngestResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A filename is required.")

    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {settings.max_upload_mb} MB limit.",
        )

    try:
        document = await service.ingest(
            filename=file.filename,
            content=content,
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except (EmptyDocumentError, ExtractionError, InvalidChunkingConfigError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    return DocumentIngestResponse.model_validate(document)