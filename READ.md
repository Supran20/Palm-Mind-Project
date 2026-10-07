# Palm Mind RAG Backend

FastAPI backend for document ingestion and conversational RAG.
This repository currently contains **API 1: Document Ingestion**.

## Stack

FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · PostgreSQL · Qdrant ·
sentence-transformers (`all-MiniLM-L6-v2`, local) · Redis (reserved for API 2)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
uvicorn app.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

> Postgres is exposed on host port **55432** (not 5432) to avoid clashing with a
> locally installed PostgreSQL. Change it in `docker-compose.yml` and `.env` if needed.

## API 1: `POST /documents/ingest`

Multipart form fields:

| Field               | Type                   | Default  | Notes                                 |
| ------------------- | ---------------------- | -------- | ------------------------------------- |
| `file`              | file                   | required | `.pdf` or `.txt`, max `MAX_UPLOAD_MB` |
| `chunking_strategy` | `fixed` \| `paragraph` | `fixed`  | see below                             |
| `chunk_size`        | int (100–4000)         | 800      | characters                            |
| `chunk_overlap`     | int (0–1000)           | 100      | `fixed` only, must be < `chunk_size`  |

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -F "file=@sample.pdf" -F "chunking_strategy=paragraph" -F "chunk_size=500"
```

Responses: `201` created · `413` too large · `415` unsupported type · `422` empty/invalid input.

### Chunking strategies

- **fixed**: sliding window of `chunk_size` characters with `chunk_overlap`. Predictable sizes.
- **paragraph**: packs whole paragraphs up to `chunk_size`; falls back to sentences, then a hard split, for oversized paragraphs. Keeps ideas together.

## Design notes

- **Layers:** routes (HTTP only) → `IngestionService` (pipeline) → extraction / chunking / embedding / vector store.
- **Interfaces:** `Embedder`, `Chunker`, and `VectorStore` are `Protocol`s, so implementations can be swapped and tests use fakes.
- **Consistency:** Qdrant and Postgres can't share a transaction. Vectors are written first; if the Postgres commit fails, the vectors are deleted. Point IDs are deterministic (`uuid5`), so re-ingesting overwrites rather than duplicates.
- **Embeddings** run in a worker thread so they don't block the event loop.
- **Typing:** `mypy --strict`, plus `ruff`.

## Tests

```bash
ruff check . && mypy app tests && pytest
```

Tests use a fake embedder, an in-memory vector store, and in-memory SQLite, so they need no Docker.

## Known limitations / next steps

- Tables are created at startup; production would use Alembic migrations.
- Scanned (image-only) PDFs aren't supported (no OCR).
- Ingestion is synchronous; very large files would call for a background queue.
