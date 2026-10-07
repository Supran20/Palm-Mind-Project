from httpx import AsyncClient

from tests.fakes import InMemoryVectorStore

URL = "/documents/ingest"


async def test_ingest_txt_returns_201(
    client: AsyncClient, store: InMemoryVectorStore
) -> None:
    response = await client.post(
        URL,
        files={"file": ("notes.txt", b"word " * 100, "text/plain")},
        data={"chunking_strategy": "fixed", "chunk_size": "100", "chunk_overlap": "10"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunking_strategy"] == "fixed"
    assert body["num_chunks"] >= 2


async def test_paragraph_strategy_is_selectable(client: AsyncClient) -> None:
    content = b"Para one.\n\n" + b"x" * 150 + b"\n\nPara three."
    response = await client.post(
        URL,
        files={"file": ("p.txt", content, "text/plain")},
        data={"chunking_strategy": "paragraph", "chunk_size": "100"},
    )
    assert response.status_code == 201
    assert response.json()["chunking_strategy"] == "paragraph"
    assert response.json()["chunk_overlap"] is None


async def test_unsupported_type_returns_415(client: AsyncClient) -> None:
    response = await client.post(URL, files={"file": ("photo.png", b"x", "image/png")})
    assert response.status_code == 415


async def test_empty_document_returns_422(client: AsyncClient) -> None:
    response = await client.post(URL, files={"file": ("empty.txt", b"   ", "text/plain")})
    assert response.status_code == 422


async def test_overlap_not_smaller_than_size_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        URL,
        files={"file": ("a.txt", b"hello world", "text/plain")},
        data={"chunking_strategy": "fixed", "chunk_size": "100", "chunk_overlap": "100"},
    )
    assert response.status_code == 422


async def test_unknown_strategy_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        URL,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"chunking_strategy": "nope"},
    )
    assert response.status_code == 422


async def test_oversized_file_returns_413(client: AsyncClient) -> None:
    too_big = b"a" * (1024 * 1024 + 1)
    response = await client.post(URL, files={"file": ("big.txt", too_big, "text/plain")})
    assert response.status_code == 413