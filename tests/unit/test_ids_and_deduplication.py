from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.qdrant import point_id


class FakeEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[list[dict], list[list[float]]]] = []

    def upsert_document(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        self.upserts.append((chunks, vectors))


def test_point_ids_are_valid_and_deterministic() -> None:
    first = point_id("doc-123", 4)
    assert first == point_id("doc-123", 4)
    assert first != point_id("doc-123", 5)
    assert len(first) == 36


def test_reingesting_same_file_uses_same_chunk_ids(text_file: Path) -> None:
    store = FakeStore()
    pipeline = IngestionPipeline(FakeEmbeddings(), store, chunk_size=100, overlap=10)  # type: ignore[arg-type]

    first = pipeline.ingest_file(text_file)
    second = pipeline.ingest_file(text_file)

    assert first == second
    assert store.upserts[0][0][0]["chunk_id"] == store.upserts[1][0][0]["chunk_id"]
    assert store.upserts[0][0][0]["content_hash"] == store.upserts[1][0][0]["content_hash"]
