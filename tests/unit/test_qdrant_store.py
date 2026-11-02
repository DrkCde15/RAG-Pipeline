from qdrant_client import QdrantClient

from app.vectorstore.qdrant import QdrantStore


def test_in_memory_qdrant_upsert_search_and_document_deletion() -> None:
    client = QdrantClient(":memory:")
    store = QdrantStore("http://unused", "unit_test_collection", 3, client=client)
    chunks = [
        {
            "document_id": "doc-1",
            "chunk_id": f"chunk-{index}",
            "source": "/docs/guide.md",
            "filename": "guide.md",
            "page": None,
            "chunk_index": index,
            "content_hash": f"hash-{index}",
            "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            "content": f"chunk {index}",
        }
        for index in range(2)
    ]

    try:
        store.upsert_document(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = store.search([1.0, 0.0, 0.0], limit=5, filters={"filename": "guide.md"})
        assert len(result) == 2
        assert result[0].payload["document_id"] == "doc-1"

        store.upsert_document(chunks[:1], [[0.0, 0.0, 1.0]])
        assert len(store.search([1.0, 0.0, 0.0], limit=5)) == 1
        assert store.list_documents()[0].chunk_count == 1
        assert store.delete_document("doc-1")
        assert not store.delete_document("doc-1")
    finally:
        client.close()
