import os
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from app.vectorstore.qdrant import QdrantStore


@pytest.mark.integration
def test_qdrant_upsert_search_and_delete() -> None:
    url = os.getenv("QDRANT_TEST_URL")
    if not url:
        pytest.skip("Defina QDRANT_TEST_URL para executar o teste real do Qdrant.")
    client = QdrantClient(url=url)
    collection = f"rag_test_{uuid4().hex}"
    store = QdrantStore(url, collection, vector_size=3)
    chunk = {
        "document_id": "test-document",
        "chunk_id": "test-chunk",
        "source": "fixture",
        "filename": "fixture.txt",
        "page": None,
        "chunk_index": 0,
        "content_hash": "hash",
        "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
        "content": "fixture",
    }

    try:
        store.upsert_document([chunk], [[1.0, 0.0, 0.0]])
        result = store.search([1.0, 0.0, 0.0], limit=1)
        assert result[0].payload["document_id"] == "test-document"
        assert store.delete_document("test-document")
    finally:
        client.delete_collection(collection)
