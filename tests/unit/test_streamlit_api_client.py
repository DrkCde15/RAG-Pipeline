import json

import httpx
import pytest

from ui.api_client import APIError, RAGAPIClient


def client_for(handler) -> RAGAPIClient:
    return RAGAPIClient(
        "http://api.test",
        transport=httpx.MockTransport(handler),
    )


def test_ingest_sends_multipart_file() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/documents/ingest"
        assert request.headers["content-type"].startswith("multipart/form-data")
        assert b"guia.pdf" in request.read()
        return httpx.Response(
            200,
            json={"documents_processed": 1, "chunks_indexed": 4, "document_ids": ["doc-1"]},
        )

    client = client_for(handler)
    try:
        result = client.ingest_file("guia.pdf", b"pdf content")
    finally:
        client.close()

    assert result["chunks_indexed"] == 4


def test_chat_sends_question_and_returns_sources() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        assert json.loads(request.read()) == {"question": "Pergunta?"}
        return httpx.Response(
            200,
            json={
                "answer": "Resposta [1]",
                "sources": [{"filename": "guia.pdf", "page": 2, "score": 0.8}],
                "retrieval_scores": [0.8],
                "processing_time": 1.2,
            },
        )

    client = client_for(handler)
    try:
        result = client.chat("Pergunta?")
    finally:
        client.close()

    assert result["sources"][0]["page"] == 2
    assert result["retrieval_scores"] == [0.8]


def test_list_and_delete_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"documents": [{"document_id": "doc-1"}]})
        assert request.method == "DELETE"
        assert request.url.path == "/documents/doc-1"
        return httpx.Response(200, json={"document_id": "doc-1", "deleted": True})

    client = client_for(handler)
    try:
        assert client.list_documents() == [{"document_id": "doc-1"}]
        assert client.delete_document("doc-1")["deleted"] is True
    finally:
        client.close()


def test_api_error_includes_server_detail() -> None:
    client = client_for(lambda request: httpx.Response(503, json={"detail": "Qdrant indisponível"}))
    try:
        with pytest.raises(APIError, match="Qdrant indisponível"):
            client.health()
    finally:
        client.close()
