from fastapi.testclient import TestClient

from app.api.routes import chat as chat_route
from app.api.routes import documents as documents_route
from app.api.routes import health as health_route
from app.core.config import get_settings
from app.main import app


def fake_qdrant_client(**kwargs):
    class Client:
        def get_collections(self):
            return object()

    return Client()


def test_health_endpoint_with_mocked_services(monkeypatch) -> None:
    monkeypatch.setattr(health_route, "QdrantClient", fake_qdrant_client)
    monkeypatch.setattr(health_route.OllamaGenerator, "health", lambda self: True)
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"api": "ok", "qdrant": "ok", "ollama": "ok"}


def test_openapi_exposes_expected_routes() -> None:
    paths = app.openapi()["paths"]
    assert {"/health", "/documents/ingest", "/documents", "/chat"} <= paths.keys()


def test_chat_returns_sources_and_scores(monkeypatch) -> None:
    payload = {
        "document_id": "doc-1",
        "source": "/docs/guide.txt",
        "filename": "guide.txt",
        "page": None,
        "chunk_id": "chunk-1",
        "content": "Informação recuperada.",
    }

    class Retriever:
        def retrieve(self, question, top_k, min_score):
            return [type("Result", (), {"payload": payload, "score": 0.8})()]

    class Generator:
        def generate(self, prompt):
            assert "Informação recuperada." in prompt
            return "Resposta fundamentada [1]."

    monkeypatch.setattr(
        chat_route, "get_services", lambda: (object(), object(), Generator(), Retriever())
    )
    response = TestClient(app).post("/chat", json={"question": "Pergunta?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Resposta fundamentada [1]."
    assert response.json()["retrieval_scores"] == [0.8]
    assert response.json()["sources"][0]["filename"] == "guide.txt"


def test_file_ingestion_endpoint_with_mocks(tmp_path, monkeypatch) -> None:
    class Embeddings:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Store:
        def upsert_document(self, chunks, vectors):
            self.chunks = chunks

    store = Store()
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(
        documents_route, "get_services", lambda: (Embeddings(), store, object(), object())
    )

    response = TestClient(app).post(
        "/documents/ingest",
        files={"file": ("sample.txt", b"Documento para API.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["documents_processed"] == 1
    assert response.json()["chunks_indexed"] == 1
