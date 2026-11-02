from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.service import EmbeddingService
from app.generation.ollama import OllamaGenerator
from app.retrieval.retriever import Retriever
from app.vectorstore.qdrant import QdrantStore


@lru_cache
def get_services() -> tuple[EmbeddingService, QdrantStore, OllamaGenerator, Retriever]:
    settings = get_settings()
    embeddings = EmbeddingService(settings.embedding_model)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, embeddings.dimension)
    generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model)
    retriever = Retriever(embeddings, store)
    return embeddings, store, generator, retriever
