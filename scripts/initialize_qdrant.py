from app.core.config import get_settings
from app.embeddings.service import EmbeddingService
from app.vectorstore.qdrant import QdrantStore


def main() -> None:
    settings = get_settings()
    embeddings = EmbeddingService(settings.embedding_model)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, embeddings.dimension)
    print(f"Collection pronta: {store.collection} ({embeddings.dimension} dimensões)")


if __name__ == "__main__":
    main()
