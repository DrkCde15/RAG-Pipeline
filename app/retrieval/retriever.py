from qdrant_client.http.models import ScoredPoint

from app.embeddings.service import EmbeddingService
from app.retrieval.reranker import IdentityReranker, Reranker
from app.vectorstore.qdrant import QdrantStore


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingService,
        store: QdrantStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.reranker = reranker or IdentityReranker()

    def retrieve(self, question: str, top_k: int, min_score: float) -> list[ScoredPoint]:
        vector = self.embeddings.embed([question])[0]
        results = self.store.search(vector, top_k)
        ranked = self.reranker.rerank(question, results)
        return [result for result in ranked if result.score >= min_score]


def configured_retriever(embeddings: EmbeddingService, store: QdrantStore) -> Retriever:
    return Retriever(embeddings, store)
