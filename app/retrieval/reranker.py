from typing import Protocol

from qdrant_client.http.models import ScoredPoint


class Reranker(Protocol):
    def rerank(self, query: str, results: list[ScoredPoint]) -> list[ScoredPoint]: ...


class IdentityReranker:
    def rerank(self, query: str, results: list[ScoredPoint]) -> list[ScoredPoint]:
        return results
