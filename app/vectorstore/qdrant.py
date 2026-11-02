import logging
from datetime import UTC, datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint

from app.core.config import get_settings
from app.schemas.documents import DocumentMetadata

logger = logging.getLogger(__name__)


def point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}"))


class QdrantStore:
    def __init__(
        self,
        url: str,
        collection: str,
        vector_size: int,
        client: QdrantClient | None = None,
    ) -> None:
        self.client = client or QdrantClient(url=url, timeout=10)
        self.collection = collection
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
        else:
            info = self.client.get_collection(self.collection)
            config = info.config.params.vectors
            if isinstance(config, models.VectorParams) and config.size != self.vector_size:
                raise ValueError(
                    f"Dimensão da collection ({config.size}) difere do modelo "
                    f"({self.vector_size}); configure uma collection nova."
                )

    def upsert_document(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("A quantidade de chunks e embeddings deve ser igual")
        if not chunks:
            return
        document_id = chunks[0]["document_id"]
        if any(chunk["document_id"] != document_id for chunk in chunks):
            raise ValueError("Todos os chunks devem pertencer ao mesmo documento")
        points = [
            models.PointStruct(
                id=point_id(document_id, chunk["chunk_index"]), vector=vector, payload=chunk
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ],
                    must_not=[
                        models.HasIdCondition(
                            has_id=[point_id(document_id, chunk["chunk_index"]) for chunk in chunks]
                        )
                    ],
                )
            ),
            wait=True,
        )

    def search(
        self, vector: list[float], limit: int, filters: dict | None = None
    ) -> list[ScoredPoint]:
        must = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (filters or {}).items()
        ]
        return self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=models.Filter(must=must) if must else None,
            with_payload=True,
        ).points

    def list_documents(self) -> list[DocumentMetadata]:
        records, _ = self.client.scroll(
            collection_name=self.collection, limit=10000, with_payload=True, with_vectors=False
        )
        grouped: dict[str, dict] = {}
        for record in records:
            payload = record.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            current = grouped.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "source": payload.get("source", ""),
                    "filename": payload.get("filename", ""),
                    "chunk_count": 0,
                    "ingestion_timestamp": payload.get("ingestion_timestamp", ""),
                },
            )
            current["chunk_count"] += 1
        return [DocumentMetadata(**item) for item in grouped.values()]

    def delete_document(self, document_id: str) -> bool:
        count = self.client.count(
            collection_name=self.collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True,
        ).count
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
        return count > 0

    def health(self) -> bool:
        return self.client.get_collections() is not None


def get_qdrant_store(vector_size: int) -> QdrantStore:
    settings = get_settings()
    return QdrantStore(settings.qdrant_url, settings.qdrant_collection, vector_size)


def make_chunk_payload(
    document_id: str,
    source: str,
    filename: str,
    chunk_index: int,
    content: str,
    content_hash: str,
    page: int | None,
) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": point_id(document_id, chunk_index),
        "source": source,
        "filename": filename,
        "page": page,
        "chunk_index": chunk_index,
        "content_hash": sha256(content.encode("utf-8")).hexdigest(),
        "document_content_hash": content_hash,
        "ingestion_timestamp": datetime.now(UTC).isoformat(),
        "content": content,
    }
