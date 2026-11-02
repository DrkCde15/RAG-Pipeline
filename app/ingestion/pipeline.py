import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import split_text
from app.ingestion.loaders import discover_documents, load_document
from app.vectorstore.qdrant import QdrantStore, make_chunk_payload

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self, embeddings: EmbeddingService, store: QdrantStore, chunk_size: int, overlap: int
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_directory(self, directory: Path) -> tuple[int, int, list[str]]:
        documents = discover_documents(directory)
        processed = 0
        chunk_count = 0
        document_ids: list[str] = []
        for path in documents:
            try:
                document, chunks = self.ingest_file(path)
            except DocumentProcessingError:
                logger.exception("Documento ignorado após falha: %s", path)
                continue
            processed += 1
            chunk_count += chunks
            document_ids.append(document)
        return processed, chunk_count, document_ids

    def ingest_file(self, path: Path) -> tuple[str, int]:
        document = load_document(path)
        chunk_records: list[dict] = []
        chunk_texts: list[str] = []
        chunk_index = 0
        for page in document.pages:
            for chunk in split_text(page.text, self.chunk_size, self.overlap):
                chunk_records.append(
                    make_chunk_payload(
                        document_id=document.document_id,
                        source=str(path.resolve()),
                        filename=path.name,
                        chunk_index=chunk_index,
                        content=chunk.content,
                        content_hash=document.content_hash,
                        page=page.page,
                    )
                )
                chunk_texts.append(chunk.content)
                chunk_index += 1
        if not chunk_records:
            raise DocumentProcessingError(f"Nenhum chunk foi gerado para {path.name}")
        vectors = self.embeddings.embed(chunk_texts)
        self.store.upsert_document(chunk_records, vectors)
        return document.document_id, len(chunk_records)


def create_pipeline() -> IngestionPipeline:
    from app.api.dependencies import get_services

    settings = get_settings()
    embeddings, store, _, _ = get_services()
    return IngestionPipeline(embeddings, store, settings.chunk_size, settings.chunk_overlap)
