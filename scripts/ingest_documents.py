import argparse
from pathlib import Path

from app.core.config import get_settings
from app.embeddings.service import EmbeddingService
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.qdrant import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere documentos locais no Qdrant.")
    parser.add_argument("--directory", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    embeddings = EmbeddingService(settings.embedding_model)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, embeddings.dimension)
    pipeline = IngestionPipeline(embeddings, store, settings.chunk_size, settings.chunk_overlap)
    processed, chunks, document_ids = pipeline.ingest_directory(args.directory or settings.data_dir)
    print(f"Documentos processados: {processed}; chunks indexados: {chunks}")
    for document_id in document_ids:
        print(document_id)


if __name__ == "__main__":
    main()
