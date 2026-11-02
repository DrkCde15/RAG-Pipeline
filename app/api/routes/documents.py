import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_services
from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.documents import (
    DeleteResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    file: Annotated[UploadFile | None, File()] = None,
    directory: Annotated[str | None, Form()] = None,
) -> IngestResponse:
    if bool(file) == bool(directory):
        raise HTTPException(400, "Informe exatamente um arquivo ou um diretório.")
    embeddings, store, _, _ = get_services()
    settings = get_settings()
    pipeline = IngestionPipeline(embeddings, store, settings.chunk_size, settings.chunk_overlap)
    if directory:
        root = settings.data_dir.resolve()
        target = IngestRequest(directory=directory).directory
        if target is None:
            raise HTTPException(400, "Diretório inválido.")
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if target != root and root not in target.parents:
            raise HTTPException(400, "O diretório deve estar dentro de DATA_DIR.")
        try:
            processed, chunks, ids = pipeline.ingest_directory(target)
        except DocumentProcessingError as error:
            raise HTTPException(400, str(error)) from error
        return IngestResponse(
            documents_processed=processed, chunks_indexed=chunks, document_ids=ids
        )

    assert file is not None
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in {".pdf", ".txt", ".md", ".markdown"}:
        raise HTTPException(400, "Formato inválido. Envie PDF, TXT ou Markdown.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.data_dir / filename
    try:
        content = file.file.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(413, "O arquivo excede o limite de 50 MB.")
        destination.write_bytes(content)
        document_id, chunks = pipeline.ingest_file(destination)
        return IngestResponse(
            documents_processed=1, chunks_indexed=chunks, document_ids=[document_id]
        )
    except DocumentProcessingError as error:
        logger.exception("Falha na ingestão de arquivo")
        raise HTTPException(400, str(error)) from error
    finally:
        file.file.close()


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    _, store, _, _ = get_services()
    return DocumentListResponse(documents=store.list_documents())


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str) -> DeleteResponse:
    _, store, _, _ = get_services()
    return DeleteResponse(document_id=document_id, deleted=store.delete_document(document_id))
