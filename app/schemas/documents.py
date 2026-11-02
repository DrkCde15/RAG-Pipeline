from pathlib import Path

from pydantic import BaseModel


class IngestRequest(BaseModel):
    directory: Path | None = None


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_indexed: int
    document_ids: list[str]


class DocumentMetadata(BaseModel):
    document_id: str
    source: str
    filename: str
    chunk_count: int
    ingestion_timestamp: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
