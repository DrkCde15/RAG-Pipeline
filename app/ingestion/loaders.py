import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz

from app.core.exceptions import DocumentProcessingError
from app.ingestion.cleaner import clean_text

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class LoadedPage:
    text: str
    page: int | None


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    document_id: str
    content_hash: str
    pages: list[LoadedPage]


def discover_documents(directory: Path) -> list[Path]:
    if not directory.exists():
        raise DocumentProcessingError(f"Diretório inexistente: {directory}")
    if not directory.is_dir():
        raise DocumentProcessingError(f"O caminho não é um diretório: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: Path) -> LoadedDocument:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(f"Formato não suportado: {path.name}")
    try:
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                pages = [
                    LoadedPage(clean_text(page.get_text()), index + 1)
                    for index, page in enumerate(pdf)
                ]
        else:
            pages = [LoadedPage(clean_text(path.read_text(encoding="utf-8")), None)]
    except (OSError, UnicodeError, fitz.FileDataError) as error:
        logger.exception("Falha ao carregar documento %s", path)
        raise DocumentProcessingError(f"Não foi possível ler {path.name}: {error}") from error

    combined = "\n".join(page.text for page in pages)
    if not combined:
        raise DocumentProcessingError(f"Documento vazio ou sem texto extraível: {path.name}")
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return LoadedDocument(path, document_id, digest, pages)
