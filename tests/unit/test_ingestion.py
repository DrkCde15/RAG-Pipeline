from pathlib import Path

import fitz
import pytest

from app.core.exceptions import DocumentProcessingError
from app.ingestion.chunker import split_text
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import discover_documents, load_document


def test_clean_text_normalizes_whitespace_and_invalid_characters() -> None:
    assert clean_text("  Olá\t mundo\x00 \n\n  teste  ") == "Olá mundo\n\nteste"


def test_chunking_respects_size_and_overlap() -> None:
    chunks = split_text("um dois três quatro cinco seis sete oito", chunk_size=16, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 16 for chunk in chunks)
    assert chunks[0].content.split()[-1] in chunks[1].content
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=4, overlap=4)


def test_load_text_file_and_generate_stable_id(text_file: Path) -> None:
    first = load_document(text_file)
    second = load_document(text_file)

    assert first.pages[0].text.startswith("Conteúdo confiável")
    assert first.document_id == second.document_id
    assert first.content_hash == second.content_hash


def test_load_pdf_extracts_text(tmp_path: Path) -> None:
    path = tmp_path / "guia.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Texto de uma página.")
        pdf.save(path)

    document = load_document(path)

    assert document.pages[0].page == 1
    assert "Texto de uma página" in document.pages[0].text


def test_empty_document_fails(tmp_path: Path) -> None:
    path = tmp_path / "vazio.md"
    path.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(DocumentProcessingError, match="vazio"):
        load_document(path)


def test_discovery_only_returns_supported_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("texto", encoding="utf-8")
    (tmp_path / "b.csv").write_text("a,b", encoding="utf-8")
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "c.md").write_text("# título", encoding="utf-8")

    assert [path.name for path in discover_documents(tmp_path)] == ["a.txt", "c.md"]
