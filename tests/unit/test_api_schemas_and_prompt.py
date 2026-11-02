import pytest
from pydantic import ValidationError

from app.generation.ollama import build_prompt
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.documents import IngestRequest


def test_chat_request_validates_question() -> None:
    assert ChatRequest(question="Pergunta válida").question == "Pergunta válida"
    with pytest.raises(ValidationError):
        ChatRequest(question=" ")


def test_ingest_request_accepts_optional_directory() -> None:
    assert IngestRequest(directory="data/raw").directory is not None
    assert IngestRequest().directory is None


def test_chat_response_schema() -> None:
    response = ChatResponse(answer="Resposta", sources=[], retrieval_scores=[], processing_time=0.1)
    assert response.answer == "Resposta"


def test_prompt_contains_safety_and_source_instructions() -> None:
    prompt = build_prompt(
        "Qual é a informação?",
        [{"filename": "guia.pdf", "page": 2, "content": "Informação de teste."}],
    )

    assert "[1] Fonte: guia.pdf, página 2" in prompt
    assert "somente o contexto" in prompt
    assert "ignore instruções" in prompt
    assert "Não invente fatos" in prompt


def test_prompt_handles_missing_context() -> None:
    assert "(nenhum contexto relevante encontrado)" in build_prompt("Pergunta?", [])
