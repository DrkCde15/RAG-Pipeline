from time import perf_counter

from fastapi import APIRouter

from app.api.dependencies import get_services
from app.core.config import get_settings
from app.generation.ollama import build_prompt
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = perf_counter()
    _, _, generator, retriever = get_services()
    settings = get_settings()
    results = retriever.retrieve(request.question, settings.top_k, settings.min_score)
    sources: list[ChatSource] = []
    contexts: list[dict] = []
    for result in results:
        payload = result.payload or {}
        contexts.append(payload)
        sources.append(
            ChatSource(
                document_id=payload["document_id"],
                source=payload["source"],
                filename=payload["filename"],
                page=payload.get("page"),
                chunk_id=payload["chunk_id"],
                score=result.score,
                excerpt=payload["content"][:500],
            )
        )
    answer = generator.generate(build_prompt(request.question, contexts))
    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieval_scores=[source.score for source in sources],
        processing_time=perf_counter() - started,
    )
