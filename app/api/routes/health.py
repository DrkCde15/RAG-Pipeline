import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.generation.ollama import OllamaGenerator

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> JSONResponse:
    settings = get_settings()
    status = {"api": "ok", "qdrant": "unavailable", "ollama": "unavailable"}
    try:
        QdrantClient(url=settings.qdrant_url, timeout=5).get_collections()
        status["qdrant"] = "ok"
    except Exception:
        logger.exception("Verificação de saúde do Qdrant falhou")
    try:
        OllamaGenerator(settings.ollama_base_url, settings.ollama_model).health()
        status["ollama"] = "ok"
    except Exception:
        logger.exception("Verificação de saúde do Ollama falhou")
    healthy = all(value == "ok" for value in status.values())
    return JSONResponse(status_code=200 if healthy else 503, content=status)
