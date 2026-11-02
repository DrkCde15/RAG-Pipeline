from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health
from app.core.exceptions import DocumentProcessingError, ServiceUnavailableError
from app.core.logging import configure_logging

configure_logging()
app = FastAPI(
    title="RAG Pipeline API",
    description="API local para ingestão, recuperação e geração de respostas com fontes.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.exception_handler(DocumentProcessingError)
async def document_error_handler(request: Request, error: DocumentProcessingError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(ServiceUnavailableError)
async def service_error_handler(request: Request, error: ServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})
