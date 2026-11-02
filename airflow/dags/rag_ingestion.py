import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from app.core.config import get_settings
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import split_text
from app.ingestion.loaders import discover_documents, load_document
from app.vectorstore.qdrant import QdrantStore, make_chunk_payload

logger = logging.getLogger(__name__)
settings = get_settings()


def _run_dir(run_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", run_id)
    path = settings.processed_dir / "airflow" / safe_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover(**context) -> None:
    paths = discover_documents(settings.data_dir)
    path = _run_dir(context["run_id"]) / "sources.json"
    path.write_text(json.dumps([str(item) for item in paths]), encoding="utf-8")
    logger.info("Descobertos %d documentos", len(paths))


def prepare(**context) -> None:
    run_dir = _run_dir(context["run_id"])
    paths = json.loads((run_dir / "sources.json").read_text(encoding="utf-8"))
    chunks = []
    for path in paths:
        document = load_document(Path(path))
        index = 0
        for page in document.pages:
            for chunk in split_text(page.text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(
                    make_chunk_payload(
                        document.document_id,
                        str(Path(path).resolve()),
                        Path(path).name,
                        index,
                        chunk.content,
                        document.content_hash,
                        page.page,
                    )
                )
                index += 1
    (run_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    logger.info("Preparados %d chunks", len(chunks))


def embed(**context) -> None:
    run_dir = _run_dir(context["run_id"])
    chunks = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
    if not chunks:
        (run_dir / "vectors.json").write_text("[]", encoding="utf-8")
        logger.info("Nenhum chunk para embedding")
        return
    embedding_service = EmbeddingService(settings.embedding_model)
    vectors = embedding_service.embed([item["content"] for item in chunks])
    (run_dir / "vectors.json").write_text(json.dumps(vectors), encoding="utf-8")
    logger.info("Gerados %d embeddings", len(vectors))


def index(**context) -> None:
    run_dir = _run_dir(context["run_id"])
    chunks = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = json.loads((run_dir / "vectors.json").read_text(encoding="utf-8"))
    if not chunks:
        logger.info("Nenhum chunk para indexação")
        return
    store = QdrantStore(
        settings.qdrant_url, settings.qdrant_collection, vector_size=len(vectors[0])
    )
    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(chunks):
        grouped.setdefault(item["document_id"], []).append(index)
    for positions in grouped.values():
        store.upsert_document(
            [chunks[position] for position in positions],
            [vectors[position] for position in positions],
        )
    logger.info("Indexados %d chunks em %d documentos", len(chunks), len(grouped))


with DAG(
    dag_id="rag_document_ingestion",
    description="Descobre, prepara, embeda e indexa documentos locais no Qdrant.",
    start_date=datetime(2025, 1, 1),
    schedule=os.getenv("AIRFLOW_SCHEDULE", "@daily"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "rag-pipeline", "retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["rag", "ingestion"],
) as dag:
    discover_task = PythonOperator(
        task_id="discover_documents", python_callable=discover, do_xcom_push=False
    )
    prepare_task = PythonOperator(
        task_id="extract_clean_chunk", python_callable=prepare, do_xcom_push=False
    )
    embed_task = PythonOperator(
        task_id="generate_embeddings", python_callable=embed, do_xcom_push=False
    )
    index_task = PythonOperator(task_id="index_qdrant", python_callable=index, do_xcom_push=False)
    discover_task >> prepare_task >> embed_task >> index_task
