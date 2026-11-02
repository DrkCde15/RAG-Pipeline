.PHONY: install test lint format compose-config up down pull-model ingest

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

compose-config:
	podman-compose config

up:
	podman-compose up --build -d

down:
	podman-compose down

pull-model:
	podman-compose exec ollama ollama pull $${OLLAMA_MODEL:-llama3.2:1b}

ingest:
	podman-compose exec api python scripts/ingest_documents.py
