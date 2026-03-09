#!/usr/bin/env sh
set -e

export ANONYMIZED_TELEMETRY=FALSE

echo ">>> Building RAG knowledge base..."
uv run python -m app.core.clients.db.rag.build || true

echo ">>> Starting FastAPI server..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /code/app