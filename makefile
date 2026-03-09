.PHONY: help install sync run dev rag build up down restart logs ps shell rebuild clean

help:
	@echo "Доступные команды:"
	@echo "  make install   - установить uv, если нужно"
	@echo "  make sync      - синхронизировать зависимости через uv"
	@echo "  make run       - локальный запуск FastAPI без reload"
	@echo "  make dev       - локальный запуск FastAPI с reload"
	@echo "  make rag       - пересобрать RAG индекс"
	@echo "  make build     - собрать docker image"
	@echo "  make up        - поднять docker compose"
	@echo "  make down      - остановить docker compose"
	@echo "  make restart   - перезапустить docker compose"
	@echo "  make logs      - смотреть логи контейнера"
	@echo "  make ps        - список контейнеров"
	@echo "  make shell     - зайти в контейнер app"
	@echo "  make rebuild   - полная пересборка контейнеров без cache"
	@echo "  make clean     - удалить контейнеры, volume и кеш"

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh

sync:
	uv sync

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

rag:
	uv run python -m app.core.clients.db.rag.build

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

restart:
	docker compose down && docker compose up

logs:
	docker compose logs -f app

ps:
	docker compose ps

shell:
	docker compose exec app sh

rebuild:
	docker compose down -v
	docker compose build --no-cache
	docker compose up

clean:
	docker compose down -v --remove-orphans
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +