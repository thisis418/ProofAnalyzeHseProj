ARG DOCKER_REGISTRY=docker.io/library
ARG PYTHON_VERSION=3.13

FROM ${DOCKER_REGISTRY}/python:${PYTHON_VERSION}-slim AS app-prod

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev

COPY . .

RUN uv sync --no-dev

# если в базовом образе нет пользователя runner, создай его
RUN id runner >/dev/null 2>&1 || useradd -m -u 10001 runner

RUN chown -R runner:root /code \
    && chmod -R g=u /code \
    && chmod -R g=u /home

USER runner

EXPOSE 8000