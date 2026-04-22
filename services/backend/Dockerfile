FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/backend/pyproject.toml services/backend/uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project --no-dev
COPY services/backend/app /app/app
COPY services/backend/scripts /app/scripts
COPY services/backend/alembic.ini /app/alembic.ini
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
CMD ["fastapi", "run", "--workers", "4", "app/main.py"]
