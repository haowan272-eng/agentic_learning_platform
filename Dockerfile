FROM ghcr.io/astral-sh/uv:0.11.18 AS uv
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_ROOT=/app/models \
    UPLOAD_DIR=/app/uploads

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        antiword \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY alembic.ini run.py worker.py ./
COPY alembic ./alembic
COPY app ./app

RUN mkdir -p /app/uploads /app/models

EXPOSE 8001

CMD ["uv", "run", "--frozen", "--no-dev", "python", "run.py"]
