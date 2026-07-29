# Agentic Learning RAG

Agentic Learning RAG is a FastAPI knowledge-work platform for source-backed
interview practice. It combines shared RAG, document indexing, conversation
memory, and an Agent runtime. PostgreSQL stores durable state, Qdrant stores
vectors, Redis handles queues/cache/streams, and Celery runs long-lived jobs.

## Layered Layout

```text
agentic_learning_rag/
|-- backend/
|   |-- app/                         # Product platform: FastAPI, auth, API, scheduler, channels
|   |-- packages/harness/deerflow/    # Agent kernel: Agent, tools, models, sandbox, memory, runtime
|   |-- alembic/                     # Backend database migrations
|   |-- run.py                       # API entry point
|   `-- worker.py                    # Celery worker entry point
|-- frontend/                        # Web client workspace
|-- skills/                          # Built-in Skill protocol and implementations
|-- contracts/                       # Cross-language schemas and contracts
|-- docker/                          # Compose, Dockerfiles, Nginx
|-- deploy/helm/                     # Kubernetes/Helm deployment layer
|-- docs/                            # Architecture and development notes
|-- evaluation/                      # RAG evaluation tooling
|-- scripts/                         # Operational helpers
|-- tests/                           # Backend test suite
|-- pyproject.toml
`-- uv.lock
```

See `docs/architecture.md` for the dependency direction between layers.

## Local Setup

```powershell
cd E:\my-project\agentic_learning_rag
uv sync --frozen --group dev
Copy-Item .env.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64)); print('REFRESH_SECRET_KEY=' + secrets.token_urlsafe(64))"
```

Put the generated `SECRET_KEY` and `REFRESH_SECRET_KEY` values into `.env`
before starting the backend or Docker stack.

Frontend dependencies:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm install
```

## Run Locally

Container stack:

```powershell
docker compose --env-file .env -f docker/docker-compose.yml up -d --build
```

Backend without Docker:

```powershell
uv run --frozen python backend/run.py
```

Workers:

```powershell
$env:PYTHONPATH = "backend;backend/packages/harness"
uv run --frozen celery -A app.core.celery:celery_app worker --loglevel=INFO --queues=document_index,agent_runtime --pool=solo --concurrency=1
uv run --frozen celery -A app.core.celery:celery_app beat --loglevel=INFO
```

Frontend dev server:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm run dev
```

Default ports:

```text
Backend:    http://localhost:8001
Frontend:   http://localhost:5173
API docs:   http://localhost:8001/docs
PostgreSQL: localhost:15432
Redis:      localhost:16379
Qdrant:     http://localhost:16333
```

## Tests

```powershell
uv run --frozen --group dev python -m pytest
uv run --frozen --group dev --group evaluation python -m pytest evaluation/tests
docker compose --env-file .env -f docker/docker-compose.yml config
docker compose --env-file .env -f docker/docker-compose.prod.yml config
```

Frontend build:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm run build
```

## Deployment

Use `.env.production.example` as the production template, then run:

```powershell
Copy-Item .env.production.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(64)); print('REFRESH_SECRET_KEY=' + secrets.token_urlsafe(64))"
docker compose --env-file .env -f docker/docker-compose.prod.yml up -d --build
```

Replace every secret placeholder in `.env`; the backend refuses to start when
JWT secrets are missing or still set to placeholder values.

Helm/Kubernetes assets belong under `deploy/helm/`.
