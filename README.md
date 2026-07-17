# 面试提优学习

面试提优学习 is a FastAPI knowledge-work platform for source-backed interview practice. It combines shared RAG, document indexing, conversation memory, and an agent runtime. The service stores structured state in PostgreSQL, vectors in Qdrant, transient queues/cache in Redis, and uses Celery workers for document indexing and agent task execution.

## Capabilities

- User authentication with access and refresh tokens.
- Knowledge base CRUD and member-role management.
- Multi-format document upload and asynchronous indexing.
- Markdown normalization, chunking, embeddings, and Qdrant retrieval.
- RAG answers with citations, conversation history, and memory context.
- Agent Runtime with task lifecycle, streaming events, tool execution, feedback, and retryable durable outbox dispatch.
- Vue frontend for source-backed interview practice, citations, and Agent review.
- Optional RAGAS evaluation tooling under `evaluation/`.

## Architecture

```text
interview_improvement_rag/
|-- app/                  # FastAPI application and domain modules
|   |-- api/              # HTTP route modules
|   |-- agent_runtime/    # Agent planning, execution, tools, LLM gateway, store
|   |-- auth/             # JWT helpers
|   |-- core/             # Config, database, Redis, Celery
|   |-- memory/           # Memory profiles, events, summaries, consolidation
|   |-- models/           # SQLAlchemy models
|   |-- rag/              # Chunking, embeddings, vectorstore, answering
|   |-- schemas/          # Pydantic schemas
|   |-- services/         # Application services
|   `-- tasks/            # Celery tasks
|-- alembic/              # Database migrations
|-- docs/                 # Architecture and development notes
|-- evaluation/           # RAGAS datasets, scripts, and evaluation tests
|-- frontend/             # Vue 3 interview-improvement workspace
|-- scripts/              # Operational and evaluation helpers
|-- tests/                # Core backend test suite
|-- Dockerfile
|-- docker-compose.yml
|-- docker-compose.prod.yml
|-- pyproject.toml
`-- uv.lock
```

## Local Setup

Install backend dependencies:

```powershell
cd E:\my-project\agentic_learning_rag
uv sync --frozen --group dev
```

Install evaluation dependencies only when needed:

```powershell
uv sync --frozen --group evaluation
```

Install frontend dependencies:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm install
```

Create local environment configuration:

```powershell
Copy-Item .env.example .env
```

Important defaults:

```env
DATABASE_URL=postgresql://interview_user:interview_password@127.0.0.1:15432/interview_improvement_rag
REDIS_URL=redis://localhost:16379/0
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=interview_improvement_chunks
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:8001,http://127.0.0.1:8001
```

## Run Locally

Start the local stack:

```powershell
docker compose up -d --build
```

Default ports:

```text
Backend:    http://localhost:8001
Frontend:   http://localhost:5173
API docs:   http://localhost:8001/docs
PostgreSQL: localhost:15432
Redis:      localhost:16379
Qdrant:     http://localhost:6333
```

Run the backend without Docker when dependencies are already available:

```powershell
uv run --frozen python run.py
```

Run workers manually:

```powershell
uv run --frozen celery -A app.core.celery:celery_app worker --loglevel=INFO --queues=document_index,agent_runtime --pool=solo --concurrency=1
uv run --frozen celery -A app.core.celery:celery_app beat --loglevel=INFO
```

Run the frontend dev server:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm run dev
```

## Tests

Backend tests:

```powershell
uv run --frozen --group dev python -m pytest
```

Evaluation script tests:

```powershell
uv run --frozen --group dev --group evaluation python -m pytest evaluation/tests
```

Docker configuration checks:

```powershell
docker compose -f docker-compose.yml config
docker compose -f docker-compose.prod.yml config
```

Frontend build:

```powershell
cd E:\my-project\agentic_learning_rag\frontend
npm run build
```

## Deployment Notes

Use `.env.production.example` as the production template. Production deployments must set:

- `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `SECRET_KEY`
- `REFRESH_SECRET_KEY`
- `CORS_ORIGINS`
- LLM provider credentials
- `MODEL_ROOT`

Production startup:

```powershell
Copy-Item .env.production.example .env
docker compose -f docker-compose.prod.yml up -d --build
```

## Evaluation

Generate RAGAS inputs by calling the running API:

```powershell
python evaluation/run_rag_on_testset.py --username alice --password secret --kb-id 1
```

Score generated samples:

```powershell
uv run --frozen --group evaluation python scripts/ragas_evaluate.py --dataset evaluation/testsets/tongchuan/ragas_eval_input.jsonl
```

Large local evaluation outputs, temporary files, and backups are intentionally ignored by Git.
