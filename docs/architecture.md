# Architecture

面试提优学习 is organized around four runtime subsystems:

1. API service
   FastAPI exposes authentication, knowledge base, document, RAG, memory, conversation, health, metrics, and agent endpoints.

2. Background workers
   Celery handles long-running document indexing and agent runtime tasks. Celery Beat periodically dispatches pending durable outbox rows for agent tasks.

3. Data stores
   PostgreSQL is the durable source of truth. Redis handles Celery transport, short-term memory, cache, and runtime streams. Qdrant stores vector embeddings and retrieval metadata.

4. Frontend workspace
   Vue 3 serves the interview-improvement workspace for knowledge-base management, document upload, source-backed interview practice, citations, and Agent Runtime review.

## Backend Modules

```text
app/api/             HTTP route modules
app/agent_runtime/   planner, runtime, tools, store, feedback, LLM gateway
app/core/            config, database, Redis, Celery
app/memory/          memory event and summary logic
app/models/          SQLAlchemy models
app/rag/             parsing, chunking, embeddings, vectorstore, answering
app/schemas/         Pydantic request/response models
app/services/        workflow-oriented application services
app/tasks/           Celery task entry points
```

## Frontend

```text
frontend/src/App.vue          Interview-improvement workspace shell and views
frontend/src/services/api.ts  Typed API client for auth, KB, document, RAG, and Agent Runtime endpoints
frontend/src/style.css        Application layout and visual system
```

The development server proxies API calls to `http://localhost:8001`. The production nginx image serves static assets and proxies backend API paths to the `backend` Compose service.

## Queues

- `document_index`: document parsing, OCR/VL enrichment, chunking, embedding, Qdrant upsert.
- `agent_runtime`: agent task execution and resume handling.

## Migrations

Alembic migrations live in `alembic/versions`. Application startup does not run migrations by default; deployment should run:

```powershell
uv run --frozen --no-dev alembic upgrade head
```

The Docker Compose files run a dedicated `migrate` service before API and worker startup.
