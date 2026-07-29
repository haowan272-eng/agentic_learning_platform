# Architecture

The repository is organized into the layered layout requested for Agentic Learning RAG.

## Code Layers

```text
backend/packages/harness/deerflow/
  Agent kernel: runtime, planner, middleware-adjacent orchestration, tools,
  model gateway, sandbox, memory hooks, durable store, feedback, and schemas.

backend/app/
  Product platform: FastAPI gateway, authentication, API routers, scheduler,
  Celery tasks, RAG services, Memory services, SQLAlchemy models, and channels.

frontend/
  Web client: Vue/Vite workspace, streaming conversation UI, artifacts,
  and Agent review surfaces.

skills/
  Prompt-level Skills: built-in guidance and generated Skill packages that
  are progressively retrieved and injected into Agent prompts.

contracts/
  Cross-language contracts: Slash Skill schema, Subagent state schema,
  and Skill Review schema.

docker/
  Local/container deployment: Compose files, backend/frontend Dockerfiles,
  and Nginx config.

deploy/helm/
  Kubernetes deployment layer: Helm chart and provisioner-facing values.
```

## Dependency Direction

`backend/app` is the product boundary. It owns HTTP, auth, persistence models,
RAG workflows, Memory workflows, and Celery task scheduling.

`deerflow` is the Agent kernel boundary. Product routes and tasks call into
`deerflow`; the kernel may call platform services where it needs persistence,
memory context, RAG tools, or learning outputs.

Memory is an internal runtime subsystem, not a registered Tool: the runtime
loads its context before planning, appends execution events during a task, and
consolidates approved learner facts only after the task completes.

Research uses a source policy. In `auto`, the Research Agent selects local RAG,
DuckDuckGo, and GitHub discovery tools for each query; an empty local retrieval
is not a task failure. `local_only` is available through the Agent task API and
disables external web and GitHub retrieval, requiring local citations before a
verified proposal can pass.

`frontend` talks only to product APIs. It does not import backend or Agent
kernel code directly.

`contracts` and `skills` are repository-level extension boundaries intended to
stay language-neutral where possible. Skills are prompt guidance, not tools:
the Agent Runtime retrieves relevant `SKILL.md` packages and injects only the
matching instructions into planner/agent prompts. External side effects remain
behind the Tool Registry permission model.

## Runtime Subsystems

1. API service: FastAPI exposes authentication, knowledge base, document, RAG,
   memory, conversation, health, metrics, and Agent endpoints.
2. Background workers: Celery handles document indexing and Agent runtime tasks.
   Celery Beat periodically dispatches pending durable outbox rows.
3. Memory system: Redis holds a bounded, TTL-based session event window for
   fast context reads; on a cache miss the runtime rebuilds it from PostgreSQL
   AgentEvent records. PostgreSQL remains the source of truth for session
   summaries, task state, and long-term learner profiles. Approved events are
   consolidated into UserMemory with confidence thresholds and time decay.
4. Data stores: PostgreSQL is the durable source of truth. Redis handles Celery,
   short-term memory, cache, and runtime streams. Qdrant stores embeddings.
5. Web workspace: The frontend serves the learning workspace and Agent review UI.
6. Skill evolution: Optional online evolution can distill durable interaction
   lessons into `skills/generated/<skill>/SKILL.md`, recording usage,
   provenance, and version snapshots under `.agentic_learning_rag/skill-evolution/`.

## Agent Runtime Routes

The Agent Runtime exposes two top-level task routes only. `supervisor_agent` is
a pure router: it creates neither answers nor tool calls.

```mermaid
flowchart TD
    user["User task"] --> supervisor["supervisor_agent"]
    supervisor --> route{"answer or research"}
    route -->|answer| answer["answer_agent"]
    answer --> final["final_response"]
    route -->|research| planner["planner_agent"]
    planner --> approval{"approval required"}
    approval -->|no| research["research_agent"]
    approval -->|yes| gate["approval_gate"]
    gate -->|approve| research
    gate -->|edit| planner
    gate -->|reject| fallback["fallback_response"]
    research --> result{"result"}
    result -->|deliverable| review["review_agent"]
    result -->|needs confirmation| gate
    result -->|insufficient evidence or failure| fallback
    review -->|approved| final
    review -->|needs confirmation| gate
    review -->|rejected| fallback
    final --> end["End"]
    fallback --> end
```

`answer_agent` may answer from model knowledge or use its registered RAG, web,
GitHub, and claim-verification tools. It can only proceed to `final_response`
or `fallback_response`; it cannot upgrade into research. `planner_agent` only
creates an executable plan, and `research_agent` owns multi-step evidence,
proposal, verification, and repair work. `review_agent` is the final
publication gate: it reads research state but has no tool permissions and can
only approve publication, request confirmation, or reject to fallback. All
other graph nodes have no tool permissions.

## Common Commands

```powershell
uv run --frozen --group dev python -m pytest
uv run --frozen python backend/run.py
$env:PYTHONPATH = "backend;backend/packages/harness"
uv run --frozen celery -A app.core.celery:celery_app worker --loglevel=INFO --queues=document_index,agent_runtime
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.prod.yml config
```

Alembic migrations live in `backend/alembic/versions`.
