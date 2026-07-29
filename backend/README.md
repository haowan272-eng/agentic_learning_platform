# Backend

The backend layer contains the product platform and backend-owned runtime
entry points.

```text
backend/app/
  FastAPI gateway, authentication, API routers, Celery scheduling, RAG,
  Memory, SQLAlchemy models, schemas, and application services.

backend/packages/harness/deerflow/
  Agent kernel package used by the product platform.

backend/alembic/
  Database migrations.
```

Run the API from the repository root:

```powershell
uv run --frozen python backend/run.py
```
