# Development

## Dependency Groups

- `dev`: pytest and core development tooling.
- `evaluation`: RAGAS, Jupyter, and evaluation-only dependencies.

Default tests intentionally cover only `tests/`. Evaluation-specific tests live under `evaluation/tests/` and require the `evaluation` dependency group.

## Useful Commands

```powershell
uv sync --frozen --group dev
uv run --frozen --group dev python -m pytest
uv run --frozen --group dev --group evaluation python -m pytest evaluation/tests
```

```powershell
cd frontend
npm ci
npm run build
```

```powershell
docker compose -f docker-compose.yml config
docker compose -f docker-compose.prod.yml config
```

## Local Artifacts

These are local-only and ignored:

- `.venv/`
- `.uv-cache/`
- `.pytest_cache/`
- `__pycache__/`
- `frontend/node_modules/`
- `frontend/dist/`
- `uploads/`
- generated RAGAS result, backup, and temporary files
