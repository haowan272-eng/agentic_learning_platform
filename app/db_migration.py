"""Database migration entry points."""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import DATABASE_URL


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def alembic_config() -> Config:
    root = _project_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    return config


def upgrade_database(revision: str = "head") -> None:
    command.upgrade(alembic_config(), revision)


__all__ = ["alembic_config", "upgrade_database"]
