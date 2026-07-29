"""数据库迁移入口。

使用 Alembic 执行 PostgreSQL schema 变更，由 main.py 启动时按需调用。
"""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import DATABASE_URL


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _alembic_config() -> Config:
    root = _project_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    return config


def upgrade_database(revision: str = "head") -> None:
    command.upgrade(_alembic_config(), revision)


__all__ = ["upgrade_database"]
