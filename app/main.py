"""FastAPI 应用入口：创建 app、生命周期、路由注册、全局异常处理。"""
from contextlib import asynccontextmanager
import logging
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.database import engine, Base, SessionLocal
from .models import User
from .api import api_router
from app.core.redis import init_redis, close_redis
from app.core.config import (
    APP_ENV,
    CORS_ORIGINS,
    CREATE_SEED_ADMIN,
    SEED_ADMIN_PASSWORD,
    SEED_ADMIN_USERNAME,
    REFRESH_SECRET_KEY,
    RUN_MIGRATIONS_ON_STARTUP,
    SECRET_KEY,
)
from .logging_config import request_id_var, setup_logging
from .db_migration import upgrade_database

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Knowledge Platform", version="1.0.0")


def _validate_production_secret(name: str, value: str) -> None:
    normalized = (value or "").strip()
    if (
        not normalized
        or len(normalized) < 32
        or normalized.lower().startswith("change-me")
    ):
        raise RuntimeError(f"生产环境必须配置安全的 {name}")

# ---- 全局异常处理器 ----

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """统一捕获未处理异常，返回标准格式，避免内部信息泄"""

    return JSONResponse(
        status_code=500,
        content={"detail": "服务内部错误，请稍后重试", "error_type": type(exc).__name__},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """参数校验异常"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )

# ---- CORS：生产环境改为具体域名 ----

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("request completed", extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return response
    finally:
        request_id_var.reset(token)

app.include_router(api_router)


def on_startup():
    """启动时：建表、创建种子用户（bcrypt 哈希）并初始化 Redis。"""
    import bcrypt

    if APP_ENV == "production":
        _validate_production_secret("SECRET_KEY", SECRET_KEY)
        _validate_production_secret("REFRESH_SECRET_KEY", REFRESH_SECRET_KEY)
    if APP_ENV == "production" and "*" in CORS_ORIGINS:
        raise RuntimeError("生产环境 CORS_ORIGINS 不允许使用通配")

    # PostgreSQL schema changes are managed by Alembic. Local tests can still
    # create lightweight databases directly from SQLAlchemy metadata.
    if engine.dialect.name == "postgresql" and RUN_MIGRATIONS_ON_STARTUP:
        upgrade_database()
    elif engine.dialect.name != "postgresql":
        Base.metadata.create_all(bind=engine)

    # 种子用户：首次启动时按配置创建管理员账号，使用 bcrypt 哈希存储密码。
    db = SessionLocal()
    try:
        if CREATE_SEED_ADMIN and not SEED_ADMIN_PASSWORD:
            raise RuntimeError("CREATE_SEED_ADMIN=true 时必须配置 SEED_ADMIN_PASSWORD")
        if CREATE_SEED_ADMIN and not db.query(User).filter(User.username == SEED_ADMIN_USERNAME).first():
            hashed = bcrypt.hashpw(SEED_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
            db.add(User(username=SEED_ADMIN_USERNAME, password=hashed))
            db.commit()
            logger.info("seed user created")
        elif CREATE_SEED_ADMIN:
            logger.info("seed user already exists")
    except Exception:
        logger.exception("database seed initialization failed")
    finally:
        db.close()

    # Redis 初始化（失败不阻塞启动，缓存降级为查库）
    try:
        init_redis()
    except Exception:
        logger.exception("redis initialization failed; cache disabled")


def on_shutdown():
    """关闭时：释放 Redis 连接"""
    close_redis()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """使用FastAPI lifespan统一管理数据库、Redis初始化与释放"""
    on_startup()
    try:
        yield
    finally:
        on_shutdown()


app.router.lifespan_context = lifespan
