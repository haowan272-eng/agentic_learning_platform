"""集中管理所有配置常量。

从 .env 文件加载配置，环境变量优先。
"""
import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
LOG_JSON = os.getenv("LOG_JSON", "true").lower() == "true"

# ==================== 数据库配置 ====================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://interview_user:interview_password@127.0.0.1:15432/interview_improvement_rag",
)
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"
RUN_MIGRATIONS_ON_STARTUP = os.getenv("RUN_MIGRATIONS_ON_STARTUP", "false").lower() == "true"

# ==================== JWT 配置 ====================
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY", "change-me-in-production-refresh")
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "1440"))

# ==================== LangChain / 回答模型 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
RAG_ANSWER_TEMPERATURE = float(os.getenv("RAG_ANSWER_TEMPERATURE", "0.1"))
RAG_ANSWER_MAX_TOKENS = max(1, int(os.getenv("RAG_ANSWER_MAX_TOKENS", "2000")))
RAG_ANSWER_TIMEOUT_SECONDS = max(1.0, float(os.getenv("RAG_ANSWER_TIMEOUT_SECONDS", "60")))
RAG_ANSWER_MAX_RETRIES = max(0, int(os.getenv("RAG_ANSWER_MAX_RETRIES", "1")))
RAG_MAX_CONTEXT_CHARS = max(1000, int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000")))
RAG_QUERY_REWRITE_ENABLED = os.getenv("RAG_QUERY_REWRITE_ENABLED", "true").lower() == "true"
RAG_QUERY_REWRITE_MODEL = os.getenv("RAG_QUERY_REWRITE_MODEL", DEEPSEEK_MODEL).strip()
RAG_QUERY_REWRITE_TEMPERATURE = float(os.getenv("RAG_QUERY_REWRITE_TEMPERATURE", "0.0"))
RAG_QUERY_REWRITE_MAX_CHARS = max(100, int(os.getenv("RAG_QUERY_REWRITE_MAX_CHARS", "800")))
RAG_RERANK_CANDIDATES = max(1, int(os.getenv("RAG_RERANK_CANDIDATES", "16")))
RAG_HISTORY_MESSAGES = max(0, int(os.getenv("RAG_HISTORY_MESSAGES", "10")))
RAG_COMPACTION_THRESHOLD = max(
    RAG_HISTORY_MESSAGES + 2,
    int(os.getenv("RAG_COMPACTION_THRESHOLD", "18")),
)
RAG_SUMMARY_MAX_CHARS = max(1000, int(os.getenv("RAG_SUMMARY_MAX_CHARS", "5000")))
RAG_MEMORY_MAX_TOKENS = max(1000, int(os.getenv("RAG_MEMORY_MAX_TOKENS", "16000")))
RAG_MEMORY_SUMMARY_MAX_TOKENS = max(200, int(os.getenv("RAG_MEMORY_SUMMARY_MAX_TOKENS", "2000")))
RAG_MEMORY_RECENT_MAX_TOKENS = max(200, int(os.getenv("RAG_MEMORY_RECENT_MAX_TOKENS", "10000")))
RAG_MEMORY_TASK_STATE_MAX_TOKENS = max(100, int(os.getenv("RAG_MEMORY_TASK_STATE_MAX_TOKENS", "800")))
RAG_MEMORY_EXTRACTION_INTERVAL_MESSAGES = max(1, int(os.getenv("RAG_MEMORY_EXTRACTION_INTERVAL_MESSAGES", "10")))
RAG_MEMORY_EXTRACTION_MIN_CHARS = max(0, int(os.getenv("RAG_MEMORY_EXTRACTION_MIN_CHARS", "30")))
RAG_MEMORY_EXTRACTION_MESSAGE_HEAD_CHARS = max(0, int(os.getenv("RAG_MEMORY_EXTRACTION_MESSAGE_HEAD_CHARS", "300")))
RAG_MEMORY_EXTRACTION_MESSAGE_TAIL_CHARS = max(0, int(os.getenv("RAG_MEMORY_EXTRACTION_MESSAGE_TAIL_CHARS", "1000")))


# ==================== Qdrant 向量数据库配置 ====================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "interview_improvement_chunks")
QDRANT_DIM = int(os.getenv("QDRANT_DIM", "1024"))
# 标量量化：int8 压缩向量内存 75%，精度损失 < 1%。空字符串表示关闭。
QDRANT_QUANTIZATION = os.getenv("QDRANT_QUANTIZATION", "int8").strip().lower()

# ==================== RAG 配置 ====================
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_DOCUMENT_SIZE_MB = max(1, int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50")))
DOCUMENT_UPLOAD_CHUNK_BYTES = max(
    64 * 1024,
    int(os.getenv("DOCUMENT_UPLOAD_CHUNK_BYTES", str(1024 * 1024))),
)
DOC_INDEX_MAX_RETRIES = max(0, int(os.getenv("DOC_INDEX_MAX_RETRIES", "2")))
DOC_INDEX_LOCK_TTL_SECONDS = min(
    600,
    max(60, int(os.getenv("DOC_INDEX_LOCK_TTL_SECONDS", "300"))),
)
DOC_INDEX_LOCK_HEARTBEAT_SECONDS = max(
    10,
    min(
        DOC_INDEX_LOCK_TTL_SECONDS // 3,
        int(os.getenv("DOC_INDEX_LOCK_HEARTBEAT_SECONDS", "60")),
    ),
)

# ==================== Qwen-VL / vLLM 视觉理解 ====================
# VISION_MODEL empty disables vision captioning; images still use OCR and nearby text.
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "http://127.0.0.1:8001/v1").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "EMPTY").strip()
VISION_TIMEOUT_SECONDS = float(os.getenv("VISION_TIMEOUT_SECONDS", "20"))
VISION_MAX_RETRIES = int(os.getenv("VISION_MAX_RETRIES", "0"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "400"))
VISION_MAX_IMAGE_EDGE = int(os.getenv("VISION_MAX_IMAGE_EDGE", "1600"))
VISION_CACHE_CAPTIONS = os.getenv("VISION_CACHE_CAPTIONS", "true").lower() == "true"
# Max vision captions per PDF; 0 disables captioning.
VISION_MAX_CAPTIONS_PER_DOCUMENT = int(os.getenv("VISION_MAX_CAPTIONS_PER_DOCUMENT", "8"))
# Use OCR directly for text-heavy images.
VISION_SKIP_CAPTION_OCR_CHARS = int(os.getenv("VISION_SKIP_CAPTION_OCR_CHARS", "120"))
# Drop duplicate caption sentences above this similarity threshold.
VISION_TEXT_DEDUP_THRESHOLD = float(os.getenv("VISION_TEXT_DEDUP_THRESHOLD", "0.72"))
# OCR concurrency.
RAG_OCR_CONCURRENCY = max(
    1,
    int(os.getenv("RAG_OCR_CONCURRENCY", "3")),
)
RAG_VISION_CONCURRENCY = max(1, int(os.getenv("RAG_VISION_CONCURRENCY", "2")))
# Global VL concurrency shared by workers.
VISION_GLOBAL_CONCURRENCY = max(1, int(os.getenv("VISION_GLOBAL_CONCURRENCY", "2")))
VISION_GLOBAL_ACQUIRE_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("VISION_GLOBAL_ACQUIRE_TIMEOUT_SECONDS", "30")),
)
VISION_GLOBAL_SLOT_TTL_SECONDS = max(
    VISION_TIMEOUT_SECONDS * (VISION_MAX_RETRIES + 1) + 5.0,
    float(os.getenv("VISION_GLOBAL_SLOT_TTL_SECONDS", "45")),
)
# Per-document parse timeout.
RAG_DOCUMENT_TIMEOUT_SECONDS = max(
    30.0,
    float(os.getenv("RAG_DOCUMENT_TIMEOUT_SECONDS", "1800")),
)
# Per-image OCR timeout; 0 disables timeout.
RAG_OCR_TIMEOUT_SECONDS = max(0.0, float(os.getenv("RAG_OCR_TIMEOUT_SECONDS", "15")))

# Nearby text constraints.
RAG_NEARBY_TEXT_MAX_VERTICAL_RATIO = float(os.getenv("RAG_NEARBY_TEXT_MAX_VERTICAL_RATIO", "0.18"))
RAG_NEARBY_TEXT_MIN_HORIZONTAL_OVERLAP = float(os.getenv("RAG_NEARBY_TEXT_MIN_HORIZONTAL_OVERLAP", "0.20"))
RAG_NEARBY_TEXT_MAX_BLOCKS = int(os.getenv("RAG_NEARBY_TEXT_MAX_BLOCKS", "2"))
RAG_NEARBY_TEXT_MAX_CHARS = int(os.getenv("RAG_NEARBY_TEXT_MAX_CHARS", "800"))

# ==================== 文件存储 ====================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "").strip() or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
)

# ==================== 服务器配置 ====================
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8001"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8001,http://127.0.0.1:8001",
    ).split(",")
    if origin.strip()
]
CREATE_SEED_ADMIN = os.getenv("CREATE_SEED_ADMIN", "false").lower() == "true"
SEED_ADMIN_USERNAME = os.getenv("SEED_ADMIN_USERNAME", "admin").strip()
SEED_ADMIN_PASSWORD = os.getenv("SEED_ADMIN_PASSWORD", "").strip()

# ==================== 知识库配置 ====================
AUTO_CREATE_DEFAULT_KB = os.getenv("AUTO_CREATE_DEFAULT_KB", "false").lower() == "true"
DEFAULT_KB_NAME = os.getenv("DEFAULT_KB_NAME", "我的知识").strip()
DEFAULT_KB_CHUNK_SIZE = int(os.getenv("DEFAULT_KB_CHUNK_SIZE", "500"))
DEFAULT_KB_CHUNK_OVERLAP = int(os.getenv("DEFAULT_KB_CHUNK_OVERLAP", "50"))

# ==================== Redis 配置 ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_USER_TTL = int(os.getenv("CACHE_USER_TTL", "300"))
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_DOCUMENT_QUEUE = os.getenv("CELERY_DOCUMENT_QUEUE", "document_index")
CELERY_AGENT_QUEUE = os.getenv("CELERY_AGENT_QUEUE", "agent_runtime")
AGENT_TASK_MAX_RETRIES = max(0, int(os.getenv("AGENT_TASK_MAX_RETRIES", "1")))
AGENT_TASK_LEASE_SECONDS = max(30, int(os.getenv("AGENT_TASK_LEASE_SECONDS", "300")))
AGENT_TASK_DEFAULT_DEADLINE_SECONDS = max(30, int(os.getenv("AGENT_TASK_DEFAULT_DEADLINE_SECONDS", "900")))
AGENT_TASK_DEFAULT_MAX_STEPS = max(2, int(os.getenv("AGENT_TASK_DEFAULT_MAX_STEPS", "8")))
AGENT_TASK_DEFAULT_MAX_TOOL_CALLS = max(2, int(os.getenv("AGENT_TASK_DEFAULT_MAX_TOOL_CALLS", "12")))
AGENT_TASK_DEFAULT_MAX_TOKENS = max(1000, int(os.getenv("AGENT_TASK_DEFAULT_MAX_TOKENS", "24000")))
AGENT_TASK_DEFAULT_MAX_COST_USD = max(0.01, float(os.getenv("AGENT_TASK_DEFAULT_MAX_COST_USD", "2.0")))
AGENT_ESTIMATED_COST_PER_1K_TOKENS_USD = max(0.0, float(os.getenv("AGENT_ESTIMATED_COST_PER_1K_TOKENS_USD", "0.002")))
AGENT_TOOL_TIMEOUT_SECONDS = max(1.0, float(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "75")))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_RETRY_ATTEMPTS = max(1, int(os.getenv("GITHUB_RETRY_ATTEMPTS", "3")))
GITHUB_CACHE_TTL_SECONDS = max(60, int(os.getenv("GITHUB_CACHE_TTL_SECONDS", "3600")))
AGENT_MEMORY_MAX_EVENT_CHARS = max(200, int(os.getenv("AGENT_MEMORY_MAX_EVENT_CHARS", "1600")))
AGENT_MEMORY_MIN_CONFIDENCE = min(1.0, max(0.0, float(os.getenv("AGENT_MEMORY_MIN_CONFIDENCE", "0.65"))))
AGENT_MEMORY_DECAY_HALF_LIFE_DAYS = max(1, int(os.getenv("AGENT_MEMORY_DECAY_HALF_LIFE_DAYS", "30")))
AGENT_MEMORY_DECAY_ENABLED = os.getenv("AGENT_MEMORY_DECAY_ENABLED", "true").lower() == "true"
AGENT_LLM_TIMEOUT_SECONDS = max(5.0, float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "60")))
AGENT_LLM_PROVIDER_FAILURE_THRESHOLD = max(1, int(os.getenv("AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", "2")))
AGENT_LLM_PROVIDER_COOLDOWN_SECONDS = max(5, int(os.getenv("AGENT_LLM_PROVIDER_COOLDOWN_SECONDS", "30")))
AGENT_PLANNER_MODEL_CHAIN = os.getenv("AGENT_PLANNER_MODEL_CHAIN", "deepseek,openai,anthropic").strip()
AGENT_ARCHITECT_MODEL_CHAIN = os.getenv("AGENT_ARCHITECT_MODEL_CHAIN", "deepseek,openai,anthropic").strip()
AGENT_JUDGE_MODEL_CHAIN = os.getenv("AGENT_JUDGE_MODEL_CHAIN", "deepseek,openai,anthropic").strip()
AGENT_PLANNER_CANDIDATE_COUNT = min(3, max(1, int(os.getenv("AGENT_PLANNER_CANDIDATE_COUNT", "2"))))
# Sandbox execution (untrusted code / tool execution in isolated containers).
# Leave AGENT_SANDBOX_URL empty to disable sandbox execution.
AGENT_SANDBOX_URL = os.getenv("AGENT_SANDBOX_URL", "").strip()
AGENT_SANDBOX_TOKEN = os.getenv("AGENT_SANDBOX_TOKEN", "").strip()
AGENT_SANDBOX_TIMEOUT_SECONDS = max(5.0, float(os.getenv("AGENT_SANDBOX_TIMEOUT_SECONDS", "30")))
# LangGraph checkpoints are the durable source of execution state.  Keep setup
# opt-in so migrations run as a deliberate deployment action, not per worker.
LANGGRAPH_CHECKPOINT_SETUP = os.getenv("LANGGRAPH_CHECKPOINT_SETUP", "false").lower() == "true"
CELERY_RESULT_EXPIRES_SECONDS = max(
    300,
    int(os.getenv("CELERY_RESULT_EXPIRES_SECONDS", "3600")),
)
CELERY_VISIBILITY_TIMEOUT_SECONDS = max(
    int(RAG_DOCUMENT_TIMEOUT_SECONDS + 600),
    int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", "3600")),
)
SHORT_TERM_MEMORY_MESSAGES = max(
    RAG_HISTORY_MESSAGES,
    int(os.getenv("SHORT_TERM_MEMORY_MESSAGES", "12")),
)
SHORT_TERM_MEMORY_TTL_SECONDS = max(
    300,
    int(os.getenv("SHORT_TERM_MEMORY_TTL_SECONDS", "86400")),
)
SHORT_TERM_MESSAGE_MAX_CHARS = max(
    500,
    int(os.getenv("SHORT_TERM_MESSAGE_MAX_CHARS", "4000")),
)

# ==================== Rate Limiting 配置 ====================
RATE_LIMIT_LOGIN_PER_MIN = int(os.getenv("RATE_LIMIT_LOGIN_PER_MIN", "5"))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
