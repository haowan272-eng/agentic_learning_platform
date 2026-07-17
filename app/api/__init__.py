from fastapi import APIRouter

from .auth import router as auth_router
from .user import router as user_router
from .document import router as document_router
from .rag import router as rag_router
from .knowledge_base import router as kb_router
from .conversation import router as conversation_router
from .memory import router as memory_router
from .health import router as health_router
from .agent import router as agent_router
from .metrics import router as metrics_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(user_router)
api_router.include_router(document_router)
api_router.include_router(rag_router)
api_router.include_router(kb_router)
api_router.include_router(conversation_router)
api_router.include_router(memory_router)
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(metrics_router)
