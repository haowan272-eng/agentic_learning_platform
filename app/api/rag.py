"""RAG回答HTTP与SSE流式入口"""
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from app.schemas.rag import AnswerRequest, AnswerResponse
from app.services.rag_service import run_rag_answer, stream_rag_answer

router = APIRouter(prefix="/embedding", tags=["RAG回答"])


@router.post("/rag/answer", response_model=AnswerResponse)
def answer_rag(
    body: AnswerRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return run_rag_answer(db, current_user, body)


@router.post("/rag/answer/stream")
def answer_rag_stream(
    body: AnswerRequest,
    current_user: str = Depends(get_current_user),
):
    def encode_events():
        for item in stream_rag_answer(current_user, body):
            payload = json.dumps(item["data"], ensure_ascii=False, separators=(",", ":"))
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return StreamingResponse(
        encode_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
