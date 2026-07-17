from fastapi import APIRouter, Depends, Response

from app.api.deps import get_current_user
from app.observability import prometheus_text


router = APIRouter(tags=["Observability"])


@router.get("/metrics", include_in_schema=False)
def metrics(_: str = Depends(get_current_user)) -> Response:
    return Response(prometheus_text(), media_type="text/plain; version=0.0.4; charset=utf-8")
