"""Prometheus 指标暴露端点。

将 in-process 计数器与延迟汇总以 Prometheus text 格式输出，
供 Prometheus/Grafana 抓取。
"""
from fastapi import APIRouter, Depends, Response

from app.api.deps import get_current_user
from app.observability import prometheus_text


router = APIRouter(tags=["Observability"])


@router.get("/metrics", include_in_schema=False)
def metrics(_: str = Depends(get_current_user)) -> Response:
    return Response(prometheus_text(), media_type="text/plain; version=0.0.4; charset=utf-8")
