"""FastAPI exception handlers for the unified API error envelope."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, ErrorCode, error_payload, fallback_code_for_status

logger = logging.getLogger(__name__)


_KNOWN_MESSAGE_CODES: dict[str, ErrorCode] = {
    "Token expired": ErrorCode.AUTH_TOKEN_EXPIRED,
    "Invalid token": ErrorCode.AUTH_TOKEN_INVALID,
    "Refresh token expired": ErrorCode.AUTH_TOKEN_EXPIRED,
    "Invalid refresh token": ErrorCode.AUTH_TOKEN_INVALID,
    "User not found": ErrorCode.AUTH_USER_NOT_FOUND,
    "用户不存在": ErrorCode.AUTH_USER_NOT_FOUND,
    "用户不存在，请注册": ErrorCode.AUTH_USER_NOT_FOUND,
    "密码错误": ErrorCode.AUTH_PASSWORD_INVALID,
    "用户名已存在": ErrorCode.AUTH_USERNAME_EXISTS,
    "Knowledge base not found": ErrorCode.KB_NOT_FOUND,
    "You are not a member of this knowledge base": ErrorCode.KB_FORBIDDEN,
    "Document not found": ErrorCode.DOCUMENT_NOT_FOUND,
    "Document is outside your personal scope": ErrorCode.DOCUMENT_FORBIDDEN,
    "Document is already being indexed": ErrorCode.DOCUMENT_ALREADY_INDEXING,
    "Original document file is not available for reindexing": ErrorCode.DOCUMENT_SOURCE_MISSING,
    "Index queue is unavailable": ErrorCode.DOCUMENT_INDEX_QUEUE_UNAVAILABLE,
    "Conversation not found": ErrorCode.RAG_CONVERSATION_NOT_FOUND,
    "Agent task not found": ErrorCode.AGENT_TASK_NOT_FOUND,
    "Agent task not found or already finished": ErrorCode.AGENT_TASK_NOT_FOUND,
    "Agent task is not waiting for user approval": ErrorCode.AGENT_TASK_NOT_WAITING_APPROVAL,
}


def _normalize_http_detail(exc: HTTPException) -> tuple[str, str, bool, dict[str, Any]]:
    detail = exc.detail
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or "Request failed")
        details = {key: value for key, value in detail.items() if key not in {"code", "message", "detail", "retryable"}}
        return (
            str(detail.get("code") or fallback_code_for_status(exc.status_code)),
            message,
            bool(detail.get("retryable", False)),
            details,
        )
    message = str(detail or "Request failed")
    return (
        str(_KNOWN_MESSAGE_CODES.get(message) or fallback_code_for_status(exc.status_code)),
        message,
        exc.status_code in {429, 500, 502, 503, 504},
        {},
    )


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.to_detail().model_dump(mode="json")},
    )


async def http_exception_handler(_request: Request, exc: HTTPException | StarletteHTTPException) -> JSONResponse:
    code, message, retryable, details = _normalize_http_detail(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code=code, message=message, retryable=retryable, details=details),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_payload(
            code=ErrorCode.VALIDATION_ERROR,
            message="请求参数校验失败",
            details={"errors": exc.errors()},
        ),
    )


async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_payload(code=ErrorCode.BAD_REQUEST, message=str(exc)),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled request error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload(
            code=ErrorCode.INTERNAL_ERROR,
            message="服务内部错误，请稍后重试",
            retryable=True,
            details={"error_type": type(exc).__name__},
        ),
    )


def register_error_handlers(app) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = ["register_error_handlers"]
