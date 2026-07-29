"""Unified API error payloads and typed application exceptions."""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import status
from pydantic import BaseModel, Field

from app.logging_config import request_id_var


class ErrorCode(StrEnum):
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    AUTH_TOKEN_INVALID = "AUTH_TOKEN_INVALID"
    AUTH_USER_NOT_FOUND = "AUTH_USER_NOT_FOUND"
    AUTH_PASSWORD_INVALID = "AUTH_PASSWORD_INVALID"
    AUTH_LOGIN_RATE_LIMITED = "AUTH_LOGIN_RATE_LIMITED"
    AUTH_USERNAME_EXISTS = "AUTH_USERNAME_EXISTS"
    AUTH_ADMIN_USERNAME_RESERVED = "AUTH_ADMIN_USERNAME_RESERVED"

    KB_NOT_FOUND = "KB_NOT_FOUND"
    KB_FORBIDDEN = "KB_FORBIDDEN"
    KB_ROLE_REQUIRED = "KB_ROLE_REQUIRED"
    KB_MEMBER_EXISTS = "KB_MEMBER_EXISTS"
    KB_MEMBER_NOT_FOUND = "KB_MEMBER_NOT_FOUND"
    KB_OWNER_IMMUTABLE = "KB_OWNER_IMMUTABLE"

    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_FORBIDDEN = "DOCUMENT_FORBIDDEN"
    DOCUMENT_TYPE_UNSUPPORTED = "DOCUMENT_TYPE_UNSUPPORTED"
    DOCUMENT_FILE_TOO_LARGE = "DOCUMENT_FILE_TOO_LARGE"
    DOCUMENT_FILE_EMPTY = "DOCUMENT_FILE_EMPTY"
    DOCUMENT_SIGNATURE_INVALID = "DOCUMENT_SIGNATURE_INVALID"
    DOCUMENT_ALREADY_INDEXING = "DOCUMENT_ALREADY_INDEXING"
    DOCUMENT_SOURCE_MISSING = "DOCUMENT_SOURCE_MISSING"
    DOCUMENT_DELETE_VECTOR_FAILED = "DOCUMENT_DELETE_VECTOR_FAILED"
    DOCUMENT_INDEX_QUEUE_UNAVAILABLE = "DOCUMENT_INDEX_QUEUE_UNAVAILABLE"

    RAG_CONVERSATION_NOT_FOUND = "RAG_CONVERSATION_NOT_FOUND"
    RAG_SCOPE_MISMATCH = "RAG_SCOPE_MISMATCH"
    RAG_RETRIEVAL_UNAVAILABLE = "RAG_RETRIEVAL_UNAVAILABLE"
    RAG_STREAM_FAILED = "RAG_STREAM_FAILED"
    RAG_QUERY_REWRITE_DEGRADED = "RAG_QUERY_REWRITE_DEGRADED"
    RAG_BM25_DEGRADED = "RAG_BM25_DEGRADED"
    RAG_GENERATION_DEGRADED = "RAG_GENERATION_DEGRADED"

    AGENT_TASK_NOT_FOUND = "AGENT_TASK_NOT_FOUND"
    AGENT_TASK_NOT_WAITING_APPROVAL = "AGENT_TASK_NOT_WAITING_APPROVAL"
    AGENT_QUEUE_DEGRADED = "AGENT_QUEUE_DEGRADED"
    AGENT_BUDGET_EXCEEDED = "AGENT_BUDGET_EXCEEDED"
    AGENT_CANCELLED = "AGENT_CANCELLED"

    VALIDATION_ERROR = "VALIDATION_ERROR"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"
    UNAUTHORIZED = "UNAUTHORIZED"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str = Field(default_factory=request_id_var.get)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class AppError(Exception):
    """Typed business error converted to the public API error envelope."""

    def __init__(
        self,
        *,
        code: ErrorCode | str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            details=self.details,
        )


class DegradationWarning(BaseModel):
    code: str
    message: str
    retryable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


def error_payload(
    *,
    code: ErrorCode | str,
    message: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "detail": ErrorDetail(
            code=str(code),
            message=message,
            retryable=retryable,
            details=details or {},
        ).model_dump(mode="json")
    }


def warning_payload(
    code: ErrorCode | str,
    message: str,
    *,
    retryable: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return DegradationWarning(
        code=str(code),
        message=message,
        retryable=retryable,
        details=details or {},
    ).model_dump(mode="json")


def fallback_code_for_status(status_code: int) -> ErrorCode:
    return {
        status.HTTP_400_BAD_REQUEST: ErrorCode.BAD_REQUEST,
        status.HTTP_401_UNAUTHORIZED: ErrorCode.UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN: ErrorCode.FORBIDDEN,
        status.HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND,
        status.HTTP_409_CONFLICT: ErrorCode.CONFLICT,
        status.HTTP_429_TOO_MANY_REQUESTS: ErrorCode.RATE_LIMITED,
        status.HTTP_503_SERVICE_UNAVAILABLE: ErrorCode.DEPENDENCY_UNAVAILABLE,
    }.get(status_code, ErrorCode.INTERNAL_ERROR if status_code >= 500 else ErrorCode.BAD_REQUEST)


__all__ = [
    "AppError",
    "DegradationWarning",
    "ErrorCode",
    "ErrorDetail",
    "error_payload",
    "fallback_code_for_status",
    "warning_payload",
]
