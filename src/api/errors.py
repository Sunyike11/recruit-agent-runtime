from typing import Any, Dict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    status_code = 400
    error_type = "ValidationError"
    retryable = False

    def __init__(self, message: str = "", *, status_code: int | None = None):
        super().__init__(message or self.error_type)
        if status_code is not None:
            self.status_code = status_code


class InvalidTenant(APIError):
    status_code = 400
    error_type = "InvalidTenant"


class TaskNotFound(APIError):
    status_code = 404
    error_type = "TaskNotFound"


class TenantAccessDenied(APIError):
    status_code = 403
    error_type = "TenantAccessDenied"


class IdempotencyConflict(APIError):
    status_code = 409
    error_type = "IdempotencyConflict"


class QueueFull(APIError):
    status_code = 503
    error_type = "QueueFull"
    retryable = True


class TaskAlreadyTerminal(APIError):
    status_code = 409
    error_type = "TaskAlreadyTerminal"


class ServiceNotReady(APIError):
    status_code = 503
    error_type = "ServiceNotReady"
    retryable = True


class RuntimeExecutionFailed(APIError):
    status_code = 500
    error_type = "RuntimeExecutionFailed"


def error_payload(error_type: str, message: str, request_id: str = "", *, retryable: bool = False) -> Dict[str, Any]:
    return {
        "error": {
            "type": str(error_type),
            "message": str(message),
            "request_id": str(request_id or ""),
            "retryable": bool(retryable),
        }
    }


async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            exc.error_type,
            str(exc),
            request_id=request.headers.get("X-Request-ID", ""),
            retryable=exc.retryable,
        ),
    )


async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            "ValidationError" if exc.status_code < 500 else "RuntimeExecutionFailed",
            str(exc.detail),
            request_id=request.headers.get("X-Request-ID", ""),
            retryable=exc.status_code in {429, 503},
        ),
    )
