from typing import Optional

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_tenant_id, safe_task_summary
from src.api.errors import APIError
from src.api.schemas import (
    CancelResponse,
    CreateMatchingTaskRequest,
    CreateMatchingTaskResponse,
    EventsResponse,
    FeedbackRequest,
    FeedbackResponse,
    TaskSummaryResponse,
)
from src.api.sse import task_event_stream


router = APIRouter()


def _manager(request: Request):
    return request.app.state.task_manager


def _tenant(x_tenant_id: Optional[str]) -> str:
    return get_tenant_id(x_tenant_id)


@router.post("/matching/tasks", response_model=CreateMatchingTaskResponse)
async def create_matching_task(
    payload: CreateMatchingTaskRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    tenant_id = _tenant(x_tenant_id)
    if not idempotency_key or len(idempotency_key) > 120:
        raise APIError("Missing or invalid Idempotency-Key", status_code=400)
    record, replayed = await _manager(request).submit(tenant_id, idempotency_key, payload)
    return CreateMatchingTaskResponse(
        task_id=record.task_id,
        session_id=record.session_id,
        status="queued" if not replayed else record.status,
        created=not replayed,
        idempotency_replayed=replayed,
    )


@router.get("/tasks/{task_id}", response_model=TaskSummaryResponse)
async def get_task(
    task_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    record = _manager(request).get_task(_tenant(x_tenant_id), task_id)
    return TaskSummaryResponse(**safe_task_summary(record))


@router.get("/tasks/{task_id}/events", response_model=EventsResponse)
async def get_task_events(
    task_id: str,
    request: Request,
    after_event_id: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=200),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    events = _manager(request).events_for_task(tenant_id, task_id, after_event_id=after_event_id, limit=limit)
    next_cursor = str(events[-1].get("event_id") if events else after_event_id)
    return EventsResponse(task_id=task_id, events=events, next_cursor=next_cursor)


@router.get("/tasks/{task_id}/stream")
async def stream_task_events(
    task_id: str,
    request: Request,
    last_event_id_query: str = Query(default="", alias="last_event_id"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    last_event_id_header: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    _manager(request).get_task(tenant_id, task_id)
    last_event_id = last_event_id_header or last_event_id_query
    return StreamingResponse(
        task_event_stream(_manager(request), tenant_id=tenant_id, task_id=task_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/tasks/{task_id}/feedback", response_model=FeedbackResponse)
async def add_feedback(
    task_id: str,
    payload: FeedbackRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    feedback = _manager(request).add_feedback(
        tenant_id,
        task_id,
        payload.feedback_type,
        {
            "rating": payload.rating,
            "comment_length": len(payload.comment or ""),
            "candidate_id": payload.candidate_id,
            "summary_only": True,
        },
    )
    return FeedbackResponse(
        feedback_id=str(feedback["feedback_id"]),
        task_id=task_id,
        feedback_type=str(feedback["feedback_type"]),
    )


@router.post("/tasks/{task_id}/cancel", response_model=CancelResponse)
async def cancel_task(
    task_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    before = _manager(request).get_task(tenant_id, task_id)
    already_terminal = before.status in {"completed", "completed_with_fallback", "failed", "cancelled"}
    record = _manager(request).cancel(tenant_id, task_id)
    return CancelResponse(
        task_id=task_id,
        cancel_requested=bool(record.cancel_requested),
        status=record.status,
        already_terminal=already_terminal,
    )
