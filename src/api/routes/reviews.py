from typing import Optional

from fastapi import APIRouter, Header, Query, Request

from src.api.dependencies import get_tenant_id
from src.api.errors import APIError, TenantAccessDenied
from src.api.schemas import (
    MemoryCandidateListResponse,
    MemoryCandidateResponse,
    MemoryListResponse,
    MemoryResponse,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewListResponse,
    ReviewResponse,
)


router = APIRouter()


def _tenant(value: Optional[str]) -> str:
    return get_tenant_id(value)


def _store(request: Request):
    return request.app.state.review_memory_store


def _manager(request: Request):
    return request.app.state.task_manager


def _runtime_store(request: Request):
    return request.app.state.runtime_store


@router.get("/reviews", response_model=ReviewListResponse)
async def list_reviews(
    request: Request,
    status: str = Query(default=""),
    review_type: str = Query(default=""),
    task_id: str = Query(default=""),
    candidate_id: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=200),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    reviews = _store(request).list_reviews(
        tenant_id=tenant_id,
        status=status,
        review_type=review_type,
        task_id=task_id,
        candidate_id=candidate_id,
        limit=limit,
    )
    return ReviewListResponse(reviews=[item.to_summary() for item in reviews])


@router.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    try:
        review = _store(request).get_review(tenant_id=_tenant(x_tenant_id), review_id=review_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this review") from exc
    return ReviewResponse(review=review.to_summary())


@router.post("/reviews/{review_id}/decision", response_model=ReviewDecisionResponse)
async def decide_review(
    review_id: str,
    payload: ReviewDecisionRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        decision, review, candidate = _store(request).decide_review(
            tenant_id=tenant_id,
            review_id=review_id,
            decision=payload.decision,
            correction=payload.correction,
            reason=payload.reason,
            promote_to_memory=payload.promote_to_memory,
            memory_candidate_type=payload.memory_candidate_type or "",
            expires_at=payload.expires_at or "",
            supersedes_memory_id=payload.supersedes_memory_id,
        )
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this review") from exc
    except ValueError as exc:
        raise APIError(str(exc), status_code=409) from exc
    _append_review_event(request, "review_decided", review.task_id, tenant_id, review.review_id, decision.decision_id, decision.decision)
    if candidate is not None:
        _append_review_event(
            request,
            "memory_candidate_created",
            review.task_id,
            tenant_id,
            review.review_id,
            decision.decision_id,
            decision.decision,
            memory_candidate_id=candidate.memory_candidate_id,
            memory_type=candidate.memory_type,
        )
    _manager(request).metrics["review_decision_count"] = _manager(request).metrics.get("review_decision_count", 0) + 1
    return ReviewDecisionResponse(
        decision=decision.to_summary(),
        review=review.to_summary(),
        memory_candidate=candidate.to_summary() if candidate is not None else None,
    )


@router.get("/memory-candidates", response_model=MemoryCandidateListResponse)
async def list_memory_candidates(
    request: Request,
    status: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=200),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    candidates = _store(request).list_memory_candidates(tenant_id=tenant_id, status=status, limit=limit)
    return MemoryCandidateListResponse(memory_candidates=[item.to_summary() for item in candidates])


@router.get("/memory-candidates/{memory_candidate_id}", response_model=MemoryCandidateResponse)
async def get_memory_candidate(
    memory_candidate_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    try:
        candidate = _store(request).get_memory_candidate(tenant_id=_tenant(x_tenant_id), memory_candidate_id=memory_candidate_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this memory candidate") from exc
    return MemoryCandidateResponse(memory_candidate=candidate.to_summary())


@router.post("/memory-candidates/{memory_candidate_id}/approve", response_model=MemoryResponse)
async def approve_memory_candidate(
    memory_candidate_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        memory = _store(request).approve_memory_candidate(tenant_id=tenant_id, memory_candidate_id=memory_candidate_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this memory candidate") from exc
    except ValueError as exc:
        raise APIError(str(exc), status_code=409) from exc
    _manager(request).metrics["memory_activation_count"] = _manager(request).metrics.get("memory_activation_count", 0) + 1
    _append_memory_event(request, "memory_activated", memory.source_task_id, tenant_id, memory.memory_id, memory.memory_type)
    return MemoryResponse(memory=memory.to_summary())


@router.post("/memory-candidates/{memory_candidate_id}/reject", response_model=MemoryCandidateResponse)
async def reject_memory_candidate(
    memory_candidate_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        candidate = _store(request).reject_memory_candidate(tenant_id=tenant_id, memory_candidate_id=memory_candidate_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this memory candidate") from exc
    except ValueError as exc:
        raise APIError(str(exc), status_code=409) from exc
    _append_memory_event(request, "memory_candidate_rejected", candidate.source_task_id, tenant_id, "", candidate.memory_type, memory_candidate_id=memory_candidate_id)
    return MemoryCandidateResponse(memory_candidate=candidate.to_summary())


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    request: Request,
    status: str = Query(default=""),
    memory_type: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=200),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    memories = _store(request).list_memories(tenant_id=tenant_id, status=status, memory_type=memory_type, limit=limit)
    return MemoryListResponse(memories=[item.to_summary() for item in memories])


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    try:
        memory = _store(request).get_memory(tenant_id=_tenant(x_tenant_id), memory_id=memory_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this memory") from exc
    return MemoryResponse(memory=memory.to_summary())


@router.post("/memories/{memory_id}/revoke", response_model=MemoryResponse)
async def revoke_memory(
    memory_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        memory = _store(request).revoke_memory(tenant_id=tenant_id, memory_id=memory_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this memory") from exc
    _manager(request).metrics["memory_revocation_count"] = _manager(request).metrics.get("memory_revocation_count", 0) + 1
    _append_memory_event(request, "memory_revoked", memory.source_task_id, tenant_id, memory.memory_id, memory.memory_type)
    return MemoryResponse(memory=memory.to_summary())


def _append_review_event(request: Request, event_type: str, task_id: str, tenant_id: str, review_id: str, decision_id: str = "", decision: str = "", **extra):
    _append_safe_event(
        request,
        event_type,
        task_id,
        {
            "tenant_id": tenant_id,
            "review_id": review_id,
            "decision_id": decision_id,
            "decision": decision,
            **extra,
        },
    )


def _append_memory_event(request: Request, event_type: str, task_id: str, tenant_id: str, memory_id: str, memory_type: str, **extra):
    _append_safe_event(
        request,
        event_type,
        task_id,
        {
            "tenant_id": tenant_id,
            "memory_id": memory_id,
            "memory_type": memory_type,
            **extra,
        },
    )


def _append_safe_event(request: Request, event_type: str, api_task_id: str, payload: dict):
    try:
        record = _manager(request).tasks.get(api_task_id)
        runtime_task_id = getattr(record, "runtime_task_id", "") if record is not None else ""
        session_id = getattr(record, "runtime_session_id", "") or getattr(record, "session_id", "")
        target_task_id = runtime_task_id or api_task_id
        _runtime_store(request).append_event(
            event_type,
            session_id=session_id,
            task_id=target_task_id,
            payload={
                "status": "completed",
                "summary_only": True,
                **payload,
            },
        )
    except Exception:
        return
