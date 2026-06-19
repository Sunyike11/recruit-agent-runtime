import hashlib
from typing import Optional

from fastapi import APIRouter, File, Header, Query, Request, UploadFile

from src.api.dependencies import get_tenant_id
from src.api.errors import APIError, IdempotencyConflict, TenantAccessDenied
from src.api.schemas import (
    CandidateProfileResponse,
    CandidateSummaryResponse,
    CreateCandidateRequest,
    CreateCandidateResponse,
    ResumeVersionListResponse,
    ResumeVersionSummaryResponse,
    ResumeVersionUploadResponse,
)
from src.domain.candidate_management import (
    MAX_UPLOAD_BYTES,
    compute_sha256,
    infer_media_type,
    safe_filename,
)


router = APIRouter()


def _tenant(value: Optional[str]) -> str:
    return get_tenant_id(value)


def _candidate_store(request: Request):
    return request.app.state.candidate_store


def _blob_store(request: Request):
    return request.app.state.resume_blob_store


def _manager(request: Request):
    return request.app.state.task_manager


def _get_candidate_or_deny(store, tenant_id: str, candidate_id: str):
    try:
        return store.get_candidate(tenant_id=tenant_id, candidate_id=candidate_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this candidate") from exc


@router.post("/candidates", response_model=CreateCandidateResponse)
async def create_candidate(
    payload: CreateCandidateRequest,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    tenant_id = _tenant(x_tenant_id)
    if not idempotency_key or len(idempotency_key) > 120:
        raise APIError("Missing or invalid Idempotency-Key", status_code=400)
    fingerprint = hashlib.sha256(
        f"{payload.external_ref}|{sorted(payload.metadata.keys())}".encode("utf-8")
    ).hexdigest()
    store = _candidate_store(request)
    try:
        created, object_id = store.remember_idempotency(
            tenant_id=tenant_id,
            key=idempotency_key,
            fingerprint=fingerprint,
            object_id="pending",
            object_type="candidate",
        )
    except ValueError as exc:
        raise IdempotencyConflict("Idempotency key already used for a different request") from exc
    if not created and object_id != "pending":
        candidate = store.get_candidate(tenant_id=tenant_id, candidate_id=object_id)
        return CreateCandidateResponse(
            candidate_id=candidate.candidate_id,
            status=candidate.status,
            created=False,
            idempotency_replayed=True,
            created_at=candidate.created_at,
        )
    candidate = store.create_candidate(
        tenant_id=tenant_id,
        external_ref=payload.external_ref,
        metadata=payload.metadata,
    )
    # Replace the pending idempotency mapping with a stable mapping by using the same key.
    with store._connect() as conn:  # repository-local compatibility update
        conn.execute(
            "UPDATE candidate_idempotency SET object_id = ? WHERE tenant_id = ? AND idempotency_key = ?",
            (candidate.candidate_id, tenant_id, idempotency_key),
        )
    _manager(request).metrics["candidate_created_count"] += 1
    return CreateCandidateResponse(
        candidate_id=candidate.candidate_id,
        status=candidate.status,
        created=True,
        idempotency_replayed=False,
        created_at=candidate.created_at,
    )


@router.get("/candidates/{candidate_id}", response_model=CandidateSummaryResponse)
async def get_candidate(
    candidate_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    store = _candidate_store(request)
    candidate = _get_candidate_or_deny(store, tenant_id, candidate_id)
    versions = store.list_resume_versions(tenant_id=tenant_id, candidate_id=candidate_id)
    return CandidateSummaryResponse(
        candidate_id=candidate.candidate_id,
        tenant_id=candidate.tenant_id,
        status=candidate.status,
        active_resume_version_id=candidate.active_resume_version_id,
        active_profile_version_id=candidate.active_profile_version_id,
        resume_version_count=len(versions),
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )


@router.post("/candidates/{candidate_id}/resume-versions", response_model=ResumeVersionUploadResponse, status_code=202)
async def upload_resume_version(
    candidate_id: str,
    request: Request,
    file: UploadFile = File(...),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    tenant_id = _tenant(x_tenant_id)
    if not idempotency_key or len(idempotency_key) > 120:
        raise APIError("Missing or invalid Idempotency-Key", status_code=400)
    filename = safe_filename(file.filename or "resume.txt")
    try:
        media_type, extension = infer_media_type(filename, file.content_type or "")
    except ValueError as exc:
        raise APIError("UnsupportedMediaType", status_code=415) from exc
    data = await file.read()
    if not data:
        raise APIError("EmptyFile", status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise APIError("FileTooLarge", status_code=413)
    content_hash = compute_sha256(data)
    store = _candidate_store(request)
    blob_store = _blob_store(request)
    upload_fingerprint = hashlib.sha256(f"{candidate_id}|{content_hash}|{len(data)}".encode("utf-8")).hexdigest()
    try:
        upload_created, mapped_version_id = store.remember_idempotency(
            tenant_id=tenant_id,
            key=idempotency_key,
            fingerprint=upload_fingerprint,
            object_id="pending",
            object_type="resume_version",
        )
    except ValueError as exc:
        raise IdempotencyConflict("Idempotency key already used for a different request") from exc
    if not upload_created and mapped_version_id != "pending":
        existing = store.get_resume_version(
            tenant_id=tenant_id,
            candidate_id=candidate_id,
            resume_version_id=mapped_version_id,
        )
        _manager(request).metrics["duplicate_upload_count"] += 1
        return ResumeVersionUploadResponse(
            candidate_id=candidate_id,
            resume_version_id=existing.resume_version_id,
            version_number=existing.version_number,
            content_hash_prefix=existing.content_hash[:12],
            status=existing.status,
            ingestion_task_id=existing.ingestion_task_id,
            created=False,
            duplicate_content=True,
        )
    storage_key = blob_store.storage_key(
        tenant_id=tenant_id,
        candidate_id=candidate_id,
        content_hash=content_hash,
        extension=extension,
    )
    record, created = store.create_resume_version(
        tenant_id=tenant_id,
        candidate_id=candidate_id,
        content_hash=content_hash,
        original_filename_safe=filename,
        media_type=media_type,
        file_size=len(data),
        storage_key=storage_key,
    )
    duplicate = not created
    if created:
        blob_store.put_bytes(storage_key=storage_key, data=data)
        ingestion_request = _IngestionRequest(
            candidate_id=candidate_id,
            resume_version_id=record.resume_version_id,
            content_hash=content_hash,
            file_size=len(data),
            storage_key=storage_key,
            original_filename_safe=filename,
            media_type=media_type,
        )
        task, replayed = await _manager(request).submit_ingestion(tenant_id, idempotency_key, ingestion_request)
        store.update_resume_status(
            tenant_id=tenant_id,
            resume_version_id=record.resume_version_id,
            status="queued",
            ingestion_task_id=task.task_id,
        )
        with store._connect() as conn:
            conn.execute(
                "UPDATE candidate_idempotency SET object_id = ? WHERE tenant_id = ? AND idempotency_key = ?",
                (record.resume_version_id, tenant_id, idempotency_key),
            )
        ingestion_task_id = task.task_id
    else:
        with store._connect() as conn:
            conn.execute(
                "UPDATE candidate_idempotency SET object_id = ? WHERE tenant_id = ? AND idempotency_key = ?",
                (record.resume_version_id, tenant_id, idempotency_key),
            )
        _manager(request).metrics["duplicate_upload_count"] += 1
        ingestion_task_id = record.ingestion_task_id
    return ResumeVersionUploadResponse(
        candidate_id=candidate_id,
        resume_version_id=record.resume_version_id,
        version_number=record.version_number,
        content_hash_prefix=record.content_hash[:12],
        status="queued" if created else record.status,
        ingestion_task_id=ingestion_task_id,
        created=created,
        duplicate_content=duplicate,
    )


@router.get("/candidates/{candidate_id}/resume-versions", response_model=ResumeVersionListResponse)
async def list_resume_versions(
    candidate_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        records = _candidate_store(request).list_resume_versions(
            tenant_id=tenant_id,
            candidate_id=candidate_id,
            limit=limit,
            offset=offset,
        )
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this candidate") from exc
    return ResumeVersionListResponse(
        candidate_id=candidate_id,
        resume_versions=[_resume_response(item) for item in records],
    )


@router.get("/candidates/{candidate_id}/resume-versions/{resume_version_id}", response_model=ResumeVersionSummaryResponse)
async def get_resume_version(
    candidate_id: str,
    resume_version_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        record = _candidate_store(request).get_resume_version(
            tenant_id=tenant_id,
            candidate_id=candidate_id,
            resume_version_id=resume_version_id,
        )
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this resume version") from exc
    return _resume_response(record)


@router.get("/candidates/{candidate_id}/profile", response_model=CandidateProfileResponse)
async def get_candidate_profile(
    candidate_id: str,
    request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    tenant_id = _tenant(x_tenant_id)
    try:
        profile = _candidate_store(request).get_profile(tenant_id=tenant_id, candidate_id=candidate_id)
    except KeyError as exc:
        raise TenantAccessDenied("Tenant cannot access this profile") from exc
    return CandidateProfileResponse(
        candidate_id=candidate_id,
        profile_version_id=profile.profile_version_id,
        resume_version_id=profile.resume_version_id,
        schema_version=profile.schema_version,
        profile=profile.profile,
    )


class _IngestionRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _resume_response(record) -> ResumeVersionSummaryResponse:
    return ResumeVersionSummaryResponse(
        candidate_id=record.candidate_id,
        resume_version_id=record.resume_version_id,
        version_number=record.version_number,
        content_hash_prefix=record.content_hash[:12],
        original_filename_safe=record.original_filename_safe,
        media_type=record.media_type,
        file_size=record.file_size,
        status=record.status,
        parser_version=record.parser_version,
        profile_version=record.profile_version,
        index_version=record.index_version,
        created_at=record.created_at,
        ready_at=record.ready_at,
        supersedes_version_id=record.supersedes_version_id,
        error_type=record.error_type,
        ingestion_task_id=record.ingestion_task_id,
    )
