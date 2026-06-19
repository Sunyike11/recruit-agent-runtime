import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,62}$")
ALLOWED_METADATA_KEYS = {"request_source", "trace_label", "experiment", "notes"}


class CreateMatchingTaskRequest(BaseModel):
    jd_text: str = Field(..., min_length=1, max_length=8000)
    candidate_source: Literal["direct", "mcp"] = "direct"
    allow_legacy_fallback: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        invalid = sorted(str(key) for key in value.keys() if str(key) not in ALLOWED_METADATA_KEYS)
        if invalid:
            raise ValueError("metadata_key_not_allowed")
        safe = {}
        for key, item in value.items():
            text = str(item)
            if len(text) > 240:
                raise ValueError("metadata_value_too_long")
            if "/" in text or "\\" in text:
                raise ValueError("metadata_path_like_value_denied")
            safe[str(key)] = item
        return safe


class CreateMatchingTaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    created: bool
    idempotency_replayed: bool
    summary_only: bool = True


class TaskSummaryResponse(BaseModel):
    task_id: str
    session_id: str
    runtime_task_id: str = ""
    tenant_id: str
    status: str
    graph_mode: str = "skill"
    candidate_source: str = "direct"
    task_type: str = "matching"
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    candidate_count: int = 0
    report_count: int = 0
    error_type: str = ""
    fallback_attempted: bool = False
    fallback_succeeded: bool = False
    cancel_requested: bool = False
    summary_only: bool = True


class EventsResponse(BaseModel):
    task_id: str
    events: List[Dict[str, Any]]
    next_cursor: str = ""
    summary_only: bool = True


class FeedbackRequest(BaseModel):
    feedback_type: Literal["approve", "reject", "correction", "comment"]
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: str = Field(default="", max_length=1000)
    candidate_id: str = Field(default="", max_length=80)


class FeedbackResponse(BaseModel):
    feedback_id: str
    task_id: str
    feedback_type: str
    summary_only: bool = True


class CancelResponse(BaseModel):
    task_id: str
    cancel_requested: bool
    status: str
    already_terminal: bool
    summary_only: bool = True


class HealthResponse(BaseModel):
    status: str
    summary_only: bool = True


class CreateCandidateRequest(BaseModel):
    external_ref: str = Field(default="", max_length=120)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return CreateMatchingTaskRequest.validate_metadata(value)


class CreateCandidateResponse(BaseModel):
    candidate_id: str
    status: str
    created: bool
    idempotency_replayed: bool
    created_at: str
    summary_only: bool = True


class CandidateSummaryResponse(BaseModel):
    candidate_id: str
    tenant_id: str
    status: str
    active_resume_version_id: str = ""
    active_profile_version_id: str = ""
    resume_version_count: int = 0
    created_at: str
    updated_at: str
    summary_only: bool = True


class ResumeVersionUploadResponse(BaseModel):
    candidate_id: str
    resume_version_id: str
    version_number: int
    content_hash_prefix: str
    status: str
    ingestion_task_id: str
    created: bool
    duplicate_content: bool
    summary_only: bool = True


class ResumeVersionSummaryResponse(BaseModel):
    candidate_id: str
    resume_version_id: str
    version_number: int
    content_hash_prefix: str
    original_filename_safe: str
    media_type: str
    file_size: int
    status: str
    parser_version: str
    profile_version: str
    index_version: str
    created_at: str
    ready_at: str = ""
    supersedes_version_id: str = ""
    error_type: str = ""
    ingestion_task_id: str = ""
    summary_only: bool = True


class ResumeVersionListResponse(BaseModel):
    candidate_id: str
    resume_versions: List[ResumeVersionSummaryResponse]
    summary_only: bool = True


class CandidateProfileResponse(BaseModel):
    candidate_id: str
    profile_version_id: str
    resume_version_id: str
    schema_version: str
    profile: Dict[str, Any]
    summary_only: bool = True


def validate_tenant_id(value: str) -> str:
    tenant = str(value or "").strip()
    if not tenant or not TENANT_RE.match(tenant):
        raise ValueError("invalid_tenant_id")
    return tenant
