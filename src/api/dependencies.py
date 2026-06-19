from pathlib import Path
import time
from typing import Any, Callable, Mapping

from src.api.schemas import CreateMatchingTaskRequest, validate_tenant_id
from src.domain.candidate_management import (
    CandidateSQLiteStore,
    ResumeBlobStore,
    ResumeEvidenceRecord,
    profile_to_index_document,
)
from src.integration.production_skill_graph import ProductionSkillGraphConfig, build_production_skill_graph_runner
from src.mcp.gateway import CandidateMCPGatewayConfig, build_candidate_mcp_retrieve_callable
from src.runtime.entry import (
    RuntimeEntryResult,
    RuntimeEntryConfig,
    RuntimeEntryHarness,
    build_default_graph_runner,
)
from src.runtime.event_envelope import build_runtime_event_payload
from src.runtime.models import TaskStatus
from src.runtime.sqlite_store import SQLiteRuntimeStore
from src.runtime.variant_runner import build_real_retriever_callable
from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutionRecorder, SkillExecutor
from src.skills.registry import SkillRegistry
from src.skills.resume_ingestion import EvidenceExtractSkill, ResumeParseSkill


DEFAULT_DB_PATH = "storage/sqlite/recruit_api_runtime.sqlite"
DEFAULT_BLOB_DIR = "storage/resume_blobs"


def build_runtime_store(db_path: str | None = None) -> SQLiteRuntimeStore:
    return SQLiteRuntimeStore(db_path or DEFAULT_DB_PATH)


def build_candidate_store(db_path: str | None = None) -> CandidateSQLiteStore:
    return CandidateSQLiteStore(db_path or DEFAULT_DB_PATH)


def build_resume_blob_store(root_dir: str | None = None) -> ResumeBlobStore:
    return ResumeBlobStore(root_dir or DEFAULT_BLOB_DIR)


def get_tenant_id(value: str | None) -> str:
    try:
        return validate_tenant_id(value or "")
    except ValueError as exc:
        from src.api.errors import InvalidTenant

        raise InvalidTenant("Invalid or missing X-Tenant-ID") from exc


def build_runtime_submitter(store: Any) -> Callable[[CreateMatchingTaskRequest, str], Any]:
    default_runner = build_default_graph_runner()
    managed_db_path = str(getattr(store, "db_path", DEFAULT_DB_PATH))

    def submit(request: CreateMatchingTaskRequest, tenant_id: str):
        real_retriever = None
        retrieve_callable = None
        if request.candidate_source == "mcp":
            retrieve_callable = build_candidate_mcp_retrieve_callable(
                direct_fallback_callable=None,
                provider_mode="managed",
                db_path=managed_db_path,
                config=CandidateMCPGatewayConfig(
                    provider_mode="managed",
                    db_path=managed_db_path,
                    tenant_id=tenant_id,
                    access_scope="managed_candidates",
                ),
            )
        else:
            real_retriever = build_real_retriever_callable()
            retrieve_callable = real_retriever
        production_runner = build_production_skill_graph_runner(
            ProductionSkillGraphConfig(
                enabled=True,
                allow_planner_fallback=False,
                use_real_retriever=True,
                use_candidate_profile_preview=True,
                candidate_source=request.candidate_source,
                rollback_on_failure=bool(request.allow_legacy_fallback),
                summary_only=True,
            ),
            retrieve_callable=retrieve_callable,
        ).run
        return RuntimeEntryHarness().run(
            request.jd_text,
            default_runner=default_runner,
            production_skill_graph_runner=production_runner,
            store=store,
            config=RuntimeEntryConfig(
                graph_mode="skill",
                legacy_fallback_enabled=bool(request.allow_legacy_fallback),
                db_path=None,
                summary_only=True,
                metadata={
                    "api": True,
                    "tenant_id": tenant_id,
                    "candidate_source": request.candidate_source,
                    "metadata_keys": sorted(str(key) for key in request.metadata.keys()),
                    "summary_only": True,
                },
            ),
        )

    return submit


def build_ingestion_submitter(
    *,
    runtime_store: Any,
    candidate_store: CandidateSQLiteStore,
    blob_store: ResumeBlobStore,
) -> Callable[[Any, str], RuntimeEntryResult]:
    registry = SkillRegistry()
    registry.register(ResumeParseSkill())
    registry.register(EvidenceExtractSkill())

    def submit(request: Any, tenant_id: str) -> RuntimeEntryResult:
        session = runtime_store.create_session(
            metadata={
                "mode": "candidate_ingestion",
                "tenant_id": tenant_id,
                "summary_only": True,
            }
        )
        task = runtime_store.create_task(
            session.session_id,
            input_payload={
                "task_type": "candidate_ingestion",
                "workflow_type": "candidate_ingestion",
                "candidate_id": request.candidate_id,
                "resume_version_id": request.resume_version_id,
                "summary_only": True,
            },
        )
        candidate_store.update_resume_status(
            tenant_id=tenant_id,
            resume_version_id=request.resume_version_id,
            status="queued",
        )
        runtime_store.update_task_status(task.task_id, TaskStatus.RUNNING)
        runtime_store.append_event(
            "workflow_started",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=_workflow_payload(session, task, "started", request),
        )
        recorder = SkillExecutionRecorder(runtime_store)
        executor = SkillExecutor(registry, recorder=recorder)
        context = SkillExecutionContext(
            session_id=session.session_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            metadata={
                "graph_mode": "skill",
                "runner_used": "candidate_ingestion",
                "node_name": "candidate_ingestion",
                "summary_only": True,
            },
        )
        try:
            candidate_store.update_resume_status(tenant_id=tenant_id, resume_version_id=request.resume_version_id, status="parsing")
            parse_started = time.perf_counter()
            content = blob_store.read_bytes(request.storage_key)
            parse_result = executor.execute(
                "resume_parse",
                {
                    "candidate_id": request.candidate_id,
                    "resume_version_id": request.resume_version_id,
                    "filename": request.original_filename_safe,
                    "media_type": request.media_type,
                    "content_bytes": content,
                    "summary_only": True,
                },
                context=context,
            )
            parse_ms = round((time.perf_counter() - parse_started) * 1000, 3)
            if not parse_result.success:
                raise RuntimeError(parse_result.error or "ResumeParseFailed")

            candidate_store.update_resume_status(tenant_id=tenant_id, resume_version_id=request.resume_version_id, status="extracting_evidence")
            evidence_started = time.perf_counter()
            parsed = dict(parse_result.output or {})
            evidence_result = executor.execute(
                "evidence_extract",
                {
                    **parsed,
                    "tenant_id": tenant_id,
                    "summary_only": True,
                },
                context=context,
            )
            evidence_ms = round((time.perf_counter() - evidence_started) * 1000, 3)
            if not evidence_result.success:
                raise RuntimeError(evidence_result.error or "EvidenceExtractionFailed")
            output = dict(evidence_result.output or {})
            evidence_records = [
                ResumeEvidenceRecord(
                    evidence_id=item["evidence_id"],
                    candidate_id=item["candidate_id"],
                    resume_version_id=item["resume_version_id"],
                    tenant_id=tenant_id,
                    field_name=item["field_name"],
                    evidence_type=item["evidence_type"],
                    safe_summary=item.get("safe_summary") or item.get("summary") or "",
                    source_locator=item.get("source_locator") or "",
                    provenance=dict(item.get("provenance") or {}),
                    created_at=item.get("created_at") or "",
                )
                for item in output.get("evidence") or []
            ]
            profile = dict(output.get("profile") or {})

            candidate_store.update_resume_status(tenant_id=tenant_id, resume_version_id=request.resume_version_id, status="indexing")
            index_started = time.perf_counter()
            profile_record = candidate_store.save_profile_and_evidence(
                tenant_id=tenant_id,
                candidate_id=request.candidate_id,
                resume_version_id=request.resume_version_id,
                profile=profile,
                evidence=evidence_records,
            )
            profile["profile_version_id"] = profile_record.profile_version_id
            document, search_text = profile_to_index_document(profile, evidence_records)
            document["resume_version_id"] = request.resume_version_id
            document["profile_version_id"] = profile_record.profile_version_id
            candidate_store.upsert_index_document(
                tenant_id=tenant_id,
                candidate_id=request.candidate_id,
                resume_version_id=request.resume_version_id,
                profile_version_id=profile_record.profile_version_id,
                document=document,
                search_text=search_text,
            )
            candidate_store.update_resume_status(tenant_id=tenant_id, resume_version_id=request.resume_version_id, status="ready")
            candidate_store.activate_version(
                tenant_id=tenant_id,
                candidate_id=request.candidate_id,
                resume_version_id=request.resume_version_id,
                profile_version_id=profile_record.profile_version_id,
            )
            index_ms = round((time.perf_counter() - index_started) * 1000, 3)
            summary = {
                "task_type": "candidate_ingestion",
                "workflow_type": "candidate_ingestion",
                "candidate_id": request.candidate_id,
                "resume_version_id": request.resume_version_id,
                "profile_version_id": profile_record.profile_version_id,
                "status": "completed",
                "parse_duration_ms": parse_ms,
                "evidence_extract_duration_ms": evidence_ms,
                "index_duration_ms": index_ms,
                "evidence_count": len(evidence_records),
                "active_version_switched": True,
                "summary_only": True,
            }
            runtime_store.append_event(
                "index_completed",
                session_id=session.session_id,
                task_id=task.task_id,
                payload=_workflow_payload(session, task, "completed", request, extra={"duration_ms": index_ms}),
            )
            runtime_store.append_event(
                "workflow_completed",
                session_id=session.session_id,
                task_id=task.task_id,
                payload=_workflow_payload(session, task, "completed", request),
            )
            runtime_store.update_task_status(task.task_id, TaskStatus.COMPLETED, result=summary)
            return RuntimeEntryResult(
                status="ok",
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                runner_used="candidate_ingestion",
                task_status="completed",
                event_count=len(runtime_store.list_events(task_id=task.task_id)),
                output_summary=summary,
            )
        except Exception as exc:
            error_type = _ingestion_error_type(exc)
            candidate_store.update_resume_status(
                tenant_id=tenant_id,
                resume_version_id=request.resume_version_id,
                status="failed",
                error_type=error_type,
            )
            runtime_store.append_event(
                "workflow_failed",
                session_id=session.session_id,
                task_id=task.task_id,
                payload=_workflow_payload(session, task, "failed", request, extra={"error_type": error_type}),
            )
            runtime_store.update_task_status(task.task_id, TaskStatus.FAILED, error=error_type)
            return RuntimeEntryResult(
                status="failed",
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                runner_used="candidate_ingestion",
                task_status="failed",
                event_count=len(runtime_store.list_events(task_id=task.task_id)),
                output_summary={
                    "task_type": "candidate_ingestion",
                    "candidate_id": request.candidate_id,
                    "resume_version_id": request.resume_version_id,
                    "status": "failed",
                    "error_type": error_type,
                    "summary_only": True,
                },
                error_type=error_type,
            )

    return submit


def _workflow_payload(session, task, status: str, request: Any, extra: Mapping[str, Any] | None = None) -> dict:
    return build_runtime_event_payload(
        session_id=session.session_id,
        task_id=task.task_id,
        thread_id=task.thread_id,
        graph_mode="skill",
        runner_name="candidate_ingestion",
        node_name="candidate_ingestion",
        skill_name="",
        status=status,
        extra={
            "task_type": "candidate_ingestion",
            "workflow_type": "candidate_ingestion",
            "candidate_id": request.candidate_id,
            "resume_version_id": request.resume_version_id,
            "summary_only": True,
            **dict(extra or {}),
        },
    )


def _ingestion_error_type(exc: Exception) -> str:
    text = str(exc)
    for marker in [
        "UnsupportedMediaType",
        "FileTooLarge",
        "EmptyFile",
        "TextExtractionUnavailable",
        "ResumeParseFailed",
        "EvidenceExtractionFailed",
        "CandidateIndexFailed",
        "IngestionCancelled",
    ]:
        if marker in text:
            return marker
    return type(exc).__name__


def safe_task_summary(record) -> dict:
    result = record.result_summary if isinstance(record.result_summary, Mapping) else {}
    return {
        "task_id": record.task_id,
        "session_id": record.session_id,
        "runtime_task_id": record.runtime_task_id,
        "tenant_id": record.tenant_id,
        "status": record.status,
        "graph_mode": str(result.get("selected_graph_mode") or result.get("graph_mode") or "skill"),
        "candidate_source": record.candidate_source,
        "task_type": getattr(record, "task_type", "matching"),
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "candidate_count": int(result.get("candidate_count") or 0),
        "report_count": int(result.get("report_count") or 0),
        "error_type": record.error_type or str(result.get("error_type") or ""),
        "fallback_attempted": bool(result.get("fallback_attempted", False)),
        "fallback_succeeded": bool(result.get("fallback_succeeded", False)),
        "cancel_requested": bool(record.cancel_requested),
        "summary_only": True,
    }
