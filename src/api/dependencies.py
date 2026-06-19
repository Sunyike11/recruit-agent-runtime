from pathlib import Path
from typing import Any, Callable, Mapping

from src.api.schemas import CreateMatchingTaskRequest, validate_tenant_id
from src.integration.production_skill_graph import ProductionSkillGraphConfig, build_production_skill_graph_runner
from src.mcp.gateway import build_candidate_mcp_retrieve_callable
from src.runtime.entry import (
    RuntimeEntryConfig,
    RuntimeEntryHarness,
    build_default_graph_runner,
)
from src.runtime.sqlite_store import SQLiteRuntimeStore
from src.runtime.variant_runner import build_real_retriever_callable


DEFAULT_DB_PATH = "storage/sqlite/recruit_api_runtime.sqlite"


def build_runtime_store(db_path: str | None = None) -> SQLiteRuntimeStore:
    return SQLiteRuntimeStore(db_path or DEFAULT_DB_PATH)


def get_tenant_id(value: str | None) -> str:
    try:
        return validate_tenant_id(value or "")
    except ValueError as exc:
        from src.api.errors import InvalidTenant

        raise InvalidTenant("Invalid or missing X-Tenant-ID") from exc


def build_runtime_submitter(store: Any) -> Callable[[CreateMatchingTaskRequest, str], Any]:
    default_runner = build_default_graph_runner()

    def submit(request: CreateMatchingTaskRequest, tenant_id: str):
        real_retriever = None
        retrieve_callable = None
        if request.candidate_source == "mcp":
            retrieve_callable = build_candidate_mcp_retrieve_callable(
                direct_fallback_callable=None,
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
