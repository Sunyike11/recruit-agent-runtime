from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class RuntimeTaskInspection:
    task_id: str
    session_id: str
    thread_id: str
    task_status: str
    runner_used: str
    event_count: int
    event_types: List[str]
    error_type: str
    error_hint: str
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    timeline_summary: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeInspector:
    """Read-only summary inspector for runtime tasks and timelines."""

    def inspect_task(self, task_id: str, store) -> RuntimeTaskInspection:
        task = store.get_task(task_id)
        events = self.inspect_events(task_id, store)
        output = _safe_output_summary(task.result)
        runner_used = _runner_used(events, output)
        return RuntimeTaskInspection(
            task_id=task.task_id,
            session_id=task.session_id,
            thread_id=task.thread_id,
            task_status=_status_value(task.status),
            runner_used=runner_used,
            event_count=len(events),
            event_types=[event["event_type"] for event in events],
            error_type=str(task.error or output.get("error_type") or ""),
            error_hint=str(output.get("error_hint") or _event_error_hint(events) or ""),
            input_summary=_input_summary(task.input),
            output_summary=output,
            timeline_summary=events,
            metadata={
                "mode": "runtime_task_inspection",
                "summary_only": True,
                "raw_payload_exposed": False,
            },
        )

    def inspect_latest_task(self, store) -> RuntimeTaskInspection:
        task = _latest_task(store)
        if task is None:
            raise KeyError("No runtime tasks found")
        return self.inspect_task(task.task_id, store)

    def inspect_events(self, task_id: str, store) -> List[Dict[str, Any]]:
        return [_event_summary(event) for event in store.list_events(task_id=task_id)]


def _latest_task(store):
    tasks = []
    if hasattr(store, "tasks") and isinstance(store.tasks, Mapping):
        tasks = list(store.tasks.values())
    elif hasattr(store, "list_sessions") and hasattr(store, "list_tasks_by_session"):
        for session in store.list_sessions():
            tasks.extend(store.list_tasks_by_session(session.session_id))
    if not tasks:
        return None
    return sorted(tasks, key=lambda task: (task.created_at, task.task_id))[-1]


def _input_summary(input_payload: Any) -> Dict[str, Any]:
    data = dict(input_payload or {}) if isinstance(input_payload, Mapping) else {}
    jd_text = data.get("jd_text") if isinstance(data.get("jd_text"), str) else ""
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    return {
        "keys": sorted(str(key) for key in data.keys()),
        "jd_length": len(jd_text),
        "metadata_keys": sorted(str(key) for key in metadata.keys()),
        "summary_only": True,
    }


def _safe_output_summary(result: Any) -> Dict[str, Any]:
    data = dict(result or {}) if isinstance(result, Mapping) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    return {
        "status": str(data.get("status") or ""),
        "candidate_count": _safe_int(data.get("candidate_count")),
        "report_count": _safe_int(data.get("report_count")),
        "top_score_present": bool(data.get("top_score_present", False)),
        "output_keys": sorted(str(key) for key in data.get("output_keys", [])),
        "graph_mode": str(data.get("graph_mode") or data.get("selected_graph_mode") or ""),
        "selected_graph_mode": str(data.get("selected_graph_mode") or data.get("graph_mode") or ""),
        "runner_name": str(data.get("runner_name") or ""),
        "selection_reason": str(data.get("selection_reason") or ""),
        "legacy_graph_invoked": bool(data.get("legacy_graph_invoked", False)),
        "skill_graph_invoked": bool(data.get("skill_graph_invoked", False)),
        "rollback_target": str(data.get("rollback_target") or ""),
        "rollback_recommended": bool(data.get("rollback_recommended", False)),
        "primary_graph_mode": str(data.get("primary_graph_mode") or ""),
        "fallback_attempted": bool(data.get("fallback_attempted", False)),
        "fallback_graph_mode": str(data.get("fallback_graph_mode") or ""),
        "fallback_succeeded": bool(data.get("fallback_succeeded", False)),
        "final_runner_used": str(data.get("final_runner_used") or ""),
        "final_status": str(data.get("final_status") or ""),
        "error_type": str(data.get("error_type") or ""),
        "error_hint": str(data.get("error_hint") or ""),
        "planner_source": str(data.get("planner_source") or ""),
        "real_planner_invoked": bool(data.get("real_planner_invoked", False)),
        "real_planner_failed": bool(data.get("real_planner_failed", False)),
        "planner_fallback_used": bool(data.get("planner_fallback_used", False)),
        "planner_fallback_type": str(data.get("planner_fallback_type") or ""),
        "fallback_not_real_planner_success": bool(data.get("fallback_not_real_planner_success", False)),
        "planner_invocation_stage": str(data.get("planner_invocation_stage") or ""),
        "planner_input_shape": _safe_planner_input_shape(data.get("planner_input_shape")),
        "planner_output_keys": sorted(str(key) for key in data.get("planner_output_keys", [])),
        "planner_expected_keys": sorted(str(key) for key in data.get("planner_expected_keys", [])),
        "planner_adapter_error_hint": str(data.get("planner_adapter_error_hint") or ""),
        "provider_error_type": str(data.get("provider_error_type") or ""),
        "planner_provider_diagnostics": _safe_planner_provider_diagnostics(
            data.get("planner_provider_diagnostics")
        ),
        "matcher_invocation_stage": str(data.get("matcher_invocation_stage") or ""),
        "matcher_input_keys": sorted(str(key) for key in data.get("matcher_input_keys", [])),
        "matcher_candidate_id": str(data.get("matcher_candidate_id") or ""),
        "matcher_candidate_name_present": bool(data.get("matcher_candidate_name_present", False)),
        "matcher_skills_count": _safe_int(data.get("matcher_skills_count")),
        "matcher_source": str(data.get("matcher_source") or ""),
        "matcher_input_source": str(data.get("matcher_input_source") or ""),
        "real_matcher_invoked": bool(data.get("real_matcher_invoked", False)),
        "real_matcher_failed": bool(data.get("real_matcher_failed", False)),
        "matcher_adapter_error_hint": str(data.get("matcher_adapter_error_hint") or ""),
        "matcher_provider_error_type": str(data.get("matcher_provider_error_type") or ""),
        "matcher_output_keys": sorted(str(key) for key in data.get("matcher_output_keys", [])),
        "runner_error_type": str(data.get("runner_error_type") or ""),
        "runner_error_stage": str(data.get("runner_error_stage") or ""),
        "graph_input_keys": sorted(str(key) for key in data.get("graph_input_keys", [])),
        "graph_config_has_thread_id": bool(data.get("graph_config_has_thread_id", False)),
        "graph_result_keys": sorted(str(key) for key in data.get("graph_result_keys", [])),
        "metadata": {
            "keys": sorted(str(key) for key in metadata.keys()),
            "summary_only": True,
        },
        "summary_only": True,
    }


def _safe_planner_input_shape(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    raw_keys = value.get("input_keys") or []
    if not isinstance(raw_keys, list):
        raw_keys = []
    return {
        "input_keys": sorted(str(key) for key in raw_keys),
        "has_messages": bool(value.get("has_messages", False)),
        "raw_text_length": _safe_int(value.get("raw_text_length")),
        "summary_only": True,
    }


def _safe_planner_provider_diagnostics(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if not value:
        return {}
    dotenv_loaded = value.get("dotenv_loaded", "skip")
    if dotenv_loaded not in {True, False, "skip"}:
        dotenv_loaded = "skip"
    return {
        "dotenv_loaded": dotenv_loaded,
        "dotenv_path_present": bool(value.get("dotenv_path_present", False)),
        "openai_api_key": "set" if value.get("openai_api_key") == "set" else "missing",
        "openai_api_base": "set" if value.get("openai_api_base") == "set" else "missing",
        "llm_model": str(value.get("llm_model") or ""),
        "planner_agent_class": str(value.get("planner_agent_class") or ""),
        "invocation_method": str(value.get("invocation_method") or ""),
        "provider_error_type": str(value.get("provider_error_type") or ""),
        "summary_only": True,
    }


def _event_summary(event) -> Dict[str, Any]:
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "session_id": event.session_id,
        "task_id": event.task_id,
        "payload_keys": sorted(str(key) for key in payload.keys()),
        "runner_used": str(payload.get("runner_used") or ""),
        "graph_mode": str(payload.get("graph_mode") or ""),
        "runner_name": str(payload.get("runner_name") or ""),
        "node_name": str(payload.get("node_name") or ""),
        "skill_name": str(payload.get("skill_name") or ""),
        "skill_version": str(payload.get("skill_version") or ""),
        "tool_name": str(payload.get("tool_name") or ""),
        "mcp_server_name": str(payload.get("mcp_server_name") or ""),
        "transport": str(payload.get("transport") or ""),
        "request_id": str(payload.get("request_id") or ""),
        "execution_id": str(payload.get("execution_id") or ""),
        "status": str(payload.get("status") or ""),
        "duration_ms": payload.get("duration_ms"),
        "error_type": str(payload.get("error_type") or ""),
        "error_hint": str(payload.get("error_hint") or ""),
        "timeout": bool(payload.get("timeout", False)),
        "permission_decision": str(payload.get("permission_decision") or ""),
        "result_count": _safe_int(payload.get("result_count")),
        "fallback_used": bool(payload.get("fallback_used", False)),
        "rollback_recommended": bool(payload.get("rollback_recommended", False)),
        "summary_only": True,
    }


def _runner_used(events: List[Dict[str, Any]], output: Mapping[str, Any]) -> str:
    for event in events:
        if event.get("runner_used"):
            return str(event["runner_used"])
    return str(output.get("runner_used") or "")


def _event_error_hint(events: List[Dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("error_hint"):
            return str(event["error_hint"])
    return ""


def _status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
