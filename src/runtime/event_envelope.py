from typing import Any, Dict, Mapping, Optional


def build_runtime_event_payload(
    *,
    session_id: str = "",
    task_id: str = "",
    thread_id: str = "",
    graph_mode: str = "",
    runner_name: str = "",
    node_name: str = "",
    skill_name: str = "",
    skill_version: str = "",
    execution_id: str = "",
    status: str = "",
    duration_ms: Optional[float] = None,
    error_type: str = "",
    error_hint: str = "",
    fallback_used: bool = False,
    rollback_recommended: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "thread_id": str(thread_id or ""),
        "graph_mode": str(graph_mode or ""),
        "runner_name": str(runner_name or ""),
        "node_name": str(node_name or ""),
        "skill_name": str(skill_name or ""),
        "skill_version": str(skill_version or ""),
        "execution_id": str(execution_id or ""),
        "status": str(status or ""),
        "duration_ms": duration_ms,
        "error_type": str(error_type or ""),
        "error_hint": str(error_hint or ""),
        "fallback_used": bool(fallback_used),
        "rollback_recommended": bool(rollback_recommended),
        "summary_only": True,
    }
    if extra:
        payload.update(_safe_extra(extra))
    return payload


def _safe_extra(extra: Mapping[str, Any]) -> Dict[str, Any]:
    blocked = {
        "raw_jd",
        "jd_text",
        "resume_text",
        "full_resume",
        "raw_chunks",
        "prompt",
        "llm_response",
        "reasoning",
        "api_key",
        "openai_api_key",
        "hf_token",
    }
    safe: Dict[str, Any] = {}
    for key, value in extra.items():
        key_text = str(key)
        if key_text.lower() in blocked:
            continue
        safe[key_text] = value
    return safe
