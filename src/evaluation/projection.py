from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


TASK_EVENT_TYPES = {
    "task_created",
    "task_started",
    "task_completed",
    "task_failed",
}
SKILL_EVENT_TYPES = {
    "skill_started",
    "skill_completed",
    "skill_failed",
}
TOOL_EVENT_TYPES = {
    "tool_started",
    "tool_completed",
    "tool_failed",
    "tool_denied",
    "tool_approval_required",
    "tool_approval_granted",
    "tool_approval_rejected",
    "tool_sandbox_denied",
}
SUPPORTED_RUNTIME_EVENT_TYPES = TASK_EVENT_TYPES | SKILL_EVENT_TYPES | TOOL_EVENT_TYPES

_ERROR_EVENT_TYPES = {
    "task_failed",
    "skill_failed",
    "tool_failed",
    "tool_denied",
    "tool_approval_rejected",
    "tool_sandbox_denied",
}


@dataclass
class EvaluationTargetProjection:
    target_id: str = ""
    target_type: str = "runtime_timeline"
    status: str = ""
    event_counts: Dict[str, int] = field(default_factory=dict)
    skill_event_counts: Dict[str, int] = field(default_factory=dict)
    tool_event_counts: Dict[str, int] = field(default_factory=dict)
    task_event_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "status": self.status,
            "event_counts": dict(self.event_counts),
            "skill_event_counts": dict(self.skill_event_counts),
            "tool_event_counts": dict(self.tool_event_counts),
            "task_event_counts": dict(self.task_event_counts),
            "errors": list(self.errors),
            "events": [dict(event) for event in self.events],
            "metadata": dict(self.metadata),
        }


def project_runtime_timeline(
    events: Iterable[Any],
    target_id: Optional[str] = None,
) -> EvaluationTargetProjection:
    event_counts = Counter()
    skill_counts = Counter()
    tool_counts = Counter()
    task_counts = Counter()
    errors: List[str] = []
    projected_events: List[Dict[str, Any]] = []
    ignored_count = 0
    resolved_target_id = target_id or ""

    for event in events or []:
        event_type = _event_value(event, "event_type", "")
        if not resolved_target_id:
            resolved_target_id = _event_value(event, "task_id", "") or ""
        if event_type not in SUPPORTED_RUNTIME_EVENT_TYPES:
            ignored_count += 1
            continue

        event_counts[event_type] += 1
        if event_type in TASK_EVENT_TYPES:
            task_counts[event_type] += 1
        elif event_type in SKILL_EVENT_TYPES:
            skill_counts[event_type] += 1
        elif event_type in TOOL_EVENT_TYPES:
            tool_counts[event_type] += 1

        projected_events.append({"event_type": event_type})
        if event_type in _ERROR_EVENT_TYPES:
            error = _payload_error(event)
            if error:
                errors.append(error)

    return EvaluationTargetProjection(
        target_id=resolved_target_id,
        status=_projection_status(event_counts),
        event_counts=dict(event_counts),
        skill_event_counts=dict(skill_counts),
        tool_event_counts=dict(tool_counts),
        task_event_counts=dict(task_counts),
        errors=errors,
        events=projected_events,
        metadata={
            "source": "runtime_timeline",
            "recognized_event_count": len(projected_events),
            "ignored_event_count": ignored_count,
            "summary_only": True,
        },
    )


def project_task_timeline(store: Any, task_id: str) -> EvaluationTargetProjection:
    projection = project_runtime_timeline(store.list_events_by_task(task_id), target_id=task_id)
    projection.metadata["source"] = "runtime_store"
    return projection


def _event_value(event: Any, key: str, default: Any) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _payload_error(event: Any) -> str:
    payload = _event_value(event, "payload", {})
    if not isinstance(payload, Mapping):
        return ""
    error = payload.get("error", "")
    return str(error) if error else ""


def _projection_status(event_counts: Counter) -> str:
    if event_counts["task_failed"]:
        return "failed"
    if event_counts["task_completed"]:
        return "completed"
    if event_counts["task_started"]:
        return "running"
    if event_counts["task_created"]:
        return "created"
    if any(event_counts[event_type] for event_type in _ERROR_EVENT_TYPES):
        return "failed"
    return ""
