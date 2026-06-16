from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


TOOL_AUDIT_EVENT_TYPES = {
    "tool_started",
    "tool_completed",
    "tool_failed",
    "tool_denied",
    "tool_approval_required",
    "tool_approval_granted",
    "tool_approval_rejected",
    "tool_sandbox_denied",
}


@dataclass
class ToolAuditEvent:
    event_id: str = ""
    task_id: Optional[str] = None
    event_type: str = ""
    tool_name: str = ""
    tool_version: str = ""
    status: str = ""
    caller_type: str = ""
    caller_name: str = ""
    permission_status: str = ""
    sandbox_status: str = ""
    duration_ms: Optional[float] = None
    error: str = ""
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_runtime_event(cls, event: Any) -> "ToolAuditEvent":
        payload = _event_value(event, "payload", {}) or {}
        if not isinstance(payload, dict):
            payload = {}
        permission_decision = payload.get("permission_decision") or {}
        sandbox_decision = payload.get("sandbox_decision") or {}
        metadata = payload.get("metadata") or {}
        return cls(
            event_id=str(_event_value(event, "event_id", "") or ""),
            task_id=_event_value(event, "task_id", None),
            event_type=str(_event_value(event, "event_type", "") or ""),
            tool_name=str(payload.get("tool_name", "") or ""),
            tool_version=str(payload.get("version", payload.get("tool_version", "")) or ""),
            status=str(payload.get("status", "") or ""),
            caller_type=str(payload.get("caller_type", "") or ""),
            caller_name=str(payload.get("caller_name", "") or ""),
            permission_status=_decision_status(permission_decision),
            sandbox_status=_decision_status(sandbox_decision),
            duration_ms=_optional_number(payload.get("duration_ms")),
            error=str(payload.get("error", "") or ""),
            created_at=_event_value(event, "created_at", None),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass
class ToolAuditReport:
    task_id: Optional[str] = None
    total_tool_events: int = 0
    started_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    denied_count: int = 0
    approval_required_count: int = 0
    approval_granted_count: int = 0
    approval_rejected_count: int = 0
    sandbox_denied_count: int = 0
    tools_called: List[str] = field(default_factory=list)
    events: List[ToolAuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolAuditReporter:
    @classmethod
    def from_events(cls, events: Iterable[Any]) -> ToolAuditReport:
        audit_events = [
            ToolAuditEvent.from_runtime_event(event)
            for event in events
            if _event_value(event, "event_type", "") in TOOL_AUDIT_EVENT_TYPES
        ]
        tools_called: List[str] = []
        for event in audit_events:
            if event.tool_name and event.tool_name not in tools_called:
                tools_called.append(event.tool_name)
        task_id = next((event.task_id for event in audit_events if event.task_id), None)
        return ToolAuditReport(
            task_id=task_id,
            total_tool_events=len(audit_events),
            started_count=_event_count(audit_events, "tool_started"),
            completed_count=_event_count(audit_events, "tool_completed"),
            failed_count=_event_count(audit_events, "tool_failed"),
            denied_count=_event_count(audit_events, "tool_denied"),
            approval_required_count=_event_count(audit_events, "tool_approval_required"),
            approval_granted_count=_event_count(audit_events, "tool_approval_granted"),
            approval_rejected_count=_event_count(audit_events, "tool_approval_rejected"),
            sandbox_denied_count=_event_count(audit_events, "tool_sandbox_denied"),
            tools_called=tools_called,
            events=audit_events,
            metadata={"recognized_event_types": sorted(TOOL_AUDIT_EVENT_TYPES)},
        )

    @classmethod
    def from_runtime_store(cls, store: Any, task_id: str) -> ToolAuditReport:
        report = cls.from_events(store.list_events_by_task(task_id))
        report.task_id = task_id
        report.metadata["source"] = "runtime_store"
        return report

    @staticmethod
    def to_dict(report: ToolAuditReport) -> Dict[str, Any]:
        return {
            "task_id": report.task_id,
            "total_tool_events": report.total_tool_events,
            "started_count": report.started_count,
            "completed_count": report.completed_count,
            "failed_count": report.failed_count,
            "denied_count": report.denied_count,
            "approval_required_count": report.approval_required_count,
            "approval_granted_count": report.approval_granted_count,
            "approval_rejected_count": report.approval_rejected_count,
            "sandbox_denied_count": report.sandbox_denied_count,
            "tools_called": list(report.tools_called),
            "events": [_event_to_dict(event) for event in report.events],
            "metadata": dict(report.metadata),
        }

    @staticmethod
    def format_text(report: ToolAuditReport) -> str:
        tools = ", ".join(report.tools_called) if report.tools_called else "none"
        return "\n".join(
            [
                f"Tool Audit Report for task {report.task_id or '<unknown>'}",
                f"Total tool events: {report.total_tool_events}",
                f"Started: {report.started_count}",
                f"Completed: {report.completed_count}",
                f"Failed: {report.failed_count}",
                f"Denied: {report.denied_count}",
                f"Approval required: {report.approval_required_count}",
                f"Approval granted: {report.approval_granted_count}",
                f"Approval rejected: {report.approval_rejected_count}",
                f"Sandbox denied: {report.sandbox_denied_count}",
                f"Tools: {tools}",
            ]
        )


def _event_value(event: Any, name: str, default: Any) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _decision_status(decision: Any) -> str:
    if isinstance(decision, dict):
        return str(decision.get("status", "") or "")
    return ""


def _optional_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _event_count(events: Iterable[ToolAuditEvent], event_type: str) -> int:
    return sum(1 for event in events if event.event_type == event_type)


def _event_to_dict(event: ToolAuditEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "task_id": event.task_id,
        "event_type": event.event_type,
        "tool_name": event.tool_name,
        "tool_version": event.tool_version,
        "status": event.status,
        "caller_type": event.caller_type,
        "caller_name": event.caller_name,
        "permission_status": event.permission_status,
        "sandbox_status": event.sandbox_status,
        "duration_ms": event.duration_ms,
        "error": event.error,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "metadata": dict(event.metadata),
    }
