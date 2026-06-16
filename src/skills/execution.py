import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.runtime.models import utc_now
from src.runtime.event_envelope import build_runtime_event_payload
from src.skills.context import SkillExecutionContext
from src.skills.models import SkillResult


def new_execution_id() -> str:
    return f"skill_execution_{uuid.uuid4()}"


@dataclass
class SkillExecutionRecord:
    execution_id: str = field(default_factory=new_execution_id)
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    skill_name: str = ""
    skill_version: str = ""
    status: str = "started"
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Optional[Dict[str, Any]] = None
    error: str = ""
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "skill_name": self.skill_name,
            "skill_version": self.skill_version,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class SkillExecutionRecorder:
    def __init__(self, runtime_store):
        self.runtime_store = runtime_store

    def skill_started(self, record: SkillExecutionRecord):
        self._append_skill_event("skill_started", record)

    def skill_completed(self, record: SkillExecutionRecord, result: SkillResult):
        record.status = "completed"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_skill_event("skill_completed", record)

    def skill_failed(self, record: SkillExecutionRecord, result: SkillResult):
        record.status = "failed"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_skill_event("skill_failed", record)

    def _append_skill_event(self, event_type: str, record: SkillExecutionRecord):
        if not record.task_id:
            return None
        graph_mode = str(record.metadata.get("graph_mode") or "")
        runner_name = str(record.metadata.get("runner_used") or record.metadata.get("runner_type") or "")
        return self.runtime_store.append_event(
            event_type,
            session_id=record.session_id,
            task_id=record.task_id,
            payload={
                **record.to_event_payload(),
                **build_runtime_event_payload(
                    session_id=record.session_id or "",
                    task_id=record.task_id or "",
                    thread_id=record.thread_id or "",
                    graph_mode=graph_mode,
                    runner_name=runner_name,
                    node_name=str(record.metadata.get("node_name") or record.skill_name),
                    skill_name=record.skill_name,
                    skill_version=record.skill_version,
                    execution_id=record.execution_id,
                    status=record.status,
                    duration_ms=record.duration_ms,
                    error_type=_safe_error_type(record.error),
                    error_hint=str(record.metadata.get("error_hint") or ""),
                    fallback_used=bool(record.metadata.get("fallback_used", False)),
                    rollback_recommended=bool(record.metadata.get("rollback_recommended", False)),
                ),
            },
        )


def _safe_error_type(error: str) -> str:
    if not error:
        return ""
    text = str(error)
    for marker in ("RuntimeError", "ValueError", "TypeError", "APIConnectionError"):
        if marker in text:
            return marker
    return text.split(":", 1)[0][:80]


class SkillExecutor:
    def __init__(self, registry, recorder: Optional[SkillExecutionRecorder] = None):
        self.registry = registry
        self.recorder = recorder

    def execute(
        self,
        skill_name: str,
        input_data: Dict[str, Any],
        context: Optional[SkillExecutionContext] = None,
        version: Optional[str] = None,
    ) -> SkillResult:
        skill = self.registry.get(skill_name, version=version)
        execution_context = context or SkillExecutionContext()
        record = SkillExecutionRecord(
            task_id=execution_context.task_id,
            session_id=execution_context.session_id,
            thread_id=execution_context.thread_id,
            skill_name=skill.spec.name,
            skill_version=skill.spec.version,
            status="started",
            input_summary=summarize_value(input_data),
            metadata=dict(execution_context.metadata),
        )

        if self.recorder is not None:
            self.recorder.skill_started(record)

        started = time.perf_counter()
        result = skill.execute(input_data, context=execution_context)
        record.duration_ms = round((time.perf_counter() - started) * 1000, 3)

        if self.recorder is not None:
            if result.success:
                self.recorder.skill_completed(record, result)
            else:
                self.recorder.skill_failed(record, result)

        return result


def summarize_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
            "size": len(value),
        }
    if isinstance(value, (list, tuple, set)):
        return {
            "type": value.__class__.__name__,
            "length": len(value),
        }
    if isinstance(value, str):
        return {
            "type": "str",
            "length": len(value),
            "repr": _truncate_repr(value),
        }
    if value is None or isinstance(value, (int, float, bool)):
        return {
            "type": type(value).__name__,
            "repr": _truncate_repr(value),
        }
    return {
        "type": value.__class__.__name__,
        "repr": _truncate_repr(value),
    }


def _truncate_repr(value: Any, limit: int = 80) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit]


def _duration_ms(started_at: datetime, completed_at: datetime) -> float:
    return round((completed_at - started_at).total_seconds() * 1000, 3)
