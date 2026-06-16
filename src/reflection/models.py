import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_reflection_id() -> str:
    return f"reflection_{uuid.uuid4()}"


class ReflectionSourceType(str, Enum):
    EVAL_REPORT = "eval_report"
    EVAL_RECORD = "eval_record"
    CORRELATION_REPORT = "correlation_report"
    MANUAL = "manual"


class ReflectionStatus(str, Enum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"
    UNKNOWN = "unknown"


REFLECTION_TARGET_TYPES = {
    "task",
    "skill_workflow",
    "tool_workflow",
    "runtime_timeline",
    "manual",
}


@dataclass
class ReflectionRecord:
    reflection_id: str = field(default_factory=new_reflection_id)
    source_type: str = ReflectionSourceType.MANUAL.value
    source_id: str = ""
    target_type: str = "manual"
    target_id: str = ""
    status: str = ReflectionStatus.UNKNOWN.value
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def validate(self) -> "ReflectionRecord":
        if not isinstance(self.reflection_id, str) or not self.reflection_id.strip():
            raise ValueError("ReflectionRecord reflection_id must be non-empty")
        if self.source_type not in {value.value for value in ReflectionSourceType}:
            raise ValueError(f"unsupported ReflectionRecord source_type: {self.source_type}")
        if self.target_type not in REFLECTION_TARGET_TYPES:
            raise ValueError(f"unsupported ReflectionRecord target_type: {self.target_type}")
        if self.status not in {value.value for value in ReflectionStatus}:
            raise ValueError(f"unsupported ReflectionRecord status: {self.status}")
        for name in ("findings", "recommended_actions", "evidence_refs", "tags"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"ReflectionRecord {name} must be a list of strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("ReflectionRecord metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionRecord":
        if not isinstance(data, dict):
            raise ValueError("ReflectionRecord payload must be a dict")
        values = dict(data)
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        return cls(**values).validate()
