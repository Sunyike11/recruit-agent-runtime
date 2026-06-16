import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.memory.models import MemoryRecord, MemorySourceType, MemoryType
from src.reflection.models import ReflectionRecord, ReflectionSourceType, ReflectionStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_candidate_id() -> str:
    return f"memory_candidate_{uuid.uuid4()}"


PROJECTION_STATUSES = {"allowed", "denied", "requires_approval"}
MAX_REFLECTION_SUMMARY_CHARS = 500


@dataclass
class MemoryCandidate:
    candidate_id: str = field(default_factory=new_candidate_id)
    source_reflection_id: str = ""
    memory_type: str = MemoryType.EPISODIC.value
    content: str = ""
    importance: float = 0.3
    tags: List[str] = field(default_factory=list)
    requires_approval: bool = True
    approval_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def validate(self) -> "MemoryCandidate":
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("MemoryCandidate candidate_id must be non-empty")
        if not isinstance(self.source_reflection_id, str) or not self.source_reflection_id.strip():
            raise ValueError("MemoryCandidate source_reflection_id must be non-empty")
        if self.memory_type not in {value.value for value in MemoryType}:
            raise ValueError(f"unsupported MemoryCandidate memory_type: {self.memory_type}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("MemoryCandidate content must be non-empty")
        if not isinstance(self.importance, (int, float)) or not 0 <= self.importance <= 1:
            raise ValueError("MemoryCandidate importance must be between 0 and 1")
        if not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("MemoryCandidate tags must be a list of strings")
        if not isinstance(self.requires_approval, bool):
            raise ValueError("MemoryCandidate requires_approval must be bool")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryCandidate metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCandidate":
        if not isinstance(data, dict):
            raise ValueError("MemoryCandidate payload must be a dict")
        values = dict(data)
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        return cls(**values).validate()

    def to_memory_record(self) -> MemoryRecord:
        """Build an unsaved MemoryRecord preview; callers own any later approval flow."""
        self.validate()
        return MemoryRecord(
            memory_type=self.memory_type,
            source_type=MemorySourceType.MANUAL.value,
            source_id=self.source_reflection_id,
            content=self.content,
            importance=self.importance,
            tags=list(self.tags),
            metadata={
                "memory_candidate_id": self.candidate_id,
                "source_reflection_id": self.source_reflection_id,
                "requires_approval": self.requires_approval,
                "approval_reason": self.approval_reason,
                "projection_source": "reflection",
                "summary_only": True,
            },
        )


@dataclass
class MemoryProjectionDecision:
    allowed: bool
    status: str
    reason: str
    source_reflection_id: str
    proposed_memory_type: str = ""
    importance: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "MemoryProjectionDecision":
        if self.status not in PROJECTION_STATUSES:
            raise ValueError(f"unsupported MemoryProjectionDecision status: {self.status}")
        if not isinstance(self.allowed, bool):
            raise ValueError("MemoryProjectionDecision allowed must be bool")
        if not isinstance(self.source_reflection_id, str) or not self.source_reflection_id.strip():
            raise ValueError("MemoryProjectionDecision source_reflection_id must be non-empty")
        if self.proposed_memory_type and self.proposed_memory_type not in {
            value.value for value in MemoryType
        }:
            raise ValueError(
                f"unsupported MemoryProjectionDecision memory_type: {self.proposed_memory_type}"
            )
        if not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("MemoryProjectionDecision tags must be a list of strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryProjectionDecision metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReflectionMemoryProjectionPolicy:
    """Create review-gated, unsaved memory candidates from safe reflection summaries."""

    def evaluate(self, reflection_record: ReflectionRecord) -> MemoryProjectionDecision:
        reflection_record.validate()
        if reflection_record.metadata.get("sensitive") is True:
            return self._denied(reflection_record, "sensitive reflection cannot be projected")
        summary = reflection_record.summary.strip()
        if not summary:
            return self._denied(reflection_record, "reflection summary is required")
        if len(summary) > MAX_REFLECTION_SUMMARY_CHARS:
            return self._denied(
                reflection_record,
                f"reflection summary exceeds {MAX_REFLECTION_SUMMARY_CHARS} characters",
            )

        memory_type, importance = _memory_type_and_importance(reflection_record.status)
        reason = "memory candidate requires explicit approval before any persistence"
        if reflection_record.source_type == ReflectionSourceType.MANUAL.value:
            reason = "manual reflection requires explicit approval before any persistence"
        return MemoryProjectionDecision(
            allowed=True,
            status="requires_approval",
            reason=reason,
            source_reflection_id=reflection_record.reflection_id,
            proposed_memory_type=memory_type,
            importance=importance,
            tags=_candidate_tags(reflection_record.status, memory_type),
            metadata={"requires_approval": True, "summary_only": True},
        ).validate()

    def project(self, reflection_record: ReflectionRecord) -> Optional[MemoryCandidate]:
        decision = self.evaluate(reflection_record)
        if not decision.allowed:
            return None
        return MemoryCandidate(
            source_reflection_id=reflection_record.reflection_id,
            memory_type=decision.proposed_memory_type,
            content=reflection_record.summary.strip(),
            importance=decision.importance,
            tags=list(decision.tags),
            requires_approval=decision.status == "requires_approval",
            approval_reason=decision.reason,
            metadata={
                "projection_status": decision.status,
                "reflection_status": reflection_record.status,
                "source_type": reflection_record.source_type,
                "summary_only": True,
            },
        ).validate()

    def project_many(self, reflection_records: Iterable[ReflectionRecord]) -> List[MemoryCandidate]:
        candidates = []
        for record in reflection_records:
            candidate = self.project(record)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _denied(reflection_record: ReflectionRecord, reason: str) -> MemoryProjectionDecision:
        return MemoryProjectionDecision(
            allowed=False,
            status="denied",
            reason=reason,
            source_reflection_id=reflection_record.reflection_id,
            metadata={"summary_only": True},
        ).validate()


def _memory_type_and_importance(status: str) -> tuple:
    if status == ReflectionStatus.FAILURE.value:
        return MemoryType.PROCEDURAL.value, 0.8
    if status == ReflectionStatus.WARNING.value:
        return MemoryType.PROCEDURAL.value, 0.6
    return MemoryType.EPISODIC.value, 0.3


def _candidate_tags(status: str, memory_type: str) -> List[str]:
    return ["reflection_candidate", status, memory_type]
