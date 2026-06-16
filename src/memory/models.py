import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_memory_id() -> str:
    return f"memory_{uuid.uuid4()}"


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PREFERENCE = "preference"
    PROCEDURAL = "procedural"


class MemorySourceType(str, Enum):
    CANDIDATE_PROFILE = "candidate_profile"
    RESUME_DOCUMENT = "resume_document"
    MATCH_REPORT = "match_report"
    SEARCH_ATTEMPT = "search_attempt"
    HUMAN_FEEDBACK = "human_feedback"
    RUNTIME_EVENT = "runtime_event"
    MANUAL = "manual"


@dataclass
class MemoryRecord:
    memory_id: str = field(default_factory=new_memory_id)
    memory_type: str = MemoryType.SEMANTIC.value
    source_type: str = MemorySourceType.MANUAL.value
    source_id: str = ""
    content: str = ""
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict):
        values = data.copy()
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        if isinstance(values.get("updated_at"), str):
            values["updated_at"] = datetime.fromisoformat(values["updated_at"])
        return cls(**values)
