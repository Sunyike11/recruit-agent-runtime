import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.memory.models import MemoryRecord


GOVERNANCE_STATUSES = {"active", "revoked", "superseded", "expired"}
DECISION_STATUSES = GOVERNANCE_STATUSES | {"unknown"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_governance_id() -> str:
    return f"governance_{uuid.uuid4()}"


@dataclass
class MemoryGovernanceRecord:
    governance_id: str = field(default_factory=new_governance_id)
    memory_id: str = ""
    status: str = "active"
    reason: str = ""
    actor: str = ""
    supersedes_memory_id: str = ""
    superseded_by_memory_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def validate(self) -> "MemoryGovernanceRecord":
        if not isinstance(self.governance_id, str) or not self.governance_id.strip():
            raise ValueError("MemoryGovernanceRecord governance_id must be non-empty")
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("MemoryGovernanceRecord memory_id must be non-empty")
        if self.status not in GOVERNANCE_STATUSES:
            raise ValueError(f"unsupported MemoryGovernanceRecord status: {self.status}")
        if not isinstance(self.reason, str):
            raise ValueError("MemoryGovernanceRecord reason must be str")
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ValueError("MemoryGovernanceRecord actor must be non-empty")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryGovernanceRecord metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryGovernanceRecord":
        if not isinstance(data, dict):
            raise ValueError("MemoryGovernanceRecord payload must be a dict")
        values = dict(data)
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        return cls(**values).validate()


@dataclass
class MemoryGovernanceDecision:
    memory_id: str
    allowed: bool
    status: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "MemoryGovernanceDecision":
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("MemoryGovernanceDecision memory_id must be non-empty")
        if not isinstance(self.allowed, bool):
            raise ValueError("MemoryGovernanceDecision allowed must be bool")
        if self.status not in DECISION_STATUSES:
            raise ValueError(f"unsupported MemoryGovernanceDecision status: {self.status}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("MemoryGovernanceDecision reason must be non-empty")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryGovernanceDecision metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InMemoryMemoryGovernanceStore:
    """Append-only local governance history, separate from memory persistence."""

    def __init__(self):
        self._records: List[MemoryGovernanceRecord] = []

    def save_record(self, record: MemoryGovernanceRecord) -> MemoryGovernanceRecord:
        record.validate()
        self._records.append(record)
        return record

    def get_latest_record(self, memory_id: str) -> Optional[MemoryGovernanceRecord]:
        for record in reversed(self._records):
            if record.memory_id == memory_id:
                return record
        return None

    def list_records(
        self,
        memory_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[MemoryGovernanceRecord]:
        records = list(self._records)
        if memory_id is not None:
            records = [record for record in records if record.memory_id == memory_id]
        if status is not None:
            records = [record for record in records if record.status == status]
        return records

    def revoke_memory(
        self,
        memory_id: str,
        reason: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryGovernanceRecord:
        return self.save_record(
            MemoryGovernanceRecord(
                memory_id=memory_id,
                status="revoked",
                reason=reason,
                actor=actor,
                metadata=dict(metadata or {}),
            )
        )

    def expire_memory(
        self,
        memory_id: str,
        reason: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryGovernanceRecord:
        return self.save_record(
            MemoryGovernanceRecord(
                memory_id=memory_id,
                status="expired",
                reason=reason,
                actor=actor,
                metadata=dict(metadata or {}),
            )
        )

    def mark_superseded(
        self,
        memory_id: str,
        superseded_by_memory_id: str,
        reason: str,
        actor: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryGovernanceRecord:
        return self.save_record(
            MemoryGovernanceRecord(
                memory_id=memory_id,
                status="superseded",
                reason=reason,
                actor=actor,
                superseded_by_memory_id=superseded_by_memory_id,
                metadata=dict(metadata or {}),
            )
        )


class MemoryGovernancePolicy:
    """Read-only lifecycle policy for a stored memory record."""

    def evaluate(
        self,
        memory_record: MemoryRecord,
        governance_store: Optional[InMemoryMemoryGovernanceStore] = None,
    ) -> MemoryGovernanceDecision:
        metadata = memory_record.metadata if isinstance(memory_record.metadata, dict) else {}
        if metadata.get("sensitive") is True:
            return self._decision(
                memory_record, False, "unknown", "sensitive memory is denied by governance", "metadata"
            )
        if metadata.get("revoked") is True:
            return self._decision(
                memory_record, False, "revoked", "memory metadata marks it revoked", "metadata"
            )
        latest = governance_store.get_latest_record(memory_record.memory_id) if governance_store else None
        if latest is None:
            return self._decision(
                memory_record, True, "active", "no governance record; memory is active by default", "default"
            )
        if latest.status == "active":
            return self._decision(memory_record, True, "active", "latest governance status is active", "store")
        return self._decision(
            memory_record,
            False,
            latest.status,
            f"latest governance status is {latest.status}",
            "store",
        )

    @staticmethod
    def _decision(
        memory_record: MemoryRecord,
        allowed: bool,
        status: str,
        reason: str,
        source: str,
    ) -> MemoryGovernanceDecision:
        return MemoryGovernanceDecision(
            memory_id=memory_record.memory_id,
            allowed=allowed,
            status=status,
            reason=reason,
            metadata={"summary_only": True, "governance_source": source},
        ).validate()
