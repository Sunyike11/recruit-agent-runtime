from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.memory.governance import MemoryGovernancePolicy
from src.memory.models import MemoryRecord, MemoryType


ELIGIBILITY_STATUSES = {"eligible", "denied", "requires_review"}
ALLOWED_PROMOTED_MEMORY_TYPES = {
    MemoryType.EPISODIC.value,
    MemoryType.PROCEDURAL.value,
}
HIGH_IMPORTANCE_THRESHOLD = 0.85


@dataclass
class MemoryEligibilityDecision:
    memory_id: str
    eligible: bool
    status: str
    reason: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "MemoryEligibilityDecision":
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("MemoryEligibilityDecision memory_id must be non-empty")
        if not isinstance(self.eligible, bool):
            raise ValueError("MemoryEligibilityDecision eligible must be bool")
        if self.status not in ELIGIBILITY_STATUSES:
            raise ValueError(f"unsupported MemoryEligibilityDecision status: {self.status}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("MemoryEligibilityDecision reason must be non-empty")
        if not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("MemoryEligibilityDecision tags must be a list of strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryEligibilityDecision metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotedMemoryAuditReport:
    total_memories: int
    promoted_from_reflection_count: int
    eligible_count: int
    denied_count: int
    requires_review_count: int
    sensitive_count: int
    missing_provenance_count: int
    decisions: List[MemoryEligibilityDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "promoted_from_reflection_count": self.promoted_from_reflection_count,
            "eligible_count": self.eligible_count,
            "denied_count": self.denied_count,
            "requires_review_count": self.requires_review_count,
            "sensitive_count": self.sensitive_count,
            "missing_provenance_count": self.missing_provenance_count,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "metadata": dict(self.metadata),
        }


class MemoryContextEligibilityPolicy:
    """Read-only policy for promoted reflection memory context previews."""

    def __init__(
        self,
        governance_policy: Optional[MemoryGovernancePolicy] = None,
        governance_store: Optional[Any] = None,
    ):
        self.governance_policy = (
            governance_policy
            if governance_policy is not None
            else MemoryGovernancePolicy() if governance_store is not None else None
        )
        self.governance_store = governance_store

    def evaluate(
        self,
        memory_record: MemoryRecord,
        target_context: Optional[Mapping[str, Any]] = None,
    ) -> MemoryEligibilityDecision:
        metadata = memory_record.metadata if isinstance(memory_record.metadata, dict) else {}
        tags = ["promoted_memory", memory_record.memory_type]
        base_metadata = {"summary_only": True, "context_preview_only": True}

        if metadata.get("promoted_from_reflection") is not True:
            return self._decision(memory_record, False, "denied", "memory is not promoted from reflection", tags, base_metadata)
        if not memory_record.content or not memory_record.content.strip():
            return self._decision(memory_record, False, "denied", "memory content is required", tags, base_metadata)
        if metadata.get("sensitive") is True:
            return self._decision(memory_record, False, "denied", "sensitive memory is not context eligible", tags, base_metadata)
        if not metadata.get("source_reflection_id"):
            return self._decision(memory_record, False, "denied", "missing source_reflection_id", tags, base_metadata)
        if not metadata.get("source_candidate_id"):
            return self._decision(memory_record, False, "denied", "missing source_candidate_id", tags, base_metadata)
        if metadata.get("dry_run") is True:
            return self._decision(memory_record, False, "denied", "dry-run memory preview is not eligible", tags, base_metadata)
        if self.governance_policy is not None:
            governance = self.governance_policy.evaluate(
                memory_record,
                governance_store=self.governance_store,
            )
            if not governance.allowed:
                governance_metadata = dict(base_metadata)
                governance_metadata["governance_status"] = governance.status
                return self._decision(
                    memory_record,
                    False,
                    "denied",
                    f"governance status {governance.status} denies context preview",
                    tags,
                    governance_metadata,
                )
        if not (metadata.get("approved_by") or metadata.get("reviewer")):
            return self._decision(memory_record, False, "requires_review", "missing reviewer approval provenance", tags, base_metadata)
        if memory_record.importance > HIGH_IMPORTANCE_THRESHOLD:
            return self._decision(memory_record, False, "requires_review", "high importance memory requires review", tags, base_metadata)
        if memory_record.memory_type not in ALLOWED_PROMOTED_MEMORY_TYPES:
            return self._decision(memory_record, False, "requires_review", "memory type requires review", tags, base_metadata)
        context_reason = _target_context_mismatch(memory_record, target_context)
        if context_reason:
            return self._decision(memory_record, False, "requires_review", context_reason, tags, base_metadata)
        return self._decision(memory_record, True, "eligible", "promoted memory is eligible for context preview", tags, base_metadata)

    def filter_eligible(
        self,
        memory_records: Iterable[MemoryRecord],
        target_context: Optional[Mapping[str, Any]] = None,
    ) -> List[MemoryRecord]:
        return [
            record
            for record in memory_records
            if self.evaluate(record, target_context=target_context).eligible
        ]

    @staticmethod
    def _decision(
        memory_record: MemoryRecord,
        eligible: bool,
        status: str,
        reason: str,
        tags: List[str],
        metadata: Dict[str, Any],
    ) -> MemoryEligibilityDecision:
        return MemoryEligibilityDecision(
            memory_id=memory_record.memory_id,
            eligible=eligible,
            status=status,
            reason=reason,
            tags=list(tags),
            metadata=dict(metadata),
        ).validate()


class PromotedMemoryAuditor:
    """Summarize promoted memory eligibility without copying memory content or provenance values."""

    def __init__(self, policy: Optional[MemoryContextEligibilityPolicy] = None):
        self.policy = policy or MemoryContextEligibilityPolicy()

    def audit(
        self,
        memory_records: Iterable[MemoryRecord],
        target_context: Optional[Mapping[str, Any]] = None,
    ) -> PromotedMemoryAuditReport:
        records = list(memory_records)
        decisions = [
            self.policy.evaluate(record, target_context=target_context) for record in records
        ]
        return PromotedMemoryAuditReport(
            total_memories=len(records),
            promoted_from_reflection_count=sum(
                1 for record in records if record.metadata.get("promoted_from_reflection") is True
            ),
            eligible_count=sum(1 for decision in decisions if decision.status == "eligible"),
            denied_count=sum(1 for decision in decisions if decision.status == "denied"),
            requires_review_count=sum(
                1 for decision in decisions if decision.status == "requires_review"
            ),
            sensitive_count=sum(1 for record in records if record.metadata.get("sensitive") is True),
            missing_provenance_count=sum(1 for record in records if _missing_provenance(record)),
            decisions=decisions,
            metadata={"summary_only": True, "context_preview_only": True},
        )


def build_eligible_memory_context_preview(
    memory_records: Iterable[MemoryRecord],
    policy: Optional[MemoryContextEligibilityPolicy] = None,
    target_context: Optional[Mapping[str, Any]] = None,
    max_items: int = 5,
    max_chars: int = 2000,
) -> str:
    evaluator = policy or MemoryContextEligibilityPolicy()
    eligible = evaluator.filter_eligible(memory_records, target_context=target_context)
    lines = ["Promoted Memory Context Preview:"]
    if not eligible:
        lines.append("No eligible promoted memory.")
    else:
        for record in eligible[: max(0, max_items)]:
            lines.append(f"[{record.memory_type}]: {record.content}")
    text = "\n".join(lines)
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def _missing_provenance(memory_record: MemoryRecord) -> bool:
    metadata = memory_record.metadata if isinstance(memory_record.metadata, dict) else {}
    if metadata.get("promoted_from_reflection") is not True:
        return False
    return not metadata.get("source_reflection_id") or not metadata.get("source_candidate_id")


def _target_context_mismatch(
    memory_record: MemoryRecord,
    target_context: Optional[Mapping[str, Any]],
) -> str:
    if target_context is None:
        return ""
    allowed_types = target_context.get("memory_types")
    if allowed_types is not None and memory_record.memory_type not in set(allowed_types):
        return "memory type does not match target context"
    required_tags = target_context.get("tags")
    if required_tags is not None and not set(required_tags).intersection(memory_record.tags):
        return "memory tags do not match target context"
    return ""
