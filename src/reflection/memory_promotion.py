from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.memory.models import MemoryRecord
from src.reflection.memory_projection import MemoryCandidate


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MemoryCandidateReviewDecision:
    candidate_id: str
    approved: bool
    reviewer: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=utc_now)

    def validate(self) -> "MemoryCandidateReviewDecision":
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("MemoryCandidateReviewDecision candidate_id must be non-empty")
        if not isinstance(self.approved, bool):
            raise ValueError("MemoryCandidateReviewDecision approved must be bool")
        if not isinstance(self.reviewer, str) or not self.reviewer.strip():
            raise ValueError("MemoryCandidateReviewDecision reviewer must be non-empty")
        if not isinstance(self.reason, str):
            raise ValueError("MemoryCandidateReviewDecision reason must be str")
        if not isinstance(self.metadata, dict):
            raise ValueError("MemoryCandidateReviewDecision metadata must be a dict")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["decided_at"] = self.decided_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryCandidateReviewDecision":
        if not isinstance(data, dict):
            raise ValueError("MemoryCandidateReviewDecision payload must be a dict")
        values = dict(data)
        if isinstance(values.get("decided_at"), str):
            values["decided_at"] = datetime.fromisoformat(values["decided_at"])
        return cls(**values).validate()


@dataclass
class MemoryPromotionResult:
    candidate_id: str
    promoted: bool
    dry_run: bool
    memory_id: Optional[str] = None
    error: str = ""
    memory_preview: Optional[MemoryRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "promoted": self.promoted,
            "dry_run": self.dry_run,
            "memory_id": self.memory_id,
            "error": self.error,
            "memory_preview": (
                self.memory_preview.to_dict() if self.memory_preview is not None else None
            ),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


class MemoryCandidatePromoter:
    """Review-gated memory promotion. Persistence is opt-in and never the default."""

    def promote(
        self,
        candidate: MemoryCandidate,
        decision: Optional[MemoryCandidateReviewDecision],
        memory_store: Optional[Any] = None,
        dry_run: bool = True,
    ) -> MemoryPromotionResult:
        candidate.validate()
        preview = self._build_memory_preview(candidate, decision, dry_run)
        if dry_run:
            approval_verified = self._has_matching_approval(candidate, decision)
            return MemoryPromotionResult(
                candidate_id=candidate.candidate_id,
                promoted=False,
                dry_run=True,
                memory_preview=preview,
                metadata={
                    "approval_verified": approval_verified,
                    "summary_only": True,
                },
            )
        decision_error = self._validate_approved_decision(candidate, decision)
        if decision_error:
            return self._failed(candidate, False, preview, decision_error)
        if memory_store is None:
            return self._failed(
                candidate,
                False,
                preview,
                "memory store is required when dry_run is False",
            )
        if not hasattr(memory_store, "save_memory"):
            return self._failed(
                candidate,
                False,
                preview,
                "memory store must provide save_memory(record)",
            )
        saved = memory_store.save_memory(preview)
        return MemoryPromotionResult(
            candidate_id=candidate.candidate_id,
            promoted=True,
            dry_run=False,
            memory_id=saved.memory_id,
            memory_preview=preview,
            metadata={"approval_verified": True, "summary_only": True},
        )

    def promote_many(
        self,
        candidates: Iterable[MemoryCandidate],
        decisions: Mapping[str, MemoryCandidateReviewDecision],
        memory_store: Optional[Any] = None,
        dry_run: bool = True,
    ) -> List[MemoryPromotionResult]:
        return [
            self.promote(
                candidate,
                decisions.get(candidate.candidate_id),
                memory_store=memory_store,
                dry_run=dry_run,
            )
            for candidate in candidates
        ]

    @staticmethod
    def _validate_approved_decision(
        candidate: MemoryCandidate,
        decision: Optional[MemoryCandidateReviewDecision],
    ) -> str:
        if decision is None:
            return "approved review decision is required"
        decision.validate()
        if decision.candidate_id != candidate.candidate_id:
            return "review decision candidate_id does not match candidate"
        if not decision.approved:
            return "memory candidate review rejected"
        return ""

    @staticmethod
    def _build_memory_preview(
        candidate: MemoryCandidate,
        decision: Optional[MemoryCandidateReviewDecision],
        dry_run: bool,
    ) -> MemoryRecord:
        preview = candidate.to_memory_record()
        approved = MemoryCandidatePromoter._has_matching_approval(candidate, decision)
        reviewer = decision.reviewer if approved else ""
        approval_reason = decision.reason if approved else ""
        preview.metadata.update(
            {
                "source_candidate_id": candidate.candidate_id,
                "approved_by": reviewer,
                "reviewer": reviewer,
                "approval_reason": approval_reason,
                "promoted_from_reflection": True,
                "dry_run": dry_run,
            }
        )
        return preview

    @staticmethod
    def _has_matching_approval(
        candidate: MemoryCandidate,
        decision: Optional[MemoryCandidateReviewDecision],
    ) -> bool:
        return bool(
            decision is not None
            and decision.approved
            and decision.candidate_id == candidate.candidate_id
            and isinstance(decision.reviewer, str)
            and decision.reviewer.strip()
        )

    @staticmethod
    def _failed(
        candidate: MemoryCandidate,
        dry_run: bool,
        preview: MemoryRecord,
        error: str,
    ) -> MemoryPromotionResult:
        return MemoryPromotionResult(
            candidate_id=candidate.candidate_id,
            promoted=False,
            dry_run=dry_run,
            error=error,
            memory_preview=preview,
            metadata={"approval_verified": False, "summary_only": True},
        )
