from typing import Any, Dict, Iterable, Mapping, Optional

from src.memory import (
    MemoryContext,
    MemoryContextEligibilityPolicy,
    MemoryGovernancePolicy,
    MemoryRecord,
    build_eligible_memory_context_preview,
)
from src.skills.context import SkillExecutionContext


class ShadowWorkflowMemoryContext(MemoryContext):
    """Read-only promoted-memory preview supplied only to shadow skill workflows."""

    def __init__(self, preview_text: str):
        super().__init__(items=[], max_chars=max(0, len(preview_text)))
        self.preview_text = preview_text

    def format_for_prompt(self) -> str:
        return self.preview_text

    def is_empty(self) -> bool:
        return "No eligible promoted memory." in self.preview_text


def build_shadow_workflow_memory_context(
    memory_records: Iterable[MemoryRecord],
    eligibility_policy: Optional[MemoryContextEligibilityPolicy] = None,
    governance_policy: Optional[MemoryGovernancePolicy] = None,
    governance_store: Optional[Any] = None,
    target_context: Optional[Mapping[str, Any]] = None,
    max_items: int = 5,
    max_chars: int = 2000,
) -> ShadowWorkflowMemoryContext:
    """Build an opt-in, filtered preview context for deterministic shadow workflow calls."""
    if eligibility_policy is not None and (
        governance_policy is not None or governance_store is not None
    ):
        raise ValueError(
            "pass eligibility_policy or governance_policy/governance_store, not both"
        )
    policy = eligibility_policy or MemoryContextEligibilityPolicy(
        governance_policy=governance_policy,
        governance_store=governance_store,
    )
    preview = build_eligible_memory_context_preview(
        memory_records,
        policy=policy,
        target_context=target_context,
        max_items=max_items,
        max_chars=max_chars,
    )
    return ShadowWorkflowMemoryContext(preview)


def create_skill_execution_context_with_memory(
    base_context: Optional[SkillExecutionContext] = None,
    memory_preview: Optional[ShadowWorkflowMemoryContext] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SkillExecutionContext:
    """Copy a skill context and attach shadow preview memory without mutating its source."""
    existing = base_context or SkillExecutionContext()
    merged_metadata = dict(existing.metadata)
    merged_metadata.update(metadata or {})
    if memory_preview is not None:
        merged_metadata.setdefault("memory_context_mode", "shadow_preview")
    return SkillExecutionContext(
        task_id=existing.task_id,
        session_id=existing.session_id,
        thread_id=existing.thread_id,
        memory_context=memory_preview if memory_preview is not None else existing.memory_context,
        runtime_context=(
            dict(existing.runtime_context)
            if isinstance(existing.runtime_context, dict)
            else existing.runtime_context
        ),
        metadata=merged_metadata,
    )
