import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.memory import MemoryContextEligibilityPolicy, MemoryGovernancePolicy
from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutor
from src.skills.memory_context_adapter import (
    build_shadow_workflow_memory_context,
    create_skill_execution_context_with_memory,
)
from src.skills.workflow import RecruitmentSkillWorkflow, SkillWorkflowResult


FEATURE_FLAG_ENV = "RECRUIT_AGENT_SKILL_BACKED_GRAPH_VARIANT"


@dataclass
class MemoryContextInjectionConfig:
    enabled: bool = False
    max_items: int = 5
    max_chars: int = 2000
    allow_memory_context: bool = False
    require_governance: bool = True
    target_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContextInjectionResult:
    built: bool
    memory_context: Optional[Any] = None
    skill_context: Optional[SkillExecutionContext] = None
    preview: str = ""
    reason: str = ""
    input_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built": self.built,
            "memory_context": self.memory_context is not None,
            "skill_context": self.skill_context is not None,
            "preview": self.preview,
            "reason": self.reason,
            "input_count": self.input_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class GraphVariantConfig:
    enabled: bool = False
    variant_name: str = "skill_backed_recruit_graph_preview"
    use_skill_planner: bool = True
    use_skill_retriever: bool = True
    use_skill_matcher: bool = True
    use_skill_refiner: bool = True
    allow_memory_context: bool = False
    memory_context_config: Optional[MemoryContextInjectionConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphVariantBuildResult:
    built: bool
    config: GraphVariantConfig
    variant: Optional["SkillBackedRecruitGraphVariant"] = None
    reason: str = ""
    errors: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built": self.built,
            "config": self.config.to_dict(),
            "variant": self.variant is not None,
            "reason": self.reason,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


class SkillBackedRecruitGraphVariant:
    """Explicit preview-only variant backed by the shadow skill workflow."""

    def __init__(
        self,
        skill_executor: SkillExecutor,
        config: Optional[GraphVariantConfig] = None,
        low_score_threshold: float = 60.0,
    ):
        self.config = config or GraphVariantConfig(enabled=True)
        self.skill_executor = skill_executor
        self.workflow = RecruitmentSkillWorkflow(
            skill_executor=skill_executor,
            low_score_threshold=low_score_threshold,
        )

    def invoke(
        self,
        state: Mapping[str, Any],
        *,
        top_k: int = 5,
        context: Optional[SkillExecutionContext] = None,
        memory_records: Optional[list] = None,
        memory_context_config: Optional[MemoryContextInjectionConfig] = None,
        eligibility_policy: Optional[MemoryContextEligibilityPolicy] = None,
        governance_policy: Optional[MemoryGovernancePolicy] = None,
        governance_store: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_jd = _extract_raw_jd(state)
        execution_context = context or SkillExecutionContext()
        memory_injection = MemoryContextInjectionResult(
            built=False,
            reason="memory context disabled",
            metadata={"summary_only": True, "read_only": True},
        )
        if self.config.allow_memory_context:
            memory_injection = create_skill_backed_variant_context(
                base_context=execution_context,
                memory_records=memory_records or [],
                variant_config=self.config,
                memory_config=memory_context_config,
                eligibility_policy=eligibility_policy,
                governance_policy=governance_policy,
                governance_store=governance_store,
            )
            execution_context = memory_injection.skill_context or execution_context
        else:
            execution_context.memory_context = None
        workflow_result = self.workflow.run(
            raw_jd=raw_jd,
            top_k=top_k,
            context=execution_context,
            metadata={
                "variant_name": self.config.variant_name,
                **dict(metadata or {}),
            },
        )
        return skill_workflow_result_to_state_update(
            workflow_result,
            config=self.config,
            raw_jd=raw_jd,
            memory_injection=memory_injection,
        )

    def __call__(self, state: Mapping[str, Any], **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)


def should_use_skill_backed_variant(
    config: Optional[GraphVariantConfig] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    if config is not None:
        return bool(config.enabled)
    value = (env or os.environ).get(FEATURE_FLAG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_skill_backed_recruit_graph_variant(
    *,
    config: Optional[GraphVariantConfig] = None,
    registry: Any = None,
    skill_executor: Optional[SkillExecutor] = None,
    low_score_threshold: float = 60.0,
) -> GraphVariantBuildResult:
    variant_config = config or GraphVariantConfig()
    metadata = {
        "mode": "skill_backed_graph_variant_preview",
        "default_production_graph_replaced": False,
        "memory_context_allowed": bool(variant_config.allow_memory_context),
        "memory_context_config_enabled": bool(
            variant_config.memory_context_config.enabled
            if variant_config.memory_context_config is not None
            else False
        ),
        "feature_flag_env": FEATURE_FLAG_ENV,
    }
    if not should_use_skill_backed_variant(variant_config):
        return GraphVariantBuildResult(
            built=False,
            config=variant_config,
            reason="feature flag disabled",
            metadata=metadata,
        )

    try:
        if skill_executor is None and registry is None:
            raise ValueError("skill-backed graph variant requires a registry or skill_executor")
        executor = skill_executor or SkillExecutor(registry)
        variant = SkillBackedRecruitGraphVariant(
            skill_executor=executor,
            config=variant_config,
            low_score_threshold=low_score_threshold,
        )
        return GraphVariantBuildResult(
            built=True,
            config=variant_config,
            variant=variant,
            reason="variant built",
            metadata=metadata,
        )
    except Exception as exc:
        return GraphVariantBuildResult(
            built=False,
            config=variant_config,
            reason="variant build failed",
            errors=[type(exc).__name__],
            metadata=metadata,
        )


def skill_workflow_result_to_state_update(
    result: SkillWorkflowResult,
    *,
    config: GraphVariantConfig,
    raw_jd: str,
    memory_injection: Optional[MemoryContextInjectionResult] = None,
) -> Dict[str, Any]:
    memory_result = memory_injection or MemoryContextInjectionResult(
        built=False,
        reason="memory context disabled",
    )
    return {
        "messages": [],
        "raw_jd_preview": raw_jd,
        "extracted_jd_preview": dict(result.job_requirement or {}),
        "candidate_pool_preview": list(result.retrieved_candidates),
        "resume_documents_preview": list(result.resume_documents),
        "retrieved_evidence_preview": list(result.evidence),
        "final_reports_preview": list(result.match_reports),
        "refined_query_preview": result.refined_query or "",
        "next_action_preview": "end" if result.success else "failed",
        "memory_context_preview": memory_result.preview if memory_result.built else "",
        "variant_metadata": {
            "variant_name": config.variant_name,
            "status": result.status,
            "success": result.success,
            "step_count": len(result.steps),
            "skill_names": [step.skill_name for step in result.steps],
            "allow_memory_context": bool(config.allow_memory_context),
            "memory_context_used": bool(memory_result.built),
            "memory_context_reason": memory_result.reason,
            "default_production_graph_replaced": False,
            "preview_only": True,
            "error_type": type(result.error).__name__ if result.error else "",
        },
    }


def build_variant_memory_context(
    memory_records,
    *,
    variant_config: Optional[GraphVariantConfig] = None,
    memory_config: Optional[MemoryContextInjectionConfig] = None,
    eligibility_policy: Optional[MemoryContextEligibilityPolicy] = None,
    governance_policy: Optional[MemoryGovernancePolicy] = None,
    governance_store: Optional[Any] = None,
) -> MemoryContextInjectionResult:
    records = list(memory_records or [])
    graph_config = variant_config or GraphVariantConfig()
    config = memory_config or graph_config.memory_context_config or MemoryContextInjectionConfig()
    base_metadata = {
        "summary_only": True,
        "read_only": True,
        "preview_only": True,
        "default_production_graph_replaced": False,
        "memory_store_written": False,
    }
    if not graph_config.enabled:
        return MemoryContextInjectionResult(
            built=False,
            reason="graph variant disabled",
            input_count=len(records),
            metadata=base_metadata,
        )
    if not graph_config.allow_memory_context:
        return MemoryContextInjectionResult(
            built=False,
            reason="graph variant memory context disabled",
            input_count=len(records),
            metadata=base_metadata,
        )
    if not config.enabled or not config.allow_memory_context:
        return MemoryContextInjectionResult(
            built=False,
            reason="memory context injection disabled",
            input_count=len(records),
            metadata=base_metadata,
        )

    effective_governance_policy = governance_policy
    if config.require_governance and eligibility_policy is None:
        effective_governance_policy = effective_governance_policy or MemoryGovernancePolicy()

    memory_context = build_shadow_workflow_memory_context(
        records,
        eligibility_policy=eligibility_policy,
        governance_policy=effective_governance_policy,
        governance_store=governance_store,
        target_context=config.target_context or None,
        max_items=config.max_items,
        max_chars=config.max_chars,
    )
    preview = memory_context.format_for_prompt()
    built = not memory_context.is_empty()
    metadata = dict(base_metadata)
    metadata.update(
        {
            "require_governance": bool(config.require_governance),
            "target_context_keys": sorted(str(key) for key in config.target_context.keys()),
        }
    )
    return MemoryContextInjectionResult(
        built=built,
        memory_context=memory_context if built else None,
        preview=preview if built else "",
        reason=(
            "eligible memory context built"
            if built
            else "no eligible memory context after eligibility/governance filtering"
        ),
        input_count=len(records),
        metadata=metadata,
    )


def create_skill_backed_variant_context(
    *,
    base_context: Optional[SkillExecutionContext] = None,
    memory_records: Optional[list] = None,
    variant_config: Optional[GraphVariantConfig] = None,
    memory_config: Optional[MemoryContextInjectionConfig] = None,
    eligibility_policy: Optional[MemoryContextEligibilityPolicy] = None,
    governance_policy: Optional[MemoryGovernancePolicy] = None,
    governance_store: Optional[Any] = None,
) -> MemoryContextInjectionResult:
    result = build_variant_memory_context(
        memory_records or [],
        variant_config=variant_config,
        memory_config=memory_config,
        eligibility_policy=eligibility_policy,
        governance_policy=governance_policy,
        governance_store=governance_store,
    )
    if result.built:
        context = create_skill_execution_context_with_memory(
            base_context=base_context,
            memory_preview=result.memory_context,
            metadata={
                "memory_context_mode": "skill_backed_variant_preview",
                "memory_context_injection": True,
            },
        )
    else:
        existing = base_context or SkillExecutionContext()
        metadata = dict(existing.metadata)
        metadata.update(
            {
                "memory_context_mode": "skill_backed_variant_preview",
                "memory_context_injection": False,
            }
        )
        context = SkillExecutionContext(
            task_id=existing.task_id,
            session_id=existing.session_id,
            thread_id=existing.thread_id,
            memory_context=None,
            runtime_context=(
                dict(existing.runtime_context)
                if isinstance(existing.runtime_context, dict)
                else existing.runtime_context
            ),
            metadata=metadata,
        )
    result.skill_context = context
    return result


def _extract_raw_jd(state: Mapping[str, Any]) -> str:
    for key in ("raw_jd", "query", "jd_text"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value

    messages = state.get("messages") or []
    if messages:
        first = messages[0]
        content = getattr(first, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(first, Mapping):
            value = first.get("content")
            if isinstance(value, str) and value.strip():
                return value
    return ""
