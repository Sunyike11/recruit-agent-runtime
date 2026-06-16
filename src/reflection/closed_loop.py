from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

from src.domain.models import CandidateProfile, JobRequirement
from src.evaluation import (
    EvalCase,
    EvalReport,
    EvalRunner,
    RuntimeAuditCorrelation,
    create_eval_record,
    project_runtime_timeline,
)
from src.memory import (
    InMemoryMemoryGovernanceStore,
    MemoryContextEligibilityPolicy,
    MemoryGovernancePolicy,
    MemoryRecord,
)
from src.reflection.derivation import reflection_from_correlation_report
from src.reflection.memory_projection import MemoryCandidate, ReflectionMemoryProjectionPolicy
from src.reflection.memory_promotion import MemoryCandidatePromoter, MemoryCandidateReviewDecision
from src.runtime import InMemoryRuntimeStore, SessionManager, TaskManager
from src.skills import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RecruitmentSkillWorkflow,
    RetrieverSkill,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillRegistry,
    build_shadow_workflow_memory_context,
    create_skill_execution_context_with_memory,
)


ApprovalInput = Optional[
    Union[
        MemoryCandidateReviewDecision,
        Callable[[MemoryCandidate], Optional[MemoryCandidateReviewDecision]],
    ]
]
PostPromotionHook = Callable[[MemoryRecord, InMemoryMemoryGovernanceStore], None]


@dataclass
class ClosedLoopDemoResult:
    status: str
    success: bool
    first_workflow_summary: Dict[str, Any] = field(default_factory=dict)
    evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    correlation_summary: Dict[str, Any] = field(default_factory=dict)
    reflection_summary: Dict[str, Any] = field(default_factory=dict)
    memory_candidate_summary: Dict[str, Any] = field(default_factory=dict)
    promotion_summary: Dict[str, Any] = field(default_factory=dict)
    eligibility_summary: Dict[str, Any] = field(default_factory=dict)
    memory_context_preview: str = ""
    second_workflow_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "first_workflow_summary": dict(self.first_workflow_summary),
            "evaluation_summary": dict(self.evaluation_summary),
            "correlation_summary": dict(self.correlation_summary),
            "reflection_summary": dict(self.reflection_summary),
            "memory_candidate_summary": dict(self.memory_candidate_summary),
            "promotion_summary": dict(self.promotion_summary),
            "eligibility_summary": dict(self.eligibility_summary),
            "memory_context_preview": self.memory_context_preview,
            "second_workflow_summary": dict(self.second_workflow_summary),
            "metadata": dict(self.metadata),
        }


class ClosedLoopDemoHarness:
    """Deterministic feedback demo that operates only on the shadow skill workflow."""

    def __init__(
        self,
        memory_store: Optional[Any] = None,
        governance_store: Optional[InMemoryMemoryGovernanceStore] = None,
        evaluation_case: Optional[EvalCase] = None,
        post_promotion_hook: Optional[PostPromotionHook] = None,
    ):
        self.memory_store = memory_store if memory_store is not None else _InMemoryDemoMemoryStore()
        self.governance_store = governance_store or InMemoryMemoryGovernanceStore()
        self.evaluation_case = evaluation_case or _default_evaluation_case()
        self.post_promotion_hook = post_promotion_hook

    def run(
        self,
        raw_jd: str,
        approval_decision: ApprovalInput = None,
        dry_run: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopDemoResult:
        if not isinstance(raw_jd, str) or not raw_jd.strip():
            raise ValueError("ClosedLoopDemoHarness raw_jd must be a non-empty string")

        runtime_store, task, base_context = _create_runtime_context()
        workflow = _build_shadow_workflow(runtime_store)
        runtime_store.append_event(
            "task_started",
            session_id=task.session_id,
            task_id=task.task_id,
            payload={"shadow_stage": "first"},
        )
        first_result = workflow.run(
            raw_jd,
            context=base_context,
            metadata={"shadow_stage": "first"},
        )
        runtime_store.append_event(
            "task_completed" if first_result.success else "task_failed",
            session_id=task.session_id,
            task_id=task.task_id,
            payload={"shadow_stage": "first"},
        )

        projection = project_runtime_timeline(
            runtime_store.list_events(task_id=task.task_id),
            target_id=task.task_id,
        )
        eval_case = replace(self.evaluation_case, input_data=projection.to_dict())
        eval_result = EvalRunner().run_case(eval_case)
        eval_report = EvalReport.from_results(
            [eval_result],
            metadata={"runner": "closed_loop_demo", "summary_only": True},
        )
        eval_record = create_eval_record(
            self.evaluation_case,
            eval_result,
            target_id=task.task_id,
            metadata={"closed_loop_demo": True, "summary_only": True},
        )
        correlation = RuntimeAuditCorrelation.correlate(eval_report, projection)
        reflection = reflection_from_correlation_report(correlation)
        candidate = ReflectionMemoryProjectionPolicy().project(reflection)
        if candidate is None:
            return self._failed_result(
                first_result,
                eval_report,
                correlation,
                reflection,
                "reflection did not produce a memory candidate",
                dry_run,
                metadata,
            )

        decision = _resolve_approval_decision(approval_decision, candidate)
        promotion = MemoryCandidatePromoter().promote(
            candidate,
            decision,
            memory_store=self.memory_store,
            dry_run=dry_run,
        )
        preview_memory = promotion.memory_preview
        if promotion.promoted and preview_memory is not None and self.post_promotion_hook is not None:
            self.post_promotion_hook(preview_memory, self.governance_store)

        eligibility_policy = MemoryContextEligibilityPolicy(
            governance_policy=MemoryGovernancePolicy(),
            governance_store=self.governance_store,
        )
        eligibility = (
            eligibility_policy.evaluate(preview_memory)
            if preview_memory is not None
            else None
        )
        memory_context = build_shadow_workflow_memory_context(
            [preview_memory] if preview_memory is not None else [],
            eligibility_policy=eligibility_policy,
        )
        second_context = create_skill_execution_context_with_memory(
            base_context,
            memory_context,
            metadata={"shadow_stage": "second"},
        )
        second_result = workflow.run(
            raw_jd,
            context=second_context,
            metadata={"shadow_stage": "second"},
        )

        return ClosedLoopDemoResult(
            status="completed" if first_result.success and second_result.success else "failed",
            success=first_result.success and second_result.success,
            first_workflow_summary=_workflow_summary(first_result),
            evaluation_summary=_evaluation_summary(eval_report, eval_record.eval_id),
            correlation_summary=_correlation_summary(correlation),
            reflection_summary=_reflection_summary(reflection),
            memory_candidate_summary=_candidate_summary(candidate),
            promotion_summary=_promotion_summary(promotion),
            eligibility_summary=_eligibility_summary(eligibility),
            memory_context_preview=memory_context.format_for_prompt(),
            second_workflow_summary=_workflow_summary(second_result),
            metadata=_result_metadata(dry_run, approval_decision, metadata),
        )

    @staticmethod
    def _failed_result(
        first_result: Any,
        eval_report: EvalReport,
        correlation: Any,
        reflection: Any,
        error: str,
        dry_run: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> ClosedLoopDemoResult:
        return ClosedLoopDemoResult(
            status="failed",
            success=False,
            first_workflow_summary=_workflow_summary(first_result),
            evaluation_summary=_evaluation_summary(eval_report, ""),
            correlation_summary=_correlation_summary(correlation),
            reflection_summary=_reflection_summary(reflection),
            promotion_summary={"promoted": False, "dry_run": dry_run, "error": error},
            metadata=_result_metadata(dry_run, None, metadata),
        )


class _InMemoryDemoMemoryStore:
    """Isolated memory persistence surface used only by the local demo harness."""

    def __init__(self):
        self._records: Dict[str, MemoryRecord] = {}

    def save_memory(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    def get_memory(self, memory_id: str) -> MemoryRecord:
        return self._records[memory_id]

    def list_memories(self) -> List[MemoryRecord]:
        return list(self._records.values())


def _create_runtime_context():
    runtime_store = InMemoryRuntimeStore()
    session = SessionManager(runtime_store).create_session(
        metadata={"phase": "5K", "summary_only": True}
    )
    task = TaskManager(runtime_store).create_task(
        session.session_id,
        jd_text="<omitted shadow input>",
        thread_id="phase5k-shadow-closed-loop",
        metadata={"summary_only": True},
    )
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"phase": "5K", "summary_only": True},
    )
    return runtime_store, task, context


def _build_shadow_workflow(runtime_store: InMemoryRuntimeStore) -> RecruitmentSkillWorkflow:
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=_fake_planner))
    registry.register(RetrieverSkill(retrieve_callable=_fake_retriever))
    registry.register(CandidateMatchSkill(match_callable=_fake_matcher))
    registry.register(QueryRefineSkill(refine_callable=_fake_refiner))
    return RecruitmentSkillWorkflow(
        SkillExecutor(registry, recorder=SkillExecutionRecorder(runtime_store))
    )


def _fake_planner(input_data: Dict[str, Any], context: SkillExecutionContext):
    return {
        "job_requirement": JobRequirement(
            job_id="job_closed_loop_demo",
            title="Agent Engineer",
            required_skills=["Python", "LangGraph"],
            metadata={"search_query": "Python LangGraph"},
        ).to_dict(),
        "extracted_keywords": ["Python", "LangGraph"],
    }


def _fake_retriever(input_data: Dict[str, Any], context: SkillExecutionContext):
    memory_seen = bool(context.memory_context is not None and not context.memory_context.is_empty())
    return {
        "candidates": [
            CandidateProfile(
                candidate_id="candidate_closed_loop_demo",
                name="Demo Candidate",
                skills=["Python", "LangGraph"],
            ).to_dict()
        ],
        "evidence": ["deterministic summary evidence"],
        "metadata": {"memory_context_seen": memory_seen},
    }


def _fake_matcher(input_data: Dict[str, Any], context: SkillExecutionContext):
    return {
        "total_score": 100.0,
        "recommendation": "strong_match",
        "match_report": {
            "job_id": input_data["job_requirement"]["job_id"],
            "candidate_id": input_data["candidate_profile"]["candidate_id"],
            "total_score": 100.0,
        },
    }


def _fake_refiner(input_data: Dict[str, Any], context: SkillExecutionContext):
    return {"refined_query": f"{input_data['query']} refined"}


def _default_evaluation_case() -> EvalCase:
    return EvalCase(
        case_id="phase5k_runtime_timeline",
        target_type="runtime_timeline",
        checks=[
            {"type": "event_type_present", "event_type": "task_completed"},
            {"type": "event_type_count_at_least", "event_type": "skill_completed", "value": 3},
            {"type": "event_type_absent", "event_type": "skill_failed"},
        ],
        tags=["phase5k", "closed_loop"],
        metadata={"summary_only": True},
    ).validate()


def _resolve_approval_decision(
    approval_decision: ApprovalInput,
    candidate: MemoryCandidate,
) -> Optional[MemoryCandidateReviewDecision]:
    if callable(approval_decision):
        return approval_decision(candidate)
    return approval_decision


def _workflow_summary(result: Any) -> Dict[str, Any]:
    memory_seen = False
    for skill_result in result.skill_results:
        if skill_result.skill_name == "resume_retrieve":
            memory_seen = bool(skill_result.output.get("metadata", {}).get("memory_context_seen"))
            break
    return {
        "status": result.status,
        "success": result.success,
        "skill_result_count": len(result.skill_results),
        "match_report_count": len(result.match_reports),
        "memory_context_seen": memory_seen,
    }


def _evaluation_summary(report: EvalReport, eval_id: str) -> Dict[str, Any]:
    return {
        "eval_id": eval_id,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "average_score": report.average_score,
        "failed_case_ids": [result.case_id for result in report.results if not result.passed],
    }


def _correlation_summary(report: Any) -> Dict[str, Any]:
    return {
        "target_id": report.target_id,
        "evaluation_passed": report.evaluation_passed,
        "average_score": report.average_score,
        "failed_cases": list(report.failed_cases),
        "event_counts": dict(report.event_counts),
        "skill_event_counts": dict(report.skill_event_counts),
        "tool_denied_count": report.tool_denied_count,
        "tool_sandbox_denied_count": report.tool_sandbox_denied_count,
        "tool_approval_required_count": report.tool_approval_required_count,
    }


def _reflection_summary(reflection: Any) -> Dict[str, Any]:
    return {
        "reflection_id": reflection.reflection_id,
        "source_type": reflection.source_type,
        "target_id": reflection.target_id,
        "status": reflection.status,
        "summary": reflection.summary,
        "recommended_actions": list(reflection.recommended_actions),
    }


def _candidate_summary(candidate: MemoryCandidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_reflection_id": candidate.source_reflection_id,
        "memory_type": candidate.memory_type,
        "importance": candidate.importance,
        "requires_approval": candidate.requires_approval,
    }


def _promotion_summary(promotion: Any) -> Dict[str, Any]:
    return {
        "candidate_id": promotion.candidate_id,
        "promoted": promotion.promoted,
        "dry_run": promotion.dry_run,
        "memory_id": promotion.memory_id,
        "error": promotion.error,
    }


def _eligibility_summary(eligibility: Any) -> Dict[str, Any]:
    if eligibility is None:
        return {"eligible": False, "status": "denied", "reason": "no memory preview available"}
    return {
        "memory_id": eligibility.memory_id,
        "eligible": eligibility.eligible,
        "status": eligibility.status,
        "reason": eligibility.reason,
    }


def _result_metadata(
    dry_run: bool,
    approval_decision: ApprovalInput,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "harness": "deterministic_shadow_closed_loop",
        "summary_only": True,
        "dry_run": dry_run,
        "approval_supplied": approval_decision is not None,
        "request_metadata_supplied": bool(metadata),
        "production_graph_used": False,
    }
