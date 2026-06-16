from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.skills.context import SkillExecutionContext
from src.skills.eval import SkillEvalCase, SkillEvalResult
from src.skills.models import SkillResult


@dataclass
class SkillWorkflowStep:
    skill_name: str
    input_data: Dict[str, Any]
    result: SkillResult


@dataclass
class SkillWorkflowResult:
    status: str
    success: bool
    job_requirement: Optional[Dict[str, Any]] = None
    retrieved_candidates: List[Dict[str, Any]] = field(default_factory=list)
    resume_documents: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Any] = field(default_factory=list)
    match_reports: List[Dict[str, Any]] = field(default_factory=list)
    refined_query: Optional[str] = None
    skill_results: List[SkillResult] = field(default_factory=list)
    steps: List[SkillWorkflowStep] = field(default_factory=list)
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "job_requirement": self.job_requirement,
            "retrieved_candidates": self.retrieved_candidates,
            "resume_documents": self.resume_documents,
            "evidence": self.evidence,
            "match_reports": self.match_reports,
            "refined_query": self.refined_query,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class SkillWorkflowEvalCase:
    case_id: str
    raw_jd: str
    top_k: int = 5
    expected_status: str = "completed"
    expected_min_match_reports: int = 0
    expected_refined_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillWorkflowEvalCase":
        return cls(
            case_id=data["case_id"],
            raw_jd=data["raw_jd"],
            top_k=data.get("top_k", 5),
            expected_status=data.get("expected_status", "completed"),
            expected_min_match_reports=data.get("expected_min_match_reports", 0),
            expected_refined_query=data.get("expected_refined_query"),
            metadata=dict(data.get("metadata", {})),
        )


class RecruitmentSkillWorkflow:
    """Shadow recruitment workflow composed from registered skills.

    This class is intentionally separate from the production LangGraph graph.
    It uses SkillExecutor so fake composition can reuse runtime skill events.
    """

    def __init__(self, skill_executor, low_score_threshold: float = 60.0):
        self.skill_executor = skill_executor
        self.low_score_threshold = float(low_score_threshold)

    def run(
        self,
        raw_jd: str,
        top_k: int = 5,
        context: Optional[SkillExecutionContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillWorkflowResult:
        workflow_context = context or SkillExecutionContext()
        workflow_metadata = dict(metadata or {})
        result = SkillWorkflowResult(
            status="running",
            success=False,
            metadata={
                "workflow": "shadow_recruitment_skill_workflow",
                "top_k": top_k,
                **workflow_metadata,
            },
        )

        planner = self._execute_step(
            result,
            "planner_extract",
            {"raw_text": raw_jd, "metadata": workflow_metadata},
            workflow_context,
        )
        if not planner.success:
            return self._failed(result, planner.error)

        job_requirement = planner.output["job_requirement"]
        result.job_requirement = job_requirement
        derived_query = _derive_query(raw_jd, job_requirement)

        retriever = self._execute_step(
            result,
            "resume_retrieve",
            {
                "job_requirement": job_requirement,
                "query": derived_query,
                "top_k": top_k,
                "metadata": workflow_metadata,
            },
            workflow_context,
        )
        if not retriever.success:
            return self._failed(result, retriever.error)

        result.retrieved_candidates = list(retriever.output.get("candidates", []))
        result.resume_documents = list(retriever.output.get("resume_documents", []))
        result.evidence = list(retriever.output.get("evidence", []))

        if result.retrieved_candidates:
            for candidate in result.retrieved_candidates:
                match = self._execute_step(
                    result,
                    "candidate_match",
                    {
                        "job_requirement": job_requirement,
                        "candidate_profile": candidate,
                        "evidence": result.evidence,
                        "metadata": workflow_metadata,
                    },
                    workflow_context,
                )
                if not match.success:
                    return self._failed(result, match.error)
                result.match_reports.append(match.output["match_report"])

        should_refine, reason = self._should_refine(result)
        if should_refine:
            refine = self._execute_step(
                result,
                "query_refine",
                {
                    "query": derived_query or raw_jd,
                    "context": reason,
                    "metadata": workflow_metadata,
                },
                workflow_context,
            )
            if not refine.success:
                return self._failed(result, refine.error)
            result.refined_query = refine.output["refined_query"]
            result.metadata["refinement_reason"] = reason

        result.status = "completed"
        result.success = True
        return result

    def _execute_step(
        self,
        workflow_result: SkillWorkflowResult,
        skill_name: str,
        input_data: Dict[str, Any],
        context: SkillExecutionContext,
    ) -> SkillResult:
        result = self.skill_executor.execute(skill_name, input_data, context=context)
        workflow_result.skill_results.append(result)
        workflow_result.steps.append(
            SkillWorkflowStep(
                skill_name=skill_name,
                input_data=input_data,
                result=result,
            )
        )
        return result

    def _should_refine(self, result: SkillWorkflowResult):
        if not result.retrieved_candidates:
            return True, "no candidates retrieved"
        scores = [
            float(report.get("total_score", 0))
            for report in result.match_reports
        ]
        if scores and max(scores) < self.low_score_threshold:
            return True, f"best score below threshold {self.low_score_threshold}"
        return False, ""

    def _failed(self, result: SkillWorkflowResult, error: str) -> SkillWorkflowResult:
        result.status = "failed"
        result.success = False
        result.error = error
        return result


def run_workflow_eval_case(
    workflow: RecruitmentSkillWorkflow,
    eval_case: SkillWorkflowEvalCase,
    context: Optional[SkillExecutionContext] = None,
) -> SkillEvalResult:
    started_result = workflow.run(
        raw_jd=eval_case.raw_jd,
        top_k=eval_case.top_k,
        context=context,
        metadata=eval_case.metadata,
    )
    checks = _workflow_checks(eval_case, started_result)
    passed = all(check["passed"] for check in checks)
    return SkillEvalResult(
        case_id=eval_case.case_id,
        skill_name="recruitment_skill_workflow",
        skill_version="shadow_v1",
        success=started_result.success,
        passed=passed,
        output=started_result.to_dict(),
        error=started_result.error,
        checks=checks,
        metadata={
            **eval_case.metadata,
            "workflow_eval": True,
        },
    )


def replay_workflow_case_from_fixture(
    fixture,
    workflow: RecruitmentSkillWorkflow,
    context: Optional[SkillExecutionContext] = None,
) -> SkillEvalResult:
    eval_case = fixture if isinstance(fixture, SkillWorkflowEvalCase) else SkillWorkflowEvalCase.from_dict(fixture)
    result = run_workflow_eval_case(workflow, eval_case, context=context)
    result.metadata.update(
        {
            "replay_mode": "workflow_fixture_full_replay",
            "full_replay": True,
        }
    )
    return result


def _workflow_checks(eval_case: SkillWorkflowEvalCase, result: SkillWorkflowResult) -> List[Dict[str, Any]]:
    checks = [
        {
            "name": "expected_status",
            "expected": eval_case.expected_status,
            "actual": result.status,
            "passed": result.status == eval_case.expected_status,
        },
        {
            "name": "expected_min_match_reports",
            "expected": eval_case.expected_min_match_reports,
            "actual": len(result.match_reports),
            "passed": len(result.match_reports) >= eval_case.expected_min_match_reports,
        },
    ]
    if eval_case.expected_refined_query is not None:
        checks.append(
            {
                "name": "expected_refined_query",
                "expected": eval_case.expected_refined_query,
                "actual": result.refined_query,
                "passed": result.refined_query == eval_case.expected_refined_query,
            }
        )
    return checks


def _derive_query(raw_jd: str, job_requirement: Dict[str, Any]) -> str:
    metadata = job_requirement.get("metadata") or {}
    search_query = metadata.get("search_query")
    if isinstance(search_query, str) and search_query.strip():
        return search_query

    skills = job_requirement.get("required_skills") or []
    if skills:
        return " ".join(str(skill) for skill in skills)

    title = job_requirement.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return raw_jd
