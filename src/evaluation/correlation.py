import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from src.evaluation.models import EvalReport
from src.evaluation.projection import EvaluationTargetProjection
from src.evaluation.store import EvalRecord


@dataclass
class CorrelationReport:
    target_id: str
    evaluation_passed: bool
    average_score: float
    failed_cases: List[str] = field(default_factory=list)
    event_counts: Dict[str, int] = field(default_factory=dict)
    skill_event_counts: Dict[str, int] = field(default_factory=dict)
    tool_event_counts: Dict[str, int] = field(default_factory=dict)
    tool_denied_count: int = 0
    tool_sandbox_denied_count: int = 0
    tool_approval_required_count: int = 0
    errors: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "evaluation_passed": self.evaluation_passed,
            "average_score": self.average_score,
            "failed_cases": list(self.failed_cases),
            "event_counts": dict(self.event_counts),
            "skill_event_counts": dict(self.skill_event_counts),
            "tool_event_counts": dict(self.tool_event_counts),
            "tool_denied_count": self.tool_denied_count,
            "tool_sandbox_denied_count": self.tool_sandbox_denied_count,
            "tool_approval_required_count": self.tool_approval_required_count,
            "errors": list(self.errors),
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


class RuntimeAuditCorrelation:
    """Correlate deterministic evaluation outcomes with safe runtime/audit summaries."""

    @staticmethod
    def correlate(
        evaluation: Union[EvalReport, Iterable[EvalRecord]],
        runtime_projection: Union[EvaluationTargetProjection, Mapping[str, Any]],
        tool_audit_report: Optional[Any] = None,
    ) -> CorrelationReport:
        projection = _projection_data(runtime_projection)
        evaluation_passed, average_score, failed_cases, evaluation_source = _evaluation_summary(
            evaluation
        )
        tool_event_counts = dict(projection.get("tool_event_counts", {}))
        denied_count = int(tool_event_counts.get("tool_denied", 0))
        sandbox_denied_count = int(tool_event_counts.get("tool_sandbox_denied", 0))
        approval_required_count = int(tool_event_counts.get("tool_approval_required", 0))
        if tool_audit_report is not None:
            denied_count = int(getattr(tool_audit_report, "denied_count", denied_count))
            sandbox_denied_count = int(
                getattr(tool_audit_report, "sandbox_denied_count", sandbox_denied_count)
            )
            approval_required_count = int(
                getattr(tool_audit_report, "approval_required_count", approval_required_count)
            )
        event_counts = dict(projection.get("event_counts", {}))
        projected_errors = [error for error in projection.get("errors", []) if error]
        report = CorrelationReport(
            target_id=str(projection.get("target_id", "") or ""),
            evaluation_passed=evaluation_passed,
            average_score=average_score,
            failed_cases=failed_cases,
            event_counts=event_counts,
            skill_event_counts=dict(projection.get("skill_event_counts", {})),
            tool_event_counts=tool_event_counts,
            tool_denied_count=denied_count,
            tool_sandbox_denied_count=sandbox_denied_count,
            tool_approval_required_count=approval_required_count,
            errors=(
                [f"{len(projected_errors)} projected runtime error(s) recorded"]
                if projected_errors
                else []
            ),
            metadata={
                "evaluation_source": evaluation_source,
                "runtime_source": "summary_projection",
                "tool_audit_correlated": tool_audit_report is not None,
                "projected_error_count": len(projected_errors),
                "summary_only": True,
            },
        )
        report.summary = _summary_text(report)
        return report


def export_correlation_report_json(report: CorrelationReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)


def export_correlation_report_text(report: CorrelationReport) -> str:
    failed = ", ".join(report.failed_cases) if report.failed_cases else "none"
    return "\n".join(
        [
            f"Evaluation Correlation Report for target {report.target_id or '<unknown>'}",
            f"Evaluation passed: {report.evaluation_passed}",
            f"Average score: {report.average_score:.4f}",
            f"Failed cases: {failed}",
            f"Total events: {sum(report.event_counts.values())}",
            f"Tool denied: {report.tool_denied_count}",
            f"Tool sandbox denied: {report.tool_sandbox_denied_count}",
            f"Tool approval required: {report.tool_approval_required_count}",
        ]
    )


def _projection_data(
    projection: Union[EvaluationTargetProjection, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if isinstance(projection, EvaluationTargetProjection):
        return projection.to_dict()
    if isinstance(projection, Mapping):
        return projection
    raise TypeError("runtime_projection must be an EvaluationTargetProjection or mapping")


def _evaluation_summary(
    evaluation: Union[EvalReport, Iterable[EvalRecord]],
) -> tuple:
    if isinstance(evaluation, EvalReport):
        return (
            evaluation.failed_cases == 0,
            evaluation.average_score,
            [result.case_id for result in evaluation.results if not result.passed],
            "eval_report",
        )
    records = list(evaluation)
    if not all(isinstance(record, EvalRecord) for record in records):
        raise TypeError("evaluation must be an EvalReport or iterable of EvalRecord")
    score = round(sum(record.score for record in records) / len(records), 4) if records else 0.0
    return (
        bool(records) and all(record.passed for record in records),
        score,
        [record.case_id for record in records if not record.passed],
        "eval_records",
    )


def _summary_text(report: CorrelationReport) -> str:
    evaluation_status = "passed" if report.evaluation_passed else "failed"
    return (
        f"evaluation={evaluation_status}; score={report.average_score:.4f}; "
        f"events={sum(report.event_counts.values())}; "
        f"tool_denied={report.tool_denied_count}; "
        f"sandbox_denied={report.tool_sandbox_denied_count}; "
        f"approval_required={report.tool_approval_required_count}"
    )
