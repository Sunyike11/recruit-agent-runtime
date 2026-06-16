from typing import List, Optional

from src.evaluation import CorrelationReport, EvalRecord, EvalReport
from src.reflection.models import ReflectionRecord, ReflectionSourceType, ReflectionStatus


def reflection_from_eval_report(
    eval_report: EvalReport,
    target_id: Optional[str] = None,
    target_type: Optional[str] = None,
) -> ReflectionRecord:
    status = (
        ReflectionStatus.SUCCESS.value
        if eval_report.failed_cases == 0
        else ReflectionStatus.WARNING.value
    )
    failed_cases = [result.case_id for result in eval_report.results if not result.passed]
    findings = [
        f"Evaluation passed cases: {eval_report.passed_cases}/{eval_report.total_cases}.",
        f"Evaluation average score: {eval_report.average_score:.4f}.",
    ]
    if failed_cases:
        findings.append(f"Failed evaluation cases: {', '.join(failed_cases)}.")
    actions = ["Review failed evaluation cases"] if failed_cases else []
    resolved_target_id = target_id or ""
    return ReflectionRecord(
        source_type=ReflectionSourceType.EVAL_REPORT.value,
        source_id=f"eval_report:{resolved_target_id or 'batch'}",
        target_type=target_type or "manual",
        target_id=resolved_target_id,
        status=status,
        summary=(
            f"Deterministic evaluation {status}: "
            f"{eval_report.passed_cases}/{eval_report.total_cases} cases passed."
        ),
        findings=findings,
        recommended_actions=actions,
        evidence_refs=failed_cases,
        tags=["evaluation", status],
        metadata={
            "total_cases": eval_report.total_cases,
            "failed_cases": eval_report.failed_cases,
            "average_score": eval_report.average_score,
            "summary_only": True,
        },
    ).validate()


def reflection_from_eval_record(eval_record: EvalRecord) -> ReflectionRecord:
    status = (
        ReflectionStatus.SUCCESS.value
        if eval_record.passed
        else ReflectionStatus.WARNING.value
    )
    actions = ["Review failed evaluation cases"] if not eval_record.passed else []
    return ReflectionRecord(
        source_type=ReflectionSourceType.EVAL_RECORD.value,
        source_id=eval_record.eval_id,
        target_type=eval_record.target_type,
        target_id=eval_record.target_id,
        status=status,
        summary=(
            f"Evaluation record {eval_record.case_id} "
            f"{'passed' if eval_record.passed else 'failed'} with score {eval_record.score:.4f}."
        ),
        findings=[
            f"Evaluation case id: {eval_record.case_id}.",
            f"Evaluation score: {eval_record.score:.4f}.",
        ],
        recommended_actions=actions,
        evidence_refs=[eval_record.eval_id, eval_record.case_id],
        tags=["evaluation", "record", status],
        metadata={"score": eval_record.score, "summary_only": True},
    ).validate()


def reflection_from_correlation_report(correlation_report: CorrelationReport) -> ReflectionRecord:
    has_policy_signal = (
        correlation_report.tool_denied_count > 0
        or correlation_report.tool_sandbox_denied_count > 0
        or correlation_report.tool_approval_required_count > 0
    )
    if not correlation_report.evaluation_passed:
        status = ReflectionStatus.FAILURE.value
    elif has_policy_signal:
        status = ReflectionStatus.WARNING.value
    else:
        status = ReflectionStatus.SUCCESS.value
    findings = [
        f"Evaluation average score: {correlation_report.average_score:.4f}.",
        f"Runtime event count: {sum(correlation_report.event_counts.values())}.",
        f"Tool denied count: {correlation_report.tool_denied_count}.",
        f"Tool sandbox denied count: {correlation_report.tool_sandbox_denied_count}.",
        f"Tool approval required count: {correlation_report.tool_approval_required_count}.",
    ]
    if correlation_report.failed_cases:
        findings.append(
            f"Failed evaluation cases: {', '.join(correlation_report.failed_cases)}."
        )
    actions = _correlation_actions(correlation_report)
    return ReflectionRecord(
        source_type=ReflectionSourceType.CORRELATION_REPORT.value,
        source_id=f"correlation_report:{correlation_report.target_id or 'unknown'}",
        target_type="runtime_timeline" if correlation_report.target_id else "manual",
        target_id=correlation_report.target_id,
        status=status,
        summary=(
            f"Evaluation/audit correlation {status}: "
            f"score {correlation_report.average_score:.4f}, "
            f"failed cases {len(correlation_report.failed_cases)}."
        ),
        findings=findings,
        recommended_actions=actions,
        evidence_refs=list(correlation_report.failed_cases),
        tags=["evaluation", "correlation", status],
        metadata={
            "event_count": sum(correlation_report.event_counts.values()),
            "tool_denied_count": correlation_report.tool_denied_count,
            "tool_sandbox_denied_count": correlation_report.tool_sandbox_denied_count,
            "tool_approval_required_count": correlation_report.tool_approval_required_count,
            "summary_only": True,
        },
    ).validate()


def _correlation_actions(correlation_report: CorrelationReport) -> List[str]:
    actions = []
    if not correlation_report.evaluation_passed:
        actions.extend(
            [
                "Review failed evaluation cases",
                "Inspect runtime timeline and audit report",
            ]
        )
    if correlation_report.tool_denied_count > 0:
        actions.append("Review tool permission policy")
    if correlation_report.tool_sandbox_denied_count > 0:
        actions.append("Review sandbox profile")
    if correlation_report.tool_approval_required_count > 0:
        actions.append("Review tool approval requirements")
    return actions
