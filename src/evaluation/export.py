import json
from typing import Any, Dict, Iterable

from src.evaluation.models import EvalReport, EvalResult
from src.evaluation.store import EvalRecord


def export_eval_report_json(report: EvalReport) -> str:
    return json.dumps(_report_summary(report), ensure_ascii=False, sort_keys=True)


def export_eval_report_text(report: EvalReport) -> str:
    lines = [
        "Evaluation Report",
        f"Total cases: {report.total_cases}",
        f"Passed: {report.passed_cases}",
        f"Failed: {report.failed_cases}",
        f"Average score: {report.average_score:.4f}",
    ]
    for result in report.results:
        status = "passed" if result.passed else "failed"
        lines.append(
            f"- {result.case_id} [{result.target_type}]: {status}; "
            f"score={result.score:.4f}; checks={len(result.checks)}"
        )
    return "\n".join(lines)


def export_eval_records_json(records: Iterable[EvalRecord]) -> str:
    return json.dumps(
        {"records": [_record_summary(record) for record in records]},
        ensure_ascii=False,
        sort_keys=True,
    )


def export_eval_records_text(records: Iterable[EvalRecord]) -> str:
    summaries = [_record_summary(record) for record in records]
    lines = ["Evaluation Records", f"Total records: {len(summaries)}"]
    for record in summaries:
        status = "passed" if record["passed"] else "failed"
        lines.append(
            f"- {record['eval_id']} case={record['case_id']} target={record['target_id']}: "
            f"{status}; score={record['score']:.4f}"
        )
    return "\n".join(lines)


def _report_summary(report: EvalReport) -> Dict[str, Any]:
    return {
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "average_score": report.average_score,
        "results": [_result_summary(result) for result in report.results],
        "metadata_summary": _metadata_summary(report.metadata),
        "summary_only": True,
    }


def _result_summary(result: EvalResult) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "target_type": result.target_type,
        "passed": result.passed,
        "score": result.score,
        "checks": [
            {
                "name": check.get("name", check.get("type", "")),
                "passed": bool(check.get("passed")),
                "error_present": bool(check.get("error")),
            }
            for check in result.checks
        ],
        "error_present": bool(result.error),
        "metadata_summary": _metadata_summary(result.metadata),
        "created_at": result.created_at.isoformat(),
    }


def _record_summary(record: EvalRecord) -> Dict[str, Any]:
    return {
        "eval_id": record.eval_id,
        "case_id": record.case_id,
        "target_type": record.target_type,
        "target_id": record.target_id,
        "passed": record.passed,
        "score": record.score,
        "report_present": bool(record.report_json),
        "created_at": record.created_at.isoformat(),
        "metadata_summary": _metadata_summary(record.metadata),
    }


def _metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "keys": sorted(str(key) for key in metadata),
        "size": len(metadata),
    }
