import json
from typing import Any, Dict, Mapping

from src.runtime.memory_influence import MemoryInfluenceEvalResult


def sanitize_memory_influence_report(result: MemoryInfluenceEvalResult) -> Dict[str, Any]:
    data = result.to_dict()
    no_memory = data["no_memory_summary"]
    with_memory = data["with_memory_summary"]
    delta = data["delta"]
    return {
        "case_id": data["case_id"],
        "decision": delta["decision"],
        "risk_level": delta["risk_level"],
        "memory_context_used": delta["memory_context_used"],
        "no_memory_summary": _run_summary(no_memory),
        "with_memory_summary": _run_summary(with_memory),
        "delta": {
            "candidate_count_changed": bool(delta["candidate_count_changed"]),
            "report_count_changed": bool(delta["report_count_changed"]),
            "top_score_changed": bool(delta["top_score_changed"]),
            "ranking_changed": bool(delta["ranking_changed"]),
            "candidate_ids_changed": bool(delta["candidate_ids_changed"]),
            "notes": [str(note) for note in delta.get("notes", [])],
        },
        "passed": data.get("passed"),
        "metadata_summary": _metadata_summary(data.get("metadata", {})),
        "summary_only": True,
    }


def export_memory_influence_result_json(result: MemoryInfluenceEvalResult) -> str:
    return json.dumps(
        sanitize_memory_influence_report(result),
        ensure_ascii=False,
        sort_keys=True,
    )


def export_memory_influence_result_text(result: MemoryInfluenceEvalResult) -> str:
    report = sanitize_memory_influence_report(result)
    no_mem = report["no_memory_summary"]
    with_mem = report["with_memory_summary"]
    delta = report["delta"]
    lines = [
        "Memory Influence Report",
        f"case_id: {report['case_id']}",
        f"decision: {report['decision']}",
        f"risk_level: {report['risk_level']}",
        f"memory_context_used: {str(report['memory_context_used']).lower()}",
        "",
        "No-memory:",
        f"  status: {no_mem['status']}",
        f"  candidate_count: {no_mem['candidate_count']}",
        f"  report_count: {no_mem['report_count']}",
        f"  top_score_present: {str(no_mem['top_score_present']).lower()}",
        f"  top_score_summary: {no_mem['top_score_summary']}",
        "",
        "With-memory:",
        f"  status: {with_mem['status']}",
        f"  candidate_count: {with_mem['candidate_count']}",
        f"  report_count: {with_mem['report_count']}",
        f"  top_score_present: {str(with_mem['top_score_present']).lower()}",
        f"  top_score_summary: {with_mem['top_score_summary']}",
        f"  memory_context_provided: {str(with_mem['memory_context_provided']).lower()}",
        f"  memory_context_eligible_count: {with_mem['memory_context_eligible_count']}",
        "",
        "Delta:",
        f"  candidate_count_changed: {str(delta['candidate_count_changed']).lower()}",
        f"  report_count_changed: {str(delta['report_count_changed']).lower()}",
        f"  ranking_changed: {str(delta['ranking_changed']).lower()}",
        f"  top_score_changed: {str(delta['top_score_changed']).lower()}",
        f"  candidate_ids_changed: {str(delta['candidate_ids_changed']).lower()}",
    ]
    return "\n".join(lines)


def _run_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    top_scores = summary.get("top_scores") if isinstance(summary.get("top_scores"), list) else []
    return {
        "status": str(summary.get("status") or "unknown"),
        "runner_used": str(summary.get("runner_used") or ""),
        "candidate_count": _safe_int(summary.get("candidate_count")),
        "report_count": _safe_int(summary.get("report_count")),
        "top_score_present": bool(summary.get("top_score_present", False)),
        "top_score_summary": _top_score_summary(top_scores),
        "candidate_id_count": len(summary.get("candidate_ids", []) if isinstance(summary.get("candidate_ids"), list) else []),
        "candidate_profile_preview_count": _safe_int(summary.get("candidate_profile_preview_count")),
        "memory_context_provided": bool(summary.get("memory_context_provided", False)),
        "memory_context_eligible_count": _safe_int(summary.get("memory_context_eligible_count")),
        "memory_context_rendered_char_count": _safe_int(summary.get("memory_context_rendered_char_count")),
        "output_key_count": len(summary.get("output_keys", []) if isinstance(summary.get("output_keys"), list) else []),
        "error_type": str(summary.get("error_type") or ""),
        "summary_only": True,
    }


def _top_score_summary(scores: Any) -> Dict[str, Any]:
    if not isinstance(scores, list) or not scores:
        return {"present": False, "first": None, "count": 0}
    return {"present": True, "first": float(scores[0]), "count": len(scores)}


def _metadata_summary(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    return {
        "summary_only": True,
        "keys": sorted(str(key) for key in data.keys()),
        "memory_source": str(data.get("memory_source") or ""),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
