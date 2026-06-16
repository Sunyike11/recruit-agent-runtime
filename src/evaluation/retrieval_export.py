import json
from typing import Any, Mapping


FORBIDDEN_EXPORT_TOKENS = [
    "resume_text",
    "raw_chunks",
    "chunk_text",
    "full_text",
    "full_resume",
    "embedding",
    "llm_response",
    "reasoning",
    "api_key",
]


def sanitize_retrieval_report(report: Any) -> dict:
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    payload = json.dumps(data, ensure_ascii=False)
    lowered = payload.lower()
    for token in FORBIDDEN_EXPORT_TOKENS:
        if token in lowered:
            raise ValueError(f"unsafe retrieval report token: {token}")
    return data


def export_retrieval_report_json(report: Any) -> str:
    return json.dumps(sanitize_retrieval_report(report), ensure_ascii=False, indent=2)


def export_retrieval_report_text(report: Any) -> str:
    data = sanitize_retrieval_report(report)
    lines = [
        "Retrieval Evaluation Report",
        f"dataset_version: {data.get('dataset_version', '')}",
        f"index_version: {data.get('index_version', '')}",
        f"job_count: {data.get('job_count', 0)}",
        f"successful_case_count: {data.get('successful_case_count', 0)}",
        f"failed_case_count: {data.get('failed_case_count', 0)}",
        f"mean_mrr: {data.get('mean_mrr', 0.0)}",
        f"initialization_duration_ms: {data.get('initialization_duration_ms', 0)}",
        f"benchmark_total_duration_ms: {data.get('benchmark_total_duration_ms', 0)}",
        f"p50_query_latency_ms: {data.get('p50_query_latency_ms', data.get('p50_latency_ms', 0.0))}",
        f"p95_query_latency_ms: {data.get('p95_query_latency_ms', data.get('p95_latency_ms', 0.0))}",
        f"macro_recall_at_k: {data.get('macro_recall_at_k', {})}",
        f"macro_precision_at_k: {data.get('macro_precision_at_k', {})}",
        f"macro_ndcg_at_k: {data.get('macro_ndcg_at_k', {})}",
        f"attack_case_summary: {_summary_attack(data.get('attack_case_summary', {}))}",
        "summary_only: true",
    ]
    return "\n".join(lines)


def _summary_attack(summary: Mapping[str, Any]) -> dict:
    return {
        "attack_candidate_count": summary.get("attack_candidate_count", 0),
        "retrieved_attack_candidate_count": summary.get("retrieved_attack_candidate_count", 0),
        "attack_types": sorted((summary.get("by_attack_type") or {}).keys()),
        "summary_only": True,
    }
