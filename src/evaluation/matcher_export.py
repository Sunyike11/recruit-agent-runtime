import json
from typing import Any


FORBIDDEN_TOKENS = {
    "api_key",
    "prompt",
    "llm_response",
    "resume_text",
    "reasoning",
}


def sanitize_matcher_eval_report(report: Any) -> dict:
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    payload = json.dumps(data, ensure_ascii=False).lower()
    for token in FORBIDDEN_TOKENS:
        if token in payload:
            raise ValueError(f"unsafe matcher report token: {token}")
    return data


def export_matcher_eval_report_json(report: Any) -> str:
    return json.dumps(sanitize_matcher_eval_report(report), ensure_ascii=False, indent=2)


def export_matcher_eval_report_text(report: Any) -> str:
    data = sanitize_matcher_eval_report(report)
    return "\n".join(
        [
            "Matcher Evaluation Report",
            f"dataset_version: {data.get('dataset_version', '')}",
            f"experiment_version: {data.get('experiment_version', '')}",
            f"input_modes: {data.get('input_modes', [])}",
            f"job_count: {data.get('job_count', 0)}",
            f"candidate_evaluation_count: {data.get('candidate_evaluation_count', 0)}",
            f"successful_count: {data.get('successful_count', 0)}",
            f"failed_count: {data.get('failed_count', 0)}",
            f"macro_spearman: {data.get('macro_spearman', 'unavailable')}",
            f"macro_pearson: {data.get('macro_pearson', 'unavailable')}",
            f"macro_pairwise_accuracy: {data.get('macro_pairwise_accuracy', 0.0)}",
            f"macro_ndcg_at_5: {data.get('macro_ndcg_at_5', 0.0)}",
            f"structured_output_success_rate: {data.get('structured_output_success_rate', 0.0)}",
            f"unsupported_claim_rate: {data.get('unsupported_claim_rate', 0.0)}",
            f"p50_latency_ms: {data.get('p50_latency_ms', 0.0)}",
            f"p95_latency_ms: {data.get('p95_latency_ms', 0.0)}",
            "summary_only: true",
        ]
    )
