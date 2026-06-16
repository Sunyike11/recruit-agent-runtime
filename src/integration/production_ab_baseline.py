import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PLACEHOLDER_IDENTITY_VALUES = {
    "",
    "未知",
    "未提供",
    "简历",
    "个人简历",
    "我的简历",
    "第二份简历",
    "resume",
    "cv",
    "none",
    "unknown",
}


@dataclass
class ProductionABBaselineConfig:
    enabled: bool = False
    run_legacy: bool = True
    run_skill_graph: bool = True
    allow_planner_fallback: bool = False
    top_k: int = 5
    max_score_delta: Optional[float] = None
    require_candidate_overlap: bool = False
    rollback_on_skill_failure: bool = True
    latency_warning_ms: Optional[int] = None
    summary_only: bool = True


@dataclass
class ProductionABRunSummary:
    runner_name: str
    status: str
    task_status: str
    candidate_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    candidate_count: int = 0
    report_count: int = 0
    ranking: List[str] = field(default_factory=list)
    top_scores: List[float] = field(default_factory=list)
    top_score_present: bool = False
    candidate_name_resolved_count: int = 0
    project_evidence_present_count: int = 0
    education_evidence_present_count: int = 0
    evidence_summary_present_count: int = 0
    refine_loop_count: int = 0
    skill_execution_count: int = 0
    event_count: int = 0
    duration_ms: int = 0
    fallback_used: bool = False
    planner_fallback_used: bool = False
    error_type: str = ""
    error_hint: str = ""
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionABComparison:
    legacy_summary: Dict[str, Any]
    skill_summary: Dict[str, Any]
    both_succeeded: bool
    candidate_overlap_count: int
    candidate_union_count: int
    candidate_overlap_rate: float
    top_k_overlap_count: int
    top_k_overlap_rate: float
    ranking_alignment: Any
    score_deltas: List[Dict[str, Any]]
    candidate_identity_alignment: Dict[str, Any]
    report_count_delta: int
    name_resolution_delta: int
    project_evidence_delta: int
    education_evidence_delta: int
    latency_delta_ms: int
    decision: str
    risk_level: str
    rollback_recommended: bool
    rollback_reason: str
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionABBaselineRunner:
    """Run summary-only legacy-vs-production-skill A/B observations."""

    def __init__(self, config: Optional[ProductionABBaselineConfig] = None):
        self.config = config or ProductionABBaselineConfig()

    def run(
        self,
        raw_jd: str,
        *,
        legacy_runner: Callable[[str], Mapping[str, Any]],
        skill_runner: Callable[[str], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            return {
                "status": "skipped",
                "error_hint": "production_ab_baseline_disabled",
                "summary_only": True,
            }

        legacy = self.run_legacy(raw_jd, legacy_runner) if self.config.run_legacy else _empty_run("legacy_default_graph")
        skill = self.run_skill_graph(raw_jd, skill_runner) if self.config.run_skill_graph else _empty_run("production_skill_graph")
        comparison = self.compare(legacy, skill)
        return {
            "status": "ok",
            "legacy_summary": legacy.to_dict(),
            "skill_summary": skill.to_dict(),
            "comparison": comparison.to_dict(),
            "rollback_recommended": comparison.rollback_recommended,
            "rollback_reason": comparison.rollback_reason,
            "risk_level": comparison.risk_level,
            "decision": comparison.decision,
            "summary_only": True,
            "metadata": {
                "baseline_version": "phase11b-v1",
                "raw_jd_length": len(raw_jd or ""),
                "legacy_graph_invoked": bool(self.config.run_legacy),
                "production_skill_graph_invoked": bool(self.config.run_skill_graph),
                "production_graph_replaced": False,
                "memory_enabled": False,
                "mcp_enabled": False,
                "summary_only": True,
            },
        }

    def run_legacy(self, raw_jd: str, runner: Callable[[str], Mapping[str, Any]]) -> ProductionABRunSummary:
        return self._run_one("legacy_default_graph", raw_jd, runner)

    def run_skill_graph(self, raw_jd: str, runner: Callable[[str], Mapping[str, Any]]) -> ProductionABRunSummary:
        return self._run_one("production_skill_graph", raw_jd, runner)

    def summarize_legacy_result(self, result: Mapping[str, Any], duration_ms: int = 0) -> ProductionABRunSummary:
        return summarize_runner_result("legacy_default_graph", result, duration_ms=duration_ms)

    def summarize_skill_result(self, result: Mapping[str, Any], duration_ms: int = 0) -> ProductionABRunSummary:
        return summarize_runner_result("production_skill_graph", result, duration_ms=duration_ms)

    def compare(self, legacy: ProductionABRunSummary, skill: ProductionABRunSummary) -> ProductionABComparison:
        legacy_identity_ids = list(legacy.candidate_ids or legacy.ranking)
        skill_identity_ids = list(skill.candidate_ids or skill.ranking)
        overlap = [identity for identity in legacy_identity_ids if identity in set(skill_identity_ids)]
        union = sorted(set(legacy_identity_ids) | set(skill_identity_ids))
        top_k_overlap_count, top_k_overlap_rate = _top_k_overlap(legacy_identity_ids, skill_identity_ids, self.config.top_k)
        score_deltas = _score_deltas(legacy_identity_ids, legacy.top_scores, skill_identity_ids, skill.top_scores)
        ranking_alignment = _spearman_alignment(legacy_identity_ids, skill_identity_ids)
        identity_alignment = {
            "legacy_unresolved_count": sum(1 for value in legacy.candidate_ids if not value),
            "skill_unresolved_count": sum(1 for value in skill.candidate_ids if not value),
            "aligned_candidate_ids": overlap,
            "unresolved_identity": bool(not overlap and (legacy.candidate_count or skill.candidate_count)),
            "summary_only": True,
        }
        rollback, reason, risk, decision = self.build_rollback_decision(
            legacy=legacy,
            skill=skill,
            overlap_count=len(overlap),
            top_k_overlap_rate=top_k_overlap_rate,
            score_deltas=score_deltas,
            ranking_alignment=ranking_alignment,
        )
        return ProductionABComparison(
            legacy_summary=legacy.to_dict(),
            skill_summary=skill.to_dict(),
            both_succeeded=_success(legacy) and _success(skill),
            candidate_overlap_count=len(overlap),
            candidate_union_count=len(union),
            candidate_overlap_rate=_safe_rate(len(overlap), len(union)),
            top_k_overlap_count=top_k_overlap_count,
            top_k_overlap_rate=top_k_overlap_rate,
            ranking_alignment=ranking_alignment,
            score_deltas=score_deltas,
            candidate_identity_alignment=identity_alignment,
            report_count_delta=skill.report_count - legacy.report_count,
            name_resolution_delta=skill.candidate_name_resolved_count - legacy.candidate_name_resolved_count,
            project_evidence_delta=skill.project_evidence_present_count - legacy.project_evidence_present_count,
            education_evidence_delta=skill.education_evidence_present_count - legacy.education_evidence_present_count,
            latency_delta_ms=skill.duration_ms - legacy.duration_ms,
            decision=decision,
            risk_level=risk,
            rollback_recommended=rollback,
            rollback_reason=reason,
            summary_only=True,
        )

    def build_rollback_decision(
        self,
        *,
        legacy: ProductionABRunSummary,
        skill: ProductionABRunSummary,
        overlap_count: int,
        top_k_overlap_rate: float,
        score_deltas: Sequence[Mapping[str, Any]],
        ranking_alignment: Any,
    ) -> Tuple[bool, str, str, str]:
        if _success(legacy) and not _success(skill) and self.config.rollback_on_skill_failure:
            return True, "legacy succeeded while production skill graph failed", "high", "rollback"
        if skill.error_type or skill.error_hint in {"schema_invalid", "retriever_failed", "matcher_failed", "planner_failed"}:
            return True, "production skill graph reported failure diagnostics", "high", "rollback"
        if skill.candidate_count == 0 and legacy.candidate_count > 0:
            return True, "production skill graph returned zero candidates while legacy returned candidates", "high", "rollback"
        if skill.report_count == 0 and legacy.report_count > 0:
            return True, "production skill graph returned zero reports while legacy returned reports", "high", "rollback"
        if self.config.require_candidate_overlap and overlap_count == 0 and legacy.candidate_count and skill.candidate_count:
            return True, "candidate identity could not be aligned", "high", "rollback"

        warnings = []
        if legacy.candidate_count and skill.candidate_count and top_k_overlap_rate < 0.5:
            warnings.append("low top-k overlap")
        max_delta = self.config.max_score_delta
        if max_delta is not None and any(float(item.get("absolute_delta") or 0) > max_delta for item in score_deltas):
            warnings.append("large score delta")
        if skill.candidate_name_resolved_count < legacy.candidate_name_resolved_count:
            warnings.append("candidate name resolution declined")
        if skill.project_evidence_present_count < legacy.project_evidence_present_count:
            warnings.append("project evidence coverage declined")
        if skill.education_evidence_present_count < legacy.education_evidence_present_count:
            warnings.append("education evidence coverage declined")
        if self.config.latency_warning_ms is not None and skill.duration_ms - legacy.duration_ms > self.config.latency_warning_ms:
            warnings.append("latency increased")
        if ranking_alignment == "unavailable":
            warnings.append("ranking alignment unavailable")
        if warnings:
            return False, "; ".join(warnings), "medium", "review"
        return False, "baseline observation passed without high-risk signals", "low", "pass_observation"

    @staticmethod
    def _run_one(name: str, raw_jd: str, runner: Callable[[str], Mapping[str, Any]]) -> ProductionABRunSummary:
        start = time.perf_counter()
        try:
            output = runner(raw_jd)
            duration_ms = int((time.perf_counter() - start) * 1000)
            return summarize_runner_result(name, output, duration_ms=duration_ms)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return ProductionABRunSummary(
                runner_name=name,
                status="failed",
                task_status="failed",
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error_hint="runner_exception",
                summary_only=True,
            )


def summarize_runner_result(runner_name: str, result: Mapping[str, Any], *, duration_ms: int = 0) -> ProductionABRunSummary:
    data = dict(result or {}) if isinstance(result, Mapping) else {}
    output_summary = data.get("output_summary") if isinstance(data.get("output_summary"), Mapping) else {}
    merged = {**data, **dict(output_summary)}
    candidates = _candidate_items(merged)
    reports = _report_items(merged)
    candidate_ids = _identity_list(candidates, fallback=merged.get("candidate_ids"))
    document_ids = _document_id_list(candidates, fallback=merged.get("document_ids"))
    enriched_reports = _enrich_reports_with_candidate_sources(reports, candidates)
    ranking = _ranking_from_reports_or_candidates(enriched_reports, candidate_ids)
    top_scores = _top_scores(merged, reports)
    audit = merged.get("candidate_preview_audit") if isinstance(merged.get("candidate_preview_audit"), Mapping) else {}
    status = str(merged.get("status") or "unknown")
    error_type = str(merged.get("error_type") or merged.get("runner_error_type") or "")
    error_hint = str(merged.get("error_hint") or "")
    report_count = _safe_int(merged.get("report_count") or merged.get("match_report_count") or len(reports))
    candidate_count = _safe_int(merged.get("candidate_count") or merged.get("candidate_profile_preview_count") or len(candidate_ids))
    if error_hint == "max_loop_exceeded" and report_count > 0 and candidate_count > 0:
        status = "completed_with_limit"
        error_type = ""
        error_hint = ""
    return ProductionABRunSummary(
        runner_name=runner_name,
        status=status,
        task_status=str(merged.get("task_status") or status),
        candidate_ids=candidate_ids,
        document_ids=document_ids,
        candidate_count=candidate_count,
        report_count=report_count,
        ranking=ranking,
        top_scores=top_scores,
        top_score_present=bool(merged.get("top_score_present") or top_scores),
        candidate_name_resolved_count=_safe_int(
            merged.get("candidate_name_resolved_count")
            or audit.get("candidate_name_resolved_count")
            or audit.get("candidate_name_present")
            or _candidate_name_count(reports)
        ),
        project_evidence_present_count=_safe_int(
            merged.get("project_evidence_present_count") or audit.get("project_keywords_present_count")
        ),
        education_evidence_present_count=_safe_int(
            merged.get("education_evidence_present_count") or audit.get("education_keywords_present_count")
        ),
        evidence_summary_present_count=_safe_int(
            merged.get("evidence_summary_present_count") or audit.get("evidence_summary_present_count")
        ),
        refine_loop_count=_safe_int(merged.get("refine_loop_count") or merged.get("loop_count")),
        skill_execution_count=_safe_int(merged.get("skill_execution_count") or merged.get("skill_event_count")),
        event_count=_safe_int(merged.get("event_count")),
        duration_ms=_safe_int(merged.get("duration_ms") or duration_ms),
        fallback_used=bool(merged.get("fallback_used")),
        planner_fallback_used=bool(merged.get("planner_fallback_used")),
        error_type=error_type,
        error_hint=error_hint,
        summary_only=True,
    )


def load_baseline_manifest(path: Any) -> Dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_identity_key(item: Mapping[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    primary_candidate_id = _normalize_identity(item.get("candidate_id") or metadata.get("candidate_id"))
    if primary_candidate_id and not primary_candidate_id.startswith("candidate_preview_"):
        return primary_candidate_id
    source_candidates = [
        item.get("source_document_id"),
        item.get("document_id"),
        metadata.get("source_document_id"),
        metadata.get("document_id"),
        item.get("source_file_name"),
        item.get("file_name"),
        metadata.get("file_name"),
        metadata.get("source"),
    ]
    for value in source_candidates:
        normalized = _normalize_source_identity(value)
        if normalized:
            return normalized
    if primary_candidate_id:
        return primary_candidate_id
    name_candidates = [
        item.get("candidate_name"),
        item.get("name"),
        metadata.get("candidate_name"),
        metadata.get("name"),
    ]
    for value in name_candidates:
        normalized = _normalize_identity(value)
        if normalized:
            return normalized
    return ""


def _empty_run(name: str) -> ProductionABRunSummary:
    return ProductionABRunSummary(runner_name=name, status="skipped", task_status="skipped")


def _candidate_items(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("candidate_previews", "candidates", "candidate_pool", "retrieved_candidates"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    ids = data.get("candidate_ids")
    if isinstance(ids, list):
        return [{"candidate_id": item} for item in ids]
    return []


def _report_items(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("match_reports", "final_reports", "reports"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _enrich_reports_with_candidate_sources(
    reports: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    by_candidate_id = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            by_candidate_id[candidate_id] = candidate
    enriched = []
    for report in reports:
        data = dict(report)
        candidate = by_candidate_id.get(str(report.get("candidate_id") or ""))
        if isinstance(candidate, Mapping):
            for key in ("source_document_id", "document_id", "source_file_name", "file_name", "candidate_name", "name"):
                if key not in data and candidate.get(key):
                    data[key] = candidate.get(key)
            candidate_metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
            if candidate_metadata:
                metadata = dict(data.get("metadata") or {})
                for key in ("source_document_id", "document_id", "file_name", "source", "candidate_name", "name"):
                    if key not in metadata and candidate_metadata.get(key):
                        metadata[key] = candidate_metadata.get(key)
                data["metadata"] = metadata
        enriched.append(data)
    return enriched


def _identity_list(items: Sequence[Mapping[str, Any]], fallback: Any = None) -> List[str]:
    identities = [build_identity_key(item) for item in items]
    identities = [item for item in identities if item]
    if identities:
        return _dedupe(identities)
    if isinstance(fallback, list):
        return _dedupe(_normalize_identity(item) for item in fallback if _normalize_identity(item))
    return []


def _document_id_list(items: Sequence[Mapping[str, Any]], fallback: Any = None) -> List[str]:
    values = []
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        values.append(
            _normalize_source_identity(
                item.get("source_document_id")
                or item.get("document_id")
                or metadata.get("document_id")
                or metadata.get("source_document_id")
                or item.get("source_file_name")
                or item.get("file_name")
                or metadata.get("file_name")
                or metadata.get("source")
            )
        )
    if isinstance(fallback, list):
        values.extend(_normalize_source_identity(item) for item in fallback)
    return _dedupe(value for value in values if value)


def _ranking_from_reports_or_candidates(reports: Sequence[Mapping[str, Any]], candidate_ids: Sequence[str]) -> List[str]:
    if reports:
        ranked = []
        sorted_reports = sorted(
            reports,
            key=lambda item: _score_value(item),
            reverse=True,
        )
        for report in sorted_reports:
            identity = build_identity_key(report)
            if identity:
                ranked.append(identity)
        if ranked:
            return _dedupe(ranked)
    return list(candidate_ids)


def _top_scores(data: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]) -> List[float]:
    existing = data.get("top_scores")
    if isinstance(existing, list):
        return [float(item) for item in existing if isinstance(item, (int, float))]
    return [_score_value(report) for report in reports if _has_score(report)]


def _score_value(item: Mapping[str, Any]) -> float:
    for key in ("total_score", "score", "match_score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _has_score(item: Mapping[str, Any]) -> bool:
    return any(isinstance(item.get(key), (int, float)) for key in ("total_score", "score", "match_score"))


def _top_k_overlap(left: Sequence[str], right: Sequence[str], top_k: int) -> Tuple[int, float]:
    k = max(1, int(top_k or 1))
    left_top = list(left)[:k]
    right_top = list(right)[:k]
    denominator = min(k, max(len(left_top), len(right_top)))
    if denominator <= 0:
        return 0, 0.0
    count = len(set(left_top) & set(right_top))
    return count, round(count / denominator, 4)


def _score_deltas(left_ids: Sequence[str], left_scores: Sequence[float], right_ids: Sequence[str], right_scores: Sequence[float]) -> List[Dict[str, Any]]:
    left = {identity: float(left_scores[index]) for index, identity in enumerate(left_ids) if index < len(left_scores)}
    right = {identity: float(right_scores[index]) for index, identity in enumerate(right_ids) if index < len(right_scores)}
    rows = []
    for identity in sorted(set(left) & set(right)):
        rows.append(
            {
                "candidate_identity": identity,
                "legacy_score": left[identity],
                "skill_score": right[identity],
                "absolute_delta": round(abs(left[identity] - right[identity]), 4),
                "summary_only": True,
            }
        )
    return rows


def _spearman_alignment(left: Sequence[str], right: Sequence[str]) -> Any:
    shared = [identity for identity in left if identity in set(right)]
    if len(shared) < 2:
        return "unavailable"
    left_rank = {identity: index + 1 for index, identity in enumerate(left)}
    right_rank = {identity: index + 1 for index, identity in enumerate(right)}
    n = len(shared)
    d2 = sum((left_rank[item] - right_rank[item]) ** 2 for item in shared)
    value = 1 - (6 * d2) / (n * (n * n - 1))
    if math.isclose(value, round(value), abs_tol=1e-9):
        return float(round(value))
    return round(value, 4)


def _success(summary: ProductionABRunSummary) -> bool:
    return summary.status in {"ok", "completed_with_limit"} and not summary.error_type


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _normalize_identity(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    name = Path(text).name
    lowered = name.lower()
    for suffix in (".pdf", ".docx", ".doc", ".txt", ".md"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    normalized = name.strip(" _-()[]")
    compact = normalized.replace(" ", "").replace("_", "").replace("-", "").lower()
    if compact in PLACEHOLDER_IDENTITY_VALUES:
        return ""
    return normalized[:96]


def _normalize_source_identity(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    name = Path(text).name
    lowered = name.lower()
    for suffix in (".pdf", ".docx", ".doc", ".txt", ".md"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip(" _-()[]")[:96]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_name_count(reports: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for report in reports:
        if _normalize_identity(report.get("candidate_name") or report.get("name")):
            count += 1
    return count


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
