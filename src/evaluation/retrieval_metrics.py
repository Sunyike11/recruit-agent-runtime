import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class RetrievalEvalCase:
    job_id: str
    query: str
    relevance_labels: Dict[str, int]
    ideal_ranking: List[str] = field(default_factory=list)
    top_k_values: List[int] = field(default_factory=lambda: [5, 10])
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query_length": len(self.query or ""),
            "relevance_label_count": len(self.relevance_labels),
            "ideal_ranking_count": len(self.ideal_ranking),
            "top_k_values": list(self.top_k_values),
            "tags": list(self.tags),
            "metadata_keys": sorted(self.metadata.keys()),
            "summary_only": True,
        }


@dataclass
class RetrievedCandidate:
    candidate_id: str
    rank: int
    score: Optional[float] = None
    source_document_id: str = ""
    chunk_id: str = ""
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCaseMetrics:
    job_id: str
    retrieved_candidate_ids: List[str] = field(default_factory=list)
    relevant_candidate_count: int = 0
    retrieved_count: int = 0
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    first_relevant_rank: Optional[int] = None
    duplicate_chunk_count: int = 0
    raw_chunk_count: int = 0
    unique_candidate_count: int = 0
    latency_ms: int = 0
    status: str = "ok"
    error_type: str = ""
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["recall_at_k"] = {str(k): v for k, v in self.recall_at_k.items()}
        data["precision_at_k"] = {str(k): v for k, v in self.precision_at_k.items()}
        data["hit_rate_at_k"] = {str(k): v for k, v in self.hit_rate_at_k.items()}
        data["ndcg_at_k"] = {str(k): v for k, v in self.ndcg_at_k.items()}
        data["summary_only"] = True
        return data


@dataclass
class RetrievalEvalReport:
    dataset_version: str
    index_version: str
    job_count: int
    successful_case_count: int
    failed_case_count: int
    macro_recall_at_k: Dict[int, float] = field(default_factory=dict)
    macro_precision_at_k: Dict[int, float] = field(default_factory=dict)
    macro_hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0
    macro_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    initialization_duration_ms: int = 0
    benchmark_total_duration_ms: int = 0
    mean_query_latency_ms: float = 0.0
    p50_query_latency_ms: float = 0.0
    p95_query_latency_ms: float = 0.0
    candidate_coverage: Dict[str, Any] = field(default_factory=dict)
    attack_case_summary: Dict[str, Any] = field(default_factory=dict)
    per_job_results: List[RetrievalCaseMetrics] = field(default_factory=list)
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "index_version": self.index_version,
            "job_count": self.job_count,
            "successful_case_count": self.successful_case_count,
            "failed_case_count": self.failed_case_count,
            "macro_recall_at_k": {str(k): v for k, v in self.macro_recall_at_k.items()},
            "macro_precision_at_k": {str(k): v for k, v in self.macro_precision_at_k.items()},
            "macro_hit_rate_at_k": {str(k): v for k, v in self.macro_hit_rate_at_k.items()},
            "mean_mrr": self.mean_mrr,
            "macro_ndcg_at_k": {str(k): v for k, v in self.macro_ndcg_at_k.items()},
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "initialization_duration_ms": self.initialization_duration_ms,
            "benchmark_total_duration_ms": self.benchmark_total_duration_ms,
            "mean_query_latency_ms": self.mean_query_latency_ms,
            "p50_query_latency_ms": self.p50_query_latency_ms,
            "p95_query_latency_ms": self.p95_query_latency_ms,
            "candidate_coverage": dict(self.candidate_coverage),
            "attack_case_summary": dict(self.attack_case_summary),
            "per_job_results": [result.to_dict() for result in self.per_job_results],
            "summary_only": True,
        }


class RetrievalMetricsEvaluator:
    def evaluate_case(
        self,
        eval_case: RetrievalEvalCase,
        retrieval_results: Sequence[Mapping[str, Any] | RetrievedCandidate],
        *,
        latency_ms: int = 0,
    ) -> RetrievalCaseMetrics:
        adapted = adapt_eval_retrieval_results(retrieval_results)
        invalid_count = sum(1 for item in adapted if not item.candidate_id)
        if invalid_count:
            return RetrievalCaseMetrics(
                job_id=eval_case.job_id,
                raw_chunk_count=len(retrieval_results),
                latency_ms=latency_ms,
                status="invalid_identity",
                error_type="missing_candidate_id",
            )

        deduped, duplicate_count = deduplicate_retrieved_candidates(adapted)
        ranking = [item.candidate_id for item in deduped]
        top_k_values = sorted(set(int(k) for k in eval_case.top_k_values if int(k) > 0))
        relevant = {cid for cid, value in eval_case.relevance_labels.items() if int(value) > 0}
        first_relevant = next((idx + 1 for idx, cid in enumerate(ranking) if cid in relevant), None)

        return RetrievalCaseMetrics(
            job_id=eval_case.job_id,
            retrieved_candidate_ids=ranking,
            relevant_candidate_count=len(relevant),
            retrieved_count=len(ranking),
            recall_at_k={k: self.compute_recall_at_k(ranking, relevant, k) for k in top_k_values},
            precision_at_k={k: self.compute_precision_at_k(ranking, relevant, k) for k in top_k_values},
            hit_rate_at_k={k: self.compute_hit_rate_at_k(ranking, relevant, k) for k in top_k_values},
            mrr=self.compute_mrr(ranking, relevant),
            ndcg_at_k={k: self.compute_ndcg_at_k(ranking, eval_case.relevance_labels, k) for k in top_k_values},
            first_relevant_rank=first_relevant,
            duplicate_chunk_count=duplicate_count,
            raw_chunk_count=len(retrieval_results),
            unique_candidate_count=len(ranking),
            latency_ms=latency_ms,
            status="ok",
        )

    def evaluate_cases(
        self,
        cases: Sequence[RetrievalEvalCase],
        retrieval_results_by_job: Mapping[str, Sequence[Mapping[str, Any] | RetrievedCandidate]],
        *,
        latency_by_job: Optional[Mapping[str, int]] = None,
        dataset_version: str = "",
        index_version: str = "",
        attack_candidate_types: Optional[Mapping[str, str]] = None,
        initialization_duration_ms: int = 0,
        benchmark_total_duration_ms: int = 0,
    ) -> RetrievalEvalReport:
        results = [
            self.evaluate_case(
                case,
                retrieval_results_by_job.get(case.job_id, []),
                latency_ms=int((latency_by_job or {}).get(case.job_id, 0)),
            )
            for case in cases
        ]
        return self.aggregate_report(
            results,
            dataset_version=dataset_version,
            index_version=index_version,
            attack_candidate_types=attack_candidate_types or {},
            initialization_duration_ms=initialization_duration_ms,
            benchmark_total_duration_ms=benchmark_total_duration_ms,
        )

    def compute_recall_at_k(self, ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        return _round(len(set(ranking[:k]) & relevant_set) / len(relevant_set))

    def compute_precision_at_k(self, ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
        if k <= 0:
            return 0.0
        relevant_set = set(relevant)
        denom = min(k, len(ranking)) if ranking else k
        if denom <= 0:
            return 0.0
        return _round(len(set(ranking[:k]) & relevant_set) / denom)

    def compute_hit_rate_at_k(self, ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        return 1.0 if any(candidate_id in relevant_set for candidate_id in ranking[:k]) else 0.0

    def compute_mrr(self, ranking: Sequence[str], relevant: Iterable[str]) -> float:
        relevant_set = set(relevant)
        for idx, candidate_id in enumerate(ranking, start=1):
            if candidate_id in relevant_set:
                return _round(1.0 / idx)
        return 0.0

    def compute_dcg(self, ranking: Sequence[str], relevance_labels: Mapping[str, int], k: int) -> float:
        dcg = 0.0
        for idx, candidate_id in enumerate(ranking[:k], start=1):
            relevance = int(relevance_labels.get(candidate_id, 0))
            gain = (2**relevance) - 1
            dcg += gain / math.log2(idx + 1)
        return _round(dcg)

    def compute_ndcg_at_k(self, ranking: Sequence[str], relevance_labels: Mapping[str, int], k: int) -> float:
        dcg = self.compute_dcg(ranking, relevance_labels, k)
        ideal_relevances = sorted((int(value) for value in relevance_labels.values()), reverse=True)[:k]
        ideal_ids = [f"ideal_{idx}" for idx, _ in enumerate(ideal_relevances)]
        ideal_labels = {candidate_id: relevance for candidate_id, relevance in zip(ideal_ids, ideal_relevances)}
        idcg = self.compute_dcg(ideal_ids, ideal_labels, k)
        if idcg == 0:
            return 0.0
        return _round(dcg / idcg)

    def aggregate_report(
        self,
        results: Sequence[RetrievalCaseMetrics],
        *,
        dataset_version: str = "",
        index_version: str = "",
        attack_candidate_types: Optional[Mapping[str, str]] = None,
        initialization_duration_ms: int = 0,
        benchmark_total_duration_ms: int = 0,
    ) -> RetrievalEvalReport:
        successful = [result for result in results if result.status == "ok"]
        failed = [result for result in results if result.status != "ok"]
        k_values = sorted({k for result in results for k in result.recall_at_k.keys()})
        retrieved_all = [cid for result in successful for cid in result.retrieved_candidate_ids]
        unique_retrieved = sorted(set(retrieved_all))
        latencies = [result.latency_ms for result in successful]
        attack_summary = build_attack_case_summary(results, attack_candidate_types or {}, k_values)
        p50_query = _percentile(latencies, 50)
        p95_query = _percentile(latencies, 95)
        mean_query = _mean(latencies)
        return RetrievalEvalReport(
            dataset_version=dataset_version,
            index_version=index_version,
            job_count=len(results),
            successful_case_count=len(successful),
            failed_case_count=len(failed),
            macro_recall_at_k={k: _mean([result.recall_at_k.get(k, 0.0) for result in successful]) for k in k_values},
            macro_precision_at_k={k: _mean([result.precision_at_k.get(k, 0.0) for result in successful]) for k in k_values},
            macro_hit_rate_at_k={k: _mean([result.hit_rate_at_k.get(k, 0.0) for result in successful]) for k in k_values},
            mean_mrr=_mean([result.mrr for result in successful]),
            macro_ndcg_at_k={k: _mean([result.ndcg_at_k.get(k, 0.0) for result in successful]) for k in k_values},
            p50_latency_ms=p50_query,
            p95_latency_ms=p95_query,
            mean_latency_ms=mean_query,
            initialization_duration_ms=initialization_duration_ms,
            benchmark_total_duration_ms=benchmark_total_duration_ms,
            mean_query_latency_ms=mean_query,
            p50_query_latency_ms=p50_query,
            p95_query_latency_ms=p95_query,
            candidate_coverage={
                "unique_retrieved_candidate_count": len(unique_retrieved),
                "retrieved_candidate_ids": unique_retrieved,
                "summary_only": True,
            },
            attack_case_summary=attack_summary,
            per_job_results=list(results),
        )


def adapt_eval_retrieval_results(results: Sequence[Mapping[str, Any] | RetrievedCandidate]) -> List[RetrievedCandidate]:
    adapted: List[RetrievedCandidate] = []
    for idx, item in enumerate(results, start=1):
        if isinstance(item, RetrievedCandidate):
            adapted.append(item)
            continue
        data = dict(item)
        metadata = dict(data.get("metadata") or {})
        candidate_id = str(data.get("candidate_id") or metadata.get("candidate_id") or "")
        source_document_id = str(
            data.get("source_document_id") or metadata.get("source_document_id") or metadata.get("document_id") or ""
        )
        chunk_id = str(data.get("chunk_id") or data.get("id") or metadata.get("chunk_id") or "")
        score = data.get("score")
        adapted.append(
            RetrievedCandidate(
                candidate_id=candidate_id,
                rank=int(data.get("rank") or idx),
                score=float(score) if isinstance(score, (int, float)) else None,
                source_document_id=source_document_id,
                chunk_id=chunk_id,
            )
        )
    return adapted


def deduplicate_retrieved_candidates(candidates: Sequence[RetrievedCandidate]) -> tuple[List[RetrievedCandidate], int]:
    seen = set()
    deduped: List[RetrievedCandidate] = []
    duplicate_count = 0
    for candidate in sorted(candidates, key=lambda item: item.rank):
        if candidate.candidate_id in seen:
            duplicate_count += 1
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped, duplicate_count


def build_attack_case_summary(
    results: Sequence[RetrievalCaseMetrics],
    attack_candidate_types: Mapping[str, str],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "attack_candidate_count": len(attack_candidate_types),
        "retrieved_attack_candidate_count": 0,
        "by_attack_type": {},
        "summary_only": True,
    }
    retrieved_attacks = set()
    for result in results:
        for k in k_values:
            for candidate_id in result.retrieved_candidate_ids[:k]:
                attack_type = attack_candidate_types.get(candidate_id)
                if not attack_type:
                    continue
                retrieved_attacks.add(candidate_id)
                bucket = summary["by_attack_type"].setdefault(
                    attack_type,
                    {"top_k_hits": {}, "retrieved_candidate_ids": [], "summary_only": True},
                )
                bucket["top_k_hits"][str(k)] = int(bucket["top_k_hits"].get(str(k), 0)) + 1
                if candidate_id not in bucket["retrieved_candidate_ids"]:
                    bucket["retrieved_candidate_ids"].append(candidate_id)
    summary["retrieved_attack_candidate_count"] = len(retrieved_attacks)
    return summary


def build_query_mode_comparison_summary(raw_report: RetrievalEvalReport, structured_report: RetrievalEvalReport) -> Dict[str, Any]:
    raw = raw_report.to_dict()
    structured = structured_report.to_dict()
    return {
        "raw_jd": {
            "recall_at_k": raw.get("macro_recall_at_k", {}),
            "precision_at_k": raw.get("macro_precision_at_k", {}),
            "hit_rate_at_k": raw.get("macro_hit_rate_at_k", {}),
            "mrr": raw.get("mean_mrr", 0.0),
            "ndcg_at_k": raw.get("macro_ndcg_at_k", {}),
            "summary_only": True,
        },
        "structured": {
            "recall_at_k": structured.get("macro_recall_at_k", {}),
            "precision_at_k": structured.get("macro_precision_at_k", {}),
            "hit_rate_at_k": structured.get("macro_hit_rate_at_k", {}),
            "mrr": structured.get("mean_mrr", 0.0),
            "ndcg_at_k": structured.get("macro_ndcg_at_k", {}),
            "summary_only": True,
        },
        "interpretation": {
            "raw_jd_better_at": _better_metric_keys(raw.get("macro_recall_at_k", {}), structured.get("macro_recall_at_k", {})),
            "structured_better_at": _better_metric_keys(structured.get("macro_recall_at_k", {}), raw.get("macro_recall_at_k", {})),
            "do_not_claim_overall_winner": True,
            "summary_only": True,
        },
        "summary_only": True,
    }


def build_worst_job_summary(
    report: RetrievalEvalReport,
    *,
    job_ids: Sequence[str] = ("job_011", "job_012"),
    job_titles: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    summaries = []
    by_job = {result.job_id: result for result in report.per_job_results}
    attack_types = report.attack_case_summary.get("by_attack_type", {}) if report.attack_case_summary else {}
    attack_candidate_ids = {
        cid
        for bucket in attack_types.values()
        for cid in bucket.get("retrieved_candidate_ids", [])
    }
    for job_id in job_ids:
        result = by_job.get(job_id)
        if not result:
            continue
        relevant_retrieved = max(result.recall_at_k.values()) * result.relevant_candidate_count if result.recall_at_k else 0
        summaries.append(
            {
                "job_id": job_id,
                "job_title": (job_titles or {}).get(job_id, ""),
                "relevant_candidate_count": result.relevant_candidate_count,
                "retrieved_candidate_ids": list(result.retrieved_candidate_ids),
                "missing_relevant_estimate": max(0, result.relevant_candidate_count - int(round(relevant_retrieved))),
                "recall_at_k": {str(k): v for k, v in result.recall_at_k.items()},
                "ndcg_at_k": {str(k): v for k, v in result.ndcg_at_k.items()},
                "retrieved_attack_candidate": any(cid in attack_candidate_ids for cid in result.retrieved_candidate_ids),
                "summary_only": True,
            }
        )
    return summaries


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return _round(mean(values))


def _percentile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    fraction = index - lower
    return _round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _round(value: float) -> float:
    return round(float(value), 6)


def _better_metric_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[str]:
    keys = sorted(set(left.keys()) | set(right.keys()), key=str)
    return [str(key) for key in keys if float(left.get(key, 0.0)) > float(right.get(key, 0.0))]
