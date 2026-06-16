import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.evaluation.dataset import RecruitmentEvalDataset, load_recruitment_eval_dataset, validate_recruitment_eval_dataset
from src.evaluation.retrieval_metrics import RetrievalEvalCase, RetrievalEvalReport, RetrievalMetricsEvaluator


@dataclass
class RetrievalEvaluationConfig:
    dataset_dir: str = "evaluation_data/v1"
    index_dir: str = "evaluation_indexes/recruitment_eval_v1_chroma"
    query_mode: str = "raw_jd"
    top_k_values: List[int] = field(default_factory=lambda: [5, 10])
    index_version: str = "phase13b-v1"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    summary_only: bool = True


class RetrievalEvaluationRunner:
    def __init__(
        self,
        config: Optional[RetrievalEvaluationConfig] = None,
        evaluator: Optional[RetrievalMetricsEvaluator] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.config = config or RetrievalEvaluationConfig()
        self.evaluator = evaluator or RetrievalMetricsEvaluator()
        self.clock = clock or time.perf_counter

    def build_cases(self, dataset: RecruitmentEvalDataset) -> List[RetrievalEvalCase]:
        label_by_job = {label.job_id: label for label in dataset.relevance_labels}
        cases = []
        for job in dataset.jobs:
            label = label_by_job[job.job_id]
            cases.append(
                RetrievalEvalCase(
                    job_id=job.job_id,
                    query=build_retrieval_query(job.to_dict(), query_mode=self.config.query_mode),
                    relevance_labels=dict(label.candidate_relevance),
                    ideal_ranking=list(label.ideal_ranking),
                    top_k_values=list(self.config.top_k_values),
                    tags=list(job.tags),
                    metadata={"query_mode": self.config.query_mode, "summary_only": True},
                )
            )
        return cases

    def run(
        self,
        retrieval_callable: Callable[[str, int], Sequence[Mapping[str, Any]]],
        *,
        dataset: Optional[RecruitmentEvalDataset] = None,
    ) -> RetrievalEvalReport:
        total_started = self.clock()
        dataset = dataset or load_recruitment_eval_dataset(self.config.dataset_dir)
        validation = validate_recruitment_eval_dataset(dataset)
        if not validation.valid:
            raise ValueError(f"dataset validation failed: {validation.errors[:3]}")

        cases = self.build_cases(dataset)
        max_k = max(self.config.top_k_values) if self.config.top_k_values else 10
        results_by_job: Dict[str, Sequence[Mapping[str, Any]]] = {}
        latency_by_job: Dict[str, int] = {}
        for case in cases:
            started = self.clock()
            try:
                results_by_job[case.job_id] = retrieval_callable(case.query, max_k)
            finally:
                latency_by_job[case.job_id] = int((self.clock() - started) * 1000)
        attack_types = {case.candidate_id: case.attack_type for case in dataset.attack_cases}
        return self.evaluator.evaluate_cases(
            cases,
            results_by_job,
            latency_by_job=latency_by_job,
            dataset_version=str(dataset.manifest.get("dataset_version") or ""),
            index_version=self.config.index_version,
            attack_candidate_types=attack_types,
            benchmark_total_duration_ms=int((self.clock() - total_started) * 1000),
        )

    def run_with_retriever_factory(
        self,
        retriever_factory: Callable[[RetrievalEvaluationConfig], Any],
        *,
        dataset: Optional[RecruitmentEvalDataset] = None,
    ) -> RetrievalEvalReport:
        total_started = self.clock()
        dataset = dataset or load_recruitment_eval_dataset(self.config.dataset_dir)
        validation = validate_recruitment_eval_dataset(dataset)
        if not validation.valid:
            raise ValueError(f"dataset validation failed: {validation.errors[:3]}")

        cases = self.build_cases(dataset)
        max_k = max(self.config.top_k_values) if self.config.top_k_values else 10
        init_started = self.clock()
        retriever = retriever_factory(self.config)
        initialization_duration_ms = int((self.clock() - init_started) * 1000)
        results_by_job: Dict[str, Sequence[Mapping[str, Any]]] = {}
        latency_by_job: Dict[str, int] = {}
        for case in cases:
            started = self.clock()
            try:
                results_by_job[case.job_id] = _search_retriever(retriever, case.query, max_k)
            finally:
                latency_by_job[case.job_id] = int((self.clock() - started) * 1000)
        attack_types = {case.candidate_id: case.attack_type for case in dataset.attack_cases}
        return self.evaluator.evaluate_cases(
            cases,
            results_by_job,
            latency_by_job=latency_by_job,
            dataset_version=str(dataset.manifest.get("dataset_version") or ""),
            index_version=self.config.index_version,
            attack_candidate_types=attack_types,
            initialization_duration_ms=initialization_duration_ms,
            benchmark_total_duration_ms=int((self.clock() - total_started) * 1000),
        )


def build_retrieval_query(job: Mapping[str, Any], query_mode: str = "raw_jd") -> str:
    if query_mode == "raw_jd":
        return str(job.get("jd_text") or "")
    if query_mode == "structured":
        parts: List[str] = []
        for key in ["required_skills", "responsibilities", "hard_constraints"]:
            value = job.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
        return "；".join(parts)
    raise ValueError(f"unsupported query_mode: {query_mode}")


def _search_retriever(retriever: Any, query: str, k: int) -> Sequence[Mapping[str, Any]]:
    if hasattr(retriever, "search"):
        return retriever.search(query, k)
    if callable(retriever):
        return retriever(query, k)
    raise TypeError("retriever must expose search(query, k) or be callable")
