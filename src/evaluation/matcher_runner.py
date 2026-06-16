import json
import contextlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.evaluation.dataset import RecruitmentCandidate, RecruitmentEvalDataset, RecruitmentJob, load_recruitment_eval_dataset
from src.evaluation.matcher_metrics import (
    MatcherEvalCase,
    MatcherMetricsEvaluator,
    MatcherRunResult,
    analyze_matcher_output,
    matcher_cache_key,
)
from src.runtime.candidate_preview import (
    build_candidate_profile_preview_v2,
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
    candidate_profile_preview_v2_to_matcher_input,
)
from src.skills.agent_adapters import CandidateMatchSkill


INPUT_FULL_RESUME = "full_resume"
INPUT_PREVIEW = "candidate_profile_preview"
INPUT_PREVIEW_V1 = "candidate_profile_preview_v1"
INPUT_PREVIEW_V2 = "candidate_profile_preview_v2"
INPUT_RETRIEVER_TOP_K_PREVIEW = "retriever_top_k_preview"
INPUT_RETRIEVER_TOP_K_PREVIEW_V1 = "retriever_top_k_preview_v1"
INPUT_RETRIEVER_TOP_K_PREVIEW_V2 = "retriever_top_k_preview_v2"
V1_PREVIEW_MODES = {INPUT_PREVIEW, INPUT_PREVIEW_V1, INPUT_RETRIEVER_TOP_K_PREVIEW, INPUT_RETRIEVER_TOP_K_PREVIEW_V1}
V2_PREVIEW_MODES = {INPUT_PREVIEW_V2, INPUT_RETRIEVER_TOP_K_PREVIEW_V2}
RETRIEVER_TOP_K_MODES = {INPUT_RETRIEVER_TOP_K_PREVIEW, INPUT_RETRIEVER_TOP_K_PREVIEW_V1, INPUT_RETRIEVER_TOP_K_PREVIEW_V2}


@dataclass
class MatcherEvaluationConfig:
    dataset_dir: str = "evaluation_data/v1"
    retrieval_result_path: str = "evaluation_results/phase13b/retrieval_eval_raw_jd.json"
    input_modes: List[str] = field(default_factory=lambda: [INPUT_FULL_RESUME, INPUT_PREVIEW, INPUT_RETRIEVER_TOP_K_PREVIEW])
    job_ids: List[str] = field(default_factory=list)
    max_jobs: int = 0
    max_candidates: int = 0
    experiment_version: str = "phase13c-v1"
    summary_only: bool = True
    sampling_strategy: str = "relevance_stratified"
    reuse_failed_cache: bool = False


class MatcherEvaluationRunner:
    def __init__(
        self,
        config: Optional[MatcherEvaluationConfig] = None,
        evaluator: Optional[MatcherMetricsEvaluator] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.config = config or MatcherEvaluationConfig()
        self.evaluator = evaluator or MatcherMetricsEvaluator()
        self.clock = clock or time.perf_counter

    def run(
        self,
        matcher_callable: Callable[[Dict[str, Any]], Mapping[str, Any]],
        *,
        dataset: Optional[RecruitmentEvalDataset] = None,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False,
        reuse_failed_cache: Optional[bool] = None,
    ):
        dataset = dataset or load_recruitment_eval_dataset(self.config.dataset_dir)
        cache = cache if cache is not None else {}
        jobs = self._select_jobs(dataset)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in dataset.candidates}
        labels_by_job = {label.job_id: label for label in dataset.relevance_labels}
        attack_by_candidate = {case.candidate_id: case for case in dataset.attack_cases}
        retrieval_top_k = load_retrieval_top_k(self.config.retrieval_result_path)
        all_cases: List[MatcherEvalCase] = []
        all_results: List[MatcherRunResult] = []
        job_metrics = []
        cache_hit_success_count = 0
        cache_retry_failed_count = 0
        fresh_execution_count = 0
        reuse_failed = self.config.reuse_failed_cache if reuse_failed_cache is None else reuse_failed_cache

        for job in jobs:
            label = labels_by_job[job.job_id]
            ideal_rank = {candidate_id: idx + 1 for idx, candidate_id in enumerate(label.ideal_ranking)}
            for input_mode in self.config.input_modes:
                candidate_ids = self._candidate_ids_for_mode(job.job_id, label.candidate_relevance, retrieval_top_k, input_mode)
                cases: List[MatcherEvalCase] = []
                results: List[MatcherRunResult] = []
                for candidate_id in candidate_ids:
                    candidate = candidates_by_id[candidate_id]
                    attack = attack_by_candidate.get(candidate_id)
                    case = MatcherEvalCase(
                        job_id=job.job_id,
                        candidate_id=candidate_id,
                        input_mode=input_mode,
                        relevance=int(label.candidate_relevance[candidate_id]),
                        ideal_rank=ideal_rank.get(candidate_id),
                        is_special_case=bool(candidate.is_special_case),
                        special_case_type=str(candidate.special_case_type or ""),
                        expected_security_flags=list(attack.expected_security_flags if attack else []),
                    )
                    cache_key = matcher_cache_key(self.config.experiment_version, job.job_id, candidate_id, input_mode)
                    cached = dict(cache.get(cache_key) or {})
                    cached_failed = cached.get("status") and cached.get("status") != "ok"
                    if cache_key in cache and not force and (not cached_failed or reuse_failed):
                        result = MatcherRunResult(**dict(cache[cache_key]))
                        if result.status == "ok":
                            cache_hit_success_count += 1
                    elif cache_key in cache and cached_failed and not force and not reuse_failed:
                        cache_retry_failed_count += 1
                        result = self._execute_matcher(job, candidate, input_mode, case, matcher_callable)
                        fresh_execution_count += 1
                        cache[cache_key] = result.to_dict()
                    else:
                        result = self._execute_matcher(job, candidate, input_mode, case, matcher_callable)
                        fresh_execution_count += 1
                        cache[cache_key] = result.to_dict()
                    cases.append(case)
                    results.append(result)
                all_cases.extend(cases)
                all_results.extend(results)
                job_metrics.append(
                    self.evaluator.evaluate_job(
                        job.job_id,
                        input_mode,
                        cases,
                        results,
                        ideal_ranking=label.ideal_ranking,
                    )
                )
        report = self.evaluator.build_report(
            job_metrics,
            all_results,
            all_cases,
            dataset_version=str(dataset.manifest.get("dataset_version") or ""),
            experiment_version=self.config.experiment_version,
        )
        report.cache_hit_success_count = cache_hit_success_count
        report.cache_retry_failed_count = cache_retry_failed_count
        report.fresh_execution_count = fresh_execution_count
        report.sampling_strategy = self.config.sampling_strategy
        report.observation_gate = build_matcher_observation_gate(job_metrics)
        return report

    def _execute_matcher(
        self,
        job: RecruitmentJob,
        candidate: RecruitmentCandidate,
        input_mode: str,
        case: MatcherEvalCase,
        matcher_callable: Callable[[Dict[str, Any]], Mapping[str, Any]],
    ) -> MatcherRunResult:
        payload = build_matcher_input_payload(job, candidate, input_mode)
        started = self.clock()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                raw_output = matcher_callable(payload)
            latency_ms = int((self.clock() - started) * 1000)
            captured_text = stdout_buffer.getvalue() + stderr_buffer.getvalue()
            return analyze_matcher_output(
                case=case,
                input_payload=payload["candidate_profile"],
                raw_output=raw_output,
                latency_ms=latency_ms,
                matcher_stdout_captured=bool(captured_text),
                matcher_stdout_sensitive_content_detected=_stdout_sensitive(captured_text),
            )
        except Exception as exc:
            latency_ms = int((self.clock() - started) * 1000)
            captured_text = stdout_buffer.getvalue() + stderr_buffer.getvalue()
            return MatcherRunResult(
                job_id=job.job_id,
                candidate_id=case.candidate_id,
                input_mode=input_mode,
                status="failed",
                error_type=type(exc).__name__,
                latency_ms=latency_ms,
                matcher_stdout_captured=bool(captured_text),
                matcher_stdout_sensitive_content_detected=_stdout_sensitive(captured_text),
            )

    def _select_jobs(self, dataset: RecruitmentEvalDataset) -> List[RecruitmentJob]:
        jobs = list(dataset.jobs)
        if self.config.job_ids:
            selected = set(self.config.job_ids)
            jobs = [job for job in jobs if job.job_id in selected]
        if self.config.max_jobs:
            jobs = jobs[: self.config.max_jobs]
        return jobs

    def _candidate_ids_for_mode(
        self,
        job_id: str,
        relevance_labels: Mapping[str, int],
        retrieval_top_k: Mapping[str, List[str]],
        input_mode: str,
    ) -> List[str]:
        if input_mode in RETRIEVER_TOP_K_MODES:
            ids = list(retrieval_top_k.get(job_id, []))
        else:
            ids = relevance_stratified_sample_ids(relevance_labels, self.config.max_candidates)
            return ids
        if self.config.max_candidates:
            ids = ids[: self.config.max_candidates]
        return ids


def build_matcher_input_payload(job: RecruitmentJob, candidate: RecruitmentCandidate, input_mode: str) -> Dict[str, Any]:
    job_requirement = {
        "job_id": job.job_id,
        "title": job.title,
        "tech_stack": list(job.required_skills),
        "required_skills": list(job.required_skills),
        "preferred_skills": list(job.preferred_skills),
        "education": job.education_requirement,
        "must_have": list(job.hard_constraints),
        "search_query": job.jd_text,
        "summary_only": True,
    }
    if input_mode == INPUT_FULL_RESUME:
        profile = {
            "candidate_id": candidate.candidate_id,
            "candidate_name": candidate.display_name,
            "name": candidate.display_name,
            "skills": list(candidate.skills),
            "education": candidate.education,
            "experience": list(candidate.work_experience),
            "projects": list(candidate.projects),
            "resume_text": candidate.resume_text,
            "is_special_case": candidate.is_special_case,
            "special_case_type": candidate.special_case_type,
            "summary_only": True,
        }
    elif input_mode in V1_PREVIEW_MODES:
        build_result = build_candidate_profile_previews_from_retrieval_results(
            [
                {
                    "text": candidate.resume_text,
                    "metadata": {
                        "candidate_id": candidate.candidate_id,
                        "candidate_name": candidate.display_name,
                        "source_document_id": candidate.candidate_id,
                        "file_name": candidate.source_file_name,
                    },
                    "score": 1.0,
                }
            ],
            raw_jd=job.jd_text,
            query=job.jd_text,
        )
        profile = candidate_profile_preview_to_matcher_input(build_result.previews[0])
        profile["is_special_case"] = candidate.is_special_case
        profile["special_case_type"] = candidate.special_case_type
        profile["preview_version"] = "v1"
    elif input_mode in V2_PREVIEW_MODES:
        preview = build_candidate_profile_preview_v2(
            {
                "candidate_id": candidate.candidate_id,
                "display_name": candidate.display_name,
                "education": candidate.education,
                "years_of_experience": candidate.years_of_experience,
                "skills": list(candidate.skills),
                "projects": list(candidate.projects),
                "work_experience": list(candidate.work_experience),
                "research_experience": list(candidate.research_experience),
                "certifications": list(candidate.certifications),
                "open_source": list(candidate.open_source),
                "awards": list(candidate.awards),
                "resume_text": candidate.resume_text,
                "source_file_name": candidate.source_file_name,
                "metadata": {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.display_name,
                    "source_document_id": candidate.candidate_id,
                    "file_name": candidate.source_file_name,
                },
            },
            raw_jd=job.jd_text,
        )
        profile = candidate_profile_preview_v2_to_matcher_input(preview)
        profile["is_special_case"] = candidate.is_special_case
        profile["special_case_type"] = candidate.special_case_type
    else:
        raise ValueError(f"unknown matcher input_mode: {input_mode}")
    return {
        "job_requirement": job_requirement,
        "candidate_profile": profile,
        "metadata": {"input_mode": input_mode, "summary_only": True},
    }


def load_retrieval_top_k(path: str | Path) -> Dict[str, List[str]]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = {}
    for item in data.get("per_job_results") or []:
        results[str(item.get("job_id"))] = [str(candidate_id) for candidate_id in item.get("retrieved_candidate_ids") or []]
    return results


def relevance_stratified_sample_ids(relevance_labels: Mapping[str, int], max_candidates: int = 0) -> List[str]:
    ordered = list(relevance_labels.keys())
    if not max_candidates or max_candidates >= len(ordered):
        return ordered
    selected: List[str] = []
    for level in (2, 1, 0):
        for candidate_id in ordered:
            if candidate_id not in selected and int(relevance_labels.get(candidate_id, 0)) == level:
                selected.append(candidate_id)
                break
    for candidate_id in ordered:
        if len(selected) >= max_candidates:
            break
        if candidate_id not in selected:
            selected.append(candidate_id)
    return selected[:max_candidates]


def build_matcher_observation_gate(job_metrics: Sequence[Any]) -> Dict[str, Any]:
    by_mode = {metric.input_mode: metric for metric in job_metrics}
    v1 = by_mode.get(INPUT_PREVIEW_V1) or by_mode.get(INPUT_PREVIEW)
    v2 = by_mode.get(INPUT_PREVIEW_V2)
    full = by_mode.get(INPUT_FULL_RESUME)
    return {
        "v2_spearman_delta_vs_v1": _delta(v2, v1, "spearman_correlation"),
        "v2_ndcg5_delta_vs_v1": _delta(v2, v1, "ndcg_at_5"),
        "v2_evidence_coverage_delta_vs_v1": _delta(v2, v1, "evidence_completeness_rate"),
        "v2_unsupported_claim_delta_vs_v1": _delta(v2, v1, "unsupported_claim_case_rate"),
        "v2_ndcg5_gap_vs_full_resume": _delta(v2, full, "ndcg_at_5"),
        "summary_only": True,
    }


def _delta(left: Any, right: Any, attr: str) -> Any:
    if not left or not right:
        return "unavailable"
    a = getattr(left, attr, "unavailable")
    b = getattr(right, attr, "unavailable")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "unavailable"
    return round(a - b, 6)


def _stdout_sensitive(text: str) -> bool:
    return bool(text and any(token in text for token in ["解析后的报告", "reasoning", "final_verdict", "total_score"]))


def build_real_candidate_match_callable():
    skill = CandidateMatchSkill()

    def call(payload: Dict[str, Any]) -> Mapping[str, Any]:
        result = skill.run(payload)
        if not result.success:
            raise RuntimeError(result.error or "candidate_match_failed")
        return result.output

    return call
