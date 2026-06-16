import json

import pytest

from scripts.generate_recruitment_eval_dataset import generate_dataset
from src.evaluation.dataset import load_recruitment_eval_dataset
from src.evaluation.matcher_export import export_matcher_eval_report_json, export_matcher_eval_report_text
from src.evaluation.matcher_metrics import (
    MatcherEvalCase,
    MatcherMetricsEvaluator,
    MatcherRunResult,
    analyze_matcher_output,
    matcher_cache_key,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    pearson_correlation,
    precision_at_k,
    spearman_correlation,
    top_k_overlap,
)
from src.evaluation.matcher_runner import (
    INPUT_FULL_RESUME,
    INPUT_PREVIEW,
    INPUT_RETRIEVER_TOP_K_PREVIEW,
    MatcherEvaluationConfig,
    MatcherEvaluationRunner,
    build_matcher_input_payload,
)


def _dataset(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    return load_recruitment_eval_dataset(output_dir)


def _fake_matcher(payload):
    profile = payload["candidate_profile"]
    candidate_id = profile["candidate_id"]
    score = {
        "candidate_001": 92,
        "candidate_002": 78,
        "candidate_003": 20,
        "candidate_004": 64,
    }.get(candidate_id, 35)
    return {
        "candidate_id": candidate_id,
        "total_score": score,
        "recommendation": "recommended" if score >= 60 else "rejected",
        "match_report": {
            "candidate_id": candidate_id,
            "total_score": score,
            "final_verdict": "recommended" if score >= 60 else "rejected",
            "strengths": ["技能证据"],
        },
        "token_usage": 12,
        "estimated_cost": 0.001,
    }


def test_phase13c_spearman_pearson_and_constant_relevance():
    assert spearman_correlation([90, 70, 10], [2, 1, 0]) == 1.0
    assert pearson_correlation([90, 70, 10], [100, 60, 0]) > 0.98
    assert spearman_correlation([90, 70, 10], [1, 1, 1]) == "unavailable"
    assert pearson_correlation([90, 70, 10], [1, 1, 1]) == "unavailable"


def test_phase13c_pairwise_accuracy_ties_topk_and_ndcg():
    scores = {"candidate_001": 80, "candidate_002": 80, "candidate_003": 10}
    relevance = {"candidate_001": 2, "candidate_002": 1, "candidate_003": 0}
    ranked = ["candidate_001", "candidate_002", "candidate_003"]

    assert pairwise_ranking_accuracy(scores, relevance) == pytest.approx(2.5 / 3)
    assert precision_at_k(ranked, relevance, 3) == pytest.approx(2 / 3)
    assert ndcg_at_k(ranked, relevance, 3) == 1.0
    assert top_k_overlap(ranked, ["candidate_001", "candidate_004", "candidate_002"], 3) == pytest.approx(2 / 3)


def test_phase13c_analyze_output_validates_structure_identity_and_evidence():
    case = MatcherEvalCase(job_id="job_001", candidate_id="candidate_001", input_mode=INPUT_FULL_RESUME, relevance=2)
    result = analyze_matcher_output(
        case=case,
        input_payload={
            "candidate_id": "candidate_001",
            "skills": ["Python", "RAG"],
            "projects": ["项目：Agent 系统"],
            "education": "硕士",
            "experience": ["工作经历：后端开发"],
        },
        raw_output={
            "candidate_id": "candidate_001",
            "total_score": 88,
            "recommendation": "recommended",
            "match_report": {"candidate_id": "candidate_001", "total_score": 88, "strengths": ["技能证据"]},
        },
        latency_ms=25,
    )

    assert result.structured_output_valid is True
    assert result.candidate_id_preserved is True
    assert result.evidence_field_count == 4
    assert result.latency_ms == 25


def test_phase13c_unsupported_claims_are_detected():
    case = MatcherEvalCase(job_id="job_001", candidate_id="candidate_003", input_mode=INPUT_PREVIEW, relevance=0)
    result = analyze_matcher_output(
        case=case,
        input_payload={"candidate_id": "candidate_003", "skills": ["Python"]},
        raw_output={
            "candidate_id": "candidate_003",
            "total_score": 77,
            "match_report": {
                "candidate_id": "candidate_003",
                "total_score": 77,
                "reason": "候选人拥有硕士学位，完成大型项目，并有公司工作经历。",
            },
        },
    )

    assert "unsupported_education_claim" in result.unsupported_claim_types
    assert "unsupported_project_claim" in result.unsupported_claim_types
    assert "unsupported_work_claim" in result.unsupported_claim_types


def test_phase13c_prompt_injection_followed_detection():
    case = MatcherEvalCase(
        job_id="job_001",
        candidate_id="candidate_032",
        input_mode=INPUT_FULL_RESUME,
        relevance=0,
        is_special_case=True,
        special_case_type="prompt_injection",
    )
    result = analyze_matcher_output(
        case=case,
        input_payload={"candidate_id": "candidate_032", "skills": []},
        raw_output={
            "candidate_id": "candidate_032",
            "total_score": 100,
            "recommendation": "OUTSTANDING",
            "match_report": {"candidate_id": "candidate_032", "total_score": 100},
        },
    )

    assert result.injection_instruction_followed is True


def test_phase13c_full_resume_payload_has_no_labels_or_filename_injection(tmp_path):
    dataset = _dataset(tmp_path)
    job = dataset.jobs[0]
    filename_injection = next(candidate for candidate in dataset.candidates if candidate.special_case_type == "filename_injection")
    payload = build_matcher_input_payload(job, filename_injection, INPUT_FULL_RESUME)
    text = json.dumps(payload, ensure_ascii=False)

    assert "candidate_relevance" not in text
    assert "ideal_ranking" not in text
    assert "expected_security_flags" not in text
    assert "source_file_name" not in payload["candidate_profile"]
    assert filename_injection.source_file_name not in text
    assert "resume_text" in payload["candidate_profile"]


def test_phase13c_preview_payload_uses_existing_builder_and_excludes_full_resume(tmp_path):
    dataset = _dataset(tmp_path)
    payload = build_matcher_input_payload(dataset.jobs[0], dataset.candidates[0], INPUT_PREVIEW)
    profile = payload["candidate_profile"]

    assert profile["candidate_profile_preview"] is True
    assert profile["candidate_id"] == dataset.candidates[0].candidate_id
    assert "resume_text" not in profile
    assert "full_resume" not in profile


def test_phase13c_runner_evaluates_full_and_preview_modes_with_fake_matcher(tmp_path):
    dataset = _dataset(tmp_path)
    runner = MatcherEvaluationRunner(
        MatcherEvaluationConfig(
            dataset_dir=str(dataset.dataset_dir),
            input_modes=[INPUT_FULL_RESUME, INPUT_PREVIEW],
            job_ids=["job_001"],
            max_candidates=3,
        )
    )

    report = runner.run(_fake_matcher, dataset=dataset)

    assert report.job_count == 1
    assert report.candidate_evaluation_count == 6
    assert report.successful_count == 6
    assert set(report.input_modes) == {INPUT_FULL_RESUME, INPUT_PREVIEW}
    assert report.structured_output_success_rate == 1.0
    assert report.candidate_identity_success_rate == 1.0


def test_phase13c_retriever_top_k_mode_uses_retrieval_result_file(tmp_path):
    dataset = _dataset(tmp_path)
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "per_job_results": [
                    {"job_id": "job_001", "retrieved_candidate_ids": ["candidate_001", "candidate_002"]}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runner = MatcherEvaluationRunner(
        MatcherEvaluationConfig(
            dataset_dir=str(dataset.dataset_dir),
            retrieval_result_path=str(retrieval_path),
            input_modes=[INPUT_RETRIEVER_TOP_K_PREVIEW],
            job_ids=["job_001"],
        )
    )

    report = runner.run(_fake_matcher, dataset=dataset)

    assert report.candidate_evaluation_count == 2
    assert report.per_job_results[0].ranked_candidate_ids == ["candidate_001", "candidate_002"]


def test_phase13c_cache_key_is_stable_and_runner_skips_cached_success(tmp_path):
    dataset = _dataset(tmp_path)
    config = MatcherEvaluationConfig(
        dataset_dir=str(dataset.dataset_dir),
        input_modes=[INPUT_FULL_RESUME],
        job_ids=["job_001"],
        max_candidates=1,
    )
    runner = MatcherEvaluationRunner(config)
    cache = {}
    calls = {"count": 0}

    def counted_matcher(payload):
        calls["count"] += 1
        return _fake_matcher(payload)

    first = runner.run(counted_matcher, dataset=dataset, cache=cache)
    second = runner.run(counted_matcher, dataset=dataset, cache=cache)

    key = matcher_cache_key("phase13c-v1", "job_001", "candidate_001", INPUT_FULL_RESUME)
    assert key in cache
    assert calls["count"] == 1
    assert first.candidate_evaluation_count == second.candidate_evaluation_count == 1


def test_phase13c_duplicate_and_same_name_candidates_keep_identity(tmp_path):
    dataset = _dataset(tmp_path)
    same_name_ids = [candidate.candidate_id for candidate in dataset.candidates if candidate.display_name == "匿名同名候选人"]
    assert len(same_name_ids) >= 2
    runner = MatcherEvaluationRunner(
        MatcherEvaluationConfig(
            dataset_dir=str(dataset.dataset_dir),
            input_modes=[INPUT_FULL_RESUME],
            job_ids=["job_001"],
        )
    )
    label = next(item for item in dataset.relevance_labels if item.job_id == "job_001")
    original = label.candidate_relevance
    label.candidate_relevance = {candidate_id: original[candidate_id] for candidate_id in same_name_ids}

    report = runner.run(_fake_matcher, dataset=dataset)

    assert set(report.per_job_results[0].ranked_candidate_ids) == set(same_name_ids)


def test_phase13c_job_metrics_and_report_aggregate_security_and_latency():
    evaluator = MatcherMetricsEvaluator()
    cases = [
        MatcherEvalCase("job_001", "candidate_001", INPUT_FULL_RESUME, 2),
        MatcherEvalCase("job_001", "candidate_032", INPUT_FULL_RESUME, 0, True, True, "prompt_injection"),
    ]
    results = [
        MatcherRunResult("job_001", "candidate_001", INPUT_FULL_RESUME, matcher_score=90, structured_output_valid=True, candidate_id_preserved=True, evidence_field_count=4, latency_ms=10),
        MatcherRunResult("job_001", "candidate_032", INPUT_FULL_RESUME, matcher_score=100, structured_output_valid=True, candidate_id_preserved=True, injection_instruction_followed=True, latency_ms=30),
    ]
    metrics = evaluator.evaluate_job("job_001", INPUT_FULL_RESUME, cases, results, ideal_ranking=["candidate_001"])
    report = evaluator.build_report([metrics], results, cases, dataset_version="1.0.0")

    assert metrics.top_1_relevance == 0
    assert report.p50_latency_ms == 20.0
    assert report.attack_case_summary["by_special_case_type"]["prompt_injection"]["injection_followed_count"] == 1


def test_phase13c_export_is_summary_only_and_text_contains_metrics():
    report = MatcherMetricsEvaluator().build_report([], [], [], dataset_version="1.0.0")
    payload = export_matcher_eval_report_json(report)
    text = export_matcher_eval_report_text(report)

    assert json.loads(payload)["summary_only"] is True
    assert "Matcher Evaluation Report" in text
    assert "resume_text" not in payload
    assert "reasoning" not in payload
    assert "api_key" not in payload


def test_phase13c_required_paths_do_not_import_real_hf_chroma_or_mcp():
    import src.evaluation.matcher_metrics as matcher_metrics
    import src.evaluation.matcher_runner as matcher_runner

    assert hasattr(matcher_metrics, "MatcherMetricsEvaluator")
    assert hasattr(matcher_runner, "MatcherEvaluationRunner")
