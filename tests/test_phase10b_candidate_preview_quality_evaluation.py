import json
from pathlib import Path

from src.evaluation.candidate_preview import (
    CandidatePreviewEvalCase,
    CandidatePreviewEvaluator,
    build_candidate_preview_quality_gate,
    export_candidate_preview_eval_json,
    export_candidate_preview_eval_text,
    validate_candidate_preview_privacy,
    validate_matcher_input_privacy,
)
from src.runtime.candidate_preview import (
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
)


FIXTURE_PATH = Path("tests/fixtures/candidate_preview_quality_cases.json")
FULL_PRIVATE_TEXT = (
    "FULL PRIVATE RESUME SHOULD NOT LEAK. 候选人本科计算机，参与 Python RAG 检索系统项目，"
    "负责自动化评估、部署、测试和工程实践。这段文本较长，用来验证隐私检测不会输出完整 chunk。"
)


def load_cases():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_single_candidate_case_passes_with_expected_metrics():
    case = CandidatePreviewEvalCase.from_dict(load_cases()[0])
    result = CandidatePreviewEvaluator().run_case(case)

    assert result.status == "passed"
    assert result.grouping_correct is True
    assert result.candidate_id_match_count == 1
    assert result.candidate_name_match_count == 1
    assert result.source_document_match_count == 1
    assert result.skills_metrics.recall == 1.0
    assert result.project_keywords_metrics.recall >= 0.85
    assert result.privacy_checks_passed is True


def test_multi_chunk_grouping_and_evidence_chunk_count():
    case = CandidatePreviewEvalCase.from_dict(load_cases()[1])
    result = CandidatePreviewEvaluator().run_case(case)
    build_result = build_candidate_profile_previews_from_retrieval_results(
        case.retrieval_chunks,
        raw_jd=case.raw_jd,
    )

    assert result.status == "passed"
    assert result.preview_count == 1
    assert build_result.previews[0].evidence_chunk_count == 2
    assert "low_evidence_chunk_count" not in build_result.previews[0].preview_quality_flags


def test_multi_candidate_separation():
    case = CandidatePreviewEvalCase.from_dict(load_cases()[2])
    result = CandidatePreviewEvaluator().run_case(case)

    assert result.status == "passed"
    assert result.preview_count == 2
    assert result.grouping_correct is True
    assert result.candidate_id_match_count == 2


def test_stable_fallback_id_for_missing_candidate_id():
    case = CandidatePreviewEvalCase.from_dict(load_cases()[3])
    first = build_candidate_profile_previews_from_retrieval_results(case.retrieval_chunks)
    second = build_candidate_profile_previews_from_retrieval_results(case.retrieval_chunks)

    assert first.previews[0].candidate_id.startswith("candidate_preview_")
    assert first.previews[0].candidate_id == second.previews[0].candidate_id


def test_candidate_name_accuracy_from_file_and_missing_name_flag():
    evaluator = CandidatePreviewEvaluator()
    file_case = CandidatePreviewEvalCase.from_dict(load_cases()[4])
    missing_case = CandidatePreviewEvalCase.from_dict(load_cases()[5])

    file_result = evaluator.run_case(file_case)
    missing_result = evaluator.run_case(missing_case)

    assert file_result.status == "passed"
    assert file_result.candidate_name_match_count == 1
    assert missing_result.status == "passed"
    assert missing_result.quality_flags_match is True


def test_skills_precision_recall_f1_and_project_education_experience_coverage():
    result = CandidatePreviewEvaluator().run_case(CandidatePreviewEvalCase.from_dict(load_cases()[6]))

    assert result.status == "passed"
    assert result.skills_metrics.precision == 1.0
    assert result.skills_metrics.recall == 1.0
    assert result.skills_metrics.f1 == 1.0
    assert result.project_keywords_metrics.recall >= 0.8
    assert result.education_keywords_metrics.recall == 1.0
    assert result.experience_keywords_metrics.recall == 1.0


def test_query_terms_comparison_and_grouping_priority():
    evaluator = CandidatePreviewEvaluator()
    privacy_case = CandidatePreviewEvalCase.from_dict(load_cases()[7])
    grouping_case = CandidatePreviewEvalCase.from_dict(load_cases()[8])

    privacy_result = evaluator.run_case(privacy_case)
    grouping_result = evaluator.run_case(grouping_case)

    assert privacy_result.matched_query_terms_metrics.recall == 1.0
    assert privacy_result.evidence_summary_truncated_count == 1
    assert grouping_result.status == "passed"
    assert grouping_result.preview_count == 2
    assert grouping_result.candidate_id_match_count == 2


def test_privacy_violation_for_full_chunk_and_forbidden_keys_fails_case():
    preview = {
        "candidate_id": "bad",
        "candidate_name": "Bad",
        "source_document_id": "bad.pdf",
        "source_file_name": "bad.pdf",
        "evidence_summary": FULL_PRIVATE_TEXT,
        "raw_text": FULL_PRIVATE_TEXT,
        "summary_only": True,
    }
    privacy = validate_candidate_preview_privacy(
        [preview],
        retrieval_chunks=[{"text": FULL_PRIVATE_TEXT, "metadata": {"file_name": "bad.pdf"}}],
        max_evidence_chars=320,
    )

    assert privacy["passed"] is False
    assert "forbidden_key:raw_text" in privacy["errors"]
    assert "evidence_summary_equals_full_chunk" in privacy["errors"]


def test_matcher_compatibility_and_privacy():
    case = CandidatePreviewEvalCase.from_dict(load_cases()[9])
    build_result = build_candidate_profile_previews_from_retrieval_results(case.retrieval_chunks, raw_jd=case.raw_jd)
    matcher_input = candidate_profile_preview_to_matcher_input(build_result.previews[0])
    privacy = validate_matcher_input_privacy([matcher_input], retrieval_chunks=case.retrieval_chunks)

    assert matcher_input["candidate_profile_preview"] is True
    assert matcher_input["metadata"]["candidate_profile_preview"] is True
    assert privacy["passed"] is True
    assert "raw_chunks" not in json.dumps(matcher_input, ensure_ascii=False)


def test_batch_report_aggregation_and_quality_gate():
    evaluator = CandidatePreviewEvaluator()
    report = evaluator.run_cases(load_cases())
    gate = build_candidate_preview_quality_gate(report.results[0])

    assert report.total_cases == 10
    assert report.passed_cases >= 9
    assert report.grouping_accuracy == 1.0
    assert report.privacy_pass_rate == 1.0
    assert report.matcher_compatibility_rate == 1.0
    assert gate["status"] == "pass"
    assert gate["summary_only"] is True


def test_deterministic_serialization_is_summary_only():
    report = CandidatePreviewEvaluator().run_cases(load_cases())
    first = export_candidate_preview_eval_json(report)
    second = export_candidate_preview_eval_json(report)
    text = export_candidate_preview_eval_text(report)

    assert first == second
    assert "Candidate Preview Evaluation Report" in text
    assert "FULL PRIVATE RESUME" not in first
    assert "raw_text" not in first
    assert "llm_response" not in first
    assert "api_key" not in first


def test_fixture_catalog_covers_required_scenarios():
    case_ids = {case["case_id"] for case in load_cases()}

    assert {
        "case_a_single_candidate_single_chunk",
        "case_b_single_candidate_multi_chunk",
        "case_c_two_candidates",
        "case_d_missing_candidate_id_stable_preview",
        "case_e_name_from_file_name",
        "case_f_name_missing",
        "case_g_skill_whitelist",
        "case_h_privacy_truncation",
        "case_i_grouping_candidate_id_priority",
        "case_j_matcher_compatibility",
    }.issubset(case_ids)


def test_default_graph_behavior_is_not_modified():
    import src.core.graph as graph

    assert "CandidatePreviewEvaluator" not in graph.create_recruit_graph.__code__.co_names
    assert "build_candidate_preview_quality_gate" not in graph.create_recruit_graph.__code__.co_names
