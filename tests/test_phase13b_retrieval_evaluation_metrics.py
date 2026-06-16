import json
from pathlib import Path

import pytest

from scripts.build_recruitment_eval_index import build_recruitment_eval_index, build_recruitment_eval_index_dry_run
from scripts.generate_recruitment_eval_dataset import generate_dataset
from src.evaluation.dataset import load_recruitment_eval_dataset
from src.evaluation.retrieval_export import export_retrieval_report_json, export_retrieval_report_text
from src.evaluation.retrieval_metrics import (
    RetrievalEvalCase,
    RetrievalMetricsEvaluator,
    adapt_eval_retrieval_results,
    build_query_mode_comparison_summary,
    build_worst_job_summary,
    deduplicate_retrieved_candidates,
)
from src.evaluation.retrieval_runner import RetrievalEvaluationConfig, RetrievalEvaluationRunner, build_retrieval_query


def _case(top_k_values=None):
    return RetrievalEvalCase(
        job_id="job_eval",
        query="Python RAG LangGraph",
        relevance_labels={
            "candidate_001": 2,
            "candidate_002": 1,
            "candidate_003": 0,
            "candidate_004": 2,
            "candidate_005": 0,
        },
        ideal_ranking=["candidate_001", "candidate_004", "candidate_002"],
        top_k_values=top_k_values or [5, 10],
    )


def test_phase13b_recall_precision_hit_mrr_are_correct():
    evaluator = RetrievalMetricsEvaluator()
    metrics = evaluator.evaluate_case(
        _case([1, 3, 5, 10]),
        [
            {"candidate_id": "candidate_003", "score": 0.9},
            {"candidate_id": "candidate_002", "score": 0.8},
            {"candidate_id": "candidate_001", "score": 0.7},
        ],
    )

    assert metrics.recall_at_k[1] == 0.0
    assert metrics.recall_at_k[3] == pytest.approx(2 / 3)
    assert metrics.recall_at_k[5] == pytest.approx(2 / 3)
    assert metrics.recall_at_k[10] == pytest.approx(2 / 3)
    assert metrics.precision_at_k[1] == 0.0
    assert metrics.precision_at_k[3] == pytest.approx(2 / 3)
    assert metrics.hit_rate_at_k[1] == 0.0
    assert metrics.hit_rate_at_k[3] == 1.0
    assert metrics.mrr == 0.5
    assert metrics.first_relevant_rank == 2


def test_phase13b_dcg_and_ndcg_use_graded_relevance():
    evaluator = RetrievalMetricsEvaluator()
    ranking = ["candidate_003", "candidate_002", "candidate_001"]
    labels = _case().relevance_labels

    dcg = evaluator.compute_dcg(ranking, labels, 3)
    ndcg = evaluator.compute_ndcg_at_k(ranking, labels, 3)

    assert dcg == pytest.approx(2.13093, abs=1e-5)
    assert ndcg == pytest.approx(0.395144, abs=1e-5)
    assert evaluator.compute_ndcg_at_k(["candidate_001", "candidate_004", "candidate_002"], labels, 3) == 1.0


def test_phase13b_empty_results_and_no_relevant_candidates():
    evaluator = RetrievalMetricsEvaluator()
    empty = evaluator.evaluate_case(_case([5]), [])
    no_relevant = evaluator.evaluate_case(
        RetrievalEvalCase(job_id="empty", query="x", relevance_labels={"candidate_001": 0}, top_k_values=[5]),
        [{"candidate_id": "candidate_001"}],
    )

    assert empty.recall_at_k[5] == 0.0
    assert empty.precision_at_k[5] == 0.0
    assert empty.hit_rate_at_k[5] == 0.0
    assert empty.mrr == 0.0
    assert no_relevant.recall_at_k[5] == 0.0
    assert no_relevant.hit_rate_at_k[5] == 0.0


def test_phase13b_duplicate_chunks_do_not_repeat_candidate_rank():
    evaluator = RetrievalMetricsEvaluator()
    metrics = evaluator.evaluate_case(
        _case([5]),
        [
            {"candidate_id": "candidate_001", "chunk_id": "a", "rank": 1},
            {"candidate_id": "candidate_001", "chunk_id": "b", "rank": 2},
            {"candidate_id": "candidate_002", "chunk_id": "c", "rank": 3},
        ],
    )

    assert metrics.retrieved_candidate_ids == ["candidate_001", "candidate_002"]
    assert metrics.raw_chunk_count == 3
    assert metrics.unique_candidate_count == 2
    assert metrics.duplicate_chunk_count == 1


def test_phase13b_adapt_results_requires_candidate_id():
    evaluator = RetrievalMetricsEvaluator()
    adapted = adapt_eval_retrieval_results([{"metadata": {"source_document_id": "candidate_001"}}])
    metrics = evaluator.evaluate_case(_case([5]), [{"metadata": {"source_document_id": "candidate_001"}}])

    assert adapted[0].candidate_id == ""
    assert metrics.status == "invalid_identity"
    assert metrics.error_type == "missing_candidate_id"


def test_phase13b_report_macro_aggregation_latency_and_attack_summary():
    evaluator = RetrievalMetricsEvaluator()
    cases = [_case([5, 10]), RetrievalEvalCase(job_id="job_2", query="x", relevance_labels={"candidate_033": 1}, top_k_values=[5, 10])]
    report = evaluator.evaluate_cases(
        cases,
        {
            "job_eval": [{"candidate_id": "candidate_001"}, {"candidate_id": "candidate_002"}],
            "job_2": [{"candidate_id": "candidate_033"}],
        },
        latency_by_job={"job_eval": 10, "job_2": 30},
        dataset_version="1.0.0",
        index_version="phase13b-v1",
        attack_candidate_types={"candidate_033": "keyword_stuffing"},
    )

    assert report.successful_case_count == 2
    assert report.failed_case_count == 0
    assert report.p50_latency_ms == 20.0
    assert report.p95_latency_ms == 29.0
    assert report.mean_latency_ms == 20.0
    assert report.attack_case_summary["retrieved_attack_candidate_count"] == 1
    assert "keyword_stuffing" in report.attack_case_summary["by_attack_type"]


def test_phase13b_query_modes_are_deterministic():
    job = {
        "jd_text": "原始 JD",
        "required_skills": ["Python", "RAG"],
        "responsibilities": ["构建检索系统"],
        "hard_constraints": ["有上线经验"],
    }

    assert build_retrieval_query(job, "raw_jd") == "原始 JD"
    assert build_retrieval_query(job, "structured") == "Python；RAG；构建检索系统；有上线经验"
    with pytest.raises(ValueError):
        build_retrieval_query(job, "planner")


def test_phase13b_runner_uses_fake_retrieval_once_per_job(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    calls = []

    def fake_retrieve(query, k):
        calls.append((query, k))
        return [{"candidate_id": "candidate_001", "score": 1.0}]

    runner = RetrievalEvaluationRunner(
        RetrievalEvaluationConfig(dataset_dir=str(output_dir), top_k_values=[5, 10], query_mode="raw_jd")
    )
    report = runner.run(fake_retrieve, dataset=dataset)

    assert len(calls) == 12
    assert all(call[1] == 10 for call in calls)
    assert report.job_count == 12
    assert report.dataset_version == "1.0.0"


def test_phase13b_retriever_factory_initialized_once_and_searches_per_case(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    events = []

    class FakeReusableRetriever:
        def search(self, query, k):
            events.append(("search", k))
            return [{"candidate_id": "candidate_001", "score": 1.0}]

    def factory(config):
        events.append(("factory", config.index_dir))
        return FakeReusableRetriever()

    runner = RetrievalEvaluationRunner(
        RetrievalEvaluationConfig(dataset_dir=str(output_dir), index_dir="fake_index", top_k_values=[5, 10])
    )
    report = runner.run_with_retriever_factory(factory, dataset=dataset)

    assert events.count(("factory", "fake_index")) == 1
    assert len([event for event in events if event[0] == "search"]) == 12
    assert all(event == ("search", 10) for event in events if event[0] == "search")
    assert report.successful_case_count == 12


def test_phase13b_initialization_and_query_latency_are_separated(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)
    dataset = load_recruitment_eval_dataset(output_dir)
    times = iter([
        0.00,  # total start
        0.10,  # init start
        0.35,  # init end = 250 ms
        0.35, 0.36,
        0.36, 0.38,
        0.38, 0.41,
        0.41, 0.45,
        0.45, 0.50,
        0.50, 0.56,
        0.56, 0.63,
        0.63, 0.71,
        0.71, 0.80,
        0.80, 0.90,
        0.90, 1.01,
        1.01, 1.13,
        1.20,  # total end
    ])

    class FakeReusableRetriever:
        def search(self, query, k):
            return [{"candidate_id": "candidate_001"}]

    runner = RetrievalEvaluationRunner(
        RetrievalEvaluationConfig(dataset_dir=str(output_dir), top_k_values=[5, 10]),
        clock=lambda: next(times),
    )
    report = runner.run_with_retriever_factory(lambda _config: FakeReusableRetriever(), dataset=dataset)

    assert report.initialization_duration_ms == 249
    assert report.benchmark_total_duration_ms == 1200
    assert report.p50_query_latency_ms == 64.5
    assert report.p95_query_latency_ms == pytest.approx(113.5)
    assert report.mean_query_latency_ms == pytest.approx(64.416667)


def test_phase13b_export_is_summary_only_and_omits_full_payload():
    evaluator = RetrievalMetricsEvaluator()
    report = evaluator.evaluate_cases(
        [_case([5])],
        {"job_eval": [{"candidate_id": "candidate_001"}]},
        dataset_version="1.0.0",
        index_version="phase13b-v1",
    )

    json_payload = export_retrieval_report_json(report)
    text_payload = export_retrieval_report_text(report)

    assert "mean_mrr" in json_payload
    assert "Retrieval Evaluation Report" in text_payload
    assert "Python RAG LangGraph" not in json_payload
    assert "resume_text" not in json_payload
    assert "raw_chunks" not in json_payload
    assert "reasoning" not in json_payload


def test_phase13b_query_mode_comparison_summary_does_not_pick_overall_winner():
    evaluator = RetrievalMetricsEvaluator()
    raw_report = evaluator.evaluate_cases(
        [_case([5, 10])],
        {"job_eval": [{"candidate_id": "candidate_001"}, {"candidate_id": "candidate_002"}]},
        dataset_version="1.0.0",
        index_version="phase13b-v1",
    )
    structured_report = evaluator.evaluate_cases(
        [_case([5, 10])],
        {"job_eval": [{"candidate_id": "candidate_004"}, {"candidate_id": "candidate_003"}, {"candidate_id": "candidate_002"}]},
        dataset_version="1.0.0",
        index_version="phase13b-v1",
    )

    summary = build_query_mode_comparison_summary(raw_report, structured_report)

    assert summary["raw_jd"]["mrr"] == 1.0
    assert summary["structured"]["mrr"] == 1.0
    assert summary["interpretation"]["do_not_claim_overall_winner"] is True
    assert summary["summary_only"] is True


def test_phase13b_worst_job_summary_is_summary_only():
    evaluator = RetrievalMetricsEvaluator()
    cases = [
        RetrievalEvalCase(job_id="job_011", query="devops", relevance_labels={"candidate_001": 1, "candidate_002": 0}, top_k_values=[5, 10]),
        RetrievalEvalCase(job_id="job_012", query="frontend", relevance_labels={"candidate_003": 1, "candidate_033": 0}, top_k_values=[5, 10]),
    ]
    report = evaluator.evaluate_cases(
        cases,
        {
            "job_011": [{"candidate_id": "candidate_002"}],
            "job_012": [{"candidate_id": "candidate_033"}],
        },
        attack_candidate_types={"candidate_033": "keyword_stuffing"},
    )

    summary = build_worst_job_summary(
        report,
        job_titles={"job_011": "DevOps / 平台工程师", "job_012": "前端工程师"},
    )

    assert [item["job_id"] for item in summary] == ["job_011", "job_012"]
    assert summary[0]["job_title"] == "DevOps / 平台工程师"
    assert summary[1]["retrieved_attack_candidate"] is True
    assert all(item["summary_only"] for item in summary)


def test_phase13b_index_builder_refuses_existing_chroma_db(tmp_path):
    output_dir = tmp_path / "eval_v1"
    generate_dataset(output_dir, seed=2026, force=True)

    with pytest.raises(ValueError):
        build_recruitment_eval_index_dry_run(output_dir, "chroma_db")


def test_phase13b_index_builder_with_fake_writer_writes_safe_metadata(tmp_path):
    output_dir = tmp_path / "eval_v1"
    index_dir = tmp_path / "eval_index"
    generate_dataset(output_dir, seed=2026, force=True)
    captured = {}

    def fake_writer(*, documents, index_dir, embedding_model, chunk_size, chunk_overlap):
        captured["documents"] = documents
        captured["index_dir"] = index_dir
        return {"chunk_count": 44}

    result = build_recruitment_eval_index(
        output_dir,
        index_dir,
        embedding_model="fake-embedding",
        chunk_size=256,
        chunk_overlap=32,
        force=True,
        index_writer=fake_writer,
    )
    manifest = json.loads((index_dir / "eval_index_manifest.json").read_text(encoding="utf-8"))
    first_metadata = captured["documents"][0]["metadata"]

    assert result["status"] == "ok"
    assert result["chunk_count"] == 44
    assert first_metadata["candidate_id"].startswith("candidate_")
    assert "candidate_relevance" not in json.dumps(captured["documents"], ensure_ascii=False)
    assert "ideal_ranking" not in json.dumps(captured["documents"], ensure_ascii=False)
    assert manifest["contains_relevance_labels"] is False
    assert manifest["contains_ideal_ranking"] is False


def test_phase13b_index_builder_refuses_overwrite_without_force(tmp_path):
    output_dir = tmp_path / "eval_v1"
    index_dir = tmp_path / "eval_index"
    generate_dataset(output_dir, seed=2026, force=True)
    index_dir.mkdir()

    with pytest.raises(FileExistsError):
        build_recruitment_eval_index(output_dir, index_dir, index_writer=lambda **_: {"chunk_count": 1})


def test_phase13b_required_modules_do_not_import_real_embedding_stack_at_module_import():
    for path in [
        Path("src/evaluation/retrieval_metrics.py"),
        Path("src/evaluation/retrieval_runner.py"),
        Path("src/evaluation/retrieval_export.py"),
    ]:
        source = path.read_text(encoding="utf-8")
        assert "llama_index" not in source
        assert "chromadb" not in source
        assert "HuggingFaceEmbedding" not in source


def test_phase13a_dataset_still_valid_after_phase13b():
    dataset_dir = Path("evaluation_data/v1")
    if not dataset_dir.exists():
        pytest.skip("Phase13A dataset not generated")

    dataset = load_recruitment_eval_dataset(dataset_dir)
    assert len(dataset.jobs) == 12
    assert len(dataset.candidates) == 40
    assert len(dataset.relevance_labels) == 12
