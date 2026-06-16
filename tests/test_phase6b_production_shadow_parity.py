import builtins
from pathlib import Path

from src.integration import (
    ProductionShadowParityBatchReport,
    ProductionShadowParityComparator,
    ProductionShadowParityFixture,
    ProductionShadowParityReport,
    load_production_shadow_parity_fixtures,
)


def aligned_fixture():
    return ProductionShadowParityFixture(
        fixture_id="aligned_case",
        raw_jd="Need Python LangGraph engineer",
        production_state={
            "messages": [{"content": "Need Python LangGraph engineer"}],
            "extracted_jd": {
                "title": "Agent Engineer",
                "required_skills": ["Python", "LangGraph"],
                "search_query": "Python LangGraph",
            },
            "candidate_pool": [{"candidate_id": "candidate_1"}],
            "final_reports": [{"candidate_id": "candidate_1", "total_score": 95}],
        },
        shadow_result={
            "status": "completed",
            "success": True,
            "job_requirement": {
                "title": "Agent Engineer",
                "required_skills": ["Python", "LangGraph"],
                "metadata": {"search_query": "Python LangGraph"},
            },
            "retrieved_candidates": [{"candidate_id": "candidate_1"}],
            "match_reports": [{"candidate_id": "candidate_1", "total_score": 92}],
            "refined_query": None,
        },
        expected_alignment={
            "job_requirement_keys": ["title", "required_skills"],
        },
    )


def test_parity_fixture_can_create_and_round_trip():
    fixture = aligned_fixture()
    restored = ProductionShadowParityFixture.from_dict(fixture.to_dict())

    assert restored.fixture_id == "aligned_case"
    assert restored.production_state["candidate_pool"][0]["candidate_id"] == "candidate_1"


def test_parity_report_can_create_and_serialize():
    report = ProductionShadowParityReport(
        fixture_id="case",
        passed=False,
        mismatched_fields=["raw_jd"],
    )

    assert report.to_dict()["mismatched_fields"] == ["raw_jd"]


def test_aligned_fixture_returns_passed_without_exact_score_comparison():
    report = ProductionShadowParityComparator().compare(aligned_fixture())

    assert report.passed is True
    assert "raw_jd" in report.aligned_fields
    assert "query" in report.aligned_fields
    assert "retrieved_candidates/candidate_pool.ids" in report.aligned_fields
    assert "match_reports.exact_scores" not in report.mismatched_fields
    assert report.metadata["real_production_graph_invoked"] is False


def test_missing_production_field_is_recorded_without_raising():
    fixture = aligned_fixture()
    fixture.production_state.pop("messages")

    report = ProductionShadowParityComparator().compare(fixture)

    assert report.passed is False
    assert "production.raw_jd/messages[-1].content" in report.missing_fields


def test_missing_shadow_field_is_recorded_without_raising():
    fixture = aligned_fixture()
    fixture.shadow_result.pop("match_reports")

    report = ProductionShadowParityComparator().compare(fixture)

    assert report.passed is False
    assert "shadow_result.match_reports" in report.missing_fields


def test_candidate_count_mismatch_is_recorded():
    fixture = aligned_fixture()
    fixture.shadow_result["retrieved_candidates"] = []

    report = ProductionShadowParityComparator().compare(fixture)

    assert report.passed is False
    assert "retrieved_candidates/candidate_pool.count" in report.mismatched_fields


def test_memory_and_reflection_context_are_preview_only_and_do_not_fail():
    fixture = aligned_fixture()
    fixture.shadow_result["memory_context"] = "approved preview only"
    fixture.shadow_result["metadata"] = {"reflection_metadata": {"status": "success"}}
    fixture.metadata["closed_loop_memory_preview"] = True

    report = ProductionShadowParityComparator().compare(fixture)

    assert report.passed is True
    assert report.preview_only_fields == [
        "memory_context",
        "reflection_metadata",
        "closed_loop_memory_preview",
    ]


def test_explicit_exact_score_requirement_can_report_mismatch():
    fixture = aligned_fixture()
    fixture.expected_alignment["compare_exact_scores"] = True

    report = ProductionShadowParityComparator().compare(fixture)

    assert report.passed is False
    assert "match_reports.exact_scores" in report.mismatched_fields


def test_compare_many_returns_batch_summary():
    mismatch = aligned_fixture()
    mismatch.fixture_id = "mismatch_case"
    mismatch.shadow_result["retrieved_candidates"] = []

    report = ProductionShadowParityComparator().compare_many([aligned_fixture(), mismatch])

    assert isinstance(report, ProductionShadowParityBatchReport)
    assert report.total_fixtures == 2
    assert report.passed_fixtures == 1
    assert report.failed_fixtures == 1
    assert report.metadata["real_production_graph_invoked"] is False


def test_json_fixture_file_loads_deterministic_fake_cases():
    fixtures = load_production_shadow_parity_fixtures(
        Path("tests/fixtures/production_shadow_parity_cases.json")
    )
    report = ProductionShadowParityComparator().compare_many(fixtures)

    assert [fixture.fixture_id for fixture in fixtures] == [
        "aligned_python_langgraph_candidate",
        "candidate_count_mismatch",
    ]
    assert report.passed_fixtures == 1
    assert report.failed_fixtures == 1


def test_comparator_does_not_run_or_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    parity_source = Path("src/integration/parity.py").read_text(encoding="utf-8")

    report = ProductionShadowParityComparator().compare(aligned_fixture())

    assert report.metadata["real_production_graph_invoked"] is False
    assert "src.integration.parity" not in graph_source
    assert "ProductionShadowParityComparator" not in graph_source
    assert "src.core.graph" not in parity_source
    assert "create_recruit_graph" not in parity_source


def test_phase6b_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase6B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = ProductionShadowParityComparator().compare(aligned_fixture())

    assert report.passed is True
    assert blocked == []
