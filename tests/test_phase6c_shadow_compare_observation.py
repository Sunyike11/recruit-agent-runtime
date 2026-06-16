import builtins
from pathlib import Path

from src.integration import (
    ProductionShadowParityFixture,
    ShadowCompareDecision,
    ShadowCompareObservation,
    ShadowCompareObserver,
    ShadowCompareReport,
)


SENSITIVE_TEXT = "FULL-JD-RESUME-API-SECRET-MUST-NOT-ENTER-OBSERVATION"


def match_fixture():
    return ProductionShadowParityFixture(
        fixture_id="match_case",
        raw_jd="Need Python LangGraph engineer",
        production_state={
            "messages": [{"content": "Need Python LangGraph engineer"}],
            "extracted_jd": {
                "title": "Agent Engineer",
                "required_skills": ["Python", "LangGraph"],
                "search_query": "Python LangGraph",
            },
            "candidate_pool": [{"candidate_id": "candidate_1"}],
            "final_reports": [{"candidate_id": "candidate_1", "total_score": 92}],
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
            "compare_exact_scores": True,
        },
    )


def test_shadow_compare_observation_can_create_and_serialize():
    observation = ShadowCompareObserver().observe(match_fixture(), target_name="matcher_preview")
    data = observation.to_dict()

    assert isinstance(observation, ShadowCompareObservation)
    assert data["target_name"] == "matcher_preview"
    assert data["parity_report"]["passed"] is True
    assert "production_state" not in data


def test_shadow_compare_decision_can_create():
    decision = ShadowCompareDecision(
        observation_id="obs_1",
        status="match",
        risk_level="low",
        reason="Aligned.",
        recommended_action="Retain evidence.",
    )

    assert decision.to_dict()["status"] == "match"


def test_observer_generates_match_decision_for_clean_passed_parity():
    observer = ShadowCompareObserver()

    decision = observer.decide(observer.observe(match_fixture()))

    assert decision.status == "match"
    assert decision.risk_level == "low"


def test_observer_generates_warning_decision_for_parity_warning():
    fixture = match_fixture()
    fixture.expected_alignment["compare_exact_scores"] = False

    decision = ShadowCompareObserver().decide(ShadowCompareObserver().observe(fixture))

    assert decision.status == "warning"
    assert decision.risk_level == "medium"


def test_mismatch_parity_generates_high_risk_decision():
    fixture = match_fixture()
    fixture.shadow_result["retrieved_candidates"] = []
    observer = ShadowCompareObserver()

    decision = observer.decide(observer.observe(fixture))

    assert decision.status == "mismatch"
    assert decision.risk_level == "high"


def test_missing_snapshot_generates_skipped_decision():
    observer = ShadowCompareObserver()
    observation = observer.observe(
        observation_id="missing_production",
        raw_jd="summary input",
        production_snapshot=None,
        shadow_snapshot={"status": "completed", "success": True},
    )

    decision = observer.decide(observation)

    assert decision.status == "skipped"
    assert decision.risk_level == "medium"
    assert observation.parity_report.missing_fields == ["production_snapshot"]


def test_observe_many_builds_report_counts_across_decision_types():
    observer = ShadowCompareObserver()
    warning = match_fixture()
    warning.fixture_id = "warning_case"
    warning.shadow_result["memory_context"] = "preview only"
    mismatch = match_fixture()
    mismatch.fixture_id = "mismatch_case"
    mismatch.shadow_result["match_reports"] = []
    skipped = observer.observe(
        observation_id="skipped_case",
        production_snapshot=None,
        shadow_snapshot={"status": "completed", "success": True},
    )

    report = observer.observe_many([match_fixture(), warning, mismatch, skipped])

    assert isinstance(report, ShadowCompareReport)
    assert report.total_observations == 4
    assert report.match_count == 1
    assert report.warning_count == 1
    assert report.mismatch_count == 1
    assert report.skipped_count == 1
    assert report.high_risk_count == 1


def test_memory_context_preview_difference_is_warning_not_automatic_failure():
    fixture = match_fixture()
    fixture.shadow_result["memory_context"] = "optional shadow preview"
    observer = ShadowCompareObserver()

    observation = observer.observe(fixture)
    decision = observer.decide(observation)

    assert observation.parity_report.passed is True
    assert observation.parity_report.preview_only_fields == ["memory_context"]
    assert decision.status == "warning"
    assert decision.risk_level == "low"


def test_observation_summary_omits_full_sensitive_payload():
    fixture = match_fixture()
    fixture.raw_jd = f"Need Python {SENSITIVE_TEXT}"
    fixture.production_state["messages"] = [{"content": fixture.raw_jd}]
    fixture.production_state["evidence"] = SENSITIVE_TEXT
    fixture.shadow_result["metadata"] = {"secret": SENSITIVE_TEXT}
    observation = ShadowCompareObserver().observe(
        fixture,
        metadata={"secret": SENSITIVE_TEXT},
    )

    serialized = str(observation.to_dict())

    assert SENSITIVE_TEXT not in serialized
    assert "raw_jd_length" in serialized
    assert "production_state" not in serialized
    assert "shadow_result" not in serialized


def test_report_serialization_is_summary_only():
    report = ShadowCompareObserver().observe_many([match_fixture()])
    data = report.to_dict()

    assert data["match_count"] == 1
    assert data["metadata"]["summary_only"] is True
    assert "production_state" not in str(data)


def test_phase6c_does_not_run_or_integrate_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    source = Path("src/integration/shadow_compare.py").read_text(encoding="utf-8")

    decision = ShadowCompareObserver().decide(ShadowCompareObserver().observe(match_fixture()))

    assert decision.status == "match"
    assert "ShadowCompareObserver" not in graph_source
    assert "src.integration.shadow_compare" not in graph_source
    assert "src.core.graph" not in source
    assert "create_recruit_graph" not in source


def test_phase6c_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase6C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    decision = ShadowCompareObserver().decide(ShadowCompareObserver().observe(match_fixture()))

    assert decision.status == "match"
    assert blocked == []
