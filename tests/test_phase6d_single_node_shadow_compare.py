import builtins
from pathlib import Path

from src.integration import (
    SingleNodeShadowCompareCase,
    SingleNodeShadowCompareHarness,
    SingleNodeShadowCompareResult,
)
from src.skills import CandidateMatchSkill, QueryRefineSkill, SkillExecutor, SkillRegistry


SENSITIVE_TEXT = "FULL-RESUME-SECRET-API-KEY-MUST-NOT-ENTER-NODE-RESULT"


def refiner_case(shadow_query="python langgraph remote"):
    return SingleNodeShadowCompareCase(
        case_id="refiner_aligned",
        node_name="query_refine",
        node_type="refiner",
        input_data={"query": "python langgraph", "context": "broaden"},
        production_callable=lambda input_data: {
            "extracted_jd": {"search_query": "python langgraph remote"}
        },
        shadow_callable=lambda input_data: {"refined_query": shadow_query},
    )


def matcher_case(production_score=91, shadow_score=84, compare_exact_scores=False):
    return SingleNodeShadowCompareCase(
        case_id="matcher_aligned",
        node_name="candidate_match",
        node_type="matcher",
        input_data={
            "job_requirement": {"job_id": "job_1", "required_skills": ["Python"]},
            "candidate_profile": {"candidate_id": "candidate_1", "skills": ["Python"]},
        },
        production_callable=lambda input_data: {
            "final_reports": [
                {"candidate_id": "candidate_1", "total_score": production_score}
            ]
        },
        shadow_callable=lambda input_data: {
            "match_report": {
                "candidate_id": "candidate_1",
                "total_score": shadow_score,
            },
            "total_score": shadow_score,
        },
        expected_alignment={"compare_exact_scores": compare_exact_scores},
    )


def test_single_node_shadow_compare_case_can_create():
    case = refiner_case()

    assert case.node_type == "refiner"
    assert case.node_name == "query_refine"
    assert callable(case.production_callable)


def test_refiner_fake_outputs_aligned_generate_match_decision():
    result = SingleNodeShadowCompareHarness().run_case(refiner_case())

    assert isinstance(result, SingleNodeShadowCompareResult)
    assert result.success is True
    assert result.parity_report.passed is True
    assert result.decision.status == "match"
    assert result.decision.risk_level == "low"


def test_refiner_mismatched_query_generates_high_risk_mismatch():
    result = SingleNodeShadowCompareHarness().run_case(
        refiner_case(shadow_query="python only")
    )

    assert result.success is False
    assert result.decision.status == "mismatch"
    assert result.decision.risk_level == "high"
    assert result.parity_report.mismatched_fields == ["refined_query"]


def test_matcher_fake_outputs_align_without_exact_score_requirement():
    result = SingleNodeShadowCompareHarness().run_case(matcher_case())

    assert result.success is True
    assert result.decision.status == "match"
    assert "match_report.candidate_id" in result.parity_report.aligned_fields
    assert "match_report.score_presence" in result.parity_report.aligned_fields


def test_matcher_score_difference_is_compared_only_when_explicitly_requested():
    default_result = SingleNodeShadowCompareHarness().run_case(matcher_case())
    exact_result = SingleNodeShadowCompareHarness().run_case(
        matcher_case(compare_exact_scores=True)
    )

    assert default_result.decision.status == "match"
    assert exact_result.decision.status == "mismatch"
    assert "match_report.total_score" in exact_result.parity_report.mismatched_fields


def test_refiner_can_execute_injected_shadow_skill_through_executor():
    registry = SkillRegistry()
    registry.register(
        QueryRefineSkill(
            refine_callable=lambda input_data, context: {
                "refined_query": "python langgraph remote"
            }
        )
    )
    case = SingleNodeShadowCompareCase(
        case_id="refiner_skill_executor",
        node_name="query_refine",
        node_type="refiner",
        input_data={"query": "python langgraph"},
        production_callable=lambda input_data: {
            "extracted_jd": {"search_query": "python langgraph remote"}
        },
        shadow_skill_name="query_refine",
        skill_executor=SkillExecutor(registry),
    )

    result = SingleNodeShadowCompareHarness().run_case(case)

    assert result.decision.status == "match"
    assert result.metadata["real_production_graph_invoked"] is False


def test_matcher_can_execute_injected_shadow_skill_through_executor():
    registry = SkillRegistry()
    registry.register(
        CandidateMatchSkill(
            match_callable=lambda input_data, context: {"total_score": 88}
        )
    )
    case = SingleNodeShadowCompareCase(
        case_id="matcher_skill_executor",
        node_name="candidate_match",
        node_type="matcher",
        input_data={
            "job_requirement": {"job_id": "job_1"},
            "candidate_profile": {"candidate_id": "candidate_1"},
        },
        production_callable=lambda input_data: {
            "final_reports": [{"candidate_id": "candidate_1", "total_score": 81}]
        },
        shadow_skill_name="candidate_match",
        skill_executor=SkillExecutor(registry),
    )

    result = SingleNodeShadowCompareHarness().run_case(case)

    assert result.success is True
    assert result.decision.status == "match"


def test_production_callable_exception_creates_sanitized_skipped_decision():
    def fail(input_data):
        raise RuntimeError(f"production failed with {SENSITIVE_TEXT}")

    case = refiner_case()
    case.production_callable = fail

    result = SingleNodeShadowCompareHarness().run_case(case)

    assert result.success is False
    assert result.decision.status == "skipped"
    assert "production callable failed" in result.decision.reason
    assert result.metadata["production_error_type"] == "RuntimeError"
    assert SENSITIVE_TEXT not in str(result.to_dict())


def test_shadow_callable_exception_creates_sanitized_skipped_decision():
    def fail(input_data):
        raise RuntimeError(f"shadow failed with {SENSITIVE_TEXT}")

    case = refiner_case()
    case.shadow_callable = fail

    result = SingleNodeShadowCompareHarness().run_case(case)

    assert result.success is False
    assert result.decision.status == "skipped"
    assert "shadow callable failed" in result.decision.reason
    assert result.metadata["shadow_error_type"] == "RuntimeError"
    assert SENSITIVE_TEXT not in str(result.to_dict())


def test_run_cases_returns_batch_of_safe_results():
    results = SingleNodeShadowCompareHarness().run_cases(
        [refiner_case(), matcher_case()]
    )

    assert [result.decision.status for result in results] == ["match", "match"]
    assert all(result.metadata["summary_only"] for result in results)


def test_result_summary_does_not_contain_complete_sensitive_payload():
    case = refiner_case()
    case.input_data["context"] = SENSITIVE_TEXT
    case.metadata["secret"] = SENSITIVE_TEXT
    case.production_callable = lambda input_data: {
        "extracted_jd": {"search_query": f"python {SENSITIVE_TEXT}"}
    }
    case.shadow_callable = lambda input_data: {
        "refined_query": f"python {SENSITIVE_TEXT}"
    }

    result = SingleNodeShadowCompareHarness().run_case(case)
    serialized = str(result.to_dict())

    assert result.decision.status == "match"
    assert SENSITIVE_TEXT not in serialized
    assert "input_data" not in serialized
    assert result.production_output_summary["refined_query_present"] is True


def test_phase6d_does_not_run_or_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    harness_source = Path("src/integration/node_shadow.py").read_text(encoding="utf-8")

    result = SingleNodeShadowCompareHarness().run_case(refiner_case())

    assert result.metadata["real_production_graph_invoked"] is False
    assert "SingleNodeShadowCompareHarness" not in graph_source
    assert "src.integration.node_shadow" not in graph_source
    assert "src.core.graph" not in harness_source
    assert "create_recruit_graph" not in harness_source


def test_phase6d_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            ("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")
        ):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase6D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = SingleNodeShadowCompareHarness().run_case(matcher_case())

    assert result.decision.status == "match"
    assert blocked == []
