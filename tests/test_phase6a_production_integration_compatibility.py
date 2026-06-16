import builtins
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.integration import (
    ProductionIntegrationCompatibilityReport,
    ProductionStateAdapter,
    ProductionStateShape,
    ShadowWorkflowShape,
    build_production_integration_plan,
    compare_production_and_shadow_shapes,
    validate_safe_migration_boundary,
)


def production_state():
    return {
        "messages": [HumanMessage(content="Need Python LangGraph engineer")],
        "extracted_jd": {"search_query": "Python LangGraph"},
        "candidate_pool": [],
        "final_reports": [],
    }


def shadow_result():
    return {
        "status": "completed",
        "success": True,
        "job_requirement": {"title": "Agent Engineer"},
        "retrieved_candidates": [{"candidate_id": "candidate_1"}],
        "match_reports": [{"candidate_id": "candidate_1", "total_score": 92}],
        "refined_query": None,
    }


def test_production_state_shape_can_create():
    shape = ProductionStateShape()

    assert shape.required_keys == ["messages"]
    assert "candidate_pool" in shape.optional_keys
    assert shape.metadata["memory_context_supported"] is False


def test_shadow_workflow_shape_can_create():
    shape = ShadowWorkflowShape()

    assert shape.required_keys == ["status", "success"]
    assert "match_reports" in shape.optional_keys
    assert shape.outputs["memory_context"] == "optional shadow preview input only"


def test_compatibility_report_can_create_and_serialize():
    report = ProductionIntegrationCompatibilityReport(
        compatible=False,
        missing_keys=["messages"],
        rollback_notes=["Keep production path."],
    )

    assert report.to_dict()["missing_keys"] == ["messages"]
    assert report.rollback_notes


def test_adapter_extracts_shadow_input_from_fake_production_state():
    mapped = ProductionStateAdapter.production_state_to_shadow_input(
        production_state(),
        top_k=3,
    )

    assert mapped["raw_jd"] == "Need Python LangGraph engineer"
    assert mapped["top_k"] == 3
    assert mapped["metadata"]["query_available"] is True
    assert mapped["metadata"]["memory_context_required"] is False


def test_adapter_maps_shadow_output_to_production_preview_only():
    update = ProductionStateAdapter.shadow_result_to_production_update(shadow_result())

    assert update["extracted_jd_preview"] == {"title": "Agent Engineer"}
    assert update["candidate_pool_preview"] == [{"candidate_id": "candidate_1"}]
    assert update["final_reports_preview"][0]["total_score"] == 92
    assert update["preview_metadata"]["apply_to_graph"] is False
    assert "candidate_pool" not in update


def test_missing_required_production_input_generates_incompatible_report():
    report = compare_production_and_shadow_shapes({"candidate_pool": []}, shadow_result())

    assert report.compatible is False
    assert "messages" in report.missing_keys
    assert "messages[-1].content/raw_jd" in report.missing_keys


def test_shadow_result_missing_match_reports_generates_migration_risk():
    result = shadow_result()
    result.pop("match_reports")

    report = compare_production_and_shadow_shapes(production_state(), result)

    assert report.compatible is True
    assert any("no match_reports" in risk for risk in report.migration_risks)


def test_memory_context_is_preview_only_not_required_production_state():
    report = validate_safe_migration_boundary(
        production_state(),
        shadow_result(),
        memory_context_requested=True,
    )

    assert report.compatible is False
    assert report.metadata["memory_context_required"] is False
    assert any("preview-only" in field for field in report.incompatible_fields)


def test_build_plan_contains_gated_steps_and_rollback_notes():
    report = compare_production_and_shadow_shapes(production_state(), shadow_result())
    steps = build_production_integration_plan(report)

    assert any("schema compatibility gates" in step for step in steps)
    assert any("explicit opt-in" in step for step in steps)
    assert any("default execution path" in note for note in report.rollback_notes)


def test_adapter_validation_reports_shape_without_executing_graph():
    state_report = ProductionStateAdapter.validate_production_state(production_state())
    shadow_report = ProductionStateAdapter.validate_shadow_result(shadow_result())

    assert state_report.compatible is True
    assert shadow_report.compatible is True
    assert state_report.metadata["mode"] == "read_only_compatibility_analysis"


def test_phase6a_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase6A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = compare_production_and_shadow_shapes(production_state(), shadow_result())

    assert report.compatible is True
    assert blocked == []


def test_phase6a_does_not_modify_or_integrate_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    integration_source = Path("src/integration/compatibility.py").read_text(encoding="utf-8")

    assert "ProductionStateAdapter" not in graph_source
    assert "src.integration" not in graph_source
    assert "SkillRegistry" not in graph_source
    assert "MemoryContext" not in graph_source
    assert "src.core.graph" not in integration_source
    assert "MemoryContextBuilder" not in integration_source
