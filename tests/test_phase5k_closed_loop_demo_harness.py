import builtins
from pathlib import Path

from src.evaluation import EvalCase
from src.memory import MemorySQLiteStore
from src.reflection import (
    ClosedLoopDemoHarness,
    MemoryCandidateReviewDecision,
)


SENSITIVE_TEXT = "RAW-JD-RESUME-API-SECRET-MUST-NOT-BE-EXPORTED-BY-CLOSED-LOOP"


def approved_decision(candidate):
    return MemoryCandidateReviewDecision(
        candidate_id=candidate.candidate_id,
        approved=True,
        reviewer="phase5k_reviewer",
        reason="Reviewed deterministic summary.",
    )


def test_closed_loop_harness_runs_to_completion():
    result = ClosedLoopDemoHarness().run("Need Python and LangGraph")

    assert result.success is True
    assert result.status == "completed"
    assert result.first_workflow_summary["success"] is True
    assert result.evaluation_summary["passed_cases"] == 1
    assert result.correlation_summary["evaluation_passed"] is True
    assert result.reflection_summary["status"] == "success"


def test_default_dry_run_does_not_write_memory_store():
    harness = ClosedLoopDemoHarness()

    result = harness.run("Need Python and LangGraph")

    assert result.promotion_summary["dry_run"] is True
    assert result.promotion_summary["promoted"] is False
    assert harness.memory_store.list_memories() == []
    assert result.eligibility_summary["eligible"] is False


def test_approved_non_dry_run_promotes_in_isolated_store_and_builds_context(tmp_path):
    store = MemorySQLiteStore(tmp_path / "phase5k_memory.sqlite3")
    harness = ClosedLoopDemoHarness(memory_store=store)

    result = harness.run(
        "Need Python and LangGraph",
        approval_decision=approved_decision,
        dry_run=False,
    )

    assert result.promotion_summary["promoted"] is True
    assert len(store.list_memories()) == 1
    assert result.eligibility_summary["eligible"] is True
    assert "Promoted Memory Context Preview:" in result.memory_context_preview
    assert "Evaluation/audit correlation success" in result.memory_context_preview
    assert result.second_workflow_summary["memory_context_seen"] is True


def test_memory_context_preview_is_built_before_second_shadow_workflow():
    result = ClosedLoopDemoHarness().run(
        "Need Python and LangGraph",
        approval_decision=approved_decision,
        dry_run=False,
    )

    assert result.memory_context_preview
    assert result.second_workflow_summary["success"] is True
    assert result.second_workflow_summary["memory_context_seen"] is True


def test_revoked_promoted_memory_does_not_enter_second_workflow_context():
    def revoke_after_promotion(memory, governance_store):
        governance_store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer")

    result = ClosedLoopDemoHarness(post_promotion_hook=revoke_after_promotion).run(
        "Need Python and LangGraph",
        approval_decision=approved_decision,
        dry_run=False,
    )

    assert result.promotion_summary["promoted"] is True
    assert result.eligibility_summary["status"] == "denied"
    assert "revoked" in result.eligibility_summary["reason"]
    assert "No eligible promoted memory." in result.memory_context_preview
    assert result.second_workflow_summary["memory_context_seen"] is False


def test_failed_evaluation_generates_failure_reflection():
    failed_case = EvalCase(
        case_id="expected_tool_failure_not_seen",
        target_type="runtime_timeline",
        checks=[{"type": "event_type_present", "event_type": "tool_failed"}],
    )

    result = ClosedLoopDemoHarness(evaluation_case=failed_case).run("Need Python")

    assert result.success is True
    assert result.evaluation_summary["failed_cases"] == 1
    assert result.correlation_summary["evaluation_passed"] is False
    assert result.reflection_summary["status"] == "failure"


def test_memory_candidate_summary_retains_reflection_reference_without_content():
    result = ClosedLoopDemoHarness().run("Need Python")

    candidate = result.memory_candidate_summary
    assert candidate["source_reflection_id"] == result.reflection_summary["reflection_id"]
    assert "content" not in candidate


def test_closed_loop_result_omits_sensitive_raw_input_and_payloads():
    result = ClosedLoopDemoHarness().run(
        f"Need Python {SENSITIVE_TEXT}",
        metadata={"secret": SENSITIVE_TEXT},
    )
    serialized = str(result.to_dict())

    assert SENSITIVE_TEXT not in serialized
    assert "raw_jd" not in serialized
    assert "evidence" not in serialized
    assert "memory_preview" not in serialized


def test_closed_loop_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    closed_loop_source = Path("src/reflection/closed_loop.py").read_text(encoding="utf-8")

    assert "ClosedLoopDemoHarness" not in graph_source
    assert "src.reflection.closed_loop" not in graph_source
    assert "RecruitmentSkillWorkflow" in closed_loop_source


def test_mismatched_approval_does_not_create_eligible_context():
    decision = MemoryCandidateReviewDecision(
        candidate_id="not_the_generated_candidate",
        approved=True,
        reviewer="reviewer",
    )

    result = ClosedLoopDemoHarness().run(
        "Need Python",
        approval_decision=decision,
        dry_run=False,
    )

    assert result.promotion_summary["promoted"] is False
    assert "does not match" in result.promotion_summary["error"]
    assert result.second_workflow_summary["memory_context_seen"] is False


def test_phase5k_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5K test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = ClosedLoopDemoHarness().run("Need Python and LangGraph")

    assert result.success is True
    assert blocked == []
