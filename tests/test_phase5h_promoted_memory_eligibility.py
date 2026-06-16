import builtins
from pathlib import Path

from src.memory import (
    HIGH_IMPORTANCE_THRESHOLD,
    MemoryContextEligibilityPolicy,
    MemoryEligibilityDecision,
    MemoryRecord,
    MemoryType,
    PromotedMemoryAuditor,
    build_eligible_memory_context_preview,
)


SECRET_TEXT = "PRIVATE-RESUME-AND-SECRET-MUST-NOT-APPEAR-IN-CONTEXT-PREVIEW"


def promoted_memory(content="Approved reflection summary.", **kwargs):
    metadata = {
        "promoted_from_reflection": True,
        "dry_run": False,
        "source_reflection_id": "reflection_1",
        "source_candidate_id": "candidate_1",
        "approved_by": "reviewer_1",
        "reviewer": "reviewer_1",
        "summary_only": True,
    }
    metadata.update(kwargs.pop("metadata", {}))
    return MemoryRecord(
        memory_type=kwargs.pop("memory_type", MemoryType.PROCEDURAL.value),
        content=content,
        importance=kwargs.pop("importance", 0.6),
        tags=kwargs.pop("tags", ["reflection_candidate", "procedure"]),
        metadata=metadata,
        **kwargs,
    )


def test_memory_eligibility_decision_can_create():
    decision = MemoryEligibilityDecision(
        memory_id="memory_1",
        eligible=True,
        status="eligible",
        reason="safe preview",
        tags=["promoted_memory"],
        metadata={"summary_only": True},
    ).validate()

    assert decision.to_dict()["status"] == "eligible"
    assert decision.eligible is True


def test_promoted_approved_memory_is_eligible():
    decision = MemoryContextEligibilityPolicy().evaluate(promoted_memory())

    assert decision.eligible is True
    assert decision.status == "eligible"


def test_dry_run_memory_is_denied():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(metadata={"dry_run": True})
    )

    assert decision.status == "denied"
    assert "dry-run" in decision.reason


def test_sensitive_memory_is_denied():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(content=SECRET_TEXT, metadata={"sensitive": True})
    )

    assert decision.status == "denied"
    assert decision.eligible is False


def test_missing_source_reflection_id_is_denied():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(metadata={"source_reflection_id": ""})
    )

    assert decision.status == "denied"
    assert "source_reflection_id" in decision.reason


def test_missing_candidate_id_is_denied():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(metadata={"source_candidate_id": ""})
    )

    assert decision.status == "denied"
    assert "source_candidate_id" in decision.reason


def test_missing_reviewer_requires_review():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(metadata={"approved_by": "", "reviewer": ""})
    )

    assert decision.status == "requires_review"
    assert decision.eligible is False


def test_high_importance_requires_review():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(importance=HIGH_IMPORTANCE_THRESHOLD + 0.01)
    )

    assert decision.status == "requires_review"
    assert "high importance" in decision.reason


def test_unsupported_memory_type_requires_review():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(memory_type=MemoryType.PREFERENCE.value)
    )

    assert decision.status == "requires_review"
    assert "type" in decision.reason


def test_target_context_mismatch_requires_review():
    decision = MemoryContextEligibilityPolicy().evaluate(
        promoted_memory(tags=["reflection_candidate"]),
        target_context={"tags": ["hiring_policy"]},
    )

    assert decision.status == "requires_review"
    assert "target context" in decision.reason


def test_filter_eligible_returns_only_eligible_records():
    allowed = promoted_memory(content="Eligible.")
    denied = promoted_memory(content=SECRET_TEXT, metadata={"sensitive": True})
    review = promoted_memory(importance=0.9)

    selected = MemoryContextEligibilityPolicy().filter_eligible([allowed, denied, review])

    assert selected == [allowed]


def test_audit_report_counts_eligibility_outcomes():
    report = PromotedMemoryAuditor().audit(
        [
            promoted_memory(content="Eligible."),
            promoted_memory(content=SECRET_TEXT, metadata={"sensitive": True}),
            promoted_memory(metadata={"source_candidate_id": ""}),
            promoted_memory(metadata={"approved_by": "", "reviewer": ""}),
        ]
    )

    assert report.total_memories == 4
    assert report.promoted_from_reflection_count == 4
    assert report.eligible_count == 1
    assert report.denied_count == 2
    assert report.requires_review_count == 1
    assert report.sensitive_count == 1
    assert report.missing_provenance_count == 1


def test_context_preview_only_contains_eligible_memory_and_no_sensitive_metadata():
    preview = build_eligible_memory_context_preview(
        [
            promoted_memory(content="Approved context preview."),
            promoted_memory(content=SECRET_TEXT, metadata={"sensitive": True}),
        ]
    )

    assert "Approved context preview." in preview
    assert SECRET_TEXT not in preview
    assert "source_reflection_id" not in preview
    assert "approved_by" not in preview


def test_eligibility_module_does_not_modify_store_builder_or_graph():
    source = Path("src/memory/eligibility.py").read_text(encoding="utf-8")
    store_source = Path("src/memory/store.py").read_text(encoding="utf-8")
    builder_source = Path("src/memory/context.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "save_memory" not in source
    assert "MemoryContextBuilder" not in source
    assert "MemoryContextEligibilityPolicy" not in store_source
    assert "MemoryContextEligibilityPolicy" not in builder_source
    assert "MemoryContextEligibilityPolicy" not in graph_source


def test_phase5h_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5H test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    decision = MemoryContextEligibilityPolicy().evaluate(promoted_memory())

    assert decision.status == "eligible"
    assert blocked == []
