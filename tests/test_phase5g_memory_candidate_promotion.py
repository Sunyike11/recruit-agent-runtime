import builtins
from pathlib import Path

from src.memory import MemorySQLiteStore
from src.reflection import (
    MemoryCandidatePromoter,
    MemoryCandidateReviewDecision,
    MemoryPromotionResult,
    ReflectionMemoryProjectionPolicy,
    ReflectionRecord,
    ReflectionSourceType,
    ReflectionStatus,
)


SECRET_TEXT = "RAW-RESUME-AND-SECRET-MUST-NOT-ENTER-PROMOTED-MEMORY-METADATA"


def make_candidate(summary="Safe reflection summary.", status=ReflectionStatus.WARNING.value):
    reflection = ReflectionRecord(
        source_type=ReflectionSourceType.EVAL_RECORD.value,
        source_id="eval_1",
        target_type="runtime_timeline",
        target_id="task_1",
        status=status,
        summary=summary,
    ).validate()
    return ReflectionMemoryProjectionPolicy().project(reflection)


def approve(candidate, reviewer="reviewer_1", reason="Reviewed summary only."):
    return MemoryCandidateReviewDecision(
        candidate_id=candidate.candidate_id,
        approved=True,
        reviewer=reviewer,
        reason=reason,
    ).validate()


def test_memory_candidate_review_decision_can_create_and_round_trip():
    candidate = make_candidate()
    decision = approve(candidate)

    restored = MemoryCandidateReviewDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.approved is True


def test_memory_promotion_result_can_create():
    candidate = make_candidate()
    preview = candidate.to_memory_record()
    result = MemoryPromotionResult(
        candidate_id=candidate.candidate_id,
        promoted=False,
        dry_run=True,
        memory_preview=preview,
    )

    serialized = result.to_dict()

    assert serialized["promoted"] is False
    assert serialized["dry_run"] is True
    assert serialized["memory_preview"]["content"] == candidate.content


def test_default_dry_run_does_not_write_store_and_returns_preview(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()

    result = MemoryCandidatePromoter().promote(candidate, None, memory_store=store)

    assert result.promoted is False
    assert result.dry_run is True
    assert result.memory_id is None
    assert result.memory_preview.content == candidate.content
    assert store.list_memories() == []


def test_approved_decision_non_dry_run_writes_to_existing_memory_store(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()

    result = MemoryCandidatePromoter().promote(
        candidate,
        approve(candidate),
        memory_store=store,
        dry_run=False,
    )

    assert result.promoted is True
    assert result.dry_run is False
    assert result.memory_id is not None
    assert store.get_memory(result.memory_id).content == candidate.content


def test_rejected_decision_does_not_write_store(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()
    rejected = MemoryCandidateReviewDecision(
        candidate_id=candidate.candidate_id,
        approved=False,
        reviewer="reviewer_2",
        reason="Not suitable for durable memory.",
    )

    result = MemoryCandidatePromoter().promote(
        candidate,
        rejected,
        memory_store=store,
        dry_run=False,
    )

    assert result.promoted is False
    assert "rejected" in result.error
    assert store.list_memories() == []


def test_non_dry_run_without_approved_decision_does_not_write(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()

    result = MemoryCandidatePromoter().promote(
        candidate,
        None,
        memory_store=store,
        dry_run=False,
    )

    assert result.promoted is False
    assert "approved review decision is required" in result.error
    assert store.list_memories() == []


def test_non_dry_run_without_store_fails_clearly():
    candidate = make_candidate()

    result = MemoryCandidatePromoter().promote(
        candidate,
        approve(candidate),
        dry_run=False,
    )

    assert result.promoted is False
    assert result.error == "memory store is required when dry_run is False"


def test_promoted_memory_metadata_contains_provenance_and_approval(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()
    decision = approve(candidate, reviewer="alice", reason="Approved after review.")

    result = MemoryCandidatePromoter().promote(
        candidate,
        decision,
        memory_store=store,
        dry_run=False,
    )
    persisted = store.get_memory(result.memory_id)

    assert persisted.metadata["source_reflection_id"] == candidate.source_reflection_id
    assert persisted.metadata["source_candidate_id"] == candidate.candidate_id
    assert persisted.metadata["approved_by"] == "alice"
    assert persisted.metadata["reviewer"] == "alice"
    assert persisted.metadata["approval_reason"] == "Approved after review."
    assert persisted.metadata["promoted_from_reflection"] is True
    assert persisted.metadata["dry_run"] is False


def test_promote_many_can_partially_succeed(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    accepted = make_candidate(summary="Accepted summary.")
    rejected = make_candidate(summary="Rejected summary.")
    decisions = {
        accepted.candidate_id: approve(accepted),
        rejected.candidate_id: MemoryCandidateReviewDecision(
            candidate_id=rejected.candidate_id,
            approved=False,
            reviewer="reviewer_1",
            reason="Reject.",
        ),
    }

    results = MemoryCandidatePromoter().promote_many(
        [accepted, rejected],
        decisions,
        memory_store=store,
        dry_run=False,
    )

    assert [result.promoted for result in results] == [True, False]
    assert len(store.list_memories()) == 1


def test_mismatched_decision_never_writes_store(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate()
    other = make_candidate(summary="Other summary.")

    result = MemoryCandidatePromoter().promote(
        candidate,
        approve(other),
        memory_store=store,
        dry_run=False,
    )

    assert result.promoted is False
    assert "does not match" in result.error
    assert store.list_memories() == []


def test_dry_run_does_not_claim_mismatched_approval():
    candidate = make_candidate()
    other = make_candidate(summary="Other summary.")

    result = MemoryCandidatePromoter().promote(candidate, approve(other))

    assert result.promoted is False
    assert result.metadata["approval_verified"] is False
    assert result.memory_preview.metadata["approved_by"] == ""


def test_sensitive_metadata_is_not_copied_into_preview_or_persisted_record(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    candidate = make_candidate(summary="Safe reviewed summary.")
    candidate.metadata["raw_payload"] = SECRET_TEXT
    decision = MemoryCandidateReviewDecision(
        candidate_id=candidate.candidate_id,
        approved=True,
        reviewer="reviewer_1",
        reason="Safe approval.",
        metadata={"secret": SECRET_TEXT},
    )

    result = MemoryCandidatePromoter().promote(
        candidate,
        decision,
        memory_store=store,
        dry_run=False,
    )
    persisted = store.get_memory(result.memory_id)

    assert SECRET_TEXT not in str(result.memory_preview.to_dict())
    assert SECRET_TEXT not in str(persisted.to_dict())


def test_promoter_uses_existing_store_interface_without_modifying_graph():
    promotion_source = Path("src/reflection/memory_promotion.py").read_text(encoding="utf-8")
    memory_store_source = Path("src/memory/store.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "save_memory" in promotion_source
    assert "MemoryCandidatePromoter" not in memory_store_source
    assert "MemoryCandidatePromoter" not in graph_source
    assert "memory_promotion" not in graph_source


def test_phase5g_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5G test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = MemoryCandidatePromoter().promote(make_candidate(), None)

    assert result.dry_run is True
    assert result.promoted is False
    assert blocked == []
