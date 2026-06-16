import builtins
from pathlib import Path

from src.memory import (
    InMemoryMemoryGovernanceStore,
    MemoryContextEligibilityPolicy,
    MemoryGovernancePolicy,
    MemoryGovernanceRecord,
    MemoryRecord,
    MemoryType,
    build_eligible_memory_context_preview,
)


def promoted_memory(content="Governed approved summary.", **kwargs):
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


def test_memory_governance_record_can_create_and_round_trip():
    record = MemoryGovernanceRecord(
        memory_id="memory_1",
        status="active",
        reason="Reviewed.",
        actor="reviewer_1",
    ).validate()

    restored = MemoryGovernanceRecord.from_dict(record.to_dict())

    assert restored == record


def test_in_memory_governance_store_can_save_get_and_list():
    store = InMemoryMemoryGovernanceStore()
    first = store.save_record(
        MemoryGovernanceRecord(memory_id="memory_1", status="active", reason="Active.", actor="alice")
    )
    second = store.save_record(
        MemoryGovernanceRecord(memory_id="memory_1", status="revoked", reason="Withdrawn.", actor="bob")
    )

    assert store.get_latest_record("memory_1") == second
    assert store.list_records(memory_id="memory_1") == [first, second]
    assert store.list_records(status="revoked") == [second]


def test_revoke_memory_creates_revoked_record():
    record = InMemoryMemoryGovernanceStore().revoke_memory(
        "memory_1", "No longer approved.", "reviewer_1"
    )

    assert record.status == "revoked"
    assert record.memory_id == "memory_1"


def test_expire_memory_creates_expired_record():
    record = InMemoryMemoryGovernanceStore().expire_memory(
        "memory_1", "Validity window ended.", "reviewer_1"
    )

    assert record.status == "expired"


def test_mark_superseded_creates_superseded_record():
    record = InMemoryMemoryGovernanceStore().mark_superseded(
        "memory_old", "memory_new", "Replaced by reviewed update.", "reviewer_1"
    )

    assert record.status == "superseded"
    assert record.superseded_by_memory_id == "memory_new"


def test_governance_policy_defaults_to_active_allowed_without_record():
    decision = MemoryGovernancePolicy().evaluate(promoted_memory())

    assert decision.allowed is True
    assert decision.status == "active"


def test_revoked_memory_is_denied():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")

    decision = MemoryGovernancePolicy().evaluate(memory, store)

    assert decision.allowed is False
    assert decision.status == "revoked"


def test_expired_memory_is_denied():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.expire_memory(memory.memory_id, "Expired.", "reviewer_1")

    decision = MemoryGovernancePolicy().evaluate(memory, store)

    assert decision.allowed is False
    assert decision.status == "expired"


def test_superseded_memory_is_denied():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.mark_superseded(memory.memory_id, "memory_new", "Updated.", "reviewer_1")

    decision = MemoryGovernancePolicy().evaluate(memory, store)

    assert decision.allowed is False
    assert decision.status == "superseded"


def test_sensitive_and_metadata_revoked_memories_are_denied():
    policy = MemoryGovernancePolicy()

    sensitive = policy.evaluate(promoted_memory(metadata={"sensitive": True}))
    revoked = policy.evaluate(promoted_memory(metadata={"revoked": True}))

    assert sensitive.allowed is False
    assert revoked.allowed is False
    assert revoked.status == "revoked"


def test_eligibility_policy_denies_revoked_memory_when_governance_is_provided():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")
    policy = MemoryContextEligibilityPolicy(governance_store=store)

    decision = policy.evaluate(memory)

    assert decision.eligible is False
    assert decision.status == "denied"
    assert "revoked" in decision.reason


def test_context_preview_omits_revoked_and_expired_memories():
    active = promoted_memory(content="Active memory.")
    revoked = promoted_memory(content="Revoked content.")
    expired = promoted_memory(content="Expired content.")
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(revoked.memory_id, "Withdrawn.", "reviewer_1")
    store.expire_memory(expired.memory_id, "Old.", "reviewer_1")
    policy = MemoryContextEligibilityPolicy(MemoryGovernancePolicy(), store)

    preview = build_eligible_memory_context_preview([active, revoked, expired], policy=policy)

    assert "Active memory." in preview
    assert "Revoked content." not in preview
    assert "Expired content." not in preview


def test_phase5h_behavior_remains_compatible_without_governance():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")

    decision = MemoryContextEligibilityPolicy().evaluate(memory)

    assert decision.eligible is True
    assert decision.status == "eligible"


def test_governance_does_not_modify_memory_store_context_builder_or_graph():
    governance_source = Path("src/memory/governance.py").read_text(encoding="utf-8")
    store_source = Path("src/memory/store.py").read_text(encoding="utf-8")
    builder_source = Path("src/memory/context.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "save_memory" not in governance_source
    assert "MemoryContextBuilder" not in governance_source
    assert "MemoryGovernance" not in store_source
    assert "MemoryGovernance" not in builder_source
    assert "MemoryGovernance" not in graph_source


def test_phase5i_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5I test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    decision = MemoryGovernancePolicy().evaluate(promoted_memory())

    assert decision.allowed is True
    assert blocked == []
