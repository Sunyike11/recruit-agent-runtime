import builtins
from pathlib import Path

from src.memory import MemoryRecord, MemoryType
from src.reflection import (
    MAX_REFLECTION_SUMMARY_CHARS,
    MemoryCandidate,
    MemoryProjectionDecision,
    ReflectionMemoryProjectionPolicy,
    ReflectionRecord,
    ReflectionSourceType,
    ReflectionStatus,
)


SECRET_TEXT = "COMPLETE-RESUME-TEXT-API-SECRET-MUST-NOT-ENTER-MEMORY-CANDIDATE"


def make_reflection(status="success", summary="Summary-only reflection.", **kwargs):
    return ReflectionRecord(
        source_type=kwargs.pop("source_type", ReflectionSourceType.EVAL_RECORD.value),
        source_id="eval_1",
        target_type="runtime_timeline",
        target_id="task_1",
        status=status,
        summary=summary,
        **kwargs,
    ).validate()


def test_memory_candidate_can_create_and_round_trip():
    candidate = MemoryCandidate(
        source_reflection_id="reflection_1",
        memory_type=MemoryType.EPISODIC.value,
        content="A safe summary.",
        importance=0.3,
        tags=["reflection_candidate"],
        requires_approval=True,
        approval_reason="review required",
    ).validate()

    restored = MemoryCandidate.from_dict(candidate.to_dict())

    assert restored == candidate
    assert candidate.candidate_id.startswith("memory_candidate_")


def test_memory_projection_decision_can_create():
    decision = MemoryProjectionDecision(
        allowed=True,
        status="requires_approval",
        reason="review required",
        source_reflection_id="reflection_1",
        proposed_memory_type=MemoryType.PROCEDURAL.value,
        importance=0.6,
        tags=["reflection_candidate"],
    ).validate()

    assert decision.allowed is True
    assert decision.to_dict()["status"] == "requires_approval"


def test_success_reflection_projects_low_importance_episodic_candidate():
    candidate = ReflectionMemoryProjectionPolicy().project(make_reflection())

    assert candidate is not None
    assert candidate.memory_type == MemoryType.EPISODIC.value
    assert candidate.importance == 0.3
    assert candidate.requires_approval is True


def test_warning_reflection_requires_approval():
    decision = ReflectionMemoryProjectionPolicy().evaluate(
        make_reflection(status=ReflectionStatus.WARNING.value)
    )

    assert decision.allowed is True
    assert decision.status == "requires_approval"
    assert decision.proposed_memory_type == MemoryType.PROCEDURAL.value
    assert decision.importance == 0.6


def test_failure_reflection_has_higher_importance_and_requires_approval():
    candidate = ReflectionMemoryProjectionPolicy().project(
        make_reflection(status=ReflectionStatus.FAILURE.value)
    )

    assert candidate.memory_type == MemoryType.PROCEDURAL.value
    assert candidate.importance == 0.8
    assert candidate.requires_approval is True


def test_empty_summary_is_denied():
    policy = ReflectionMemoryProjectionPolicy()
    record = make_reflection(summary="   ")

    decision = policy.evaluate(record)

    assert decision.allowed is False
    assert decision.status == "denied"
    assert policy.project(record) is None


def test_overlong_summary_is_denied_instead_of_truncated():
    policy = ReflectionMemoryProjectionPolicy()
    record = make_reflection(summary="x" * (MAX_REFLECTION_SUMMARY_CHARS + 1))

    decision = policy.evaluate(record)

    assert decision.status == "denied"
    assert "exceeds" in decision.reason
    assert policy.project(record) is None


def test_sensitive_metadata_is_denied():
    policy = ReflectionMemoryProjectionPolicy()
    record = make_reflection(summary="Safe-looking text.", metadata={"sensitive": True})

    decision = policy.evaluate(record)

    assert decision.status == "denied"
    assert policy.project(record) is None


def test_manual_source_requires_approval():
    decision = ReflectionMemoryProjectionPolicy().evaluate(
        make_reflection(source_type=ReflectionSourceType.MANUAL.value)
    )

    assert decision.status == "requires_approval"
    assert "manual reflection" in decision.reason


def test_project_many_skips_denied_records():
    policy = ReflectionMemoryProjectionPolicy()
    records = [
        make_reflection(summary="Allowed summary."),
        make_reflection(summary=""),
        make_reflection(status=ReflectionStatus.WARNING.value, summary="Warning summary."),
    ]

    candidates = policy.project_many(records)

    assert len(candidates) == 2
    assert [candidate.importance for candidate in candidates] == [0.3, 0.6]


def test_to_memory_record_is_an_unsaved_preview_with_reflection_reference():
    candidate = ReflectionMemoryProjectionPolicy().project(make_reflection())

    memory = candidate.to_memory_record()

    assert isinstance(memory, MemoryRecord)
    assert memory.source_id == candidate.source_reflection_id
    assert memory.metadata["source_reflection_id"] == candidate.source_reflection_id
    assert memory.metadata["requires_approval"] is True
    assert memory.metadata["projection_source"] == "reflection"


def test_candidate_and_preview_do_not_copy_sensitive_reflection_fields():
    record = make_reflection(
        summary="Review-safe summary.",
        findings=[SECRET_TEXT],
        recommended_actions=[SECRET_TEXT],
        evidence_refs=[SECRET_TEXT],
        tags=[SECRET_TEXT],
        metadata={"raw_payload": SECRET_TEXT},
    )

    candidate = ReflectionMemoryProjectionPolicy().project(record)
    memory = candidate.to_memory_record()

    assert SECRET_TEXT not in str(candidate.to_dict())
    assert SECRET_TEXT not in str(memory.to_dict())


def test_projection_module_does_not_write_memory_store_or_modify_graph():
    projection_source = Path("src/reflection/memory_projection.py").read_text(encoding="utf-8")
    memory_store_source = Path("src/memory/store.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "MemorySQLiteStore" not in projection_source
    assert "save_memory" not in projection_source
    assert "MemoryCandidate" not in memory_store_source
    assert "ReflectionMemoryProjectionPolicy" not in graph_source
    assert "MemoryCandidate" not in graph_source


def test_phase5f_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5F test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    candidate = ReflectionMemoryProjectionPolicy().project(make_reflection())

    assert candidate.content == "Summary-only reflection."
    assert blocked == []
