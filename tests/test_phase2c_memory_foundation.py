import builtins
import sys

import pytest

from src.domain import CandidateProfile, HumanFeedback, MatchReport, SearchAttempt
from src.memory import (
    MemoryRecord,
    MemorySQLiteStore,
    MemorySourceType,
    MemoryType,
    memory_from_candidate_profile,
    memory_from_human_feedback,
    memory_from_match_report,
    memory_from_search_attempt,
)


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase2C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_memory_record_can_create_and_roundtrip():
    record = MemoryRecord(
        memory_type=MemoryType.SEMANTIC.value,
        source_type=MemorySourceType.CANDIDATE_PROFILE.value,
        source_id="candidate_1",
        content="Candidate has PyTorch skills.",
        importance=0.7,
        tags=["candidate", "skills"],
        metadata={"candidate_id": "candidate_1"},
    )

    restored = MemoryRecord.from_dict(record.to_dict())

    assert restored == record


def test_memory_store_can_save_and_read(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    record = MemoryRecord(content="Manual memory", tags=["manual"])

    store.save_memory(record)
    loaded = store.get_memory(record.memory_id)

    assert loaded == record


def test_memory_store_filters_by_type_and_source(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    semantic = MemoryRecord(
        memory_type=MemoryType.SEMANTIC.value,
        source_type=MemorySourceType.CANDIDATE_PROFILE.value,
        source_id="candidate_1",
        content="Candidate semantic memory",
    )
    episodic = MemoryRecord(
        memory_type=MemoryType.EPISODIC.value,
        source_type=MemorySourceType.MATCH_REPORT.value,
        source_id="match_1",
        content="Match episodic memory",
    )
    store.save_memory(semantic)
    store.save_memory(episodic)

    assert store.list_memories(memory_type=MemoryType.SEMANTIC.value) == [semantic]
    assert store.list_memories(source_type=MemorySourceType.MATCH_REPORT.value) == [episodic]
    assert store.list_memories(source_type=MemorySourceType.CANDIDATE_PROFILE.value, source_id="candidate_1") == [semantic]


def test_memory_store_search_by_tag(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    tagged = MemoryRecord(content="Tagged memory", tags=["feedback", "preference"])
    other = MemoryRecord(content="Other memory", tags=["candidate"])
    store.save_memory(tagged)
    store.save_memory(other)

    assert store.search_memories_by_tag("preference") == [tagged]


def test_delete_memory_removes_record(tmp_path):
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    record = MemoryRecord(content="Delete me")
    store.save_memory(record)

    assert store.delete_memory(record.memory_id) is True
    with pytest.raises(KeyError):
        store.get_memory(record.memory_id)


def test_memory_store_survives_reinstantiation(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = MemorySQLiteStore(db_path)
    record = MemoryRecord(content="Persistent memory", tags=["persist"])
    store.save_memory(record)

    reopened = MemorySQLiteStore(db_path)

    assert reopened.get_memory(record.memory_id) == record


def test_memory_from_candidate_profile_derives_semantic_memory():
    candidate = CandidateProfile(candidate_id="candidate_1", name="候选人A", skills=["Python", "LangGraph"])

    memory = memory_from_candidate_profile(candidate)

    assert memory.memory_type == MemoryType.SEMANTIC.value
    assert memory.source_type == MemorySourceType.CANDIDATE_PROFILE.value
    assert memory.source_id == "candidate_1"
    assert "Python, LangGraph" in memory.content
    assert "skills" in memory.tags


def test_memory_from_match_report_derives_episodic_memory():
    report = MatchReport(
        match_id="match_1",
        job_id="job_1",
        candidate_id="candidate_1",
        total_score=91,
        recommendation="OUTSTANDING",
    )

    memory = memory_from_match_report(report)

    assert memory.memory_type == MemoryType.EPISODIC.value
    assert memory.source_type == MemorySourceType.MATCH_REPORT.value
    assert memory.source_id == "match_1"
    assert "score 91" in memory.content
    assert memory.metadata["job_id"] == "job_1"


def test_memory_from_human_feedback_derives_preference_memory():
    feedback = HumanFeedback(
        feedback_id="feedback_1",
        task_id="task_1",
        target_type="candidate",
        target_id="candidate_1",
        feedback_type="approve",
        payload={"reason": "项目相关"},
    )

    memory = memory_from_human_feedback(feedback)

    assert memory.memory_type == MemoryType.PREFERENCE.value
    assert memory.source_type == MemorySourceType.HUMAN_FEEDBACK.value
    assert memory.source_id == "feedback_1"
    assert "Human feedback on task task_1" in memory.content
    assert "approve" in memory.tags


def test_memory_from_search_attempt_derives_episodic_memory():
    attempt = SearchAttempt(
        search_id="search_1",
        job_id="job_1",
        query="PyTorch 3D",
        retrieved_candidate_ids=["candidate_1", "candidate_2"],
    )

    memory = memory_from_search_attempt(attempt)

    assert memory.memory_type == MemoryType.EPISODIC.value
    assert memory.source_type == MemorySourceType.SEARCH_ATTEMPT.value
    assert memory.source_id == "search_1"
    assert "PyTorch 3D" in memory.content
    assert "candidate_1, candidate_2" in memory.content


def test_phase2c_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store = MemorySQLiteStore(tmp_path / "memory.sqlite3")
    record = MemoryRecord(content="No retrieval dependency")

    store.save_memory(record)

    assert store.get_memory(record.memory_id) == record
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
