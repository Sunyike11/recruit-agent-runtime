import builtins
import sys
from datetime import datetime, timezone

from src.memory import (
    MemoryContextBuilder,
    MemoryRecord,
    MemorySQLiteStore,
    MemorySourceType,
    MemoryType,
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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase2D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_store(tmp_path):
    return MemorySQLiteStore(tmp_path / "memory.sqlite3")


def save_records(store, records):
    for record in records:
        store.save_memory(record)
    return records


def test_can_build_empty_context(tmp_path):
    store = make_store(tmp_path)
    context = MemoryContextBuilder(store).build()

    assert context.items == []
    assert "No relevant memory" in context.format_for_prompt()


def test_can_read_memory_records_from_store(tmp_path):
    store = make_store(tmp_path)
    record = MemoryRecord(content="Candidate Alice has Python skills.")
    store.save_memory(record)

    context = MemoryContextBuilder(store).build()

    assert len(context.items) == 1
    assert context.items[0].content == "Candidate Alice has Python skills."


def test_can_filter_by_memory_type(tmp_path):
    store = make_store(tmp_path)
    semantic = MemoryRecord(memory_type=MemoryType.SEMANTIC.value, content="Semantic candidate memory.")
    preference = MemoryRecord(memory_type=MemoryType.PREFERENCE.value, content="Preference feedback memory.")
    save_records(store, [semantic, preference])

    context = MemoryContextBuilder(store).build(memory_types=[MemoryType.PREFERENCE.value])

    assert [item.memory_id for item in context.items] == [preference.memory_id]


def test_can_filter_by_source_type(tmp_path):
    store = make_store(tmp_path)
    candidate = MemoryRecord(
        source_type=MemorySourceType.CANDIDATE_PROFILE.value,
        content="Candidate profile memory.",
    )
    feedback = MemoryRecord(
        source_type=MemorySourceType.HUMAN_FEEDBACK.value,
        content="Human feedback memory.",
    )
    save_records(store, [candidate, feedback])

    context = MemoryContextBuilder(store).build(source_types=[MemorySourceType.HUMAN_FEEDBACK.value])

    assert [item.memory_id for item in context.items] == [feedback.memory_id]


def test_can_filter_by_tag(tmp_path):
    store = make_store(tmp_path)
    tagged = MemoryRecord(content="Relevant feedback memory.", tags=["feedback", "preference"])
    other = MemoryRecord(content="Other candidate memory.", tags=["candidate"])
    save_records(store, [tagged, other])

    context = MemoryContextBuilder(store).build(tags=["preference"])

    assert [item.memory_id for item in context.items] == [tagged.memory_id]


def test_can_filter_by_min_importance(tmp_path):
    store = make_store(tmp_path)
    low = MemoryRecord(content="Low importance memory.", importance=0.2)
    high = MemoryRecord(content="High importance memory.", importance=0.9)
    save_records(store, [low, high])

    context = MemoryContextBuilder(store).build(min_importance=0.5)

    assert [item.memory_id for item in context.items] == [high.memory_id]


def test_ranking_prefers_higher_importance(tmp_path):
    store = make_store(tmp_path)
    newer_low = MemoryRecord(
        content="Newer but lower importance.",
        importance=0.4,
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    older_high = MemoryRecord(
        content="Older but higher importance.",
        importance=0.9,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    save_records(store, [newer_low, older_high])

    context = MemoryContextBuilder(store).build()

    assert context.items[0].memory_id == older_high.memory_id


def test_max_items_limits_context(tmp_path):
    store = make_store(tmp_path)
    records = [
        MemoryRecord(content="First", importance=0.9),
        MemoryRecord(content="Second", importance=0.8),
        MemoryRecord(content="Third", importance=0.7),
    ]
    save_records(store, records)

    context = MemoryContextBuilder(store).build(max_items=2)

    assert len(context.items) == 2


def test_max_chars_limits_formatted_context(tmp_path):
    store = make_store(tmp_path)
    store.save_memory(MemoryRecord(content="A very long memory content " * 20))

    formatted = MemoryContextBuilder(store).build(max_chars=80).format_for_prompt()

    assert len(formatted) == 80


def test_formatted_context_contains_memory_type_and_content(tmp_path):
    store = make_store(tmp_path)
    record = MemoryRecord(
        memory_type=MemoryType.PREFERENCE.value,
        source_type=MemorySourceType.HUMAN_FEEDBACK.value,
        content="Human feedback prefers production LangGraph experience.",
    )
    store.save_memory(record)

    formatted = MemoryContextBuilder(store).build().format_for_prompt()

    assert "Recruit Agent Memory Context:" in formatted
    assert "[preference]" in formatted
    assert "Human feedback prefers production LangGraph experience." in formatted


def test_fake_consumer_can_receive_formatted_context(tmp_path):
    class FakePromptConsumer:
        def consume(self, memory_context: str):
            return f"received:{memory_context}"

    store = make_store(tmp_path)
    store.save_memory(MemoryRecord(content="Candidate Alice has PyTorch experience."))

    formatted = MemoryContextBuilder(store).build_prompt()
    response = FakePromptConsumer().consume(formatted)

    assert "received:Recruit Agent Memory Context:" in response
    assert "Candidate Alice has PyTorch experience." in response


def test_phase2d_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store = make_store(tmp_path)
    store.save_memory(MemoryRecord(content="No retrieval dependency", tags=["safe"]))

    context = MemoryContextBuilder(store).build(tags=["safe"])

    assert context.items[0].content == "No retrieval dependency"
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
