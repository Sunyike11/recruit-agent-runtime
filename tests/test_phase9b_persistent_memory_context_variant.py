import builtins
import json
from pathlib import Path

from src.memory import InMemoryMemoryGovernanceStore, MemoryRecord, MemorySQLiteStore, MemoryType
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.memory_context import (
    RuntimeMemorySourceConfig,
    build_readonly_runtime_memory_context,
)
from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner
from scripts.run_recruit_runtime import run_cli


SENSITIVE_TEXT = "PRIVATE-PERSISTENT-MEMORY-MUST-NOT-LEAK"


def promoted_memory(content="prefer Python LangGraph runtime experience", **kwargs):
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
        tags=kwargs.pop("tags", ["runtime_demo", "persistent"]),
        metadata=metadata,
        **kwargs,
    )


def source_config(db_path=None, **kwargs):
    values = {
        "source": "sqlite",
        "memory_db_path": str(db_path) if db_path is not None else None,
        "tags": ["runtime_demo"],
        "max_items": 5,
        "max_chars": 800,
        "require_governance": True,
    }
    values.update(kwargs)
    return RuntimeMemorySourceConfig(**values)


def seed_store(db_path, records):
    store = MemorySQLiteStore(db_path)
    for record in records:
        store.save_memory(record)
    return store


def fake_planner(_input_data, _context=None):
    return {"job_requirement": {"job_id": "job_phase9b", "required_skills": ["Python", "LangGraph"]}}


def fake_retriever(_input_data, context=None):
    preview = context.memory_context.format_for_prompt() if context and context.memory_context else ""
    return {
        "candidates": [
            {
                "candidate_id": "candidate_phase9b",
                "name": "Alice",
                "skills": ["Python", "LangGraph"],
            }
        ],
        "evidence": [{"summary_only": True, "memory_seen": "prefer Python" in preview}],
        "metadata": {"memory_context_seen": "prefer Python" in preview},
    }


def fake_matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 94,
        "recommendation": "strong_match",
        "match_report": {"candidate_id": candidate["candidate_id"], "total_score": 94},
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_memory_source_none_does_not_build_context():
    result = build_readonly_runtime_memory_context(RuntimeMemorySourceConfig(source="none"))

    assert result.enabled is False
    assert result.provided is False
    assert result.to_summary()["memory_source"] == "none"
    assert result.to_summary()["memory_store_loaded"] is False


def test_memory_source_demo_preserves_phase9a_behavior():
    result = build_readonly_runtime_memory_context(
        RuntimeMemorySourceConfig(
            source="demo",
            tags=["runtime_demo"],
            max_items=3,
            max_chars=800,
            metadata={"demo_memory_context": True},
        ),
        target_context={"tags": ["runtime_demo"]},
    )
    summary = result.to_summary()

    assert result.provided is True
    assert summary["memory_source"] == "demo"
    assert summary["memory_store_loaded"] is True
    assert summary["memory_records_seen"] == 1
    assert summary["eligible_count"] == 1


def test_sqlite_source_loads_records_read_only(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory()])

    result = build_readonly_runtime_memory_context(source_config(db_path))
    summary = result.to_summary()

    assert result.provided is True
    assert "prefer Python" in result.preview_text
    assert summary["memory_source"] == "sqlite"
    assert summary["memory_db_path_present"] is True
    assert summary["memory_store_loaded"] is True
    assert summary["memory_records_seen"] == 1
    assert summary["eligible_count"] == 1


def test_sqlite_filters_ineligible_memory_and_keeps_counts(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    records = [
        promoted_memory(content="eligible runtime memory"),
        promoted_memory(content="dry run memory", metadata={"dry_run": True}),
        promoted_memory(content=SENSITIVE_TEXT, metadata={"sensitive": True}),
        promoted_memory(content="missing provenance", metadata={"source_candidate_id": ""}),
        promoted_memory(content="wrong tag memory", tags=["other_tag"]),
        promoted_memory(content="high importance memory", importance=0.95),
    ]
    seed_store(db_path, records)

    result = build_readonly_runtime_memory_context(source_config(db_path))
    serialized = json.dumps(result.to_summary(), ensure_ascii=False)

    assert result.provided is True
    assert result.eligible_count == 1
    assert result.denied_count == 3
    assert result.requires_review_count == 1
    assert result.to_summary()["memory_records_seen"] == 6
    assert SENSITIVE_TEXT not in serialized
    assert "dry run memory" not in result.preview_text
    assert "missing provenance" not in result.preview_text
    assert "wrong tag memory" not in result.preview_text
    assert "high importance memory" not in result.preview_text


def test_sqlite_filters_revoked_expired_and_superseded_with_governance(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    revoked = promoted_memory(content="revoked persistent memory")
    expired = promoted_memory(content="expired persistent memory")
    superseded = promoted_memory(content="superseded persistent memory")
    seed_store(db_path, [revoked, expired, superseded])
    governance = InMemoryMemoryGovernanceStore()
    governance.revoke_memory(revoked.memory_id, "Withdrawn.", "reviewer_1")
    governance.expire_memory(expired.memory_id, "Expired.", "reviewer_1")
    governance.mark_superseded(superseded.memory_id, "replacement", "Updated.", "reviewer_1")

    result = build_readonly_runtime_memory_context(source_config(db_path), governance_store=governance)
    summary = result.to_summary()

    assert result.provided is False
    assert summary["revoked_filtered_count"] == 1
    assert summary["expired_filtered_count"] == 1
    assert summary["superseded_filtered_count"] == 1
    assert "revoked persistent memory" not in result.preview_text
    assert "expired persistent memory" not in result.preview_text
    assert "superseded persistent memory" not in result.preview_text


def test_nonexistent_sqlite_db_returns_graceful_unavailable(tmp_path):
    missing_path = tmp_path / "missing.sqlite3"

    result = build_readonly_runtime_memory_context(source_config(missing_path))
    summary = result.to_summary()

    assert result.provided is False
    assert result.reason == "sqlite memory db unavailable"
    assert summary["memory_source"] == "sqlite"
    assert summary["memory_db_path_present"] is False
    assert summary["memory_store_loaded"] is False
    assert missing_path.exists() is False


def test_variant_path_receives_persistent_memory_context(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory()])
    memory_result = build_readonly_runtime_memory_context(source_config(db_path))
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    result = RuntimeEntryHarness().run(
        "Need Python LangGraph engineer",
        default_runner=lambda _jd: {"status": "ok"},
        variant_runner=runner,
        memory_context=memory_result.memory_context_preview,
        config=RuntimeEntryConfig(
            use_skill_backed_variant=True,
            allow_memory_context=True,
            metadata={"memory_context_summary": memory_result.to_summary()},
        ),
    )

    assert result.status == "ok"
    assert result.runner_used == "skill_backed_variant"
    assert result.output_summary["memory_source"] == "sqlite"
    assert result.output_summary["memory_context_provided"] is True
    assert result.output_summary["memory_context_eligible_count"] == 1


def test_default_graph_path_ignores_persistent_memory_context(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory()])
    memory_result = build_readonly_runtime_memory_context(source_config(db_path))

    result = RuntimeEntryHarness().run(
        "Need Python",
        default_runner=lambda _jd: {"status": "ok", "candidate_count": 1},
        memory_context=memory_result.memory_context_preview,
        config=RuntimeEntryConfig(allow_memory_context=True, use_skill_backed_variant=False),
    )

    assert result.runner_used == "default_graph"
    assert result.output_summary["memory_context_provided"] is False
    assert result.output_summary["memory_source"] == "none"


def test_cli_sqlite_memory_source_is_summary_only(tmp_path, capsys):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory(content="persistent memory visible only in preview")])

    code = run_cli(
        [
            "--jd",
            "Need Python LangGraph engineer",
            "--use-skill-backed-variant",
            "--allow-memory-context",
            "--memory-source",
            "sqlite",
            "--memory-db-path",
            str(db_path),
            "--memory-tag",
            "runtime_demo",
            "--json",
        ],
        default_runner=lambda _jd: {"status": "ok"},
    )
    payload = json.loads(capsys.readouterr().out)
    output = payload["output_summary"]
    serialized = json.dumps(payload, ensure_ascii=False)

    assert code == 0
    assert output["memory_source"] == "sqlite"
    assert output["memory_db_path_present"] is True
    assert output["memory_store_loaded"] is True
    assert output["memory_records_seen"] == 1
    assert output["memory_context_provided"] is True
    assert "persistent memory visible only in preview" not in serialized


def test_no_save_memory_or_promote_called_during_runtime_load(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory()])

    def fail_save(*_args, **_kwargs):
        raise AssertionError("runtime memory source must be read-only")

    monkeypatch.setattr(MemorySQLiteStore, "save_memory", fail_save)
    result = build_readonly_runtime_memory_context(source_config(db_path))

    assert result.provided is True


def test_max_items_and_max_chars_are_respected(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(
        db_path,
        [
            promoted_memory(content="first eligible memory " + "A" * 80),
            promoted_memory(content="second eligible memory " + "B" * 80),
        ],
    )

    result = build_readonly_runtime_memory_context(
        source_config(db_path, max_items=1, max_chars=80)
    )

    assert result.rendered_char_count <= 80
    assert "second eligible memory" not in result.preview_text


def test_phase9b_does_not_import_real_llm_chroma_hf_or_mcp(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.sqlite3"
    seed_store(db_path, [promoted_memory()])
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "sentence_transformers", "mcp", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase9B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = build_readonly_runtime_memory_context(source_config(db_path))

    assert result.provided is True
    assert blocked == []


def test_default_graph_source_is_not_modified_for_persistent_memory_context():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "RuntimeMemorySourceConfig" not in graph_source
    assert "MemorySQLiteStore" not in graph_source
    assert "memory_context_preview" not in graph_source
