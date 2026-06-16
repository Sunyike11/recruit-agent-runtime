import builtins
import json
from pathlib import Path

from src.memory import InMemoryMemoryGovernanceStore, MemoryRecord, MemoryType
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.memory_context import (
    RuntimeMemoryContextConfig,
    build_runtime_memory_context,
)
from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner
from scripts.run_recruit_runtime import run_cli


SENSITIVE_TEXT = "PRIVATE-MEMORY-CONTENT-MUST-NOT-LEAK"


def promoted_memory(content="prefer LangGraph for agent workflow", **kwargs):
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
        tags=kwargs.pop("tags", ["runtime_demo", "agent"]),
        metadata=metadata,
        **kwargs,
    )


def memory_config(**kwargs):
    values = {
        "enabled": True,
        "tags": ["runtime_demo"],
        "max_items": 5,
        "max_chars": 800,
        "require_governance": True,
        "metadata": {"demo_memory_context": True},
    }
    values.update(kwargs)
    return RuntimeMemoryContextConfig(**values)


def fake_planner(_input_data, _context=None):
    return {
        "job_requirement": {
            "job_id": "job_phase9a",
            "required_skills": ["Python", "LangGraph"],
        }
    }


def fake_retriever(input_data, context=None):
    preview = context.memory_context.format_for_prompt() if context and context.memory_context else ""
    return {
        "candidates": [
            {
                "candidate_id": "candidate_phase9a",
                "name": "Alice",
                "skills": ["Python", "LangGraph"],
                "metadata": {"memory_context_seen": "prefer LangGraph" in preview},
            }
        ],
        "evidence": [{"summary_only": True}],
        "metadata": {"memory_context_seen": "prefer LangGraph" in preview},
    }


def fake_matcher(input_data, context=None):
    candidate = input_data["candidate_profile"]
    preview_seen = bool(context and context.memory_context is not None)
    return {
        "total_score": 96,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": candidate["candidate_id"],
            "total_score": 96,
            "metadata": {"memory_context_seen": preview_seen},
        },
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_runtime_memory_context_config_defaults_disabled():
    config = RuntimeMemoryContextConfig()
    result = build_runtime_memory_context([promoted_memory()], config=config)

    assert config.enabled is False
    assert config.require_governance is True
    assert result.provided is False
    assert result.reason == "runtime memory context disabled"


def test_allow_memory_context_false_does_not_pass_context_to_variant():
    seen = {"memory_context": "unset"}

    def variant_runner(_raw_jd, memory_context=None, metadata=None):
        seen["memory_context"] = memory_context
        return {
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "metadata": {"metadata_keys": sorted((metadata or {}).keys()), "summary_only": True},
        }

    result = RuntimeEntryHarness().run(
        "Need Python LangGraph engineer",
        default_runner=lambda _jd: {"status": "ok"},
        variant_runner=variant_runner,
        memory_context=object(),
        config=RuntimeEntryConfig(use_skill_backed_variant=True, allow_memory_context=False),
    )

    assert result.status == "ok"
    assert seen["memory_context"] is None
    assert result.runner_used == "skill_backed_variant"


def test_allow_memory_context_true_but_no_source_is_graceful(capsys):
    code = run_cli(
        [
            "--jd",
            "Need Python LangGraph engineer",
            "--use-skill-backed-variant",
            "--allow-memory-context",
            "--json",
        ],
        default_runner=lambda _jd: {"status": "ok"},
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["runner_used"] == "skill_backed_variant"
    output = payload["output_summary"]
    assert output["memory_context_requested"] is True
    assert output["memory_context_provided"] is False
    assert output["memory_context_eligible_count"] == 0


def test_eligible_memory_enters_runtime_variant_context():
    memory_result = build_runtime_memory_context(
        [promoted_memory()],
        config=memory_config(),
        target_context={"tags": ["runtime_demo"]},
    )
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
    assert result.output_summary["memory_context_requested"] is True
    assert result.output_summary["memory_context_provided"] is True
    assert result.output_summary["memory_context_eligible_count"] == 1
    assert result.output_summary["memory_context_governance_applied"] is True
    assert result.output_summary["memory_context_rendered_char_count"] > 0


def test_revoked_expired_and_superseded_memory_are_filtered():
    revoked = promoted_memory(content="revoked memory text")
    expired = promoted_memory(content="expired memory text")
    superseded = promoted_memory(content="superseded memory text")
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(revoked.memory_id, "Withdrawn.", "reviewer_1")
    store.expire_memory(expired.memory_id, "Expired.", "reviewer_1")
    store.mark_superseded(superseded.memory_id, "replacement", "Updated.", "reviewer_1")

    result = build_runtime_memory_context(
        [revoked, expired, superseded],
        config=memory_config(),
        governance_store=store,
    )
    serialized = json.dumps(result.to_summary(), ensure_ascii=False)

    assert result.provided is False
    assert result.revoked_filtered_count == 1
    assert result.expired_filtered_count == 1
    assert result.superseded_filtered_count == 1
    assert "revoked memory text" not in serialized
    assert "expired memory text" not in serialized
    assert "superseded memory text" not in serialized


def test_sensitive_dry_run_and_missing_provenance_memory_are_filtered():
    result = build_runtime_memory_context(
        [
            promoted_memory(content=SENSITIVE_TEXT, metadata={"sensitive": True}),
            promoted_memory(content="dry run memory", metadata={"dry_run": True}),
            promoted_memory(content="missing provenance", metadata={"source_reflection_id": ""}),
        ],
        config=memory_config(),
    )
    serialized = json.dumps(result.to_summary(), ensure_ascii=False)

    assert result.provided is False
    assert result.eligible_count == 0
    assert result.denied_count == 3
    assert SENSITIVE_TEXT not in serialized
    assert "dry run memory" not in serialized
    assert "missing provenance" not in serialized


def test_default_graph_path_does_not_receive_memory_context():
    calls = {"default": 0}

    def default_runner(_raw_jd):
        calls["default"] += 1
        return {"status": "ok", "candidate_count": 1, "report_count": 1}

    result = RuntimeEntryHarness().run(
        "Need Python",
        default_runner=default_runner,
        memory_context=object(),
        config=RuntimeEntryConfig(allow_memory_context=True, use_skill_backed_variant=False),
    )

    assert result.runner_used == "default_graph"
    assert calls["default"] == 1
    assert result.output_summary["memory_context_provided"] is False


def test_cli_demo_memory_context_reaches_explicit_variant(capsys):
    code = run_cli(
        [
            "--jd",
            "Need Python LangGraph engineer",
            "--use-skill-backed-variant",
            "--allow-memory-context",
            "--use-demo-memory-context",
            "--json",
        ],
        default_runner=lambda _jd: {"status": "ok"},
    )
    payload = json.loads(capsys.readouterr().out)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert code == 0
    assert payload["runner_used"] == "skill_backed_variant"
    output = payload["output_summary"]
    assert output["memory_context_requested"] is True
    assert output["memory_context_provided"] is True
    assert output["memory_context_eligible_count"] == 1
    assert output["memory_context_governance_applied"] is True
    assert "Prefer LangGraph" not in serialized


def test_runtime_memory_context_does_not_write_or_promote_memory():
    source = Path("src/runtime/memory_context.py").read_text(encoding="utf-8")

    assert ".save_memory(" not in source
    assert "promote_memory" not in source
    assert ".save_" not in source


def test_phase9a_does_not_import_real_llm_chroma_hf_or_mcp(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "sentence_transformers", "mcp", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase9A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = build_runtime_memory_context(
        [promoted_memory()],
        config=memory_config(),
        target_context={"tags": ["runtime_demo"]},
    )

    assert result.provided is True
    assert blocked == []


def test_default_graph_source_is_not_modified_for_runtime_memory_context():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "RuntimeMemoryContextConfig" not in graph_source
    assert "build_runtime_memory_context" not in graph_source
    assert "memory_context_preview" not in graph_source
