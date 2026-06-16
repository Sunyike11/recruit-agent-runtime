import builtins
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.core.graph import create_recruit_graph
from src.core.state import create_initial_state
from src.integration.graph_variant import (
    GraphVariantConfig,
    MemoryContextInjectionConfig,
    build_variant_memory_context,
    create_skill_backed_variant_context,
    create_skill_backed_recruit_graph_variant,
)
from src.memory import InMemoryMemoryGovernanceStore, MemoryRecord, MemoryType
from src.skills import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RetrieverSkill,
    SkillRegistry,
)
from src.skills.context import SkillExecutionContext
from src.skills.memory_context_adapter import build_shadow_workflow_memory_context


SECRET_TEXT = "PRIVATE-RESUME-CONTENT-MUST-NOT-ENTER-VARIANT-CONTEXT"


def promoted_memory(content="prefer LangGraph for agent workflows", **kwargs):
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


def build_memory_aware_registry(score=95.0):
    registry = SkillRegistry()
    registry.register(
        PlannerExtractSkill(
            extract_callable=lambda input_data, context: {
                "job_requirement": {
                    "job_id": "job_phase7c",
                    "raw_text": input_data["raw_text"],
                    "required_skills": ["Python", "LangGraph"],
                },
                "extracted_keywords": ["Python", "LangGraph"],
            }
        )
    )

    def retrieve(input_data, context):
        preview = (
            context.memory_context.format_for_prompt()
            if context and context.memory_context is not None
            else ""
        )
        seen = "prefer LangGraph" in preview
        return {
            "candidates": [
                {
                    "candidate_id": "candidate_phase7c",
                    "name": "Alice",
                    "skills": ["Python", "LangGraph"],
                }
            ],
            "evidence": ["deterministic evidence"],
            "metadata": {"memory_context_seen": seen},
        }

    registry.register(RetrieverSkill(retrieve_callable=retrieve))
    registry.register(
        CandidateMatchSkill(
            match_callable=lambda input_data, context: {
                "match_report": {
                    "candidate_id": input_data["candidate_profile"]["candidate_id"],
                    "total_score": score,
                    "metadata": {
                        "memory_context_seen": bool(
                            context and context.memory_context is not None
                        )
                    },
                },
                "total_score": score,
                "recommendation": "strong_match",
            }
        )
    )
    registry.register(
        QueryRefineSkill(
            refine_callable=lambda input_data, context: {
                "refined_query": f"{input_data['query']} refined",
            }
        )
    )
    return registry


def enabled_graph_config(memory_config=None, allow_memory_context=True):
    return GraphVariantConfig(
        enabled=True,
        allow_memory_context=allow_memory_context,
        memory_context_config=memory_config,
    )


def enabled_memory_config():
    return MemoryContextInjectionConfig(enabled=True, allow_memory_context=True)


def test_memory_context_injection_config_defaults_disabled():
    config = MemoryContextInjectionConfig()

    assert config.enabled is False
    assert config.allow_memory_context is False
    assert config.require_governance is True
    assert config.max_items == 5


def test_graph_variant_config_defaults_memory_context_disabled():
    config = GraphVariantConfig()

    assert config.enabled is False
    assert config.allow_memory_context is False
    assert config.memory_context_config is None


def test_feature_flag_disabled_does_not_build_memory_context():
    result = build_variant_memory_context(
        [promoted_memory()],
        variant_config=GraphVariantConfig(enabled=False, allow_memory_context=True),
        memory_config=enabled_memory_config(),
    )

    assert result.built is False
    assert result.preview == ""
    assert "graph variant disabled" in result.reason


def test_variant_memory_disabled_does_not_build_memory_context():
    result = build_variant_memory_context(
        [promoted_memory()],
        variant_config=enabled_graph_config(allow_memory_context=False),
        memory_config=enabled_memory_config(),
    )

    assert result.built is False
    assert result.reason == "graph variant memory context disabled"


def test_disabled_memory_injection_clears_existing_context_to_prevent_bypass():
    existing_preview = build_shadow_workflow_memory_context([promoted_memory()])
    base_context = SkillExecutionContext(memory_context=existing_preview)

    result = create_skill_backed_variant_context(
        base_context=base_context,
        memory_records=[promoted_memory()],
        variant_config=enabled_graph_config(allow_memory_context=False),
        memory_config=enabled_memory_config(),
    )

    assert result.built is False
    assert result.skill_context.memory_context is None
    assert base_context.memory_context is existing_preview


def test_enabled_variant_and_memory_config_build_context():
    result = build_variant_memory_context(
        [promoted_memory()],
        variant_config=enabled_graph_config(memory_config=enabled_memory_config()),
    )

    assert result.built is True
    assert "prefer LangGraph" in result.preview
    assert result.memory_context is not None
    assert result.metadata["memory_store_written"] is False


def test_eligible_promoted_memory_enters_skill_execution_context():
    build = create_skill_backed_recruit_graph_variant(
        config=enabled_graph_config(memory_config=enabled_memory_config()),
        registry=build_memory_aware_registry(),
    )

    update = build.variant.invoke(
        {"raw_jd": "Need Python LangGraph engineer"},
        memory_records=[promoted_memory()],
    )

    assert update["variant_metadata"]["memory_context_used"] is True
    assert "prefer LangGraph" in update["memory_context_preview"]
    assert (
        update["final_reports_preview"][0]["metadata"]["memory_context_seen"]
        is True
    )


def test_revoked_memory_is_excluded_from_variant_context():
    memory = promoted_memory(content="revoked memory text")
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")
    build = create_skill_backed_recruit_graph_variant(
        config=enabled_graph_config(memory_config=enabled_memory_config()),
        registry=build_memory_aware_registry(),
    )

    update = build.variant.invoke(
        {"raw_jd": "Need Python"},
        memory_records=[memory],
        governance_store=store,
    )

    assert update["variant_metadata"]["memory_context_used"] is False
    assert "revoked memory text" not in update["memory_context_preview"]
    assert update["final_reports_preview"][0]["metadata"]["memory_context_seen"] is False


def test_expired_and_superseded_memory_are_excluded_from_variant_context():
    expired = promoted_memory(content="expired memory text")
    superseded = promoted_memory(content="superseded memory text")
    store = InMemoryMemoryGovernanceStore()
    store.expire_memory(expired.memory_id, "Expired.", "reviewer_1")
    store.mark_superseded(superseded.memory_id, "replacement", "Updated.", "reviewer_1")

    result = build_variant_memory_context(
        [expired, superseded],
        variant_config=enabled_graph_config(memory_config=enabled_memory_config()),
        governance_store=store,
    )

    assert result.built is False
    assert "expired memory text" not in result.preview
    assert "superseded memory text" not in result.preview


def test_sensitive_dry_run_and_missing_provenance_memory_are_excluded():
    result = build_variant_memory_context(
        [
            promoted_memory(content=SECRET_TEXT, metadata={"sensitive": True}),
            promoted_memory(content="dry run text", metadata={"dry_run": True}),
            promoted_memory(content="missing provenance", metadata={"source_reflection_id": ""}),
        ],
        variant_config=enabled_graph_config(memory_config=enabled_memory_config()),
    )

    assert result.built is False
    assert SECRET_TEXT not in result.preview
    assert "dry run text" not in result.preview
    assert "missing provenance" not in result.preview


def test_variant_output_uses_preview_keys_and_does_not_write_production_fields():
    build = create_skill_backed_recruit_graph_variant(
        config=enabled_graph_config(memory_config=enabled_memory_config()),
        registry=build_memory_aware_registry(),
    )

    update = build.variant.invoke(
        {"messages": [HumanMessage(content="Need Python LangGraph")]},
        memory_records=[promoted_memory()],
    )

    assert "memory_context_preview" in update
    assert "candidate_pool" not in update
    assert "final_reports" not in update
    assert update["variant_metadata"]["preview_only"] is True
    assert update["variant_metadata"]["default_production_graph_replaced"] is False


def test_default_create_recruit_graph_behavior_remains_unchanged():
    app = create_recruit_graph(
        planner=lambda state: {"next_action": "retrieve", "extracted_jd": {"search_query": "x"}},
        retriever=lambda state: {"next_action": "match_evaluation", "candidate_pool": []},
        matcher=lambda state: {"next_action": "end", "final_reports": []},
        refiner=lambda state: {},
        interrupt_before=[],
    )

    result = app.invoke(
        create_initial_state("Need Python"),
        {"configurable": {"thread_id": "phase7c-default-graph"}},
    )

    assert "memory_context_preview" not in result
    assert "variant_metadata" not in result


def test_phase7c_does_not_import_real_llm_retriever_chroma_or_llamaindex(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase7C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    build = create_skill_backed_recruit_graph_variant(
        config=enabled_graph_config(memory_config=enabled_memory_config()),
        registry=build_memory_aware_registry(),
    )
    update = build.variant.invoke(
        {"raw_jd": "Need Python"},
        memory_records=[promoted_memory()],
    )

    assert update["variant_metadata"]["memory_context_used"] is True
    assert blocked == []


def test_production_graph_source_does_not_reference_memory_variant_injection():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "MemoryContextInjectionConfig" not in graph_source
    assert "build_variant_memory_context" not in graph_source
    assert "memory_context_preview" not in graph_source
