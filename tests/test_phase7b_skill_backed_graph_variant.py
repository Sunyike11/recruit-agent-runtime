import builtins
import importlib

from langchain_core.messages import HumanMessage

from src.core.graph import create_recruit_graph
from src.core.state import create_initial_state
from src.integration.graph_variant import (
    FEATURE_FLAG_ENV,
    GraphVariantConfig,
    create_skill_backed_recruit_graph_variant,
    should_use_skill_backed_variant,
)
from src.skills import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RetrieverSkill,
    SkillRegistry,
)


def build_fake_registry(score=88.0):
    registry = SkillRegistry()
    registry.register(
        PlannerExtractSkill(
            extract_callable=lambda input_data, context: {
                "job_requirement": {
                    "job_id": "job_variant_1",
                    "raw_text": input_data["raw_text"],
                    "required_skills": ["Python", "LangGraph"],
                },
                "extracted_keywords": ["Python", "LangGraph"],
            }
        )
    )
    registry.register(
        RetrieverSkill(
            retrieve_callable=lambda input_data, context: {
                "candidates": [
                    {
                        "candidate_id": "candidate_variant_1",
                        "name": "Alice",
                        "skills": ["Python", "LangGraph"],
                    }
                ],
                "evidence": ["Alice has Python and LangGraph experience"],
            }
        )
    )
    registry.register(
        CandidateMatchSkill(
            match_callable=lambda input_data, context: {
                "match_report": {
                    "candidate_id": input_data["candidate_profile"]["candidate_id"],
                    "total_score": score,
                    "recommendation": "strong_match",
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


def test_graph_variant_config_defaults_disabled():
    config = GraphVariantConfig()

    assert config.enabled is False
    assert config.allow_memory_context is False
    assert config.use_skill_planner is True
    assert config.variant_name == "skill_backed_recruit_graph_preview"


def test_should_use_skill_backed_variant_defaults_false_and_env_opt_in():
    assert should_use_skill_backed_variant() is False
    assert should_use_skill_backed_variant(env={FEATURE_FLAG_ENV: "true"}) is True
    assert should_use_skill_backed_variant(GraphVariantConfig(enabled=True)) is True
    assert should_use_skill_backed_variant(GraphVariantConfig(enabled=False)) is False


def test_disabled_config_does_not_build_variant_even_with_registry():
    result = create_skill_backed_recruit_graph_variant(
        config=GraphVariantConfig(enabled=False),
        registry=build_fake_registry(),
    )

    assert result.built is False
    assert result.variant is None
    assert result.reason == "feature flag disabled"
    assert result.metadata["default_production_graph_replaced"] is False


def test_enabled_config_builds_variant_with_fake_registry():
    result = create_skill_backed_recruit_graph_variant(
        config=GraphVariantConfig(enabled=True, variant_name="test_variant"),
        registry=build_fake_registry(),
    )

    assert result.built is True
    assert result.variant is not None
    assert result.config.variant_name == "test_variant"


def test_fake_variant_runs_tiny_jd_and_returns_preview_state_shape():
    build = create_skill_backed_recruit_graph_variant(
        config=GraphVariantConfig(enabled=True, variant_name="test_variant"),
        registry=build_fake_registry(),
    )

    update = build.variant.invoke(create_initial_state("Need Python LangGraph engineer"), top_k=1)

    assert update["raw_jd_preview"] == "Need Python LangGraph engineer"
    assert update["extracted_jd_preview"]["required_skills"] == ["Python", "LangGraph"]
    assert update["candidate_pool_preview"][0]["candidate_id"] == "candidate_variant_1"
    assert update["final_reports_preview"][0]["candidate_id"] == "candidate_variant_1"
    assert update["variant_metadata"]["variant_name"] == "test_variant"
    assert update["variant_metadata"]["preview_only"] is True
    assert update["variant_metadata"]["default_production_graph_replaced"] is False


def test_low_score_variant_can_emit_refined_query_preview():
    build = create_skill_backed_recruit_graph_variant(
        config=GraphVariantConfig(enabled=True),
        registry=build_fake_registry(score=42.0),
    )

    update = build.variant.invoke({"messages": [HumanMessage(content="Need Python")]}, top_k=1)

    assert update["refined_query_preview"].endswith(" refined")
    assert "query_refine" in update["variant_metadata"]["skill_names"]


def test_default_create_recruit_graph_behavior_is_not_modified():
    def planner(state):
        return {"extracted_jd": {"search_query": "Python"}, "next_action": "retrieve"}

    def retriever(state):
        return {"candidate_pool": [{"text": "Alice", "metadata": {}}], "next_action": "match_evaluation"}

    def matcher(state):
        return {"final_reports": [{"candidate_id": "candidate_1", "total_score": 90}], "next_action": "end"}

    def refiner(state):
        return {"extracted_jd": {"search_query": "Python refined"}}

    app = create_recruit_graph(
        planner=planner,
        retriever=retriever,
        matcher=matcher,
        refiner=refiner,
        interrupt_before=[],
    )
    result = app.invoke(
        create_initial_state("Need Python"),
        {"configurable": {"thread_id": "phase7b-default-graph"}},
    )

    assert result["final_reports"][0]["candidate_id"] == "candidate_1"
    assert "variant_metadata" not in result


def test_variant_build_failure_does_not_block_default_graph_build():
    build = create_skill_backed_recruit_graph_variant(config=GraphVariantConfig(enabled=True), registry=None)

    assert build.built is False
    assert build.errors
    app = create_recruit_graph(
        planner=lambda state: {"next_action": "retrieve", "extracted_jd": {"search_query": "x"}},
        retriever=lambda state: {"next_action": "match_evaluation", "candidate_pool": []},
        matcher=lambda state: {"next_action": "end", "final_reports": []},
        refiner=lambda state: {},
        interrupt_before=[],
    )
    assert app is not None


def test_memory_context_default_disabled_in_variant_metadata():
    build = create_skill_backed_recruit_graph_variant(
        config=GraphVariantConfig(enabled=True),
        registry=build_fake_registry(),
    )

    update = build.variant.invoke({"raw_jd": "Need Python"}, top_k=1)

    assert update["variant_metadata"]["allow_memory_context"] is False
    assert update["variant_metadata"]["memory_context_used"] is False


def test_variant_module_does_not_import_real_retrieval_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("src.agents.retriever", "src.services.retriever", "llama_index", "chromadb")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase7B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("src.integration.graph_variant")
    build = module.create_skill_backed_recruit_graph_variant(
        config=module.GraphVariantConfig(enabled=True),
        registry=build_fake_registry(),
    )

    update = build.variant.invoke({"raw_jd": "Need Python"}, top_k=1)

    assert update["variant_metadata"]["preview_only"] is True
    assert blocked == []


def test_production_graph_source_does_not_reference_variant_factory():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "create_skill_backed_recruit_graph_variant" not in graph_source
    assert "GraphVariantConfig" not in graph_source
    assert "SkillRegistry" not in graph_source
