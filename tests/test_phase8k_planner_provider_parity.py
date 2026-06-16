import builtins
import json
import os
import sys

from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner
from src.skills.agent_adapters import (
    build_planner_provider_diagnostics,
    create_planner_agent_for_skill,
    invoke_planner_agent_for_skill,
)


SENSITIVE_KEY = "TEST_OPENAI_KEY_SHOULD_NOT_LEAK"
SENSITIVE_ERROR = "FULL-PROVIDER-ERROR-SHOULD-NOT-LEAK"


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8K test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FakePlannerAgent:
    def __call__(self, state):
        return {
            "extracted_jd": {
                "tech_stack": ["Python", "RAG"],
                "education": "",
                "must_have": [],
                "search_query": "Python RAG",
            },
            "messages": state.get("messages", []),
        }


class ProviderFailingPlannerAgent:
    def __call__(self, _state):
        raise RuntimeError(SENSITIVE_ERROR)


def fake_retriever(input_data, _context=None):
    return {
        "candidates": [
            {
                "candidate_id": "candidate_phase8k",
                "name": "Candidate",
                "skills": ["Python", "RAG"],
            }
        ],
        "evidence": [{"candidate_id": "candidate_phase8k"}],
    }


def fake_matcher(input_data, _context=None):
    return {
        "total_score": 90,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": input_data["candidate_profile"]["candidate_id"],
            "total_score": 90,
        },
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_provider_diagnostics_are_summary_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENSITIVE_KEY)
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid/full/path")
    monkeypatch.setenv("RECRUIT_AGENT_LLM_MODEL", "test-model")

    diagnostics = build_planner_provider_diagnostics(
        provider_error_type="APIConnectionError",
        state={"messages": ["present"], "candidate_pool": []},
    )
    serialized = json.dumps(diagnostics, ensure_ascii=False)

    assert diagnostics["openai_api_key"] == "set"
    assert diagnostics["openai_api_base"] == "set"
    assert diagnostics["llm_model"] == "test-model"
    assert diagnostics["planner_agent_class"] == "PlannerAgent"
    assert diagnostics["invocation_method"] == "__call__"
    assert diagnostics["summary_only"] is True
    assert SENSITIVE_KEY not in serialized
    assert "example.invalid" not in serialized


def test_create_planner_for_skill_uses_production_like_constructor():
    planner = create_planner_agent_for_skill(planner_factory=lambda: FakePlannerAgent())

    assert isinstance(planner, FakePlannerAgent)


def test_provider_error_is_classified_without_full_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENSITIVE_KEY)
    try:
        invoke_planner_agent_for_skill(
            raw_text="Need Python",
            planner_factory=lambda: ProviderFailingPlannerAgent(),
        )
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("expected provider failure")

    assert "planner_wrapper_failed" in error
    assert "planner_invocation_stage=invoke_planner_agent" in error
    assert "provider_error_type=RuntimeError" in error
    assert "planner_provider_openai_api_key=set" in error
    assert SENSITIVE_ERROR not in error
    assert SENSITIVE_KEY not in error


def test_deterministic_fallback_is_default_off():
    try:
        invoke_planner_agent_for_skill(
            raw_text="Need Python RAG",
            planner_factory=lambda: ProviderFailingPlannerAgent(),
        )
    except RuntimeError as exc:
        assert "planner_wrapper_failed" in str(exc)
    else:
        raise AssertionError("fallback should be disabled by default")


def test_deterministic_fallback_marks_not_real_success():
    output = invoke_planner_agent_for_skill(
        raw_text="Need Python RAG engineer",
        planner_factory=lambda: ProviderFailingPlannerAgent(),
        allow_deterministic_fallback=True,
    )

    metadata = output["job_requirement"]["metadata"]
    assert metadata["planner_fallback_used"] is True
    assert metadata["planner_fallback_type"] == "deterministic"
    assert metadata["real_planner_invoked"] is True
    assert metadata["real_planner_failed"] is True
    assert output["job_requirement"]["required_skills"] == ["Python", "RAG"]


def test_variant_runner_fallback_summary_is_explicit(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    runner = build_real_skill_wrapper_variant_runner(
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
        allow_planner_deterministic_fallback=True,
        llm_readiness_checker=lambda **_: {
            "openai_api_key": "set",
            "openai_api_base": "set",
            "dotenv_loaded": "skip",
            "dotenv_path_present": False,
            "summary_only": True,
        },
    )

    # Monkeypatching real PlannerAgent import would be heavier than an injected
    # failing callable here, so exercise fallback through explicit planner helper.
    fallback_planner = lambda input_data, _context=None: invoke_planner_agent_for_skill(
        raw_text=input_data["raw_text"],
        planner_factory=lambda: ProviderFailingPlannerAgent(),
        allow_deterministic_fallback=True,
    )
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("Need Python RAG engineer")

    assert summary["status"] == "ok"
    assert summary["planner_fallback_used"] is True
    assert summary["planner_fallback_type"] == "deterministic"
    assert summary["real_planner_invoked"] is True
    assert summary["real_planner_failed"] is True
    assert summary["error_hint"] == ""


def test_default_graph_behavior_is_not_modified():
    import src.core.graph as graph

    assert "PlannerExtractSkill" not in graph.create_recruit_graph.__code__.co_names
    assert "SkillRegistry" not in graph.create_recruit_graph.__code__.co_names


def test_phase8k_required_path_does_not_import_retriever_or_memory(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    invoke_planner_agent_for_skill(
        raw_text="Need Python",
        planner_factory=lambda: FakePlannerAgent(),
    )

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
    assert "llama_index.core" not in sys.modules
    assert "chromadb" not in sys.modules
