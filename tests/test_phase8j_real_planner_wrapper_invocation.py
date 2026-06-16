import builtins
import json
import sys

from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner
from src.skills import PlannerExtractSkill, SkillExecutor, SkillRegistry
from src.skills.agent_adapters import (
    CandidateMatchSkill,
    QueryRefineSkill,
    RetrieverSkill,
    build_planner_state_for_skill,
    invoke_planner_agent_for_skill,
)
from src.skills.context import SkillExecutionContext
from src.skills.workflow import RecruitmentSkillWorkflow


SENSITIVE_TEXT = "FULL-JD-OR-LLM-TEXT-SHOULD-NOT-LEAK"


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
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8J test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def extracted_jd():
    return {
        "tech_stack": ["Python", "RAG", "LangGraph"],
        "education": "Bachelor",
        "must_have": ["agent workflow"],
        "search_query": "Python RAG LangGraph",
    }


class FakePlannerAgent:
    def __call__(self, state):
        assert "messages" in state
        assert state["messages"][-1].content
        return {
            "extracted_jd": extracted_jd(),
            "next_action": "retrieve_candidates",
        }


class FailingPlannerAgent:
    def __call__(self, _state):
        raise RuntimeError(SENSITIVE_TEXT)


def fake_retriever(input_data, _context=None):
    return {
        "candidates": [
            {
                "candidate_id": "candidate_phase8j",
                "name": "Candidate",
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "evidence": [{"candidate_id": "candidate_phase8j"}],
    }


def fake_matcher(input_data, _context=None):
    return {
        "total_score": 95,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": input_data["candidate_profile"]["candidate_id"],
            "total_score": 95,
        },
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"], "reason": "not needed"}


def test_build_planner_state_for_skill_matches_production_like_shape():
    state = build_planner_state_for_skill("Need Python RAG")

    assert sorted(state.keys()) == [
        "candidate_pool",
        "extracted_jd",
        "final_reports",
        "human_feedback",
        "loop_count",
        "messages",
        "next_action",
        "refinement_advice",
    ]
    assert state["messages"][-1].content == "Need Python RAG"
    assert state["candidate_pool"] == []
    assert state["final_reports"] == []


def test_invoke_planner_agent_for_skill_calls_production_like_callable():
    output = invoke_planner_agent_for_skill(
        raw_text="Need Python RAG LangGraph",
        planner_factory=lambda: FakePlannerAgent(),
    )

    assert output["job_requirement"]["required_skills"] == ["Python", "RAG", "LangGraph"]
    assert output["job_requirement"]["search_query"] == "Python RAG LangGraph"
    assert output["extracted_keywords"] == ["Python", "RAG", "LangGraph"]


def test_invoke_planner_agent_failure_is_summary_only():
    try:
        invoke_planner_agent_for_skill(
            raw_text=f"Need Python {SENSITIVE_TEXT}",
            planner_factory=lambda: FailingPlannerAgent(),
        )
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("expected planner invocation failure")

    assert "planner_wrapper_failed" in error
    assert "planner_invocation_stage=invoke_planner_agent" in error
    assert "provider_error_type=RuntimeError" in error
    assert "planner_raw_text_length=" in error
    assert SENSITIVE_TEXT not in error


def test_variant_runner_reports_planner_invocation_diagnostics(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    def failing_planner(_input, _context=None):
        return invoke_planner_agent_for_skill(
            raw_text=f"Need Python {SENSITIVE_TEXT}",
            planner_factory=lambda: FailingPlannerAgent(),
        )

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=failing_planner,
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner(f"Need Python {SENSITIVE_TEXT}")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "planner_wrapper_failed"
    assert summary["planner_invocation_stage"] == "invoke_planner_agent"
    assert summary["planner_input_shape"]["has_messages"] is True
    assert "messages" in summary["planner_input_shape"]["input_keys"]
    assert summary["provider_error_type"] == "RuntimeError"
    assert SENSITIVE_TEXT not in serialized


def test_injected_callable_path_still_accepts_extracted_jd():
    result = PlannerExtractSkill(extract_callable=lambda _input, _context: {"extracted_jd": extracted_jd()}).execute(
        {"raw_text": "Need Python"}
    )

    assert result.success is True
    assert result.output["job_requirement"]["search_query"] == "Python RAG LangGraph"


def test_recruitment_skill_workflow_consumes_aligned_planner_output():
    registry = SkillRegistry()
    registry.register(
        PlannerExtractSkill(
            extract_callable=lambda input_data, _context: invoke_planner_agent_for_skill(
                raw_text=input_data["raw_text"],
                planner_factory=lambda: FakePlannerAgent(),
            )
        )
    )
    registry.register(RetrieverSkill(retrieve_callable=fake_retriever))
    registry.register(CandidateMatchSkill(match_callable=fake_matcher))
    registry.register(QueryRefineSkill(refine_callable=fake_refiner))

    result = RecruitmentSkillWorkflow(SkillExecutor(registry)).run(
        "Need Python RAG LangGraph engineer",
        context=SkillExecutionContext(),
    )

    assert result.success is True
    assert len(result.match_reports) == 1
    assert result.job_requirement["search_query"] == "Python RAG LangGraph"


def test_default_graph_behavior_is_not_modified():
    import src.core.graph as graph

    assert "PlannerExtractSkill" not in graph.create_recruit_graph.__code__.co_names
    assert "SkillRegistry" not in graph.create_recruit_graph.__code__.co_names


def test_phase8j_required_path_does_not_import_real_retriever_or_memory(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    invoke_planner_agent_for_skill(
        raw_text="Need Python",
        planner_factory=lambda: FakePlannerAgent(),
    )

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
    assert "llama_index.core" not in sys.modules
    assert "chromadb" not in sys.modules
