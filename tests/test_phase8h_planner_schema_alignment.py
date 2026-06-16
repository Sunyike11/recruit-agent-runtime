import builtins
import json
import sys

from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner
from src.skills import PlannerExtractSkill, SkillExecutor, SkillRegistry
from src.skills.agent_adapters import normalize_planner_output_to_job_requirement
from src.skills.context import SkillExecutionContext
from src.skills.workflow import RecruitmentSkillWorkflow
from src.skills.agent_adapters import CandidateMatchSkill, QueryRefineSkill, RetrieverSkill


SENSITIVE_LLM_TEXT = "FULL-PLANNER-LLM-RESPONSE-SHOULD-NOT-LEAK"


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
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8H test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def extracted_jd():
    return {
        "tech_stack": ["Python", "RAG", "LangGraph"],
        "education": "Bachelor",
        "must_have": ["agent workflow"],
        "search_query": "Python RAG LangGraph",
    }


def fake_retriever(input_data, _context=None):
    return {
        "candidates": [
            {
                "candidate_id": "candidate_phase8h",
                "name": "Candidate",
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "evidence": [{"candidate_id": "candidate_phase8h"}],
    }


def fake_matcher(input_data, _context=None):
    return {
        "total_score": 93,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": input_data["candidate_profile"]["candidate_id"],
            "total_score": 93,
        },
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"], "reason": "not needed"}


def test_normalize_supports_extracted_jd_wrapper():
    normalized = normalize_planner_output_to_job_requirement({"extracted_jd": extracted_jd()}, raw_text="Need Python")

    job = normalized["job_requirement"]
    assert job["required_skills"] == ["Python", "RAG", "LangGraph"]
    assert job["tech_stack"] == ["Python", "RAG", "LangGraph"]
    assert job["education"] == "Bachelor"
    assert job["must_have"] == ["agent workflow"]
    assert job["search_query"] == "Python RAG LangGraph"
    assert job["metadata"]["search_query"] == "Python RAG LangGraph"


def test_normalize_supports_job_requirement_wrapper():
    normalized = normalize_planner_output_to_job_requirement(
        {
            "job_requirement": {
                "required_skills": ["Python"],
                "metadata": {"search_query": "Python"},
            }
        }
    )

    assert normalized["job_requirement"]["required_skills"] == ["Python"]
    assert normalized["job_requirement"]["tech_stack"] == ["Python"]
    assert normalized["job_requirement"]["search_query"] == "Python"


def test_normalize_supports_direct_extracted_jd_dict():
    normalized = normalize_planner_output_to_job_requirement(extracted_jd())

    assert normalized["job_requirement"]["tech_stack"] == ["Python", "RAG", "LangGraph"]
    assert normalized["extracted_keywords"] == ["Python", "RAG", "LangGraph"]


def test_normalize_missing_fields_is_graceful():
    normalized = normalize_planner_output_to_job_requirement({"tech_stack": ["Python"]})

    assert normalized["job_requirement"]["tech_stack"] == ["Python"]
    assert normalized["job_requirement"]["education"] == ""
    assert normalized["job_requirement"]["must_have"] == []
    assert normalized["job_requirement"]["search_query"] == ""


def test_planner_extract_injected_callable_returning_extracted_jd_succeeds():
    skill = PlannerExtractSkill(extract_callable=lambda _input, _context: {"extracted_jd": extracted_jd()})

    result = skill.execute({"raw_text": "Need Python RAG LangGraph"})

    assert result.success is True
    assert result.output["job_requirement"]["required_skills"] == ["Python", "RAG", "LangGraph"]
    assert result.output["job_requirement"]["search_query"] == "Python RAG LangGraph"


def test_recruitment_skill_workflow_consumes_normalized_planner_output():
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=lambda _input, _context: {"extracted_jd": extracted_jd()}))
    registry.register(RetrieverSkill(retrieve_callable=fake_retriever))
    registry.register(CandidateMatchSkill(match_callable=fake_matcher))
    registry.register(QueryRefineSkill(refine_callable=fake_refiner))
    workflow = RecruitmentSkillWorkflow(SkillExecutor(registry))

    result = workflow.run("Need Python RAG LangGraph engineer", context=SkillExecutionContext())

    assert result.success is True
    assert result.job_requirement["search_query"] == "Python RAG LangGraph"
    assert len(result.match_reports) == 1


def test_variant_runner_with_fake_real_wrapper_planner_output_no_longer_planner_failed(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=lambda _input, _context: {"extracted_jd": extracted_jd()},
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("Need Python RAG LangGraph engineer")

    assert summary["status"] == "ok"
    assert summary["error_hint"] == ""
    assert summary["match_count"] == 1


def test_planner_schema_diagnostics_are_summary_only(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=lambda _input, _context: {"unexpected": SENSITIVE_LLM_TEXT},
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("Need Python")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "planner_wrapper_failed"
    assert summary["planner_adapter_error_hint"] == "planner_schema_adapter_failed"
    assert summary["planner_output_keys"] == ["unexpected"]
    assert SENSITIVE_LLM_TEXT not in serialized


def test_phase8h_required_path_does_not_import_real_retriever_dependencies(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    PlannerExtractSkill(extract_callable=lambda _input, _context: {"extracted_jd": extracted_jd()}).execute(
        {"raw_text": "Need Python"}
    )

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
