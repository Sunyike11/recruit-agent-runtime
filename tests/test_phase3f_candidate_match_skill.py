import builtins
import importlib
import sys

from src.domain.models import CandidateProfile, JobRequirement
from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    CandidateMatchSkill,
    SkillEvalCase,
    SkillEvalRunner,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillNodeAdapter,
    SkillRegistry,
)


def block_real_agent_and_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.matcher",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real agent/retrieval import in Phase3F test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_domain_input():
    job = JobRequirement(
        job_id="job_1",
        title="Agent Engineer",
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG"],
        education="Bachelor",
        experience_years=3,
    )
    candidate = CandidateProfile(
        candidate_id="candidate_1",
        name="Alice",
        skills=["Python", "LangGraph", "RAG"],
        education="Bachelor",
        experience=["Built agent workflows"],
        projects=["Recruit matching agent"],
    )
    return {
        "job_requirement": job.to_dict(),
        "candidate_profile": candidate.to_dict(),
        "evidence": ["Python and LangGraph overlap"],
    }


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "3F"})
    task = TaskManager(store).create_task(session.session_id, jd_text="招聘JD", thread_id="thread-candidate-match")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3f-test"},
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_candidate_match_skill_with_fake_callable_executes_successfully():
    skill = CandidateMatchSkill(
        match_callable=lambda input_data, context: {
            "total_score": 88,
            "recommendation": "strong_match",
        }
    )

    result = skill.execute(make_domain_input())

    assert result.success is True
    assert result.output["total_score"] == 88.0
    assert result.output["recommendation"] == "strong_match"
    assert result.output["match_report"]["candidate_id"] == "candidate_1"


def test_candidate_match_skill_accepts_job_requirement_and_candidate_profile_dicts():
    captured = {}

    def match(input_data, context):
        captured["job"] = input_data["job_requirement"]
        captured["candidate"] = input_data["candidate_profile"]
        return {"total_score": 91}

    result = CandidateMatchSkill(match_callable=match).execute(make_domain_input())

    assert result.success is True
    assert captured["job"]["job_id"] == "job_1"
    assert captured["candidate"]["candidate_id"] == "candidate_1"


def test_candidate_match_generates_match_report_when_missing():
    result = CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 72}).execute(make_domain_input())

    report = result.output["match_report"]
    assert report["job_id"] == "job_1"
    assert report["candidate_id"] == "candidate_1"
    assert report["total_score"] == 72.0
    assert report["recommendation"] == "possible_match"
    assert report["evidence"] == ["Python and LangGraph overlap"]


def test_candidate_match_preserves_complete_match_report():
    full_report = {
        "match_id": "match_1",
        "job_id": "job_1",
        "candidate_id": "candidate_1",
        "total_score": 95,
        "strengths": ["Strong Python"],
        "weaknesses": [],
        "evidence": ["project evidence"],
        "recommendation": "hire",
    }
    skill = CandidateMatchSkill(
        match_callable=lambda input_data, context: {
            "total_score": "95",
            "recommendation": "hire",
            "match_report": full_report,
            "metadata": {"strategy": "fake"},
        }
    )

    result = skill.execute(make_domain_input())

    assert result.success is True
    assert result.output["match_report"]["match_id"] == "match_1"
    assert result.output["match_report"]["strengths"] == ["Strong Python"]
    assert result.output["metadata"] == {"strategy": "fake"}


def test_candidate_match_total_score_is_coerced_to_float():
    result = CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": "66.5"}).execute(
        make_domain_input()
    )

    assert result.success is True
    assert result.output["total_score"] == 66.5
    assert isinstance(result.output["total_score"], float)


def test_candidate_match_missing_total_score_fails():
    result = CandidateMatchSkill(match_callable=lambda input_data, context: {"recommendation": "missing score"}).execute(
        make_domain_input()
    )

    assert result.success is False
    assert "total_score" in result.error


def test_candidate_match_callable_exception_is_wrapped_as_failed_result():
    def fail(input_data, context):
        raise RuntimeError("match failed")

    result = CandidateMatchSkill(match_callable=fail).execute(make_domain_input())

    assert result.success is False
    assert result.error == "match failed"


def test_candidate_match_skill_can_register_to_skill_registry():
    skill = CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 80})
    registry = make_registry(skill)

    assert registry.get("candidate_match") is skill


def test_candidate_match_skill_executor_records_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    registry = make_registry(CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 83}))
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("candidate_match", make_domain_input(), context=context)

    assert result.success is True
    assert result.output["total_score"] == 83.0
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]
    assert store.list_events_by_task(task.task_id)[-1].payload["skill_name"] == "candidate_match"


def test_candidate_match_skill_eval_runner_can_run_fixture_case():
    registry = make_registry(CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 90}))
    runner = SkillEvalRunner(registry)

    result = runner.run_case(
        SkillEvalCase(
            case_id="candidate_match_case",
            skill_name="candidate_match",
            input_data=make_domain_input(),
            expected_output={"total_score": 90.0, "recommendation": "strong_match"},
        )
    )

    assert result.passed is True


def test_candidate_match_skill_node_adapter_maps_fake_state():
    registry = make_registry(CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 77}))
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="candidate_match",
        input_mapper=lambda state: {
            "job_requirement": state["job_requirement"],
            "candidate_profile": state["candidate_profile"],
        },
        output_mapper=lambda result, state: {
            "final_reports": [result.output["match_report"]],
            "match_result": {
                "candidate_id": result.output["match_report"]["candidate_id"],
                "total_score": result.output["total_score"],
                "recommendation": result.output["recommendation"],
            },
        },
    )
    input_data = make_domain_input()

    update = adapter(
        {
            "job_requirement": input_data["job_requirement"],
            "candidate_profile": input_data["candidate_profile"],
        }
    )

    assert update["final_reports"][0]["candidate_id"] == "candidate_1"
    assert update["match_result"]["total_score"] == 77.0
    assert update["skill_execution_metadata"]["skill_name"] == "candidate_match"


def test_importing_agent_adapters_does_not_import_real_matcher_or_retrieval_modules(monkeypatch):
    block_real_agent_and_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.agent_adapters", None)
    sys.modules.pop("src.agents.matcher", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    module = importlib.import_module("src.skills.agent_adapters")
    skill = module.CandidateMatchSkill(match_callable=lambda input_data, context: {"total_score": 75})
    result = skill.execute(make_domain_input())

    assert result.success is True
    assert "src.agents.matcher" not in sys.modules
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_phase3f_does_not_modify_real_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "candidate_match" not in graph_source
