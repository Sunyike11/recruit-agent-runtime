import builtins
import importlib
import sys

from src.domain.models import CandidateProfile, JobRequirement
from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RecruitmentSkillWorkflow,
    RetrieverSkill,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillRegistry,
    SkillWorkflowEvalCase,
    replay_workflow_case_from_fixture,
    run_workflow_eval_case,
)


def block_real_agent_and_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.planner",
            "src.agents.refiner",
            "src.agents.matcher",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real agent/retrieval import in Phase3J test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_job_requirement(raw_text="Need Python LangGraph engineer"):
    return JobRequirement(
        job_id="job_shadow_1",
        raw_text=raw_text,
        title="Agent Engineer",
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG"],
        metadata={"search_query": "Python LangGraph"},
    ).to_dict()


def make_candidate(candidate_id, name, skills):
    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        skills=skills,
        education="Bachelor",
        experience=["Built deterministic workflow tests"],
    ).to_dict()


def fake_planner(input_data, context):
    return {
        "job_requirement": make_job_requirement(input_data["raw_text"]),
        "extracted_keywords": ["Python", "LangGraph"],
    }


def fake_retriever_with_candidates(input_data, context):
    return {
        "candidates": [
            make_candidate("candidate_1", "Alice", ["Python", "LangGraph", "RAG"]),
            make_candidate("candidate_2", "Bob", ["Excel"]),
        ],
        "evidence": ["Alice has Python and LangGraph evidence", "Bob has operations evidence"],
    }


def fake_retriever_empty(input_data, context):
    return {"evidence": []}


def fake_retriever_low_score(input_data, context):
    return {
        "candidates": [make_candidate("candidate_3", "Casey", ["Excel"])],
        "evidence": ["Casey evidence does not overlap required skills"],
    }


def fake_matcher(input_data, context):
    required = set(input_data["job_requirement"].get("required_skills", []))
    candidate = input_data["candidate_profile"]
    candidate_skills = set(candidate.get("skills", []))
    matched = sorted(required.intersection(candidate_skills))
    score = 0.0 if not required else round((len(matched) / len(required)) * 100, 2)
    return {
        "total_score": score,
        "recommendation": "strong_match" if score >= 80 else "not_recommended",
        "match_report": {
            "job_id": input_data["job_requirement"]["job_id"],
            "candidate_id": candidate["candidate_id"],
            "total_score": score,
            "matched_skills": matched,
            "evidence": list(input_data.get("evidence", [])),
        },
    }


def fake_refiner(input_data, context):
    return {
        "refined_query": f"{input_data['query']} remote backend",
        "reason": input_data.get("context", ""),
    }


def make_registry(retrieve_callable=fake_retriever_with_candidates, planner_callable=fake_planner):
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=planner_callable))
    registry.register(RetrieverSkill(retrieve_callable=retrieve_callable))
    registry.register(CandidateMatchSkill(match_callable=fake_matcher))
    registry.register(QueryRefineSkill(refine_callable=fake_refiner))
    return registry


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "3J"})
    task = TaskManager(store).create_task(session.session_id, jd_text="Need Python LangGraph engineer", thread_id="thread-shadow-workflow")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3j-test"},
    )
    return store, task, context


def make_workflow(registry, store=None):
    recorder = SkillExecutionRecorder(store) if store is not None else None
    executor = SkillExecutor(registry, recorder=recorder)
    return RecruitmentSkillWorkflow(executor, low_score_threshold=60)


def completed_skill_names(store, task_id):
    return [
        event.payload["skill_name"]
        for event in store.list_events_by_task(task_id)
        if event.event_type == "skill_completed"
    ]


def test_fake_recruitment_skill_workflow_runs_planner_retriever_matcher():
    workflow = make_workflow(make_registry())

    result = workflow.run("Need Python LangGraph engineer", top_k=2)

    assert result.success is True
    assert result.status == "completed"
    assert result.job_requirement["required_skills"] == ["Python", "LangGraph"]
    assert [candidate["candidate_id"] for candidate in result.retrieved_candidates] == ["candidate_1", "candidate_2"]
    assert len(result.match_reports) == 2
    assert result.match_reports[0]["total_score"] == 100.0
    assert result.refined_query is None


def test_workflow_generates_match_reports_when_candidates_exist():
    workflow = make_workflow(make_registry())

    result = workflow.run("Need Python LangGraph engineer")

    assert result.success is True
    assert [report["candidate_id"] for report in result.match_reports] == ["candidate_1", "candidate_2"]
    assert result.skill_results[-1].skill_name == "candidate_match"


def test_workflow_triggers_query_refine_when_no_candidates():
    workflow = make_workflow(make_registry(retrieve_callable=fake_retriever_empty))

    result = workflow.run("Need Python LangGraph engineer")

    assert result.success is True
    assert result.retrieved_candidates == []
    assert result.match_reports == []
    assert result.refined_query == "Python LangGraph remote backend"
    assert result.metadata["refinement_reason"] == "no candidates retrieved"


def test_workflow_triggers_query_refine_when_all_scores_are_low():
    workflow = make_workflow(make_registry(retrieve_callable=fake_retriever_low_score))

    result = workflow.run("Need Python LangGraph engineer")

    assert result.success is True
    assert result.match_reports[0]["total_score"] == 0.0
    assert result.refined_query == "Python LangGraph remote backend"
    assert "best score below threshold" in result.metadata["refinement_reason"]


def test_workflow_uses_skill_executor_and_records_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    workflow = make_workflow(make_registry(), store=store)

    result = workflow.run("Need Python LangGraph engineer", top_k=1, context=context)

    assert result.success is True
    assert completed_skill_names(store, task.task_id) == [
        "planner_extract",
        "resume_retrieve",
        "candidate_match",
    ]


def test_task_timeline_can_show_all_four_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    workflow = make_workflow(make_registry(retrieve_callable=fake_retriever_empty), store=store)

    result = workflow.run("Need Python LangGraph engineer", context=context)

    assert result.refined_query == "Python LangGraph remote backend"
    assert completed_skill_names(store, task.task_id) == [
        "planner_extract",
        "resume_retrieve",
        "query_refine",
    ]
    event_types = [event.event_type for event in store.list_events_by_task(task.task_id)]
    assert event_types.count("skill_started") == 3
    assert event_types.count("skill_completed") == 3


def test_workflow_result_contains_expected_state_fields():
    workflow = make_workflow(make_registry())

    result = workflow.run("Need Python LangGraph engineer")

    state = result.to_dict()
    assert state["job_requirement"]["job_id"] == "job_shadow_1"
    assert len(state["retrieved_candidates"]) == 2
    assert len(state["match_reports"]) == 2
    assert state["status"] == "completed"


def test_workflow_failure_returns_clear_failure_result():
    def failing_planner(input_data, context):
        raise RuntimeError("planner failed")

    workflow = make_workflow(make_registry(planner_callable=failing_planner))

    result = workflow.run("Need Python LangGraph engineer")

    assert result.success is False
    assert result.status == "failed"
    assert result.error == "planner failed"
    assert result.skill_results[0].success is False


def test_workflow_level_eval_fixture_can_validate_deterministic_output():
    workflow = make_workflow(make_registry())
    eval_case = SkillWorkflowEvalCase(
        case_id="shadow_workflow_case",
        raw_jd="Need Python LangGraph engineer",
        expected_status="completed",
        expected_min_match_reports=2,
    )

    result = run_workflow_eval_case(workflow, eval_case)

    assert result.passed is True
    assert result.output["job_requirement"]["required_skills"] == ["Python", "LangGraph"]


def test_workflow_fixture_replay_marks_full_replay():
    workflow = make_workflow(make_registry(retrieve_callable=fake_retriever_empty))
    fixture = {
        "case_id": "shadow_workflow_replay",
        "raw_jd": "Need Python LangGraph engineer",
        "expected_status": "completed",
        "expected_min_match_reports": 0,
        "expected_refined_query": "Python LangGraph remote backend",
    }

    result = replay_workflow_case_from_fixture(fixture, workflow)

    assert result.passed is True
    assert result.metadata["replay_mode"] == "workflow_fixture_full_replay"
    assert result.metadata["full_replay"] is True


def test_importing_skill_workflow_does_not_import_real_agents_or_retrieval_modules(monkeypatch):
    block_real_agent_and_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.workflow", None)
    for module_name in (
        "src.agents.planner",
        "src.agents.refiner",
        "src.agents.matcher",
        "src.agents.retriever",
        "src.services.retriever",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("src.skills.workflow")

    assert hasattr(module, "RecruitmentSkillWorkflow")
    assert "src.agents.planner" not in sys.modules
    assert "src.agents.refiner" not in sys.modules
    assert "src.agents.matcher" not in sys.modules
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_phase3j_does_not_modify_production_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "RecruitmentSkillWorkflow" not in graph_source
    assert "planner_extract" not in graph_source
    assert "resume_retrieve" not in graph_source
    assert "candidate_match" not in graph_source
    assert "query_refine" not in graph_source
