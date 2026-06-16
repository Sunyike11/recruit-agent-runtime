import builtins
import importlib
import sys

from src.domain.models import JobRequirement
from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    PlannerExtractSkill,
    SkillEvalCase,
    SkillEvalRunner,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillNodeAdapter,
    SkillRegistry,
    normalize_planner_output_to_job_requirement,
)


def block_real_agent_and_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.planner",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real agent/retrieval import in Phase3G test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_job_requirement():
    return JobRequirement(
        job_id="job_1",
        raw_text="Need Python LangGraph agent engineer",
        title="Agent Engineer",
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG"],
        education="Bachelor",
        experience_years=3,
        location="Remote",
    )


def make_input():
    return {
        "raw_text": "Need Python LangGraph agent engineer",
        "metadata": {"source": "unit-test"},
    }


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "3G"})
    task = TaskManager(store).create_task(session.session_id, jd_text="招聘JD", thread_id="thread-planner-extract")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3g-test"},
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_planner_extract_skill_with_fake_callable_executes_successfully():
    skill = PlannerExtractSkill(
        extract_callable=lambda input_data, context: {
            "job_requirement": make_job_requirement().to_dict(),
            "extracted_keywords": ["Python", "LangGraph"],
            "metadata": {"strategy": "fake"},
        }
    )

    result = skill.execute(make_input())

    assert result.success is True
    assert result.output["job_requirement"]["job_id"] == "job_1"
    assert result.output["extracted_keywords"] == ["Python", "LangGraph"]
    assert result.output["metadata"] == {"strategy": "fake"}


def test_planner_extract_skill_receives_raw_text():
    captured = {}

    def extract(input_data, context):
        captured["raw_text"] = input_data["raw_text"]
        return {"job_requirement": make_job_requirement().to_dict()}

    result = PlannerExtractSkill(extract_callable=extract).execute(make_input())

    assert result.success is True
    assert captured["raw_text"] == "Need Python LangGraph agent engineer"


def test_planner_extract_callable_returning_job_requirement_dict_succeeds():
    job = make_job_requirement().to_dict()

    result = PlannerExtractSkill(extract_callable=lambda input_data, context: {"job_requirement": job}).execute(
        make_input()
    )

    assert result.success is True
    assert result.output["job_requirement"]["required_skills"] == ["Python", "LangGraph"]


def test_planner_extract_callable_returning_job_requirement_object_converts_to_dict():
    result = PlannerExtractSkill(
        extract_callable=lambda input_data, context: {
            "job_requirement": make_job_requirement(),
            "extracted_keywords": ["Python"],
        }
    ).execute(make_input())

    assert result.success is True
    assert result.output["job_requirement"]["title"] == "Agent Engineer"
    assert result.output["extracted_keywords"] == ["Python"]


def test_planner_extract_callable_returning_top_level_job_requirement_object_converts_to_dict():
    result = PlannerExtractSkill(extract_callable=lambda input_data, context: make_job_requirement()).execute(
        make_input()
    )

    assert result.success is True
    assert result.output["job_requirement"]["job_id"] == "job_1"


def test_planner_extract_missing_job_requirement_fails():
    result = PlannerExtractSkill(extract_callable=lambda input_data, context: {"extracted_keywords": ["Python"]}).execute(
        make_input()
    )

    assert result.success is False
    assert "job_requirement" in result.error


def test_planner_extract_non_dict_job_requirement_fails():
    result = PlannerExtractSkill(extract_callable=lambda input_data, context: {"job_requirement": "not a dict"}).execute(
        make_input()
    )

    assert result.success is False
    assert "job_requirement" in result.error


def test_planner_extract_callable_exception_is_wrapped_as_failed_result():
    def fail(input_data, context):
        raise RuntimeError("extract failed")

    result = PlannerExtractSkill(extract_callable=fail).execute(make_input())

    assert result.success is False
    assert result.error == "extract failed"


def test_planner_extract_skill_can_register_to_skill_registry():
    skill = PlannerExtractSkill(extract_callable=lambda input_data, context: {"job_requirement": make_job_requirement()})
    registry = make_registry(skill)

    assert registry.get("planner_extract") is skill


def test_planner_extract_skill_executor_records_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    registry = make_registry(
        PlannerExtractSkill(extract_callable=lambda input_data, context: {"job_requirement": make_job_requirement()})
    )
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("planner_extract", make_input(), context=context)

    assert result.success is True
    assert result.output["job_requirement"]["title"] == "Agent Engineer"
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]
    assert store.list_events_by_task(task.task_id)[-1].payload["skill_name"] == "planner_extract"


def test_planner_extract_skill_eval_runner_can_run_fixture_case():
    registry = make_registry(
        PlannerExtractSkill(extract_callable=lambda input_data, context: {"job_requirement": make_job_requirement()})
    )
    runner = SkillEvalRunner(registry)

    result = runner.run_case(
        SkillEvalCase(
            case_id="planner_extract_case",
            skill_name="planner_extract",
            input_data=make_input(),
            expected_output=normalize_planner_output_to_job_requirement({"job_requirement": make_job_requirement()}),
        )
    )

    assert result.passed is True


def test_planner_extract_skill_node_adapter_maps_fake_state():
    registry = make_registry(
        PlannerExtractSkill(
            extract_callable=lambda input_data, context: {
                "job_requirement": make_job_requirement().to_dict(),
                "extracted_keywords": ["Python", "LangGraph"],
            }
        )
    )
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="planner_extract",
        input_mapper=lambda state: {"raw_text": state["jd_text"]},
        output_mapper=lambda result, state: {
            "job_requirement": result.output["job_requirement"],
            "extracted_jd": {
                "tech_stack": result.output.get("extracted_keywords", []),
                "search_query": " ".join(result.output.get("extracted_keywords", [])),
            },
        },
    )

    update = adapter({"jd_text": "Need Python LangGraph"})

    assert update["job_requirement"]["job_id"] == "job_1"
    assert update["extracted_jd"]["tech_stack"] == ["Python", "LangGraph"]
    assert update["skill_execution_metadata"]["skill_name"] == "planner_extract"


def test_importing_agent_adapters_does_not_import_real_planner_or_retrieval_modules(monkeypatch):
    block_real_agent_and_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.agent_adapters", None)
    sys.modules.pop("src.agents.planner", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    module = importlib.import_module("src.skills.agent_adapters")
    skill = module.PlannerExtractSkill(
        extract_callable=lambda input_data, context: {"job_requirement": make_job_requirement()}
    )
    result = skill.execute(make_input())

    assert result.success is True
    assert "src.agents.planner" not in sys.modules
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_phase3g_does_not_modify_real_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "planner_extract" not in graph_source
