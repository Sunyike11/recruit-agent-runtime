import builtins
import importlib
import sys

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    QueryRefineSkill,
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
            "src.agents.refiner",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real agent/retrieval import in Phase3E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "3E"})
    task = TaskManager(store).create_task(session.session_id, jd_text="招聘JD", thread_id="thread-query-refine")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3e-test"},
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_query_refine_skill_with_fake_callable_executes_successfully():
    skill = QueryRefineSkill(
        refine_callable=lambda input_data, context: {
            "refined_query": f"{input_data['query']} remote",
            "reason": "append remote",
        }
    )

    result = skill.execute({"query": "python engineer", "context": "low recall"})

    assert result.success is True
    assert result.output["refined_query"] == "python engineer remote"
    assert result.output["reason"] == "append remote"


def test_query_refine_callable_returning_str_is_normalized():
    skill = QueryRefineSkill(refine_callable=lambda input_data, context: "python backend rag")

    result = skill.execute({"query": "python"})

    assert result.success is True
    assert result.output == {"refined_query": "python backend rag"}


def test_query_refine_callable_returning_dict_preserves_reason_and_metadata():
    skill = QueryRefineSkill(
        refine_callable=lambda input_data, context: {
            "refined_query": "python langgraph agent",
            "reason": "broadened skill terms",
            "metadata": {"strategy": "broaden"},
        }
    )

    result = skill.execute({"query": "python agent"})

    assert result.success is True
    assert result.output == {
        "refined_query": "python langgraph agent",
        "reason": "broadened skill terms",
        "metadata": {"strategy": "broaden"},
    }


def test_query_refine_callable_returning_dict_without_refined_query_fails():
    skill = QueryRefineSkill(refine_callable=lambda input_data, context: {"reason": "missing output"})

    result = skill.execute({"query": "python"})

    assert result.success is False
    assert "refined_query" in result.error


def test_query_refine_callable_exception_is_wrapped_as_failed_result():
    def fail(input_data, context):
        raise RuntimeError("refine failed")

    result = QueryRefineSkill(refine_callable=fail).execute({"query": "python"})

    assert result.success is False
    assert result.error == "refine failed"


def test_query_refine_skill_can_register_to_skill_registry():
    skill = QueryRefineSkill(refine_callable=lambda input_data, context: "refined")
    registry = make_registry(skill)

    assert registry.get("query_refine") is skill


def test_query_refine_skill_executor_records_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    registry = make_registry(QueryRefineSkill(refine_callable=lambda input_data, context: "python rag"))
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("query_refine", {"query": "python"}, context=context)

    assert result.success is True
    assert result.output == {"refined_query": "python rag"}
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]
    assert store.list_events_by_task(task.task_id)[-1].payload["skill_name"] == "query_refine"


def test_query_refine_skill_eval_runner_can_run_fixture_case():
    registry = make_registry(QueryRefineSkill(refine_callable=lambda input_data, context: "python langgraph"))
    runner = SkillEvalRunner(registry)

    result = runner.run_case(
        SkillEvalCase(
            case_id="query_refine_case",
            skill_name="query_refine",
            input_data={"query": "python"},
            expected_output={"refined_query": "python langgraph"},
        )
    )

    assert result.passed is True


def test_query_refine_skill_node_adapter_maps_fake_state():
    registry = make_registry(
        QueryRefineSkill(
            refine_callable=lambda input_data, context: {
                "refined_query": f"{input_data['query']} langgraph",
                "reason": input_data["context"],
            }
        )
    )
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="query_refine",
        input_mapper=lambda state: {
            "query": state["extracted_jd"]["search_query"],
            "context": state["refinement_advice"],
        },
        output_mapper=lambda result, state: {
            "extracted_jd": {
                **state["extracted_jd"],
                "search_query": result.output["refined_query"],
            },
            "refine_reason": result.output.get("reason", ""),
        },
    )

    update = adapter(
        {
            "extracted_jd": {"search_query": "python"},
            "refinement_advice": "broaden query",
        }
    )

    assert update["extracted_jd"]["search_query"] == "python langgraph"
    assert update["refine_reason"] == "broaden query"
    assert update["skill_execution_metadata"]["skill_name"] == "query_refine"


def test_importing_agent_adapters_does_not_import_real_agent_or_retrieval_modules(monkeypatch):
    block_real_agent_and_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.agent_adapters", None)
    sys.modules.pop("src.agents.refiner", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    module = importlib.import_module("src.skills.agent_adapters")
    skill = module.QueryRefineSkill(refine_callable=lambda input_data, context: "safe query")
    result = skill.execute({"query": "safe"})

    assert result.success is True
    assert "src.agents.refiner" not in sys.modules
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_phase3e_does_not_modify_real_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "query_refine" not in graph_source
