import builtins
import sys

from src.runtime import RuntimeRunner, SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    BaseSkill,
    CandidateMatchStubSkill,
    EchoSkill,
    KeywordExtractSkill,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillNodeAdapter,
    SkillRegistry,
    SkillSpec,
)


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase3C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FailingSkill(BaseSkill):
    spec = SkillSpec(name="failing_execution_demo", version="v1")

    def run(self, input_data, context=None):
        raise RuntimeError("skill execution failed")


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    sessions = SessionManager(store)
    tasks = TaskManager(store)
    session = sessions.create_session(metadata={"phase": "3C"})
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-skill")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3c-test"},
    )
    return store, session, task, context


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_skill_executor_can_execute_echo_skill():
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry)

    result = executor.execute("echo", {"message": "hello"})

    assert result.success is True
    assert result.output == {"message": "hello"}


def test_skill_executor_without_recorder_still_executes(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=None)

    result = executor.execute("echo", {"message": "no recorder"}, context=context)

    assert result.success is True
    assert event_types(store, task.task_id) == ["task_created"]


def test_skill_executor_with_recorder_and_no_task_id_does_not_write_events(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("echo", {"message": "no task"}, context=SkillExecutionContext())

    assert result.success is True
    assert store.list_events() == []


def test_skill_executor_with_recorder_writes_started_and_completed_events(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("echo", {"message": "record me"}, context=context)

    events = store.list_events_by_task(task.task_id)
    assert result.success is True
    assert [event.event_type for event in events] == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]


def test_failing_skill_writes_skill_failed_event(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(FailingSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("failing_execution_demo", {"value": 1}, context=context)

    events = store.list_events_by_task(task.task_id)
    assert result.success is False
    assert [event.event_type for event in events] == [
        "task_created",
        "skill_started",
        "skill_failed",
    ]
    assert events[-1].payload["error"] == "skill execution failed"


def test_event_payload_contains_skill_identity_status_and_duration(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    executor.execute("echo", {"message": "payload"}, context=context)

    completed = store.list_events_by_task(task.task_id)[-1].payload
    assert completed["skill_name"] == "echo"
    assert completed["skill_version"] == "v1"
    assert completed["status"] == "completed"
    assert completed["duration_ms"] is not None
    assert completed["duration_ms"] >= 0


def test_input_and_output_summary_do_not_store_full_complex_objects(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))
    complex_input = {
        "candidate": {"name": "Alice", "private_note": "do not store this"},
        "scores": [1, 2, 3],
        "long_text": "secret" * 50,
    }

    executor.execute("echo", complex_input, context=context)

    completed = store.list_events_by_task(task.task_id)[-1].payload
    assert completed["input_summary"] == {
        "type": "dict",
        "keys": ["candidate", "long_text", "scores"],
        "size": 3,
    }
    assert completed["output_summary"] == {
        "type": "dict",
        "keys": ["candidate", "long_text", "scores"],
        "size": 3,
    }
    assert "private_note" not in str(completed["input_summary"])
    assert "secretsecret" not in str(completed["output_summary"])


def test_task_timeline_can_read_skill_events(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))
    runner = RuntimeRunner(store, graph_factory=lambda: None)

    executor.execute("echo", {"message": "timeline"}, context=context)
    timeline = runner.get_task_timeline(task.task_id)

    assert [event.event_type for event in timeline] == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]


def test_fake_skill_workflow_records_multiple_skill_events_on_one_task(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(KeywordExtractSkill(), CandidateMatchStubSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    keyword_result = executor.execute(
        "keyword_extract_stub",
        {"text": "Need Python and LangGraph"},
        context=context,
    )
    match_result = executor.execute(
        "candidate_match_stub",
        {
            "required_skills": keyword_result.output["keywords"],
            "candidate_skills": ["Python", "LangGraph"],
        },
        context=context,
    )

    assert match_result.output["score"] == 100.0
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
        "skill_started",
        "skill_completed",
    ]


def test_skill_node_adapter_can_record_with_skill_executor(tmp_path):
    store, _, task, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="echo",
        input_mapper=lambda state: {"message": state["message"]},
        output_mapper=lambda result, state: {"echo_output": result.output},
        context_builder=lambda state: context,
        skill_executor=executor,
    )

    update = adapter({"message": "adapter recording"})

    assert update["echo_output"] == {"message": "adapter recording"}
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]


def test_phase3c_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store, _, _, context = make_runtime(tmp_path)
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("echo", {"safe": True}, context=context)

    assert result.success is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
