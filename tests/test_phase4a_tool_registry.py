import builtins

import pytest

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    BaseTool,
    CandidateLookupFakeTool,
    EchoTool,
    ResumeTextParseFakeTool,
    ToolAlreadyRegisteredError,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class FailingTool(BaseTool):
    spec = ToolSpec(name="failing_tool", version="v1")

    def run(self, input_data, context=None):
        raise RuntimeError("tool failed")


def make_registry(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_context(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4A"})
    task = TaskManager(store).create_task(session.session_id, jd_text="tool test", thread_id="thread-tool")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="skill",
        caller_name="planner_extract",
        permissions=["candidate:read"],
        metadata={"source": "phase4a-test"},
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_tool_spec_can_create():
    spec = ToolSpec(
        name="demo_tool",
        version="v1",
        description="demo",
        category="demo",
        permissions_required=["demo:read"],
        side_effects="read",
        timeout_seconds=5,
        metadata={"phase": "4A"},
    )

    assert spec.name == "demo_tool"
    assert spec.permissions_required == ["demo:read"]
    assert spec.side_effects == "read"


def test_tool_result_can_create():
    result = ToolResult(tool_name="demo_tool", version="v1", success=True, output={"ok": True})

    assert result.success is True
    assert result.output == {"ok": True}


def test_echo_tool_executes_successfully():
    result = EchoTool().execute({"message": "hello"})

    assert result.success is True
    assert result.output == {"message": "hello"}


def test_base_tool_execute_catches_exception_as_failed_result():
    result = FailingTool().execute({"value": 1})

    assert result.success is False
    assert result.error == "tool failed"


def test_tool_registry_register_get_list_unregister():
    tool = EchoTool()
    registry = make_registry(tool)

    assert registry.get("echo_tool") is tool
    assert registry.list_tools() == [tool]
    registry.unregister("echo_tool")
    assert registry.list_tools() == []


def test_tool_registry_same_name_different_versions_can_coexist():
    v1 = EchoTool()
    v2 = EchoTool(spec=ToolSpec(name="echo_tool", version="v2"))
    registry = make_registry(v1, v2)

    assert registry.get("echo_tool", version="v1") is v1
    assert registry.get("echo_tool", version="v2") is v2
    assert registry.get("echo_tool") is v2


def test_tool_registry_duplicate_name_version_raises_by_default():
    registry = make_registry(EchoTool())

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(EchoTool())


def test_tool_executor_executes_tool():
    registry = make_registry(CandidateLookupFakeTool())
    executor = ToolExecutor(registry)

    result = executor.execute("candidate_lookup_fake", {"skill": "Python"})

    assert result.success is True
    assert result.output["candidates"][0]["candidate_id"] == "candidate_fake_1"


def test_tool_executor_without_recorder_still_executes():
    registry = make_registry(ResumeTextParseFakeTool())
    executor = ToolExecutor(registry)

    result = executor.execute("resume_text_parse_fake", {"text": "Python LangGraph RAG"})

    assert result.success is True
    assert result.output["keywords"] == ["Python", "LangGraph", "RAG"]


def test_tool_executor_with_recorder_writes_started_and_completed_events(tmp_path):
    store, task, context = make_context(tmp_path)
    registry = make_registry(EchoTool())
    executor = ToolExecutor(registry, recorder=ToolExecutionRecorder(store))

    result = executor.execute("echo_tool", {"message": "hello"}, context=context)

    assert result.success is True
    assert event_types(store, task.task_id) == ["task_created", "tool_started", "tool_completed"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["tool_name"] == "echo_tool"
    assert payload["version"] == "v1"
    assert payload["status"] == "completed"
    assert isinstance(payload["duration_ms"], float)
    assert payload["caller_type"] == "skill"
    assert payload["caller_name"] == "planner_extract"


def test_failing_tool_writes_tool_failed_event(tmp_path):
    store, task, context = make_context(tmp_path)
    registry = make_registry(FailingTool())
    executor = ToolExecutor(registry, recorder=ToolExecutionRecorder(store))

    result = executor.execute("failing_tool", {"value": 1}, context=context)

    assert result.success is False
    assert event_types(store, task.task_id) == ["task_created", "tool_started", "tool_failed"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["tool_name"] == "failing_tool"
    assert payload["status"] == "failed"
    assert payload["error"] == "tool failed"


def test_tool_event_payload_contains_required_metadata(tmp_path):
    store, task, context = make_context(tmp_path)
    registry = make_registry(CandidateLookupFakeTool())
    executor = ToolExecutor(registry, recorder=ToolExecutionRecorder(store))

    executor.execute("candidate_lookup_fake", {"skill": "Python"}, context=context)

    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["tool_name"] == "candidate_lookup_fake"
    assert payload["version"] == "v1"
    assert payload["caller_type"] == "skill"
    assert payload["caller_name"] == "planner_extract"
    assert payload["permissions_required"] == ["candidate:read"]


def test_input_and_output_summary_do_not_store_complete_complex_objects(tmp_path):
    store, task, context = make_context(tmp_path)
    registry = make_registry(EchoTool())
    executor = ToolExecutor(registry, recorder=ToolExecutionRecorder(store))
    complex_input = {
        "secret": "x" * 200,
        "items": [{"large": "payload"}],
    }

    executor.execute("echo_tool", complex_input, context=context)

    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["input_summary"] == {"type": "dict", "keys": ["items", "secret"], "size": 2}
    assert payload["output_summary"] == {"type": "dict", "keys": ["items", "secret"], "size": 2}
    assert "x" * 100 not in str(payload)


def test_phase4a_tools_do_not_import_retrieval_or_mcp(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
            "mcp",
        )
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = EchoTool().execute({"ok": True})

    assert result.success is True
    assert blocked == []
