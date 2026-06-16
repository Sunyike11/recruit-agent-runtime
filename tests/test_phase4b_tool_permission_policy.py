import builtins

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    BaseTool,
    CandidateLookupFakeTool,
    EchoTool,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolPermissionDecision,
    ToolPermissionPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_tool_execution_contract,
)


class CountingTool(BaseTool):
    spec = ToolSpec(
        name="counting_tool",
        version="v1",
        permissions_required=["count:run"],
        side_effects="read",
    )

    def __init__(self, spec=None):
        super().__init__(spec=spec)
        self.run_count = 0

    def run(self, input_data, context=None):
        self.run_count += 1
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"run_count": self.run_count},
        )


class WriteNoteFakeTool(CountingTool):
    spec = ToolSpec(
        name="write_note_fake",
        version="v1",
        permissions_required=["note:write"],
        side_effects="write",
    )


class ExternalSearchFakeTool(CountingTool):
    spec = ToolSpec(
        name="external_search_fake",
        version="v1",
        permissions_required=["search:external"],
        side_effects="external",
    )


def make_registry(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_runtime_context(tmp_path, permissions=None):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4B"})
    task = TaskManager(store).create_task(session.session_id, jd_text="tool policy", thread_id="thread-tool-policy")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="skill",
        caller_name="candidate_match",
        permissions=list(permissions or []),
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_tool_permission_decision_can_create():
    decision = ToolPermissionDecision(
        allowed=False,
        status="denied",
        reason="missing required permissions",
        missing_permissions=["candidate:read"],
        required_permissions=["candidate:read"],
        side_effects="read",
    )

    assert decision.allowed is False
    assert decision.status == "denied"
    assert decision.to_dict()["missing_permissions"] == ["candidate:read"]


def test_default_policy_allows_when_permissions_are_satisfied():
    policy = ToolPermissionPolicy()
    decision = policy.evaluate(
        CandidateLookupFakeTool.spec,
        ToolExecutionContext(permissions=["candidate:read"]),
    )

    assert decision.allowed is True
    assert decision.status == "allowed"


def test_default_policy_denies_when_required_permission_is_missing():
    decision = ToolPermissionPolicy().evaluate(
        CandidateLookupFakeTool.spec,
        ToolExecutionContext(permissions=[]),
    )

    assert decision.allowed is False
    assert decision.status == "denied"
    assert decision.missing_permissions == ["candidate:read"]


def test_write_side_effects_require_allow_write_permission():
    decision = ToolPermissionPolicy().evaluate(
        WriteNoteFakeTool.spec,
        ToolExecutionContext(permissions=["note:write"]),
    )

    assert decision.allowed is False
    assert decision.status == "requires_approval"
    assert decision.metadata["approval_permission"] == "allow_write"


def test_external_side_effects_require_allow_external_permission():
    decision = ToolPermissionPolicy().evaluate(
        ExternalSearchFakeTool.spec,
        ToolExecutionContext(permissions=["search:external"]),
    )

    assert decision.allowed is False
    assert decision.status == "requires_approval"
    assert decision.metadata["approval_permission"] == "allow_external"


def test_tool_executor_denied_does_not_call_tool_run():
    tool = CountingTool()
    registry = make_registry(tool)
    executor = ToolExecutor(registry, permission_policy=ToolPermissionPolicy())

    result = executor.execute("counting_tool", {"value": 1}, context=ToolExecutionContext(permissions=[]))

    assert result.success is False
    assert result.metadata["permission_decision"]["status"] == "denied"
    assert tool.run_count == 0


def test_tool_executor_requires_approval_does_not_call_tool_run():
    tool = WriteNoteFakeTool()
    registry = make_registry(tool)
    executor = ToolExecutor(registry, permission_policy=ToolPermissionPolicy())

    result = executor.execute(
        "write_note_fake",
        {"note": "hello"},
        context=ToolExecutionContext(permissions=["note:write"]),
    )

    assert result.success is False
    assert result.error == "approval required"
    assert result.metadata["permission_decision"]["status"] == "requires_approval"
    assert tool.run_count == 0


def test_denied_result_contains_permission_decision_metadata():
    result = ToolExecutor(
        make_registry(CandidateLookupFakeTool()),
        permission_policy=ToolPermissionPolicy(),
    ).execute("candidate_lookup_fake", {"skill": "Python"}, context=ToolExecutionContext())

    decision = result.metadata["permission_decision"]
    assert decision["status"] == "denied"
    assert decision["missing_permissions"] == ["candidate:read"]
    assert result.error == "missing required permissions"


def test_allowed_execution_still_returns_tool_result():
    tool = CountingTool()
    registry = make_registry(tool)
    executor = ToolExecutor(registry, permission_policy=ToolPermissionPolicy())

    result = executor.execute(
        "counting_tool",
        {"value": 1},
        context=ToolExecutionContext(permissions=["count:run"]),
    )

    assert result.success is True
    assert result.output == {"run_count": 1}
    assert tool.run_count == 1


def test_recorder_writes_tool_denied_event(tmp_path):
    store, task, context = make_runtime_context(tmp_path, permissions=[])
    executor = ToolExecutor(
        make_registry(CandidateLookupFakeTool()),
        recorder=ToolExecutionRecorder(store),
        permission_policy=ToolPermissionPolicy(),
    )

    result = executor.execute("candidate_lookup_fake", {"skill": "Python"}, context=context)

    assert result.success is False
    assert event_types(store, task.task_id) == ["task_created", "tool_denied"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["permission_decision"]["status"] == "denied"
    assert payload["status"] == "denied"


def test_recorder_writes_tool_approval_required_event(tmp_path):
    store, task, context = make_runtime_context(tmp_path, permissions=["note:write"])
    executor = ToolExecutor(
        make_registry(WriteNoteFakeTool()),
        recorder=ToolExecutionRecorder(store),
        permission_policy=ToolPermissionPolicy(),
    )

    result = executor.execute("write_note_fake", {"note": "hello"}, context=context)

    assert result.success is False
    assert event_types(store, task.task_id) == ["task_created", "tool_approval_required"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["permission_decision"]["status"] == "requires_approval"
    assert payload["error"] == "approval required"


def test_allowed_execution_events_include_permission_decision_summary(tmp_path):
    store, task, context = make_runtime_context(tmp_path, permissions=["candidate:read"])
    executor = ToolExecutor(
        make_registry(CandidateLookupFakeTool()),
        recorder=ToolExecutionRecorder(store),
        permission_policy=ToolPermissionPolicy(),
    )

    result = executor.execute("candidate_lookup_fake", {"skill": "Python"}, context=context)

    assert result.success is True
    assert event_types(store, task.task_id) == ["task_created", "tool_started", "tool_completed"]
    completed_payload = store.list_events_by_task(task.task_id)[-1].payload
    assert completed_payload["permission_decision"]["status"] == "allowed"
    assert completed_payload["permission_decision"]["required_permissions"] == ["candidate:read"]


def test_tool_execution_contract_can_build_from_spec():
    contract = build_tool_execution_contract(WriteNoteFakeTool.spec)

    assert contract.tool_name == "write_note_fake"
    assert contract.tool_version == "v1"
    assert contract.required_permissions == ["note:write"]
    assert contract.side_effects == "write"
    assert contract.approval_required is True


def test_phase4b_does_not_import_mcp_or_real_external_dependencies(monkeypatch):
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
            raise ModuleNotFoundError(f"blocked import in Phase4B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = ToolExecutor(make_registry(EchoTool()), permission_policy=ToolPermissionPolicy()).execute(
        "echo_tool",
        {"ok": True},
        context=ToolExecutionContext(),
    )

    assert result.success is True
    assert blocked == []
