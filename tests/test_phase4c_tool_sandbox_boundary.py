import builtins

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    BaseTool,
    EchoTool,
    SandboxDecision,
    SandboxPolicy,
    SandboxProfile,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolPermissionPolicy,
    ToolRegistry,
    ToolResult,
    ToolSandboxContext,
    ToolSpec,
)


class CountingTool(BaseTool):
    spec = ToolSpec(name="counting_tool", version="v1")

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


def tool_with_metadata(name, metadata=None, side_effects="none", permissions_required=None):
    return CountingTool(
        spec=ToolSpec(
            name=name,
            version="v1",
            side_effects=side_effects,
            permissions_required=list(permissions_required or []),
            metadata=dict(metadata or {}),
        )
    )


def make_registry(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_runtime_context(tmp_path, permissions=None):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4C"})
    task = TaskManager(store).create_task(session.session_id, jd_text="tool sandbox", thread_id="thread-tool-sandbox")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="skill",
        caller_name="query_refine",
        permissions=list(permissions or []),
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_sandbox_profile_can_create():
    profile = SandboxProfile(
        profile_name="locked",
        allow_network=False,
        allow_file_read=True,
        allow_file_write=False,
        allow_subprocess=False,
        allowed_paths=["/tmp/safe"],
        blocked_paths=["/tmp/blocked"],
    )

    assert profile.profile_name == "locked"
    assert profile.allow_network is False
    assert profile.allowed_paths == ["/tmp/safe"]


def test_sandbox_decision_can_create():
    decision = SandboxDecision(
        allowed=False,
        status="denied",
        reason="network denied",
        violated_rules=["network_not_allowed"],
        profile_name="locked",
    )

    assert decision.allowed is False
    assert decision.to_dict()["violated_rules"] == ["network_not_allowed"]


def test_default_sandbox_policy_allows_safe_tool():
    decision = SandboxPolicy().evaluate(
        EchoTool.spec,
        ToolExecutionContext(),
        ToolSandboxContext(sandbox_profile=SandboxProfile(profile_name="locked")),
    )

    assert decision.allowed is True
    assert decision.status == "allowed"


def test_network_capability_denied_when_network_not_allowed():
    tool = tool_with_metadata("network_fake", metadata={"requested_capabilities": ["network"]})

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), ToolSandboxContext())

    assert decision.allowed is False
    assert "network_not_allowed" in decision.violated_rules


def test_file_write_capability_denied_when_file_write_not_allowed():
    tool = tool_with_metadata("file_write_fake", metadata={"requested_capabilities": ["file_write"]})

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), ToolSandboxContext())

    assert decision.allowed is False
    assert "file_write_not_allowed" in decision.violated_rules


def test_subprocess_capability_denied_when_subprocess_not_allowed():
    tool = tool_with_metadata("subprocess_fake", metadata={"requested_capabilities": ["subprocess"]})

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), ToolSandboxContext())

    assert decision.allowed is False
    assert "subprocess_not_allowed" in decision.violated_rules


def test_external_side_effects_denied_when_external_side_effects_not_allowed():
    tool = tool_with_metadata("external_fake", side_effects="external")

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), ToolSandboxContext())

    assert decision.allowed is False
    assert "external_side_effects_not_allowed" in decision.violated_rules


def test_blocked_paths_hit_is_denied(tmp_path):
    blocked = tmp_path / "blocked"
    requested = blocked / "secret.txt"
    tool = tool_with_metadata("path_fake", metadata={"requested_paths": [str(requested)]})
    context = ToolSandboxContext(
        sandbox_profile=SandboxProfile(blocked_paths=[str(blocked)]),
    )

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), context)

    assert decision.allowed is False
    assert "blocked_path_requested" in decision.violated_rules


def test_allowed_paths_nonempty_denies_path_outside_allowed_paths(tmp_path):
    allowed = tmp_path / "allowed"
    requested = tmp_path / "other" / "file.txt"
    tool = tool_with_metadata("path_fake", metadata={"requested_paths": [str(requested)]})
    context = ToolSandboxContext(
        sandbox_profile=SandboxProfile(allowed_paths=[str(allowed)]),
    )

    decision = SandboxPolicy().evaluate(tool.spec, ToolExecutionContext(), context)

    assert decision.allowed is False
    assert "path_not_in_allowed_paths" in decision.violated_rules


def test_tool_executor_sandbox_denied_does_not_call_tool_run():
    tool = tool_with_metadata("network_fake", metadata={"requested_capabilities": ["network"]})
    executor = ToolExecutor(make_registry(tool), sandbox_policy=SandboxPolicy())

    result = executor.execute("network_fake", {}, context=ToolExecutionContext())

    assert result.success is False
    assert result.metadata["sandbox_decision"]["status"] == "denied"
    assert tool.run_count == 0


def test_sandbox_denied_result_contains_sandbox_decision_metadata():
    tool = tool_with_metadata("file_write_fake", metadata={"requested_capabilities": ["file_write"]})

    result = ToolExecutor(make_registry(tool), sandbox_policy=SandboxPolicy()).execute(
        "file_write_fake",
        {},
        context=ToolExecutionContext(),
    )

    decision = result.metadata["sandbox_decision"]
    assert decision["status"] == "denied"
    assert "file_write_not_allowed" in decision["violated_rules"]


def test_recorder_writes_tool_sandbox_denied_event(tmp_path):
    store, task, context = make_runtime_context(tmp_path)
    tool = tool_with_metadata("network_fake", metadata={"requested_capabilities": ["network"]})
    executor = ToolExecutor(
        make_registry(tool),
        recorder=ToolExecutionRecorder(store),
        sandbox_policy=SandboxPolicy(),
    )

    result = executor.execute("network_fake", {}, context=context)

    assert result.success is False
    assert event_types(store, task.task_id) == ["task_created", "tool_sandbox_denied"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["sandbox_decision"]["status"] == "denied"
    assert payload["status"] == "sandbox_denied"


def test_permission_denied_short_circuits_before_sandbox_check():
    tool = tool_with_metadata(
        "permission_first_fake",
        metadata={"requested_capabilities": ["network"]},
        permissions_required=["tool:run"],
    )
    executor = ToolExecutor(
        make_registry(tool),
        permission_policy=ToolPermissionPolicy(),
        sandbox_policy=SandboxPolicy(),
    )

    result = executor.execute("permission_first_fake", {}, context=ToolExecutionContext(permissions=[]))

    assert result.success is False
    assert result.metadata["permission_decision"]["status"] == "denied"
    assert "sandbox_decision" not in result.metadata
    assert tool.run_count == 0


def test_allowed_sandbox_executes_tool():
    tool = tool_with_metadata("safe_fake")
    executor = ToolExecutor(make_registry(tool), sandbox_policy=SandboxPolicy())

    result = executor.execute("safe_fake", {}, context=ToolExecutionContext())

    assert result.success is True
    assert result.output == {"run_count": 1}
    assert tool.run_count == 1


def test_allowed_execution_events_include_sandbox_decision_summary(tmp_path):
    store, task, context = make_runtime_context(tmp_path)
    tool = tool_with_metadata("safe_fake")
    executor = ToolExecutor(
        make_registry(tool),
        recorder=ToolExecutionRecorder(store),
        sandbox_policy=SandboxPolicy(),
    )

    result = executor.execute("safe_fake", {}, context=context)

    assert result.success is True
    assert event_types(store, task.task_id) == ["task_created", "tool_started", "tool_completed"]
    payload = store.list_events_by_task(task.task_id)[-1].payload
    assert payload["sandbox_decision"]["status"] == "allowed"
    assert payload["sandbox_decision"]["profile_name"] == "default"


def test_phase4c_does_not_import_mcp_or_real_external_dependencies(monkeypatch):
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
            raise ModuleNotFoundError(f"blocked import in Phase4C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = ToolExecutor(make_registry(EchoTool()), sandbox_policy=SandboxPolicy()).execute(
        "echo_tool",
        {"ok": True},
        context=ToolExecutionContext(),
    )

    assert result.success is True
    assert blocked == []
