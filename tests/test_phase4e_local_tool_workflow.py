import builtins

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    BaseTool,
    CandidateLookupFakeTool,
    EchoTool,
    LocalToolWorkflow,
    ResumeTextParseFakeTool,
    SandboxPolicy,
    ToolAuditReporter,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolPermissionPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolWorkflowResult,
    ToolWorkflowStep,
)


class StateCaptureFakeTool(BaseTool):
    spec = ToolSpec(name="state_capture_fake", version="v1", side_effects="none")

    def __init__(self):
        super().__init__()
        self.received = None

    def run(self, input_data, context=None):
        self.received = dict(input_data)
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"received_keyword_count": len(input_data.get("keywords", []))},
        )


class NetworkDeclaredFakeTool(BaseTool):
    spec = ToolSpec(
        name="network_declared_fake",
        version="v1",
        side_effects="none",
        metadata={"requested_capabilities": ["network"]},
    )

    def __init__(self):
        super().__init__()
        self.run_count = 0

    def run(self, input_data, context=None):
        self.run_count += 1
        return ToolResult(tool_name=self.spec.name, version=self.spec.version, success=True, output={"ok": True})


def make_runtime_context(tmp_path, permissions=None):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4E"})
    task = TaskManager(store).create_task(session.session_id, jd_text="fake tool workflow", thread_id="thread-tools")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="workflow",
        caller_name="local_fake_tool_workflow",
        permissions=list(permissions or []),
    )
    return store, task, context


def make_registry(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_workflow(tmp_path, tools, steps, permissions=None):
    store, task, context = make_runtime_context(tmp_path, permissions=permissions)
    registry = make_registry(*tools)
    executor = ToolExecutor(
        registry,
        recorder=ToolExecutionRecorder(store),
        permission_policy=ToolPermissionPolicy(),
        sandbox_policy=SandboxPolicy(),
    )
    workflow = LocalToolWorkflow(registry, executor, context=context, steps=steps)
    return workflow, store, task


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_tool_workflow_step_and_result_can_create():
    step = ToolWorkflowStep("echo_tool", lambda state: {"value": state["value"]}, "echo")
    result = ToolWorkflowResult(status="completed", success=True)

    assert step.output_key == "echo"
    assert result.success is True


def test_local_tool_workflow_executes_multiple_fake_tools_in_order(tmp_path):
    steps = [
        ToolWorkflowStep("resume_text_parse_fake", lambda state: {"text": state["resume_text"]}, "parsed"),
        ToolWorkflowStep("candidate_lookup_fake", lambda state: {"skill": state["parsed"]["keywords"][0]}, "lookup"),
    ]
    workflow, store, task = make_workflow(
        tmp_path,
        [ResumeTextParseFakeTool(), CandidateLookupFakeTool()],
        steps,
        permissions=["candidate:read"],
    )

    result = workflow.run({"resume_text": "Python LangGraph"})

    assert result.status == "completed"
    assert result.success is True
    assert [item["tool_name"] for item in result.tool_results] == [
        "resume_text_parse_fake",
        "candidate_lookup_fake",
    ]
    assert event_types(store, task.task_id)[-4:] == [
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
    ]


def test_workflow_state_passes_previous_output_to_next_input_mapper(tmp_path):
    capture = StateCaptureFakeTool()
    steps = [
        ToolWorkflowStep("resume_text_parse_fake", lambda state: {"text": state["resume_text"]}, "parsed"),
        ToolWorkflowStep("state_capture_fake", lambda state: {"keywords": state["parsed"]["keywords"]}, "captured"),
    ]
    workflow, _, _ = make_workflow(tmp_path, [ResumeTextParseFakeTool(), capture], steps)

    result = workflow.run({"resume_text": "Python RAG"})

    assert result.success is True
    assert capture.received == {"keywords": ["Python", "RAG"]}


def test_permission_policy_allowed_workflow_is_completed(tmp_path):
    steps = [
        ToolWorkflowStep("candidate_lookup_fake", lambda state: {"skill": "Python"}, "candidates"),
        ToolWorkflowStep("echo_tool", lambda state: {"lookup_available": "candidates" in state}, "echo"),
    ]
    workflow, _, _ = make_workflow(
        tmp_path,
        [CandidateLookupFakeTool(), EchoTool()],
        steps,
        permissions=["candidate:read"],
    )

    result = workflow.run()

    assert result.status == "completed"
    assert result.success is True
    assert result.outputs["candidates"]["type"] == "dict"


def test_permission_denied_stops_following_steps_by_default(tmp_path):
    steps = [
        ToolWorkflowStep("candidate_lookup_fake", lambda state: {"skill": "Python"}, "candidates"),
        ToolWorkflowStep("echo_tool", lambda state: {"should_not_run": True}, "echo"),
    ]
    workflow, store, task = make_workflow(tmp_path, [CandidateLookupFakeTool(), EchoTool()], steps)

    result = workflow.run()

    assert result.status == "failed"
    assert result.success is False
    assert len(result.steps) == 1
    assert event_types(store, task.task_id)[-1:] == ["tool_denied"]


def test_sandbox_denied_stops_workflow_and_does_not_run_tool(tmp_path):
    network_tool = NetworkDeclaredFakeTool()
    steps = [
        ToolWorkflowStep("network_declared_fake", lambda state: {}, "network"),
        ToolWorkflowStep("echo_tool", lambda state: {"should_not_run": True}, "echo"),
    ]
    workflow, store, task = make_workflow(tmp_path, [network_tool, EchoTool()], steps)

    result = workflow.run()

    assert result.status == "failed"
    assert network_tool.run_count == 0
    assert event_types(store, task.task_id)[-1:] == ["tool_sandbox_denied"]


def test_runtime_timeline_records_multiple_tool_events(tmp_path):
    steps = [
        ToolWorkflowStep("echo_tool", lambda state: {"first": True}, "first"),
        ToolWorkflowStep("resume_text_parse_fake", lambda state: {"text": "Python"}, "parsed"),
    ]
    workflow, store, task = make_workflow(tmp_path, [EchoTool(), ResumeTextParseFakeTool()], steps)

    workflow.run()

    assert event_types(store, task.task_id) == [
        "task_created",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
    ]


def test_tool_audit_report_summarizes_workflow_timeline(tmp_path):
    steps = [
        ToolWorkflowStep("echo_tool", lambda state: {"first": True}, "first"),
        ToolWorkflowStep("resume_text_parse_fake", lambda state: {"text": "Python"}, "parsed"),
    ]
    workflow, store, task = make_workflow(tmp_path, [EchoTool(), ResumeTextParseFakeTool()], steps)

    workflow.run()
    report = ToolAuditReporter.from_runtime_store(store, task.task_id)

    assert report.started_count == 2
    assert report.completed_count == 2
    assert report.tools_called == ["echo_tool", "resume_text_parse_fake"]


def test_continue_on_failure_runs_later_safe_step_and_marks_partial(tmp_path):
    steps = [
        ToolWorkflowStep(
            "candidate_lookup_fake",
            lambda state: {"skill": "Python"},
            "candidates",
            continue_on_failure=True,
        ),
        ToolWorkflowStep("echo_tool", lambda state: {"continued": True}, "echo"),
    ]
    workflow, store, task = make_workflow(tmp_path, [CandidateLookupFakeTool(), EchoTool()], steps)

    result = workflow.run()
    report = ToolAuditReporter.from_runtime_store(store, task.task_id)

    assert result.status == "partial"
    assert result.success is False
    assert report.denied_count == 1
    assert report.completed_count == 1


def test_workflow_result_contains_summaries_not_sensitive_output_payload(tmp_path):
    secret = "sensitive-candidate-content-do-not-return"
    steps = [ToolWorkflowStep("echo_tool", lambda state: {"secret": secret}, "echo")]
    workflow, _, _ = make_workflow(tmp_path, [EchoTool()], steps)

    result = workflow.run()
    result_data = result.to_dict()

    assert result.outputs["echo"] == {"type": "dict", "keys": ["secret"], "size": 1}
    assert secret not in str(result_data)


def test_phase4e_does_not_import_mcp_or_real_external_dependencies(monkeypatch, tmp_path):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = ("llama_index", "chromadb", "src.agents.retriever", "src.services.retriever", "mcp")
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    steps = [ToolWorkflowStep("echo_tool", lambda state: {"ok": True}, "echo")]
    workflow, _, _ = make_workflow(tmp_path, [EchoTool()], steps)

    result = workflow.run()

    assert result.success is True
    assert blocked == []
