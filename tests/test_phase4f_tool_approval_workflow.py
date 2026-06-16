import builtins

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    BaseTool,
    InMemoryToolApprovalStore,
    SandboxPolicy,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolAuditReporter,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolPermissionPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class WriteNoteFakeTool(BaseTool):
    spec = ToolSpec(
        name="write_note_fake",
        version="v1",
        permissions_required=["note:write"],
        side_effects="write",
    )

    def __init__(self):
        super().__init__()
        self.run_count = 0

    def run(self, input_data, context=None):
        self.run_count += 1
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"written": True},
        )


class ExternalSearchFakeTool(WriteNoteFakeTool):
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


def make_context(tmp_path, permissions):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4F"})
    task = TaskManager(store).create_task(session.session_id, jd_text="approval", thread_id="thread-approval")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="workflow",
        caller_name="local_fake_tool_workflow",
        permissions=list(permissions),
    )
    return store, task, context


def make_executor(tool, store=None, approval_store=None, sandbox_policy=None):
    return ToolExecutor(
        make_registry(tool),
        recorder=ToolExecutionRecorder(store) if store is not None else None,
        permission_policy=ToolPermissionPolicy(),
        sandbox_policy=sandbox_policy,
        approval_store=approval_store,
    )


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_tool_approval_request_can_create():
    request = ToolApprovalRequest(tool_name="write_note_fake", tool_version="v1", side_effects="write")

    assert request.status == "pending"
    assert request.approval_id.startswith("tool_approval_")


def test_tool_approval_decision_can_create():
    decision = ToolApprovalDecision(approval_id="approval-1", approved=True, decided_by="reviewer")

    assert decision.approved is True
    assert decision.to_dict()["decided_by"] == "reviewer"


def test_in_memory_approval_store_creates_and_reads_request():
    store = InMemoryToolApprovalStore()
    request = store.create_request(ToolApprovalRequest(task_id="task-1", tool_name="write_note_fake"))

    assert store.get_request(request.approval_id) is request
    assert store.list_requests_by_task("task-1") == [request]
    assert store.list_pending_requests("task-1") == [request]


def test_approval_decision_is_recorded_and_updates_request_status():
    store = InMemoryToolApprovalStore()
    request = store.create_request(ToolApprovalRequest(tool_name="write_note_fake"))
    decision = ToolApprovalDecision(approval_id=request.approval_id, approved=False, decided_by="reviewer")

    store.record_decision(decision)

    assert store.get_decision(request.approval_id) is decision
    assert store.get_request(request.approval_id).status == "rejected"
    assert store.list_pending_requests() == []


def test_requires_approval_does_not_run_tool_and_creates_request(tmp_path):
    runtime_store, _, context = make_context(tmp_path, ["note:write"])
    approval_store = InMemoryToolApprovalStore()
    tool = WriteNoteFakeTool()
    executor = make_executor(tool, runtime_store, approval_store)

    result = executor.execute("write_note_fake", {"note": "safe summary only"}, context=context)

    assert result.success is False
    assert result.error == "approval required"
    assert tool.run_count == 0
    assert result.metadata["approval_id"]
    request = approval_store.get_request(result.metadata["approval_id"])
    assert request.status == "pending"
    assert request.input_summary == {"type": "dict", "keys": ["note"], "size": 1}


def test_approval_required_runtime_event_includes_approval_id(tmp_path):
    runtime_store, task, context = make_context(tmp_path, ["note:write"])
    executor = make_executor(WriteNoteFakeTool(), runtime_store, InMemoryToolApprovalStore())

    result = executor.execute("write_note_fake", {"note": "hello"}, context=context)
    payload = runtime_store.list_events_by_task(task.task_id)[-1].payload

    assert event_types(runtime_store, task.task_id)[-1] == "tool_approval_required"
    assert payload["approval_id"] == result.metadata["approval_id"]
    assert payload["approval_status"] == "pending"


def test_approved_decision_allows_retry_to_execute_tool(tmp_path):
    runtime_store, task, context = make_context(tmp_path, ["note:write"])
    approval_store = InMemoryToolApprovalStore()
    tool = WriteNoteFakeTool()
    executor = make_executor(tool, runtime_store, approval_store)
    waiting = executor.execute("write_note_fake", {"note": "hello"}, context=context)
    approval_store.record_decision(
        ToolApprovalDecision(approval_id=waiting.metadata["approval_id"], approved=True, decided_by="reviewer")
    )

    result = executor.execute(
        "write_note_fake",
        {"note": "hello"},
        context=context,
        approval_id=waiting.metadata["approval_id"],
    )

    assert result.success is True
    assert result.metadata["approval_status"] == "approved"
    assert tool.run_count == 1
    assert event_types(runtime_store, task.task_id)[-3:] == [
        "tool_approval_granted",
        "tool_started",
        "tool_completed",
    ]


def test_rejected_decision_does_not_execute_tool(tmp_path):
    runtime_store, task, context = make_context(tmp_path, ["note:write"])
    approval_store = InMemoryToolApprovalStore()
    tool = WriteNoteFakeTool()
    executor = make_executor(tool, runtime_store, approval_store)
    waiting = executor.execute("write_note_fake", {"note": "hello"}, context=context)
    approval_store.record_decision(
        ToolApprovalDecision(approval_id=waiting.metadata["approval_id"], approved=False, decided_by="reviewer")
    )

    result = executor.execute(
        "write_note_fake",
        {"note": "hello"},
        context=context,
        approval_id=waiting.metadata["approval_id"],
    )

    assert result.success is False
    assert result.error == "approval rejected"
    assert tool.run_count == 0
    assert event_types(runtime_store, task.task_id)[-1] == "tool_approval_rejected"


def test_approved_decision_cannot_bypass_sandbox_denial(tmp_path):
    runtime_store, task, context = make_context(tmp_path, ["search:external"])
    approval_store = InMemoryToolApprovalStore()
    tool = ExternalSearchFakeTool()
    executor = make_executor(tool, runtime_store, approval_store, sandbox_policy=SandboxPolicy())
    waiting = executor.execute("external_search_fake", {"query": "local fake"}, context=context)
    approval_store.record_decision(
        ToolApprovalDecision(approval_id=waiting.metadata["approval_id"], approved=True, decided_by="reviewer")
    )

    result = executor.execute(
        "external_search_fake",
        {"query": "local fake"},
        context=context,
        approval_id=waiting.metadata["approval_id"],
    )

    assert result.success is False
    assert result.metadata["sandbox_decision"]["status"] == "denied"
    assert tool.run_count == 0
    assert event_types(runtime_store, task.task_id)[-2:] == ["tool_approval_granted", "tool_sandbox_denied"]


def test_approved_decision_cannot_bypass_missing_base_permission(tmp_path):
    _, _, context = make_context(tmp_path, [])
    approval_store = InMemoryToolApprovalStore()
    tool = WriteNoteFakeTool()
    request = approval_store.create_request(
        ToolApprovalRequest(
            task_id=context.task_id,
            tool_name=tool.spec.name,
            tool_version=tool.spec.version,
            side_effects=tool.spec.side_effects,
        )
    )
    approval_store.record_decision(ToolApprovalDecision(approval_id=request.approval_id, approved=True))
    executor = make_executor(tool, approval_store=approval_store)

    result = executor.execute("write_note_fake", {"note": "hello"}, context=context, approval_id=request.approval_id)

    assert result.success is False
    assert result.metadata["permission_decision"]["status"] == "denied"
    assert tool.run_count == 0


def test_audit_report_counts_required_granted_and_rejected_events(tmp_path):
    runtime_store, task, context = make_context(tmp_path, ["note:write"])
    approval_store = InMemoryToolApprovalStore()
    tool = WriteNoteFakeTool()
    executor = make_executor(tool, runtime_store, approval_store)

    approved_request = executor.execute("write_note_fake", {"note": "one"}, context=context)
    approval_store.record_decision(ToolApprovalDecision(approved_request.metadata["approval_id"], approved=True))
    executor.execute("write_note_fake", {"note": "one"}, context=context, approval_id=approved_request.metadata["approval_id"])

    rejected_request = executor.execute("write_note_fake", {"note": "two"}, context=context)
    approval_store.record_decision(ToolApprovalDecision(rejected_request.metadata["approval_id"], approved=False))
    executor.execute("write_note_fake", {"note": "two"}, context=context, approval_id=rejected_request.metadata["approval_id"])

    report = ToolAuditReporter.from_runtime_store(runtime_store, task.task_id)

    assert report.approval_required_count == 2
    assert report.approval_granted_count == 1
    assert report.approval_rejected_count == 1
    assert "Approval granted: 1" in ToolAuditReporter.format_text(report)


def test_phase4f_does_not_import_mcp_or_real_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = ("llama_index", "chromadb", "src.agents.retriever", "src.services.retriever", "mcp")
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4F test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    approval_store = InMemoryToolApprovalStore()
    result = make_executor(WriteNoteFakeTool(), approval_store=approval_store).execute(
        "write_note_fake",
        {"note": "hello"},
        context=ToolExecutionContext(permissions=["note:write"]),
    )

    assert result.error == "approval required"
    assert blocked == []
