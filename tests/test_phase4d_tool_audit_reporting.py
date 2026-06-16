import builtins

from src.runtime import Event, SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    EchoTool,
    SandboxPolicy,
    ToolAuditEvent,
    ToolAuditReport,
    ToolAuditReporter,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolRegistry,
)


def make_runtime_task(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4D"})
    task = TaskManager(store).create_task(session.session_id, jd_text="tool audit", thread_id="thread-tool-audit")
    return store, session, task


def append_tool_event(store, session, task, event_type, **payload):
    default_payload = {
        "tool_name": "echo_tool",
        "version": "v1",
        "status": event_type.replace("tool_", ""),
        "caller_type": "skill",
        "caller_name": "planner_extract",
        "duration_ms": 1.25,
    }
    default_payload.update(payload)
    return store.append_event(
        event_type,
        session_id=session.session_id,
        task_id=task.task_id,
        payload=default_payload,
    )


def test_tool_audit_event_parses_runtime_event_fields(tmp_path):
    store, session, task = make_runtime_task(tmp_path)
    event = append_tool_event(
        store,
        session,
        task,
        "tool_completed",
        permission_decision={"status": "allowed"},
        sandbox_decision={"status": "allowed"},
        metadata={"phase": "4D"},
    )

    audit_event = ToolAuditEvent.from_runtime_event(event)

    assert audit_event.tool_name == "echo_tool"
    assert audit_event.tool_version == "v1"
    assert audit_event.permission_status == "allowed"
    assert audit_event.sandbox_status == "allowed"
    assert audit_event.duration_ms == 1.25


def test_tool_audit_report_can_create():
    report = ToolAuditReport(task_id="task-1", total_tool_events=1, completed_count=1, tools_called=["echo_tool"])

    assert report.task_id == "task-1"
    assert report.completed_count == 1


def test_reporter_counts_all_supported_event_outcomes(tmp_path):
    store, session, task = make_runtime_task(tmp_path)
    for event_type in (
        "tool_started",
        "tool_completed",
        "tool_failed",
        "tool_denied",
        "tool_approval_required",
        "tool_sandbox_denied",
    ):
        append_tool_event(store, session, task, event_type)

    report = ToolAuditReporter.from_events(store.list_events_by_task(task.task_id))

    assert report.total_tool_events == 6
    assert report.started_count == 1
    assert report.completed_count == 1
    assert report.failed_count == 1
    assert report.denied_count == 1
    assert report.approval_required_count == 1
    assert report.sandbox_denied_count == 1


def test_reporter_from_raw_events_ignores_non_tool_events():
    events = [
        Event(event_id="1", event_type="task_created", task_id="task-1"),
        Event(event_id="2", event_type="tool_completed", task_id="task-1", payload={"tool_name": "echo_tool"}),
    ]

    report = ToolAuditReporter.from_events(events)

    assert report.total_tool_events == 1
    assert report.completed_count == 1


def test_reporter_reads_events_from_sqlite_runtime_store(tmp_path):
    store, session, task = make_runtime_task(tmp_path)
    append_tool_event(store, session, task, "tool_started")
    append_tool_event(store, session, task, "tool_completed")

    report = ToolAuditReporter.from_runtime_store(store, task.task_id)

    assert report.task_id == task.task_id
    assert report.completed_count == 1
    assert report.metadata["source"] == "runtime_store"


def test_format_text_is_summary_only_and_does_not_include_payload_content(tmp_path):
    store, session, task = make_runtime_task(tmp_path)
    secret = "sensitive-resume-line-do-not-render"
    append_tool_event(
        store,
        session,
        task,
        "tool_completed",
        input_summary={"preview": secret},
        output_summary={"preview": secret},
    )

    text = ToolAuditReporter.format_text(ToolAuditReporter.from_runtime_store(store, task.task_id))

    assert "Tool Audit Report for task" in text
    assert "Completed: 1" in text
    assert "echo_tool" in text
    assert secret not in text


def test_missing_payload_fields_have_graceful_fallback():
    event = Event(event_id="event-minimal", event_type="tool_failed", task_id="task-minimal", payload={})

    audit_event = ToolAuditEvent.from_runtime_event(event)
    report = ToolAuditReporter.from_events([event])

    assert audit_event.tool_name == ""
    assert audit_event.duration_ms is None
    assert audit_event.permission_status == ""
    assert report.failed_count == 1


def test_tools_called_are_deduplicated_in_first_seen_order():
    events = [
        Event(event_id="1", event_type="tool_started", payload={"tool_name": "echo_tool"}),
        Event(event_id="2", event_type="tool_completed", payload={"tool_name": "echo_tool"}),
        Event(event_id="3", event_type="tool_denied", payload={"tool_name": "write_fake"}),
    ]

    report = ToolAuditReporter.from_events(events)

    assert report.tools_called == ["echo_tool", "write_fake"]


def test_to_dict_exposes_audit_fields_without_input_output_payload():
    event = Event(
        event_id="event-1",
        event_type="tool_completed",
        task_id="task-1",
        payload={
            "tool_name": "echo_tool",
            "input_summary": {"secret": "hidden"},
            "output_summary": {"secret": "hidden"},
        },
    )

    data = ToolAuditReporter.to_dict(ToolAuditReporter.from_events([event]))

    assert data["events"][0]["tool_name"] == "echo_tool"
    assert "input_summary" not in data["events"][0]
    assert "output_summary" not in data["events"][0]
    assert "hidden" not in str(data)


def test_reporter_consumes_events_written_by_tool_executor(tmp_path):
    store, session, task = make_runtime_task(tmp_path)
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(
        registry,
        recorder=ToolExecutionRecorder(store),
        sandbox_policy=SandboxPolicy(),
    )
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="skill",
        caller_name="query_refine",
    )

    result = executor.execute("echo_tool", {"query": "python"}, context=context)
    report = ToolAuditReporter.from_runtime_store(store, task.task_id)

    assert result.success is True
    assert report.started_count == 1
    assert report.completed_count == 1
    assert report.events[-1].sandbox_status == "allowed"


def test_phase4d_does_not_import_mcp_or_real_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = ("llama_index", "chromadb", "src.agents.retriever", "src.services.retriever", "mcp")
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = ToolAuditReporter.from_events(
        [Event(event_id="event-1", event_type="tool_completed", payload={"tool_name": "echo_tool"})]
    )

    assert report.completed_count == 1
    assert blocked == []
