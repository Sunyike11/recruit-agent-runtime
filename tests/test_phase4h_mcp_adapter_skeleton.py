import builtins
import socket
from pathlib import Path

import pytest

from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.tools import (
    FakeMCPClient,
    InMemoryToolApprovalStore,
    MCPToolAdapter,
    MCPToolCatalogBridge,
    MCPToolDescriptor,
    SandboxPolicy,
    ToolAuditReporter,
    ToolExecutionContext,
    ToolExecutionRecorder,
    ToolExecutor,
    ToolManifest,
    ToolPermissionPolicy,
    ToolRegistry,
)


def descriptor(name="mcp_echo_fake", **overrides):
    values = {
        "name": name,
        "version": "v1",
        "description": "MCP-like local fake descriptor",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "category": "mcp_fake",
        "permissions_required": [],
        "side_effects": "none",
        "timeout_seconds": 2,
        "sandbox_requirements": {},
        "approval_required": False,
        "metadata": {"phase": "4H"},
    }
    values.update(overrides)
    return MCPToolDescriptor(**values)


def make_client(*descriptors, handlers=None):
    return FakeMCPClient(descriptors, handlers=handlers or {})


def register_single(tool_descriptor, handler):
    client = make_client(tool_descriptor, handlers={tool_descriptor.name: handler})
    registry = ToolRegistry()
    MCPToolCatalogBridge.register_mcp_tools(registry, client)
    return registry, client


def make_runtime_context(tmp_path, permissions=None):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "4H"})
    task = TaskManager(store).create_task(session.session_id, jd_text="mcp fake", thread_id="thread-mcp-fake")
    context = ToolExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        caller_type="workflow",
        caller_name="mcp_adapter_skeleton",
        permissions=list(permissions or []),
    )
    return store, task, context


def test_mcp_tool_descriptor_can_create():
    tool_descriptor = descriptor(permissions_required=["catalog:read"])

    assert tool_descriptor.name == "mcp_echo_fake"
    assert tool_descriptor.permissions_required == ["catalog:read"]


def test_fake_mcp_client_lists_descriptors():
    first = descriptor("first_fake")
    second = descriptor("second_fake")

    assert make_client(first, second).list_tools() == [first, second]


def test_fake_mcp_client_calls_local_fake_handler():
    client = make_client(descriptor(), handlers={"mcp_echo_fake": lambda data: {"echo": data["value"]}})

    assert client.call_tool("mcp_echo_fake", {"value": "hello"}) == {"echo": "hello"}


def test_fake_mcp_client_missing_handler_raises_clear_error():
    with pytest.raises(RuntimeError, match="no fake MCP handler registered"):
        make_client(descriptor()).call_tool("mcp_echo_fake", {})


def test_descriptor_converts_to_tool_manifest():
    manifest = MCPToolCatalogBridge.descriptors_to_manifests([descriptor()])[0]

    assert isinstance(manifest, ToolManifest)
    assert manifest.implementation_type == "mcp_fake"
    assert manifest.implementation_ref == "mcp_echo_fake"


def test_descriptor_manifest_to_spec_preserves_safety_declarations():
    manifest = descriptor(
        permissions_required=["search:external"],
        side_effects="external",
        sandbox_requirements={"requested_capabilities": ["network"]},
        approval_required=True,
    ).to_manifest()

    spec = manifest.to_tool_spec()

    assert spec.permissions_required == ["search:external"]
    assert spec.side_effects == "external"
    assert spec.metadata["requested_capabilities"] == ["network"]
    assert spec.metadata["approval_required"] is True
    assert spec.metadata["implementation_type"] == "mcp_fake"


def test_mcp_fake_tool_registers_in_tool_registry():
    registry, _ = register_single(descriptor(), lambda data: data)

    assert isinstance(registry.get("mcp_echo_fake"), MCPToolAdapter)


def test_mcp_fake_tool_executes_through_tool_executor():
    registry, _ = register_single(descriptor(), lambda data: {"echo": data["message"]})

    result = ToolExecutor(registry).execute("mcp_echo_fake", {"message": "hello"})

    assert result.success is True
    assert result.output == {"echo": "hello"}
    assert result.metadata["adapter_type"] == "mcp_fake"


def test_fake_handler_exception_returns_failed_tool_result():
    def failing_handler(data):
        raise RuntimeError("fake MCP handler failed")

    registry, _ = register_single(descriptor(), failing_handler)

    result = ToolExecutor(registry).execute("mcp_echo_fake", {})

    assert result.success is False
    assert result.error == "fake MCP handler failed"


def test_permission_declaration_is_enforced_by_tool_executor():
    registry, _ = register_single(
        descriptor(permissions_required=["candidate:read"]),
        lambda data: {"ok": True},
    )

    result = ToolExecutor(registry, permission_policy=ToolPermissionPolicy()).execute(
        "mcp_echo_fake",
        {},
        context=ToolExecutionContext(permissions=[]),
    )

    assert result.success is False
    assert result.metadata["permission_decision"]["status"] == "denied"


def test_sandbox_declaration_is_enforced_by_tool_executor():
    registry, _ = register_single(
        descriptor(sandbox_requirements={"requested_capabilities": ["network"]}),
        lambda data: {"ok": True},
    )

    result = ToolExecutor(registry, sandbox_policy=SandboxPolicy()).execute(
        "mcp_echo_fake",
        {},
        context=ToolExecutionContext(),
    )

    assert result.success is False
    assert "network_not_allowed" in result.metadata["sandbox_decision"]["violated_rules"]


def test_external_mcp_fake_tool_uses_approval_contract():
    registry, _ = register_single(
        descriptor(
            name="mcp_external_fake",
            permissions_required=["search:external"],
            side_effects="external",
            approval_required=True,
        ),
        lambda data: {"results": []},
    )
    approval_store = InMemoryToolApprovalStore()
    executor = ToolExecutor(
        registry,
        permission_policy=ToolPermissionPolicy(),
        approval_store=approval_store,
    )

    result = executor.execute(
        "mcp_external_fake",
        {"query": "local"},
        context=ToolExecutionContext(permissions=["search:external"]),
    )

    assert result.success is False
    assert result.error == "approval required"
    assert result.metadata["approval_id"]
    assert approval_store.get_request(result.metadata["approval_id"]).tool_name == "mcp_external_fake"


def test_mcp_fake_tool_success_and_failure_are_recorded_and_audited(tmp_path):
    store, task, context = make_runtime_context(tmp_path)
    success = descriptor("mcp_success_fake")
    failure = descriptor("mcp_failure_fake")
    client = make_client(
        success,
        failure,
        handlers={
            "mcp_success_fake": lambda data: {"ok": True},
            "mcp_failure_fake": lambda data: (_ for _ in ()).throw(RuntimeError("failure")),
        },
    )
    registry = ToolRegistry()
    MCPToolCatalogBridge.register_mcp_tools(registry, client)
    executor = ToolExecutor(registry, recorder=ToolExecutionRecorder(store))

    assert executor.execute("mcp_success_fake", {}, context=context).success is True
    assert executor.execute("mcp_failure_fake", {}, context=context).success is False
    report = ToolAuditReporter.from_runtime_store(store, task.task_id)

    assert report.started_count == 2
    assert report.completed_count == 1
    assert report.failed_count == 1
    assert report.tools_called == ["mcp_success_fake", "mcp_failure_fake"]


def test_mcp_fake_adapter_does_not_access_network(monkeypatch):
    def blocked_socket(*args, **kwargs):
        raise AssertionError("network access is not permitted in Phase4H")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    registry, _ = register_single(descriptor(), lambda data: {"local": True})

    assert ToolExecutor(registry).execute("mcp_echo_fake", {}).success is True


def test_phase4h_does_not_import_real_mcp_sdk_or_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = ("mcp", "llama_index", "chromadb", "src.agents.retriever", "src.services.retriever")
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4H test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    registry, _ = register_single(descriptor(), lambda data: {"ok": True})

    assert ToolExecutor(registry).execute("mcp_echo_fake", {}).success is True
    assert blocked == []


def test_phase4h_does_not_modify_production_graph():
    graph_source = (Path(__file__).parents[1] / "src" / "core" / "graph.py").read_text(encoding="utf-8")

    assert "MCPToolAdapter" not in graph_source
    assert "MCPToolCatalogBridge" not in graph_source
    assert "ToolRegistry" not in graph_source
