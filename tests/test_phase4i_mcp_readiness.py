import builtins
import socket
from pathlib import Path

import pytest

from src.tools import (
    FakeMCPClient,
    MCPClientProtocol,
    MCPReadinessCheck,
    MCPReadinessReport,
    MCPToolDescriptor,
    ToolManifest,
    build_mcp_readiness_report,
    validate_mcp_descriptor_compatibility,
    validate_mcp_descriptors_for_catalog,
)


def descriptor(**overrides):
    values = {
        "name": "mcp_readiness_fake",
        "version": "v1",
        "description": "local readiness descriptor",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "category": "mcp_fake",
        "permissions_required": ["candidate:read"],
        "side_effects": "read",
        "timeout_seconds": 2,
        "sandbox_requirements": {"requested_capabilities": ["file_read"]},
        "approval_required": False,
        "metadata": {"phase": "4I"},
    }
    values.update(overrides)
    return MCPToolDescriptor(**values)


def client_for(*descriptors):
    return FakeMCPClient(descriptors, handlers={item.name: (lambda data: data) for item in descriptors})


def test_fake_mcp_client_satisfies_client_protocol():
    client = client_for(descriptor())

    assert isinstance(client, MCPClientProtocol)


def test_valid_descriptor_compatibility_returns_manifest():
    manifest = validate_mcp_descriptor_compatibility(descriptor())

    assert isinstance(manifest, ToolManifest)
    assert manifest.implementation_type == "mcp_fake"
    assert manifest.to_tool_spec().name == "mcp_readiness_fake"


@pytest.mark.parametrize(
    "invalid, error_text",
    [
        (descriptor(name=""), "name"),
        (descriptor(input_schema=[]), "input_schema"),
        (descriptor(output_schema=[]), "output_schema"),
        (descriptor(side_effects="unsafe"), "side_effects"),
        (descriptor(permissions_required="candidate:read"), "permissions_required"),
        (descriptor(approval_required="yes"), "approval_required"),
        (descriptor(sandbox_requirements=[]), "sandbox_requirements"),
    ],
)
def test_invalid_descriptor_compatibility_fails_clearly(invalid, error_text):
    with pytest.raises(ValueError, match=error_text):
        validate_mcp_descriptor_compatibility(invalid)


def test_descriptor_list_becomes_manifests_and_specs():
    manifests = validate_mcp_descriptors_for_catalog(
        [descriptor(name="first"), descriptor(name="second", version="v2")]
    )

    assert [manifest.name for manifest in manifests] == ["first", "second"]
    assert [manifest.to_tool_spec().version for manifest in manifests] == ["v1", "v2"]


def test_duplicate_descriptor_identity_fails_catalog_validation():
    with pytest.raises(ValueError, match="duplicate MCP descriptor"):
        validate_mcp_descriptors_for_catalog([descriptor(), descriptor()])


def test_readiness_report_records_ok_fail_and_skip():
    report = MCPReadinessReport(
        checks=[
            MCPReadinessCheck("one", "OK"),
            MCPReadinessCheck("two", "FAIL"),
            MCPReadinessCheck("three", "SKIP", required=False),
        ]
    )

    assert report.ok_count == 1
    assert report.fail_count == 1
    assert report.skip_count == 1
    assert report.ready_for_real_integration is False
    assert "SUMMARY: OK=1 FAIL=1 SKIP=1" in report.format_text()


def test_readiness_helper_accepts_fake_client_and_compatible_descriptors():
    report = build_mcp_readiness_report(
        client=client_for(descriptor()),
        optional_sdk_available=False,
    )

    statuses = {check.name: check.status for check in report.checks}
    assert statuses["client_protocol"] == "OK"
    assert statuses["descriptor_compatibility"] == "OK"
    assert statuses["safety_declarations"] == "OK"
    assert report.metadata["network_attempted"] is False


def test_missing_optional_mcp_sdk_is_skip_not_fail():
    report = build_mcp_readiness_report(
        client=client_for(descriptor()),
        optional_sdk_available=False,
    )
    sdk_check = next(check for check in report.checks if check.name == "optional_mcp_sdk")

    assert sdk_check.status == "SKIP"
    assert sdk_check.required is False
    assert report.fail_count == 0


def test_no_real_server_configured_is_graceful_skip():
    report = build_mcp_readiness_report(client=client_for(descriptor()), optional_sdk_available=False)
    server_check = next(check for check in report.checks if check.name == "real_server_connection")

    assert server_check.status == "SKIP"
    assert "no real MCP server configured" in server_check.detail


def test_readiness_helper_does_not_perform_network_call(monkeypatch):
    def blocked_socket(*args, **kwargs):
        raise AssertionError("network access is not permitted in Phase4I")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    report = build_mcp_readiness_report(
        client=client_for(descriptor()),
        optional_sdk_available=False,
        real_server_configured=True,
    )

    assert report.fail_count == 0
    assert report.metadata["network_attempted"] is False


def test_permission_sandbox_and_approval_declarations_are_preserved():
    manifest = validate_mcp_descriptor_compatibility(
        descriptor(
            permissions_required=["search:external"],
            side_effects="external",
            sandbox_requirements={"requested_capabilities": ["network"]},
            approval_required=True,
        )
    )
    spec = manifest.to_tool_spec()

    assert spec.permissions_required == ["search:external"]
    assert spec.side_effects == "external"
    assert spec.metadata["requested_capabilities"] == ["network"]
    assert spec.metadata["approval_required"] is True


def test_phase4i_does_not_require_real_mcp_sdk_import(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4I test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = build_mcp_readiness_report(
        client=client_for(descriptor()),
        optional_sdk_available=False,
    )

    assert report.fail_count == 0
    assert blocked == []


def test_phase4i_does_not_modify_production_graph():
    graph_source = (Path(__file__).parents[1] / "src" / "core" / "graph.py").read_text(encoding="utf-8")

    assert "MCPClientProtocol" not in graph_source
    assert "build_mcp_readiness_report" not in graph_source
    assert "ToolRegistry" not in graph_source
