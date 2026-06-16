import builtins
import socket
from pathlib import Path

from src.tools import (
    MCPServerConfig,
    MCPServerTrustPolicy,
    MCPToolDescriptor,
    validate_mcp_descriptor_compatibility,
)


def _descriptor(
    name: str = "search_candidates",
    *,
    permissions_required=None,
    approval_required: bool = False,
) -> MCPToolDescriptor:
    return MCPToolDescriptor(
        name=name,
        version="v1",
        description="fake MCP descriptor",
        input_schema={"query": "string"},
        output_schema={"results": "list"},
        permissions_required=list(permissions_required or []),
        side_effects="read",
        sandbox_requirements={"requested_capabilities": ["read"]},
        approval_required=approval_required,
        metadata={"source": "fake"},
    )


def _config(**overrides) -> MCPServerConfig:
    data = {
        "server_id": "local-mcp-1",
        "name": "Local Fake MCP",
        "transport": "local",
        "endpoint": "unused://local",
        "trust_level": "trusted",
    }
    data.update(overrides)
    return MCPServerConfig.from_dict(data)


def test_mcp_server_config_can_be_created_from_dict():
    config = _config(
        allowed_tools=["search_candidates"],
        required_permissions=["resume:read"],
        sandbox_profile_name="restricted-read",
        approval_required_by_default=True,
    )

    assert config.server_id == "local-mcp-1"
    assert config.allowed_tools == ["search_candidates"]
    assert config.required_permissions == ["resume:read"]
    assert config.approval_required_by_default is True


def test_invalid_transport_is_denied():
    decision = MCPServerTrustPolicy().evaluate_server(_config(transport="smtp"))

    assert decision.allowed is False
    assert decision.status == "denied"
    assert "transport" in decision.reason


def test_empty_server_id_is_denied():
    decision = MCPServerTrustPolicy().evaluate_server(_config(server_id=""))

    assert decision.allowed is False
    assert decision.status == "denied"


def test_trusted_server_is_allowed():
    decision = MCPServerTrustPolicy().evaluate_server(_config(trust_level="trusted"))

    assert decision.allowed is True
    assert decision.status == "allowed"


def test_untrusted_server_is_denied():
    filtered, decision = MCPServerTrustPolicy().filter_descriptors(
        _config(trust_level="untrusted"),
        [_descriptor()],
    )

    assert filtered == []
    assert decision.status == "denied"
    assert decision.denied_tool_names == ["search_candidates"]


def test_restricted_server_forces_approval_declaration():
    filtered, decision = MCPServerTrustPolicy().filter_descriptors(
        _config(trust_level="restricted"),
        [_descriptor()],
    )

    assert decision.status == "restricted"
    assert filtered[0].approval_required is True


def test_denied_tools_take_precedence_over_allowed_tools():
    filtered, decision = MCPServerTrustPolicy().filter_descriptors(
        _config(
            allowed_tools=["search_candidates", "parse_resume"],
            denied_tools=["search_candidates"],
        ),
        [_descriptor(), _descriptor("parse_resume")],
    )

    assert [item.name for item in filtered] == ["parse_resume"]
    assert decision.denied_tool_names == ["search_candidates"]


def test_nonempty_allowlist_filters_descriptors():
    filtered, decision = MCPServerTrustPolicy().filter_descriptors(
        _config(allowed_tools=["parse_resume"]),
        [_descriptor(), _descriptor("parse_resume")],
    )

    assert [item.name for item in filtered] == ["parse_resume"]
    assert decision.filtered_tool_names == ["parse_resume"]
    assert "search_candidates" in decision.denied_tool_names


def test_denied_tool_does_not_enter_manifest_pipeline():
    filtered, _ = MCPServerTrustPolicy().filter_descriptors(
        _config(denied_tools=["search_candidates"]),
        [_descriptor()],
    )

    assert filtered == []


def test_default_approval_is_preserved_in_manifest_and_spec():
    filtered, _ = MCPServerTrustPolicy().filter_descriptors(
        _config(approval_required_by_default=True),
        [_descriptor()],
    )
    manifest = filtered[0].to_manifest()
    spec = manifest.to_tool_spec()

    assert filtered[0].approval_required is True
    assert filtered[0].metadata["approval_required_by_server"] is True
    assert manifest.approval_required is True
    assert spec.metadata["approval_required"] is True


def test_server_permissions_are_merged_without_mutating_source_descriptor():
    original = _descriptor(permissions_required=["candidate:read"])
    filtered, _ = MCPServerTrustPolicy().filter_descriptors(
        _config(required_permissions=["resume:read", "candidate:read"]),
        [original],
    )

    assert filtered[0].permissions_required == ["candidate:read", "resume:read"]
    assert original.permissions_required == ["candidate:read"]


def test_sandbox_profile_name_is_preserved_in_metadata():
    filtered, _ = MCPServerTrustPolicy().filter_descriptors(
        _config(sandbox_profile_name="mcp-read-only"),
        [_descriptor()],
    )

    assert filtered[0].metadata["sandbox_profile_name"] == "mcp-read-only"
    assert filtered[0].to_manifest().metadata["sandbox_profile_name"] == "mcp-read-only"


def test_filtered_descriptors_still_convert_to_manifest_and_tool_spec():
    filtered, _ = MCPServerTrustPolicy().filter_descriptors(_config(), [_descriptor()])

    manifest = validate_mcp_descriptor_compatibility(filtered[0])
    spec = manifest.to_tool_spec()

    assert manifest.name == "search_candidates"
    assert spec.name == "search_candidates"


def test_policy_does_not_access_network(monkeypatch):
    def blocked_socket(*args, **kwargs):
        raise AssertionError("network access is not allowed in Phase4J")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    filtered, decision = MCPServerTrustPolicy().filter_descriptors(_config(), [_descriptor()])

    assert filtered
    assert decision.metadata["network_attempted"] is False


def test_phase4j_does_not_import_real_mcp_sdk(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise AssertionError("real MCP SDK must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert MCPServerTrustPolicy().evaluate_server(_config()).allowed is True


def test_phase4j_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "MCPServerTrustPolicy" not in graph_source
    assert "MCPToolDescriptor" not in graph_source
