import builtins
from pathlib import Path

import pytest

from src.tools import (
    CandidateLookupFakeTool,
    DEFAULT_FAKE_TOOL_FACTORIES,
    EchoTool,
    ResumeTextParseFakeTool,
    ToolCatalog,
    ToolCatalogError,
    ToolExecutionContext,
    ToolExecutor,
    ToolManifest,
    ToolManifestValidationError,
    ToolRegistry,
)


def manifest_data(name="echo_tool", version="v1", implementation_ref="echo_tool", **overrides):
    data = {
        "name": name,
        "version": version,
        "description": "local fake tool",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "category": "demo",
        "permissions_required": [],
        "side_effects": "none",
        "timeout_seconds": 3,
        "sandbox_requirements": {},
        "approval_required": False,
        "implementation_type": "fake",
        "implementation_ref": implementation_ref,
        "metadata": {"phase": "4G"},
    }
    data.update(overrides)
    return data


def test_tool_manifest_can_create_from_dict():
    manifest = ToolManifest.from_dict(manifest_data())

    assert manifest.name == "echo_tool"
    assert manifest.implementation_ref == "echo_tool"
    assert manifest.timeout_seconds == 3


def test_tool_manifest_converts_to_tool_spec_with_policy_declarations():
    manifest = ToolManifest.from_dict(
        manifest_data(
            name="candidate_lookup_fake",
            implementation_ref="candidate_lookup_fake",
            permissions_required=["candidate:read"],
            side_effects="read",
            sandbox_requirements={"requested_capabilities": ["file_read"]},
            approval_required=True,
        )
    )

    spec = manifest.to_tool_spec()

    assert spec.permissions_required == ["candidate:read"]
    assert spec.side_effects == "read"
    assert spec.metadata["requested_capabilities"] == ["file_read"]
    assert spec.metadata["approval_required"] is True
    assert spec.metadata["implementation_ref"] == "candidate_lookup_fake"


def test_invalid_manifests_raise_clear_validation_error():
    invalid_entries = [
        manifest_data(name=""),
        manifest_data(input_schema=[]),
        manifest_data(side_effects="delete"),
        manifest_data(permissions_required="candidate:read"),
        manifest_data(implementation_type="dynamic"),
        manifest_data(implementation_ref=""),
        manifest_data(timeout_seconds=0),
    ]

    for invalid in invalid_entries:
        with pytest.raises(ToolManifestValidationError):
            ToolManifest.from_dict(invalid)


def test_tool_catalog_can_create_from_dict_and_list_manifests():
    catalog = ToolCatalog.from_dict({"manifests": [manifest_data(), manifest_data(version="v2")]})

    assert [manifest.version for manifest in catalog.list_manifests()] == ["v1", "v2"]
    assert catalog.get_manifest("echo_tool").version == "v2"
    assert catalog.get_manifest("echo_tool", version="v1").version == "v1"


def test_tool_catalog_rejects_duplicate_name_and_version():
    with pytest.raises(ToolCatalogError, match="duplicate tool manifest"):
        ToolCatalog.from_dict({"manifests": [manifest_data(), manifest_data()]})


def test_tool_catalog_loads_json_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "tool_catalog.json"

    catalog = ToolCatalog.from_json_file(fixture_path)

    assert [manifest.name for manifest in catalog.list_manifests()] == [
        "echo_tool",
        "candidate_lookup_fake",
        "resume_text_parse_fake",
    ]


def test_register_tools_builds_fake_tools_with_explicit_factory_map():
    catalog = ToolCatalog.from_dict(
        {
            "manifests": [
                manifest_data(),
                manifest_data("candidate_lookup_fake", implementation_ref="candidate_lookup_fake"),
                manifest_data("resume_text_parse_fake", implementation_ref="resume_text_parse_fake"),
            ]
        }
    )
    registry = ToolRegistry()

    tools = catalog.register_tools(registry, DEFAULT_FAKE_TOOL_FACTORIES)

    assert isinstance(tools[0], EchoTool)
    assert isinstance(registry.get("candidate_lookup_fake"), CandidateLookupFakeTool)
    assert isinstance(registry.get("resume_text_parse_fake"), ResumeTextParseFakeTool)


def test_register_tools_missing_factory_raises_clear_error():
    catalog = ToolCatalog.from_dict(
        {"manifests": [manifest_data(name="unknown_fake", implementation_ref="unknown_fake")]}
    )

    with pytest.raises(ToolCatalogError, match="no registered factory"):
        catalog.register_tools(ToolRegistry())


def test_catalog_registration_does_not_use_dynamic_import_or_eval(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "importlib" or name.startswith("src.agents"):
            raise AssertionError(f"unexpected dynamic import: {name}")
        return real_import(name, *args, **kwargs)

    def guarded_eval(*args, **kwargs):
        raise AssertionError("unexpected eval")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "eval", guarded_eval)

    catalog = ToolCatalog.from_dict({"manifests": [manifest_data()]})
    registered = catalog.register_tools(ToolRegistry())

    assert registered[0].spec.name == "echo_tool"


def test_registered_echo_tool_executes_through_tool_executor():
    registry = ToolRegistry()
    ToolCatalog.from_dict({"manifests": [manifest_data()]}).register_tools(registry)

    result = ToolExecutor(registry).execute("echo_tool", {"message": "catalog"})

    assert result.success is True
    assert result.output == {"message": "catalog"}


def test_registered_tool_uses_manifest_permissions_and_side_effects():
    registry = ToolRegistry()
    catalog = ToolCatalog.from_dict(
        {
            "manifests": [
                manifest_data(
                    name="candidate_lookup_fake",
                    implementation_ref="candidate_lookup_fake",
                    permissions_required=["candidate:read"],
                    side_effects="read",
                )
            ]
        }
    )
    catalog.register_tools(registry)

    tool = registry.get("candidate_lookup_fake")

    assert tool.spec.permissions_required == ["candidate:read"]
    assert tool.spec.side_effects == "read"


def test_phase4g_does_not_import_mcp_or_real_external_dependencies(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = ("llama_index", "chromadb", "src.agents.retriever", "src.services.retriever", "mcp")
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase4G test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    registry = ToolRegistry()
    ToolCatalog.from_dict({"manifests": [manifest_data()]}).register_tools(registry)
    result = ToolExecutor(registry).execute("echo_tool", {"ok": True}, context=ToolExecutionContext())

    assert result.success is True
    assert blocked == []
