import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from src.tools.manifest import ALLOWED_SIDE_EFFECTS, ToolManifest, ToolManifestValidationError
from src.tools.mcp_adapter import MCPToolDescriptor


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Future real clients must expose the same local adapter-facing shape."""

    def list_tools(self) -> List[MCPToolDescriptor]:
        ...

    def call_tool(self, name: str, input_data: Dict[str, Any]) -> Any:
        ...


@dataclass
class MCPIntegrationRequirement:
    name: str
    description: str
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPReadinessCheck:
    name: str
    status: str
    detail: str = ""
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass
class MCPReadinessReport:
    checks: List[MCPReadinessCheck] = field(default_factory=list)
    requirements: List[MCPIntegrationRequirement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok_count(self) -> int:
        return _status_count(self.checks, "OK")

    @property
    def fail_count(self) -> int:
        return _status_count(self.checks, "FAIL")

    @property
    def skip_count(self) -> int:
        return _status_count(self.checks, "SKIP")

    @property
    def ready_for_real_integration(self) -> bool:
        return self.fail_count == 0 and not any(
            check.required and check.status != "OK" for check in self.checks
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "ok": self.ok_count,
                "fail": self.fail_count,
                "skip": self.skip_count,
                "ready_for_real_integration": self.ready_for_real_integration,
            },
            "requirements": [
                {
                    "name": requirement.name,
                    "description": requirement.description,
                    "required": requirement.required,
                    "metadata": dict(requirement.metadata),
                }
                for requirement in self.requirements
            ],
            "metadata": dict(self.metadata),
        }

    def format_text(self) -> str:
        lines = [
            f"[{check.status}] {check.name} - {check.detail}"
            for check in self.checks
        ]
        lines.append(f"SUMMARY: OK={self.ok_count} FAIL={self.fail_count} SKIP={self.skip_count}")
        return "\n".join(lines)


DEFAULT_MCP_INTEGRATION_REQUIREMENTS = [
    MCPIntegrationRequirement(
        name="client_protocol",
        description="Client exposes list_tools() and call_tool(name, input_data).",
    ),
    MCPIntegrationRequirement(
        name="descriptor_compatibility",
        description="Descriptors convert through ToolManifest and ToolSpec without dropping safety declarations.",
    ),
    MCPIntegrationRequirement(
        name="executor_safety_chain",
        description="Real adapter tools execute through permission, sandbox, approval, event, and audit boundaries.",
    ),
    MCPIntegrationRequirement(
        name="server_transport",
        description="Real server endpoint, transport lifecycle, authentication, and cancellation policy.",
        required=False,
    ),
]


def validate_mcp_descriptor_compatibility(descriptor: MCPToolDescriptor) -> ToolManifest:
    if not isinstance(descriptor, MCPToolDescriptor):
        raise ValueError("descriptor must be an MCPToolDescriptor")
    if not isinstance(descriptor.name, str) or not descriptor.name.strip():
        raise ValueError("MCP descriptor name must be non-empty")
    if not isinstance(descriptor.version, str) or not descriptor.version.strip():
        raise ValueError("MCP descriptor version must be non-empty")
    if not isinstance(descriptor.input_schema, dict):
        raise ValueError("MCP descriptor input_schema must be a dict")
    if not isinstance(descriptor.output_schema, dict):
        raise ValueError("MCP descriptor output_schema must be a dict")
    if descriptor.side_effects not in ALLOWED_SIDE_EFFECTS:
        raise ValueError("MCP descriptor side_effects is unsupported")
    if not isinstance(descriptor.permissions_required, list) or not all(
        isinstance(permission, str) for permission in descriptor.permissions_required
    ):
        raise ValueError("MCP descriptor permissions_required must be a list of strings")
    if not isinstance(descriptor.approval_required, bool):
        raise ValueError("MCP descriptor approval_required must be a bool")
    if descriptor.sandbox_requirements is not None and not isinstance(descriptor.sandbox_requirements, dict):
        raise ValueError("MCP descriptor sandbox_requirements must be a dict or None")

    try:
        manifest = ToolManifest(
            name=descriptor.name,
            version=descriptor.version,
            description=descriptor.description,
            input_schema=dict(descriptor.input_schema),
            output_schema=dict(descriptor.output_schema),
            category=descriptor.category,
            permissions_required=list(descriptor.permissions_required),
            side_effects=descriptor.side_effects,
            timeout_seconds=descriptor.timeout_seconds,
            sandbox_requirements=dict(descriptor.sandbox_requirements or {}),
            approval_required=descriptor.approval_required,
            implementation_type="mcp_fake",
            implementation_ref=descriptor.name,
            metadata=dict(descriptor.metadata),
        ).validate()
        manifest.to_tool_spec()
    except ToolManifestValidationError as exc:
        raise ValueError(f"MCP descriptor is not ToolManifest compatible: {exc}") from exc
    return manifest


def validate_mcp_descriptors_for_catalog(descriptors: Iterable[MCPToolDescriptor]) -> List[ToolManifest]:
    manifests = []
    seen = set()
    for descriptor in descriptors:
        manifest = validate_mcp_descriptor_compatibility(descriptor)
        identity = (manifest.name, manifest.version)
        if identity in seen:
            raise ValueError(f"duplicate MCP descriptor: {manifest.name}@{manifest.version}")
        seen.add(identity)
        manifests.append(manifest)
    return manifests


def build_mcp_readiness_report(
    client: Optional[Any] = None,
    descriptors: Optional[Iterable[MCPToolDescriptor]] = None,
    *,
    optional_sdk_available: Optional[bool] = None,
    real_server_configured: bool = False,
) -> MCPReadinessReport:
    checks: List[MCPReadinessCheck] = []
    client_matches_protocol = client is not None and isinstance(client, MCPClientProtocol)
    if client is None:
        checks.append(MCPReadinessCheck("client_protocol", "SKIP", "no local or real MCP client supplied"))
    elif client_matches_protocol:
        checks.append(MCPReadinessCheck("client_protocol", "OK", "client exposes required protocol shape"))
    else:
        checks.append(MCPReadinessCheck("client_protocol", "FAIL", "client does not satisfy MCPClientProtocol"))

    selected_descriptors: List[MCPToolDescriptor] = list(descriptors or [])
    if descriptors is None and client_matches_protocol:
        selected_descriptors = list(client.list_tools())
    if not selected_descriptors:
        checks.append(MCPReadinessCheck("descriptor_compatibility", "SKIP", "no descriptors supplied"))
        checks.append(MCPReadinessCheck("safety_declarations", "SKIP", "no descriptors supplied"))
    else:
        try:
            manifests = validate_mcp_descriptors_for_catalog(selected_descriptors)
            checks.append(
                MCPReadinessCheck(
                    "descriptor_compatibility",
                    "OK",
                    f"{len(manifests)} descriptor(s) compatible with ToolManifest/ToolSpec",
                )
            )
            checks.append(
                MCPReadinessCheck(
                    "safety_declarations",
                    "OK",
                    "permission, sandbox, and approval declaration fields are preserved",
                )
            )
        except ValueError as exc:
            checks.append(MCPReadinessCheck("descriptor_compatibility", "FAIL", str(exc)))
            checks.append(MCPReadinessCheck("safety_declarations", "FAIL", "descriptor validation failed"))

    sdk_available = _optional_mcp_sdk_available() if optional_sdk_available is None else optional_sdk_available
    checks.append(
        MCPReadinessCheck(
            "optional_mcp_sdk",
            "OK" if sdk_available else "SKIP",
            "MCP SDK is available" if sdk_available else "MCP SDK is optional and is not installed",
            required=False,
        )
    )
    checks.append(
        MCPReadinessCheck(
            "real_server_connection",
            "SKIP",
            (
                "real server configured, but network connection is intentionally not attempted"
                if real_server_configured
                else "no real MCP server configured"
            ),
            required=False,
        )
    )
    return MCPReadinessReport(
        checks=checks,
        requirements=list(DEFAULT_MCP_INTEGRATION_REQUIREMENTS),
        metadata={"network_attempted": False, "mode": "local_readiness_only"},
    )


def _optional_mcp_sdk_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


def _status_count(checks: Iterable[MCPReadinessCheck], status: str) -> int:
    return sum(1 for check in checks if check.status == status)
