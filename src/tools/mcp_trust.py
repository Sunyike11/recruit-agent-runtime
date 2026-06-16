from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Tuple

from src.tools.mcp_adapter import MCPToolDescriptor
from src.tools.mcp_readiness import validate_mcp_descriptor_compatibility


ALLOWED_MCP_TRANSPORTS = {"local", "stdio", "http", "websocket"}
ALLOWED_MCP_TRUST_LEVELS = {"untrusted", "restricted", "trusted"}


@dataclass
class MCPServerConfig:
    server_id: str
    name: str
    transport: str = "local"
    endpoint: str = ""
    trust_level: str = "restricted"
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    sandbox_profile_name: str = ""
    approval_required_by_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        if not isinstance(data, dict):
            raise ValueError("MCP server config must be a dict")
        config = cls(
            server_id=data.get("server_id", ""),
            name=data.get("name", ""),
            transport=data.get("transport", "local"),
            endpoint=data.get("endpoint", ""),
            trust_level=data.get("trust_level", "restricted"),
            allowed_tools=data.get("allowed_tools", []),
            denied_tools=data.get("denied_tools", []),
            required_permissions=data.get("required_permissions", []),
            sandbox_profile_name=data.get("sandbox_profile_name", ""),
            approval_required_by_default=data.get("approval_required_by_default", False),
            metadata=data.get("metadata", {}),
        )
        config.validate_shape()
        config.allowed_tools = list(config.allowed_tools)
        config.denied_tools = list(config.denied_tools)
        config.required_permissions = list(config.required_permissions)
        config.metadata = dict(config.metadata)
        return config

    def validate_shape(self) -> "MCPServerConfig":
        string_fields = {
            "server_id": self.server_id,
            "name": self.name,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "trust_level": self.trust_level,
            "sandbox_profile_name": self.sandbox_profile_name,
        }
        for name, value in string_fields.items():
            if not isinstance(value, str):
                raise ValueError(f"MCP server config {name} must be a string")
        for name, value in {
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "required_permissions": self.required_permissions,
        }.items():
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"MCP server config {name} must be a list of strings")
        if not isinstance(self.approval_required_by_default, bool):
            raise ValueError("MCP server config approval_required_by_default must be a bool")
        if not isinstance(self.metadata, dict):
            raise ValueError("MCP server config metadata must be a dict")
        return self


@dataclass
class MCPToolAllowlist:
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)

    def is_allowed(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        return not self.allowed_tools or tool_name in self.allowed_tools


@dataclass
class MCPServerTrustDecision:
    allowed: bool
    status: str
    reason: str
    server_id: str
    trust_level: str
    filtered_tool_names: List[str] = field(default_factory=list)
    denied_tool_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "server_id": self.server_id,
            "trust_level": self.trust_level,
            "filtered_tool_names": list(self.filtered_tool_names),
            "denied_tool_names": list(self.denied_tool_names),
            "metadata": dict(self.metadata),
        }


class MCPServerTrustPolicy:
    def evaluate_server(self, config: MCPServerConfig) -> MCPServerTrustDecision:
        config.validate_shape()
        if not config.server_id.strip() or not config.name.strip():
            return self._denied(config, "server_id and name are required")
        if config.transport not in ALLOWED_MCP_TRANSPORTS:
            return self._denied(config, f"unsupported MCP transport: {config.transport}")
        if config.trust_level not in ALLOWED_MCP_TRUST_LEVELS:
            return self._denied(config, f"unsupported MCP trust level: {config.trust_level}")
        if config.trust_level == "untrusted":
            return self._denied(config, "untrusted MCP servers cannot provide tools")
        if config.trust_level == "restricted":
            return MCPServerTrustDecision(
                allowed=True,
                status="restricted",
                reason="restricted server tools require approval before execution",
                server_id=config.server_id,
                trust_level=config.trust_level,
                metadata={"network_attempted": False},
            )
        return MCPServerTrustDecision(
            allowed=True,
            status="allowed",
            reason="trusted MCP server configuration accepted",
            server_id=config.server_id,
            trust_level=config.trust_level,
            metadata={"network_attempted": False},
        )

    def filter_descriptors(
        self,
        config: MCPServerConfig,
        descriptors: Iterable[MCPToolDescriptor],
    ) -> Tuple[List[MCPToolDescriptor], MCPServerTrustDecision]:
        source_descriptors = list(descriptors)
        server_decision = self.evaluate_server(config)
        if not server_decision.allowed:
            return [], replace(
                server_decision,
                denied_tool_names=[descriptor.name for descriptor in source_descriptors],
            )

        rules = MCPToolAllowlist(
            allowed_tools=list(config.allowed_tools),
            denied_tools=list(config.denied_tools),
        )
        filtered_descriptors: List[MCPToolDescriptor] = []
        denied_tool_names: List[str] = []
        for descriptor in source_descriptors:
            validate_mcp_descriptor_compatibility(descriptor)
            if not rules.is_allowed(descriptor.name):
                denied_tool_names.append(descriptor.name)
                continue
            filtered_descriptor = self._apply_server_contract(config, descriptor)
            validate_mcp_descriptor_compatibility(filtered_descriptor)
            filtered_descriptors.append(filtered_descriptor)

        return filtered_descriptors, replace(
            server_decision,
            filtered_tool_names=[descriptor.name for descriptor in filtered_descriptors],
            denied_tool_names=denied_tool_names,
            metadata={
                **server_decision.metadata,
                "allowlist_applied": bool(config.allowed_tools),
                "denylist_applied": bool(config.denied_tools),
            },
        )

    @staticmethod
    def _apply_server_contract(
        config: MCPServerConfig,
        descriptor: MCPToolDescriptor,
    ) -> MCPToolDescriptor:
        permissions = list(
            dict.fromkeys([*descriptor.permissions_required, *config.required_permissions])
        )
        approval_required = (
            descriptor.approval_required
            or config.approval_required_by_default
            or config.trust_level == "restricted"
        )
        metadata = {
            **descriptor.metadata,
            "mcp_server_id": config.server_id,
            "mcp_server_name": config.name,
            "mcp_transport": config.transport,
            "mcp_trust_level": config.trust_level,
            "sandbox_profile_name": config.sandbox_profile_name,
            "approval_required_by_server": config.approval_required_by_default,
        }
        return replace(
            descriptor,
            permissions_required=permissions,
            sandbox_requirements=dict(descriptor.sandbox_requirements or {}),
            approval_required=approval_required,
            metadata=metadata,
        )

    @staticmethod
    def _denied(config: MCPServerConfig, reason: str) -> MCPServerTrustDecision:
        return MCPServerTrustDecision(
            allowed=False,
            status="denied",
            reason=reason,
            server_id=config.server_id,
            trust_level=config.trust_level,
            metadata={"network_attempted": False},
        )
