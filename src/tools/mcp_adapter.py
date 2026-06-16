from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.tools.base import BaseTool
from src.tools.manifest import ToolManifest
from src.tools.models import ToolExecutionContext, ToolResult
from src.tools.registry import ToolRegistry


@dataclass
class MCPToolDescriptor:
    """Local MCP-like tool declaration used without a protocol client."""

    name: str
    version: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    permissions_required: List[str] = field(default_factory=list)
    side_effects: str = "none"
    timeout_seconds: Optional[float] = None
    sandbox_requirements: Dict[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> ToolManifest:
        return ToolManifest(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema),
            category=self.category,
            permissions_required=list(self.permissions_required),
            side_effects=self.side_effects,
            timeout_seconds=self.timeout_seconds,
            sandbox_requirements=dict(self.sandbox_requirements),
            approval_required=self.approval_required,
            implementation_type="mcp_fake",
            implementation_ref=self.name,
            metadata=dict(self.metadata),
        ).validate()


class FakeMCPClient:
    """In-memory MCP-like client for adapter contract tests only."""

    def __init__(
        self,
        descriptors: Iterable[MCPToolDescriptor],
        handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
    ):
        self._descriptors = list(descriptors)
        self._handlers = dict(handlers or {})

    def list_tools(self) -> List[MCPToolDescriptor]:
        return list(self._descriptors)

    def call_tool(self, name: str, input_data: Dict[str, Any]) -> Any:
        handler = self._handlers.get(name)
        if handler is None:
            raise RuntimeError(f"no fake MCP handler registered for tool: {name}")
        return handler(dict(input_data))


class MCPToolAdapter(BaseTool):
    """BaseTool wrapper around a local FakeMCPClient handler."""

    def __init__(self, descriptor: MCPToolDescriptor, client: FakeMCPClient):
        self.descriptor = descriptor
        self.client = client
        super().__init__(spec=descriptor.to_manifest().to_tool_spec())

    def run(
        self,
        input_data: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        output = self.client.call_tool(self.spec.name, input_data)
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output=output,
            metadata={"adapter_type": "mcp_fake"},
        )


class MCPToolCatalogBridge:
    """Maps local MCP-like descriptors into validated executable adapters."""

    @staticmethod
    def descriptors_to_manifests(descriptors: Iterable[MCPToolDescriptor]) -> List[ToolManifest]:
        return [descriptor.to_manifest() for descriptor in descriptors]

    @staticmethod
    def register_mcp_tools(
        registry: ToolRegistry,
        fake_client: FakeMCPClient,
        descriptors: Optional[Iterable[MCPToolDescriptor]] = None,
    ) -> List[MCPToolAdapter]:
        selected = list(descriptors) if descriptors is not None else fake_client.list_tools()
        tools = [MCPToolAdapter(descriptor, fake_client) for descriptor in selected]
        for tool in tools:
            registry.register(tool)
        return tools
