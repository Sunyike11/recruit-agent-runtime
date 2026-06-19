import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CandidateMCPError(RuntimeError):
    error_type = "MCPError"


class MCPTransportError(CandidateMCPError):
    error_type = "MCPTransportError"


class MCPTimeoutError(CandidateMCPError):
    error_type = "MCPTimeoutError"


class MCPSchemaError(CandidateMCPError):
    error_type = "MCPSchemaError"


@dataclass
class CandidateMCPClientConfig:
    dataset_dir: str = "evaluation_data/v1"
    provider_mode: str = "evaluation"
    db_path: str = "storage/sqlite/recruit_api_runtime.sqlite"
    command: str = sys.executable
    server_script: str = str(PROJECT_ROOT / "scripts" / "run_candidate_mcp_server.py")
    cwd: str = str(PROJECT_ROOT)
    timeout_seconds: float = 8.0
    env: Optional[Dict[str, str]] = None
    transport: str = "stdio"
    summary_only: bool = True


@dataclass
class CandidateMCPToolCall:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


class CandidateMCPClient:
    def __init__(self, config: Optional[CandidateMCPClientConfig] = None):
        self.config = config or CandidateMCPClientConfig()

    def list_tools(self) -> List[str]:
        return asyncio.run(self._list_tools())

    def call_tool(self, tool_name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return asyncio.run(self._call_tool(tool_name, dict(arguments or {})))

    def call_tools(self, calls: List[CandidateMCPToolCall]) -> Dict[str, Any]:
        return asyncio.run(self._call_tools(calls))

    async def _list_tools(self) -> List[str]:
        async with self._session() as session:
            tools = await session.list_tools()
            return [tool.name for tool in tools.tools]

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async with self._session() as session:
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(seconds=float(self.config.timeout_seconds)),
            )
        return _parse_tool_result(result)

    async def _call_tools(self, calls: List[CandidateMCPToolCall]) -> Dict[str, Any]:
        output: Dict[str, Any] = {"results": [], "summary_only": True}
        async with self._session() as session:
            raw_results = []
            for call in calls:
                result = await session.call_tool(
                    call.tool_name,
                    call.arguments,
                    read_timeout_seconds=timedelta(seconds=float(self.config.timeout_seconds)),
                )
                raw_results.append((call.tool_name, result))
        for tool_name, result in raw_results:
            output["results"].append(
                {
                    "tool_name": tool_name,
                    "output": _parse_tool_result(result),
                    "summary_only": True,
                }
            )
        return output

    def _server_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.config.command,
            args=[
                self.config.server_script,
                "--dataset-dir",
                self.config.dataset_dir,
                "--provider-mode",
                self.config.provider_mode,
                "--db-path",
                self.config.db_path,
            ],
            cwd=self.config.cwd,
            env=self.config.env,
        )

    def _session(self):
        return _CandidateMCPSession(self._server_parameters(), self.config.timeout_seconds)


class _CandidateMCPSession:
    def __init__(self, server_params: StdioServerParameters, timeout_seconds: float):
        self.server_params = server_params
        self.timeout_seconds = timeout_seconds
        self._stdio_cm = None
        self._session_cm = None
        self.session = None

    async def __aenter__(self):
        try:
            self._stdio_cm = stdio_client(self.server_params)
            read, write = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=float(self.timeout_seconds)),
            )
            self.session = await self._session_cm.__aenter__()
            await self.session.initialize()
            return self.session
        except TimeoutError as exc:
            raise MCPTimeoutError("mcp_timeout") from exc
        except Exception as exc:
            raise MCPTransportError(type(exc).__name__) from exc

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)


def _parse_tool_result(result: Any) -> Dict[str, Any]:
    if getattr(result, "isError", False):
        raise MCPSchemaError("mcp_tool_error")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if isinstance(structured.get("result"), dict):
            return dict(structured["result"])
        return dict(structured)
    content = getattr(result, "content", None) or []
    if not content:
        return {"summary_only": True}
    first = content[0]
    text = getattr(first, "text", None)
    if isinstance(text, str):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if isinstance(data.get("result"), dict):
                    return dict(data["result"])
                return data
        except json.JSONDecodeError:
            return {"text_present": bool(text), "summary_only": True}
    if isinstance(first, Mapping):
        return dict(first)
    raise MCPSchemaError("unsupported_mcp_tool_result")
