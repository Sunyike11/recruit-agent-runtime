from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.tools.base import BaseTool


class ToolAlreadyRegisteredError(ValueError):
    pass


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    """In-memory tool registry with deterministic latest-version lookup."""

    def __init__(self):
        self._tools: Dict[Tuple[str, str], BaseTool] = {}
        self._versions_by_name: Dict[str, List[str]] = defaultdict(list)

    def register(self, tool: BaseTool, allow_replace: bool = False) -> BaseTool:
        key = (tool.spec.name, tool.spec.version)
        if key in self._tools and not allow_replace:
            raise ToolAlreadyRegisteredError(f"Tool already registered: {tool.spec.name}@{tool.spec.version}")
        if key not in self._tools:
            self._versions_by_name[tool.spec.name].append(tool.spec.version)
        self._tools[key] = tool
        return tool

    def get(self, name: str, version: Optional[str] = None) -> BaseTool:
        selected_version = version or self._latest_version(name)
        key = (name, selected_version)
        if key not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}@{selected_version}")
        return self._tools[key]

    def list_tools(self) -> List[BaseTool]:
        return [
            self._tools[(name, version)]
            for name in sorted(self._versions_by_name)
            for version in self._versions_by_name[name]
        ]

    def unregister(self, name: str, version: Optional[str] = None):
        if version is None:
            if name not in self._versions_by_name:
                raise ToolNotFoundError(f"Tool not found: {name}")
            for existing_version in list(self._versions_by_name[name]):
                self._tools.pop((name, existing_version), None)
            self._versions_by_name.pop(name, None)
            return

        key = (name, version)
        if key not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}@{version}")
        self._tools.pop(key)
        versions = self._versions_by_name[name]
        versions.remove(version)
        if not versions:
            self._versions_by_name.pop(name, None)

    def _latest_version(self, name: str) -> str:
        versions = self._versions_by_name.get(name)
        if not versions:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return versions[-1]
