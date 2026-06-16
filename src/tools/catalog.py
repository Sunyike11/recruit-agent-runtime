import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from src.tools.base import BaseTool, CandidateLookupFakeTool, EchoTool, ResumeTextParseFakeTool
from src.tools.manifest import ToolManifest
from src.tools.models import ToolSpec
from src.tools.registry import ToolRegistry


class ToolCatalogError(ValueError):
    pass


ToolFactory = Callable[[ToolSpec], BaseTool]


def _build_echo_tool(spec: ToolSpec) -> BaseTool:
    return EchoTool(spec=spec)


def _build_candidate_lookup_fake_tool(spec: ToolSpec) -> BaseTool:
    return CandidateLookupFakeTool(spec=spec)


def _build_resume_text_parse_fake_tool(spec: ToolSpec) -> BaseTool:
    return ResumeTextParseFakeTool(spec=spec)


DEFAULT_FAKE_TOOL_FACTORIES: Dict[str, ToolFactory] = {
    "echo_tool": _build_echo_tool,
    "candidate_lookup_fake": _build_candidate_lookup_fake_tool,
    "resume_text_parse_fake": _build_resume_text_parse_fake_tool,
}


class ToolCatalog:
    """Validated local tool manifests backed by an explicit factory allowlist."""

    def __init__(self, manifests: Iterable[ToolManifest]):
        self.manifests = list(manifests)
        self.validate()

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> "ToolCatalog":
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("manifests", data.get("tools"))
        else:
            raise ToolCatalogError("tool catalog must be a dict or list")
        if not isinstance(entries, list):
            raise ToolCatalogError("tool catalog must contain a manifests list")
        return cls([ToolManifest.from_dict(entry) for entry in entries])

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "ToolCatalog":
        try:
            with Path(path).open("r", encoding="utf-8") as catalog_file:
                return cls.from_dict(json.load(catalog_file))
        except json.JSONDecodeError as exc:
            raise ToolCatalogError(f"invalid tool catalog JSON: {exc}") from exc

    def validate(self) -> "ToolCatalog":
        seen = set()
        for manifest in self.manifests:
            manifest.validate()
            key = (manifest.name, manifest.version)
            if key in seen:
                raise ToolCatalogError(f"duplicate tool manifest: {manifest.name}@{manifest.version}")
            seen.add(key)
        return self

    def list_manifests(self) -> List[ToolManifest]:
        return list(self.manifests)

    def get_manifest(self, name: str, version: Optional[str] = None) -> ToolManifest:
        matching = [manifest for manifest in self.manifests if manifest.name == name]
        if version is not None:
            matching = [manifest for manifest in matching if manifest.version == version]
        if not matching:
            suffix = f"@{version}" if version is not None else ""
            raise ToolCatalogError(f"tool manifest not found: {name}{suffix}")
        return matching[-1]

    @staticmethod
    def to_tool_spec(manifest: ToolManifest) -> ToolSpec:
        return manifest.to_tool_spec()

    def register_tools(
        self,
        registry: ToolRegistry,
        factory_map: Optional[Dict[str, ToolFactory]] = None,
    ) -> List[BaseTool]:
        factories = factory_map if factory_map is not None else DEFAULT_FAKE_TOOL_FACTORIES
        tools = []
        for manifest in self.manifests:
            factory = factories.get(manifest.implementation_ref)
            if factory is None:
                raise ToolCatalogError(
                    f"no registered factory for implementation_ref: {manifest.implementation_ref}"
                )
            tool = factory(manifest.to_tool_spec())
            if not isinstance(tool, BaseTool):
                raise ToolCatalogError(f"factory did not return BaseTool: {manifest.implementation_ref}")
            if (tool.spec.name, tool.spec.version) != (manifest.name, manifest.version):
                raise ToolCatalogError(f"factory returned mismatched ToolSpec: {manifest.implementation_ref}")
            registry.register(tool)
            tools.append(tool)
        return tools
