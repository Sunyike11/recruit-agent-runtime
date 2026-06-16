from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.models import ToolExecutionContext, ToolSpec


@dataclass
class SandboxProfile:
    profile_name: str = "default"
    allow_network: bool = False
    allow_file_read: bool = True
    allow_file_write: bool = False
    allow_subprocess: bool = False
    allow_external_side_effects: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxDecision:
    allowed: bool
    status: str
    reason: str = ""
    violated_rules: List[str] = field(default_factory=list)
    profile_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "violated_rules": list(self.violated_rules),
            "profile_name": self.profile_name,
            "metadata": dict(self.metadata),
        }


@dataclass
class ToolSandboxContext:
    sandbox_profile: SandboxProfile = field(default_factory=SandboxProfile)
    requested_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxPolicy:
    """Declarative sandbox boundary policy.

    This policy does not isolate execution. It only decides whether a tool's
    declared capabilities violate the current sandbox profile.
    """

    def evaluate(
        self,
        tool_spec: ToolSpec,
        execution_context: ToolExecutionContext,
        sandbox_context: Optional[ToolSandboxContext] = None,
    ) -> SandboxDecision:
        context = sandbox_context or ToolSandboxContext()
        profile = context.sandbox_profile
        capabilities = _requested_capabilities(tool_spec, context)
        requested_paths = _requested_paths(tool_spec, context)

        violations = []
        if (tool_spec.side_effects or "none") == "external" and not profile.allow_external_side_effects:
            violations.append("external_side_effects_not_allowed")
        if "network" in capabilities and not profile.allow_network:
            violations.append("network_not_allowed")
        if "file_read" in capabilities and not profile.allow_file_read:
            violations.append("file_read_not_allowed")
        if "file_write" in capabilities and not profile.allow_file_write:
            violations.append("file_write_not_allowed")
        if "subprocess" in capabilities and not profile.allow_subprocess:
            violations.append("subprocess_not_allowed")

        path_violation = _path_violation(requested_paths, profile)
        if path_violation:
            violations.append(path_violation)

        if violations:
            return SandboxDecision(
                allowed=False,
                status="denied",
                reason="sandbox policy denied requested capabilities",
                violated_rules=violations,
                profile_name=profile.profile_name,
                metadata={
                    "requested_capabilities": capabilities,
                    "requested_paths": requested_paths,
                },
            )

        return SandboxDecision(
            allowed=True,
            status="allowed",
            reason="sandbox policy allowed requested capabilities",
            profile_name=profile.profile_name,
            metadata={
                "requested_capabilities": capabilities,
                "requested_paths": requested_paths,
            },
        )


def _requested_capabilities(tool_spec: ToolSpec, sandbox_context: ToolSandboxContext) -> List[str]:
    capabilities = []
    capabilities.extend(tool_spec.metadata.get("requested_capabilities", []) or [])
    capabilities.extend(sandbox_context.requested_capabilities)
    return sorted(set(str(capability) for capability in capabilities))


def _requested_paths(tool_spec: ToolSpec, sandbox_context: ToolSandboxContext) -> List[str]:
    paths = []
    paths.extend(tool_spec.metadata.get("requested_paths", []) or [])
    paths.extend(sandbox_context.metadata.get("requested_paths", []) or [])
    return [str(path) for path in paths]


def _path_violation(requested_paths: List[str], profile: SandboxProfile) -> Optional[str]:
    if not requested_paths:
        return None

    allowed_paths = [_normalize_path(path) for path in profile.allowed_paths]
    blocked_paths = [_normalize_path(path) for path in profile.blocked_paths]
    for requested in requested_paths:
        normalized_requested = _normalize_path(requested)
        if any(_is_same_or_child(normalized_requested, blocked) for blocked in blocked_paths):
            return "blocked_path_requested"
        if allowed_paths and not any(_is_same_or_child(normalized_requested, allowed) for allowed in allowed_paths):
            return "path_not_in_allowed_paths"
    return None


def _normalize_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _is_same_or_child(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents
