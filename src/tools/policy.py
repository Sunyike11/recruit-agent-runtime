from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.models import ToolExecutionContext, ToolSpec


@dataclass
class ToolPermissionDecision:
    allowed: bool
    status: str
    reason: str = ""
    missing_permissions: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    side_effects: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "missing_permissions": list(self.missing_permissions),
            "required_permissions": list(self.required_permissions),
            "side_effects": self.side_effects,
            "metadata": dict(self.metadata),
        }


@dataclass
class ToolExecutionContract:
    tool_name: str
    tool_version: str
    required_permissions: List[str] = field(default_factory=list)
    side_effects: str = "none"
    timeout_seconds: Optional[float] = None
    approval_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolPermissionPolicy:
    """Conservative deterministic permission policy for local tools."""

    def evaluate(self, tool_spec: ToolSpec, execution_context: ToolExecutionContext) -> ToolPermissionDecision:
        available_permissions = set(execution_context.permissions)
        required_permissions = list(tool_spec.permissions_required)
        missing_permissions = [
            permission
            for permission in required_permissions
            if permission not in available_permissions
        ]
        side_effects = tool_spec.side_effects or "none"

        if missing_permissions:
            return ToolPermissionDecision(
                allowed=False,
                status="denied",
                reason="missing required permissions",
                missing_permissions=missing_permissions,
                required_permissions=required_permissions,
                side_effects=side_effects,
            )

        if side_effects == "write" and "allow_write" not in available_permissions:
            return ToolPermissionDecision(
                allowed=False,
                status="requires_approval",
                reason="write side effects require allow_write permission",
                required_permissions=required_permissions,
                side_effects=side_effects,
                metadata={"approval_permission": "allow_write"},
            )

        if side_effects == "external" and "allow_external" not in available_permissions:
            return ToolPermissionDecision(
                allowed=False,
                status="requires_approval",
                reason="external side effects require allow_external permission",
                required_permissions=required_permissions,
                side_effects=side_effects,
                metadata={"approval_permission": "allow_external"},
            )

        return ToolPermissionDecision(
            allowed=True,
            status="allowed",
            reason="permissions satisfied",
            required_permissions=required_permissions,
            side_effects=side_effects,
        )


def build_tool_execution_contract(tool_spec: ToolSpec) -> ToolExecutionContract:
    return ToolExecutionContract(
        tool_name=tool_spec.name,
        tool_version=tool_spec.version,
        required_permissions=list(tool_spec.permissions_required),
        side_effects=tool_spec.side_effects or "none",
        timeout_seconds=tool_spec.timeout_seconds,
        approval_required=(tool_spec.side_effects in {"write", "external"}),
        metadata=dict(tool_spec.metadata),
    )
