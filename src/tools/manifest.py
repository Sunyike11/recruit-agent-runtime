from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.models import ToolSpec


ALLOWED_SIDE_EFFECTS = {"none", "read", "write", "external"}
ALLOWED_IMPLEMENTATION_TYPES = {"fake", "local", "mcp_fake"}


class ToolManifestValidationError(ValueError):
    pass


@dataclass
class ToolManifest:
    """Declarative local catalog entry for a tool implementation."""

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
    implementation_type: str = "fake"
    implementation_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolManifest":
        if not isinstance(data, dict):
            raise ToolManifestValidationError("tool manifest must be a dict")
        manifest = cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            category=data.get("category", ""),
            permissions_required=data.get("permissions_required", []),
            side_effects=data.get("side_effects", "none"),
            timeout_seconds=data.get("timeout_seconds"),
            sandbox_requirements=data.get("sandbox_requirements", {}),
            approval_required=data.get("approval_required", False),
            implementation_type=data.get("implementation_type", "fake"),
            implementation_ref=data.get("implementation_ref", ""),
            metadata=data.get("metadata", {}),
        )
        manifest.validate()
        return manifest

    def validate(self) -> "ToolManifest":
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolManifestValidationError("tool manifest name must be non-empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ToolManifestValidationError("tool manifest version must be non-empty")
        if not isinstance(self.input_schema, dict):
            raise ToolManifestValidationError("input_schema must be a dict")
        if not isinstance(self.output_schema, dict):
            raise ToolManifestValidationError("output_schema must be a dict")
        if self.side_effects not in ALLOWED_SIDE_EFFECTS:
            raise ToolManifestValidationError(
                f"side_effects must be one of: {', '.join(sorted(ALLOWED_SIDE_EFFECTS))}"
            )
        if not isinstance(self.permissions_required, list) or not all(
            isinstance(permission, str) for permission in self.permissions_required
        ):
            raise ToolManifestValidationError("permissions_required must be a list of strings")
        if self.implementation_type not in ALLOWED_IMPLEMENTATION_TYPES:
            raise ToolManifestValidationError(
                f"implementation_type must be one of: {', '.join(sorted(ALLOWED_IMPLEMENTATION_TYPES))}"
            )
        if not isinstance(self.implementation_ref, str) or not self.implementation_ref.strip():
            raise ToolManifestValidationError("implementation_ref must be non-empty")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ToolManifestValidationError("timeout_seconds must be a positive number or None")
        if not isinstance(self.sandbox_requirements, dict):
            raise ToolManifestValidationError("sandbox_requirements must be a dict")
        if not isinstance(self.approval_required, bool):
            raise ToolManifestValidationError("approval_required must be a bool")
        if not isinstance(self.metadata, dict):
            raise ToolManifestValidationError("metadata must be a dict")
        return self

    def to_tool_spec(self) -> ToolSpec:
        self.validate()
        spec_metadata = dict(self.metadata)
        spec_metadata["sandbox_requirements"] = dict(self.sandbox_requirements)
        spec_metadata["approval_required"] = self.approval_required
        spec_metadata["implementation_type"] = self.implementation_type
        spec_metadata["implementation_ref"] = self.implementation_ref
        for key in ("requested_capabilities", "requested_paths"):
            if key in self.sandbox_requirements and key not in spec_metadata:
                spec_metadata[key] = list(self.sandbox_requirements[key])
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema),
            category=self.category,
            permissions_required=list(self.permissions_required),
            side_effects=self.side_effects,
            timeout_seconds=self.timeout_seconds,
            metadata=spec_metadata,
        )
