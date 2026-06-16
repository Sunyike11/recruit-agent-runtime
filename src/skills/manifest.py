from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.skills.models import SkillSpec


REQUIRED_MANIFEST_FIELDS = ("name", "version")


@dataclass
class SkillManifest:
    name: str
    version: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    required_memory_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_skill_spec(self) -> SkillSpec:
        return SkillSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema),
            required_tools=list(self.required_tools),
            required_memory_types=list(self.required_memory_types),
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )


def validate_skill_manifest(data: Dict[str, Any]) -> bool:
    missing = [field_name for field_name in REQUIRED_MANIFEST_FIELDS if not data.get(field_name)]
    if missing:
        raise ValueError(f"Skill manifest missing required fields: {', '.join(missing)}")
    return True


def load_skill_manifest_from_dict(data: Dict[str, Any]) -> SkillManifest:
    validate_skill_manifest(data)
    return SkillManifest(
        name=data["name"],
        version=data["version"],
        description=data.get("description", ""),
        input_schema=dict(data.get("input_schema", {})),
        output_schema=dict(data.get("output_schema", {})),
        required_tools=list(data.get("required_tools", [])),
        required_memory_types=list(data.get("required_memory_types", [])),
        tags=list(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
    )
