from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SkillSpec:
    name: str
    version: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    required_memory_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    skill_name: str
    version: str
    success: bool
    output: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
