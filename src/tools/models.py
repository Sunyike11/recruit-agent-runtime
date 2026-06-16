from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    version: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    permissions_required: List[str] = field(default_factory=list)
    side_effects: str = "none"
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionContext:
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    caller_type: str = ""
    caller_name: str = ""
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_name: str
    version: str
    success: bool
    output: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
