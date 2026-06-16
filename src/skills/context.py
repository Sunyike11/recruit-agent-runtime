from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.memory import MemoryContext


@dataclass
class SkillExecutionContext:
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_context: Optional[MemoryContext] = None
    runtime_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
