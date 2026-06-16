from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FALLBACK = "completed_with_fallback"
    FAILED = "failed"


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str
    session_id: str
    thread_id: str
    input: Dict[str, Any]
    status: TaskStatus = TaskStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    result: Optional[Any] = None
    error: Optional[str] = None
    status_history: List[TaskStatus] = field(default_factory=lambda: [TaskStatus.CREATED])


@dataclass
class Event:
    event_id: str
    event_type: str
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
