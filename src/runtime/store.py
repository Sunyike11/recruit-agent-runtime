import uuid
from typing import Dict, List, Optional

from src.runtime.models import Event, Session, Task, TaskStatus, utc_now


class InMemoryRuntimeStore:
    """Minimal in-memory runtime store for Phase1A lifecycle tests."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.tasks: Dict[str, Task] = {}
        self.events: List[Event] = []

    def create_session(self, metadata=None) -> Session:
        session = Session(
            session_id=str(uuid.uuid4()),
            metadata=metadata or {},
        )
        self.sessions[session.session_id] = session
        self.append_event("session_created", session_id=session.session_id)
        return session

    def get_session(self, session_id: str) -> Session:
        return self.sessions[session_id]

    def create_task(self, session_id: str, input_payload: Dict, thread_id: Optional[str] = None) -> Task:
        self.get_session(session_id)
        task = Task(
            task_id=str(uuid.uuid4()),
            session_id=session_id,
            thread_id=thread_id or str(uuid.uuid4()),
            input=input_payload,
        )
        self.tasks[task.task_id] = task
        self.append_event(
            "task_created",
            session_id=session_id,
            task_id=task.task_id,
            payload={"thread_id": task.thread_id},
        )
        return task

    def get_task(self, task_id: str) -> Task:
        return self.tasks[task_id]

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result=None,
        error: Optional[str] = None,
    ) -> Task:
        task = self.get_task(task_id)
        task.status = status
        task.updated_at = utc_now()
        task.result = result
        task.error = error
        task.status_history.append(status)
        return task

    def append_event(self, event_type: str, session_id=None, task_id=None, payload=None) -> Event:
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            session_id=session_id,
            task_id=task_id,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    def list_events(self, session_id: Optional[str] = None, task_id: Optional[str] = None) -> List[Event]:
        events = self.events
        if session_id is not None:
            events = [event for event in events if event.session_id == session_id]
        if task_id is not None:
            events = [event for event in events if event.task_id == task_id]
        return events
