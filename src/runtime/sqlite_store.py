import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.runtime.models import Event, Session, Task, TaskStatus, utc_now


def _dt_to_text(value: datetime) -> str:
    return value.isoformat()


def _text_to_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str], default):
    if value is None:
        return default
    return json.loads(value)


class SQLiteRuntimeStore:
    """SQLite-backed runtime metadata store for Phase1B."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    thread_id TEXT,
                    checkpoint_ref TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create_session(self, metadata=None) -> Session:
        session = Session(
            session_id=str(uuid.uuid4()),
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, metadata_json)
                VALUES (?, ?, ?)
                """,
                (session.session_id, _dt_to_text(session.created_at), _json_dumps(session.metadata)),
            )
        self.append_event("session_created", session_id=session.session_id)
        return session

    def get_session(self, session_id: str) -> Session:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, created_at, metadata_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return self._session_from_row(row)

    def list_sessions(self) -> List[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, metadata_json FROM sessions ORDER BY created_at, rowid"
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def create_task(self, session_id: str, input_payload: Dict, thread_id: Optional[str] = None) -> Task:
        self.get_session(session_id)
        task = Task(
            task_id=str(uuid.uuid4()),
            session_id=session_id,
            thread_id=thread_id or str(uuid.uuid4()),
            input=input_payload,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, session_id, thread_id, status, input_json,
                    output_json, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.session_id,
                    task.thread_id,
                    task.status.value,
                    _json_dumps(task.input),
                    None,
                    task.error,
                    _dt_to_text(task.created_at),
                    _dt_to_text(task.updated_at),
                ),
            )
        self.append_event(
            "task_created",
            session_id=session_id,
            task_id=task.task_id,
            payload={"thread_id": task.thread_id},
        )
        return task

    def get_task(self, task_id: str) -> Task:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT task_id, session_id, thread_id, status, input_json,
                       output_json, error, created_at, updated_at
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._task_from_row(row)

    def list_tasks_by_session(self, session_id: str) -> List[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, session_id, thread_id, status, input_json,
                       output_json, error, created_at, updated_at
                FROM tasks
                WHERE session_id = ?
                ORDER BY created_at, rowid
                """,
                (session_id,),
            ).fetchall()
        return [self._task_from_row(row) for row in rows]

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result=None,
        error: Optional[str] = None,
    ) -> Task:
        current = self.get_task(task_id)
        output_json = _json_dumps(result) if result is not None else (
            _json_dumps(current.result) if current.result is not None else None
        )
        updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, output_json = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status.value, output_json, error, _dt_to_text(updated_at), task_id),
            )
        return self.get_task(task_id)

    def append_event(self, event_type: str, session_id=None, task_id=None, payload=None) -> Event:
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            session_id=session_id,
            task_id=task_id,
            payload=payload or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, session_id, task_id, event_type, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.session_id,
                    event.task_id,
                    event.event_type,
                    _json_dumps(event.payload),
                    _dt_to_text(event.created_at),
                ),
            )
        return event

    def list_events(self, session_id: Optional[str] = None, task_id: Optional[str] = None) -> List[Event]:
        query = """
            SELECT event_id, session_id, task_id, event_type, payload_json, created_at
            FROM events
        """
        clauses = []
        params = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, rowid"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    def list_events_by_task(self, task_id: str) -> List[Event]:
        return self.list_events(task_id=task_id)

    def record_graph_checkpoint(self, task_id: str, thread_id: str, checkpoint_ref: str, metadata=None) -> Dict:
        checkpoint = {
            "checkpoint_id": str(uuid.uuid4()),
            "task_id": task_id,
            "thread_id": thread_id,
            "checkpoint_ref": checkpoint_ref,
            "created_at": utc_now(),
            "metadata": metadata or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_checkpoints (
                    checkpoint_id, task_id, thread_id, checkpoint_ref, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint["checkpoint_id"],
                    checkpoint["task_id"],
                    checkpoint["thread_id"],
                    checkpoint["checkpoint_ref"],
                    _dt_to_text(checkpoint["created_at"]),
                    _json_dumps(checkpoint["metadata"]),
                ),
            )
        return checkpoint

    def list_graph_checkpoints_by_task(self, task_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT checkpoint_id, task_id, thread_id, checkpoint_ref, created_at, metadata_json
                FROM graph_checkpoints
                WHERE task_id = ?
                ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        return [self._checkpoint_from_row(row) for row in rows]

    def add_human_feedback(self, task_id: str, feedback_type: str, payload) -> Dict:
        self.get_task(task_id)
        feedback = {
            "feedback_id": str(uuid.uuid4()),
            "task_id": task_id,
            "feedback_type": feedback_type,
            "payload": payload,
            "created_at": utc_now(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO human_feedback (
                    feedback_id, task_id, feedback_type, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback["feedback_id"],
                    feedback["task_id"],
                    feedback["feedback_type"],
                    _json_dumps(feedback["payload"]),
                    _dt_to_text(feedback["created_at"]),
                ),
            )
        task = self.get_task(task_id)
        self.append_event(
            "feedback_added",
            session_id=task.session_id,
            task_id=task_id,
            payload={"feedback_id": feedback["feedback_id"], "feedback_type": feedback_type},
        )
        return feedback

    def list_human_feedback_by_task(self, task_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT feedback_id, task_id, feedback_type, payload_json, created_at
                FROM human_feedback
                WHERE task_id = ?
                ORDER BY created_at, rowid
                """,
                (task_id,),
            ).fetchall()
        return [self._feedback_from_row(row) for row in rows]

    def _session_from_row(self, row) -> Session:
        return Session(
            session_id=row["session_id"],
            created_at=_text_to_dt(row["created_at"]),
            metadata=_json_loads(row["metadata_json"], {}),
        )

    def _task_from_row(self, row) -> Task:
        status = TaskStatus(row["status"])
        return Task(
            task_id=row["task_id"],
            session_id=row["session_id"],
            thread_id=row["thread_id"],
            input=_json_loads(row["input_json"], {}),
            status=status,
            created_at=_text_to_dt(row["created_at"]),
            updated_at=_text_to_dt(row["updated_at"]),
            result=_json_loads(row["output_json"], None),
            error=row["error"],
            status_history=[status],
        )

    def _event_from_row(self, row) -> Event:
        return Event(
            event_id=row["event_id"],
            event_type=row["event_type"],
            session_id=row["session_id"],
            task_id=row["task_id"],
            payload=_json_loads(row["payload_json"], {}),
            created_at=_text_to_dt(row["created_at"]),
        )

    def _checkpoint_from_row(self, row) -> Dict:
        return {
            "checkpoint_id": row["checkpoint_id"],
            "task_id": row["task_id"],
            "thread_id": row["thread_id"],
            "checkpoint_ref": row["checkpoint_ref"],
            "created_at": _text_to_dt(row["created_at"]),
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _feedback_from_row(self, row) -> Dict:
        return {
            "feedback_id": row["feedback_id"],
            "task_id": row["task_id"],
            "feedback_type": row["feedback_type"],
            "payload": _json_loads(row["payload_json"], {}),
            "created_at": _text_to_dt(row["created_at"]),
        }
