import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.memory.models import MemoryRecord


class MemorySQLiteStore:
    """SQLite store for durable memory records."""

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
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    source_type TEXT,
                    source_id TEXT,
                    content TEXT NOT NULL,
                    importance REAL,
                    tags_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

    def save_memory(self, record: MemoryRecord) -> MemoryRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, memory_type, source_type, source_id, content,
                    importance, tags_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    source_type = excluded.source_type,
                    source_id = excluded.source_id,
                    content = excluded.content,
                    importance = excluded.importance,
                    tags_json = excluded.tags_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                self._record_to_row(record),
            )
        return record

    def get_memory(self, memory_id: str) -> MemoryRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record_from_row(row)

    def list_memories(
        self,
        memory_type: Optional[str] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        query = "SELECT * FROM memories"
        clauses = []
        params = []
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, rowid"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._record_from_row(row) for row in rows]

    def search_memories_by_tag(self, tag: str):
        return [record for record in self.list_memories() if tag in record.tags]

    def delete_memory(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0

    def _record_to_row(self, record: MemoryRecord):
        return (
            record.memory_id,
            record.memory_type,
            record.source_type,
            record.source_id,
            record.content,
            record.importance,
            json.dumps(record.tags, ensure_ascii=False),
            json.dumps(record.metadata, ensure_ascii=False),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )

    def _record_from_row(self, row) -> MemoryRecord:
        return MemoryRecord.from_dict(
            {
                "memory_id": row["memory_id"],
                "memory_type": row["memory_type"],
                "source_type": row["source_type"] or "",
                "source_id": row["source_id"] or "",
                "content": row["content"],
                "importance": row["importance"],
                "tags": json.loads(row["tags_json"] or "[]"),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
