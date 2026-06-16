from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.memory.models import MemoryRecord
from src.memory.store import MemorySQLiteStore


@dataclass
class MemoryContextItem:
    memory_id: str
    memory_type: str
    source_type: str
    content: str
    importance: float
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record: MemoryRecord):
        return cls(
            memory_id=record.memory_id,
            memory_type=record.memory_type,
            source_type=record.source_type,
            content=record.content,
            importance=record.importance,
            tags=list(record.tags),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def format_line(self) -> str:
        source = f" {self.source_type}" if self.source_type else ""
        return f"[{self.memory_type}]{source}: {self.content}"


@dataclass
class MemoryContext:
    items: List[MemoryContextItem] = field(default_factory=list)
    max_chars: int = 2000

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def format_for_prompt(self) -> str:
        if self.is_empty:
            text = "Recruit Agent Memory Context:\nNo relevant memory."
        else:
            lines = ["Recruit Agent Memory Context:"]
            lines.extend(item.format_line() for item in self.items)
            text = "\n".join(lines)
        return self._limit_chars(text)

    def _limit_chars(self, text: str) -> str:
        if self.max_chars <= 0:
            return ""
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars]


class MemoryContextBuilder:
    """Build deterministic prompt context from durable memory records."""

    def __init__(self, store: MemorySQLiteStore):
        self.store = store

    def build(
        self,
        memory_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        max_items: int = 5,
        max_chars: int = 2000,
    ) -> MemoryContext:
        records = self.store.list_memories()
        selected = [
            record
            for record in records
            if self._matches(
                record,
                memory_types=memory_types,
                source_types=source_types,
                tags=tags,
                min_importance=min_importance,
            )
        ]
        ranked = sorted(
            selected,
            key=lambda record: self._ranking_key(record, tags=tags),
        )
        items = [MemoryContextItem.from_record(record) for record in ranked[: max(0, max_items)]]
        return MemoryContext(items=items, max_chars=max_chars)

    def build_prompt(
        self,
        memory_types: Optional[List[str]] = None,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        max_items: int = 5,
        max_chars: int = 2000,
    ) -> str:
        return self.build(
            memory_types=memory_types,
            source_types=source_types,
            tags=tags,
            min_importance=min_importance,
            max_items=max_items,
            max_chars=max_chars,
        ).format_for_prompt()

    def _matches(
        self,
        record: MemoryRecord,
        memory_types: Optional[List[str]],
        source_types: Optional[List[str]],
        tags: Optional[List[str]],
        min_importance: Optional[float],
    ) -> bool:
        if memory_types is not None and record.memory_type not in set(memory_types):
            return False
        if source_types is not None and record.source_type not in set(source_types):
            return False
        if tags is not None and self._tag_match_count(record, tags) == 0:
            return False
        if min_importance is not None and record.importance < min_importance:
            return False
        return True

    def _ranking_key(self, record: MemoryRecord, tags: Optional[List[str]]):
        return (
            -record.importance,
            -self._timestamp(record.updated_at),
            -self._timestamp(record.created_at),
            -self._tag_match_count(record, tags),
            record.memory_id,
        )

    def _tag_match_count(self, record: MemoryRecord, tags: Optional[List[str]]) -> int:
        if not tags:
            return 0
        requested = set(tags)
        return len(requested.intersection(record.tags))

    def _timestamp(self, value: Optional[datetime]) -> float:
        if value is None:
            return 0.0
        return value.timestamp()
