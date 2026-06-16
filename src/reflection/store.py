from typing import Dict, List, Optional

from src.reflection.models import ReflectionRecord


class InMemoryReflectionStore:
    """Non-durable reflection record store for deterministic, summary-only records."""

    def __init__(self):
        self._records: Dict[str, ReflectionRecord] = {}

    def save_reflection(self, record: ReflectionRecord) -> ReflectionRecord:
        record.validate()
        self._records[record.reflection_id] = record
        return record

    def get_reflection(self, reflection_id: str) -> ReflectionRecord:
        if reflection_id not in self._records:
            raise KeyError(reflection_id)
        return self._records[reflection_id]

    def list_reflections(
        self,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        source_type: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[ReflectionRecord]:
        records = list(self._records.values())
        if target_type is not None:
            records = [record for record in records if record.target_type == target_type]
        if target_id is not None:
            records = [record for record in records if record.target_id == target_id]
        if source_type is not None:
            records = [record for record in records if record.source_type == source_type]
        if tag is not None:
            records = [record for record in records if tag in record.tags]
        return records

    def delete_reflection(self, reflection_id: str) -> bool:
        if reflection_id not in self._records:
            return False
        del self._records[reflection_id]
        return True
