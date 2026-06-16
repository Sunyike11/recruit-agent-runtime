from typing import Optional

from src.runtime.store import InMemoryRuntimeStore


class TaskManager:
    def __init__(self, store: InMemoryRuntimeStore):
        self.store = store

    def create_task(self, session_id: str, jd_text: str, thread_id: Optional[str] = None, metadata=None):
        input_payload = {
            "jd_text": jd_text,
            "metadata": metadata or {},
        }
        return self.store.create_task(session_id, input_payload=input_payload, thread_id=thread_id)

    def get_task(self, task_id: str):
        return self.store.get_task(task_id)
