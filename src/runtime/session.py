from src.runtime.store import InMemoryRuntimeStore


class SessionManager:
    def __init__(self, store: InMemoryRuntimeStore):
        self.store = store

    def create_session(self, metadata=None):
        return self.store.create_session(metadata=metadata)

    def get_session(self, session_id: str):
        return self.store.get_session(session_id)
