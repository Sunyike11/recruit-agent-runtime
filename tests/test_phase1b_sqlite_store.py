import builtins
import sqlite3
import sys

from src.runtime import RuntimeRunner, SQLiteRuntimeStore, SessionManager, TaskManager, TaskStatus


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase1B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FakeWorkflow:
    def stream(self, state, config):
        yield {
            "fake_node": {
                "jd_text": state["messages"][0].content,
                "thread_id": config["configurable"]["thread_id"],
            }
        }


class FailingWorkflow:
    def stream(self, state, config):
        raise RuntimeError("sqlite fake workflow failed")


def make_store(tmp_path):
    return SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")


def make_runtime(tmp_path):
    store = make_store(tmp_path)
    return store, SessionManager(store), TaskManager(store)


def event_types(store, task_id=None):
    return [event.event_type for event in store.list_events_by_task(task_id)] if task_id else [
        event.event_type for event in store.list_events()
    ]


def test_sqlite_store_initializes_tables(tmp_path):
    db_path = tmp_path / "runtime.sqlite3"
    SQLiteRuntimeStore(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        table_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"sessions", "tasks", "events"}.issubset(table_names)


def test_create_and_read_session(tmp_path):
    store, sessions, _ = make_runtime(tmp_path)

    session = sessions.create_session(metadata={"owner": "hr"})
    loaded = store.get_session(session.session_id)

    assert loaded.session_id == session.session_id
    assert loaded.metadata == {"owner": "hr"}
    assert store.list_sessions()[0].session_id == session.session_id


def test_create_and_read_task(tmp_path):
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()

    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-sqlite")
    loaded = store.get_task(task.task_id)

    assert loaded.task_id == task.task_id
    assert loaded.session_id == session.session_id
    assert loaded.thread_id == "thread-sqlite"
    assert loaded.input["jd_text"] == "招聘JD"
    assert loaded.status == TaskStatus.CREATED
    assert store.list_tasks_by_session(session.session_id)[0].task_id == task.task_id


def test_update_task_status(tmp_path):
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")

    updated = store.update_task_status(task.task_id, TaskStatus.RUNNING)

    assert updated.status == TaskStatus.RUNNING
    assert store.get_task(task.task_id).status == TaskStatus.RUNNING


def test_append_and_read_events(tmp_path):
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")

    store.append_event("custom_event", session_id=session.session_id, task_id=task.task_id, payload={"ok": True})
    events = store.list_events_by_task(task.task_id)

    assert [event.event_type for event in events] == ["task_created", "custom_event"]
    assert events[-1].payload == {"ok": True}


def test_data_survives_store_reinstantiation(tmp_path):
    db_path = tmp_path / "runtime.sqlite3"
    store = SQLiteRuntimeStore(db_path)
    sessions = SessionManager(store)
    tasks = TaskManager(store)
    session = sessions.create_session(metadata={"round": 1})
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-persist")
    store.append_event("custom_event", session_id=session.session_id, task_id=task.task_id, payload={"persist": True})

    reopened = SQLiteRuntimeStore(db_path)

    assert reopened.get_session(session.session_id).metadata == {"round": 1}
    assert reopened.get_task(task.task_id).thread_id == "thread-persist"
    assert [event.event_type for event in reopened.list_events_by_task(task.task_id)] == [
        "task_created",
        "custom_event",
    ]


def test_runner_with_sqlite_store_success_persists_task_and_events(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-runner-success")
    runner = RuntimeRunner(store, graph_factory=lambda: FakeWorkflow())

    completed = runner.run_task(task.task_id)
    reopened = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    loaded = reopened.get_task(task.task_id)

    assert completed.status == TaskStatus.COMPLETED
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.result["events"][0]["fake_node"]["thread_id"] == "thread-runner-success"
    assert event_types(reopened, task.task_id) == [
        "task_created",
        "task_started",
        "task_completed",
    ]
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_runner_with_sqlite_store_failure_persists_error(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: FailingWorkflow())

    failed = runner.run_task(task.task_id)
    reopened = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    loaded = reopened.get_task(task.task_id)

    assert failed.status == TaskStatus.FAILED
    assert loaded.status == TaskStatus.FAILED
    assert loaded.error == "sqlite fake workflow failed"
    assert event_types(reopened, task.task_id) == [
        "task_created",
        "task_started",
        "task_failed",
    ]


def test_phase1b_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: FakeWorkflow())

    runner.run_task(task.task_id)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
