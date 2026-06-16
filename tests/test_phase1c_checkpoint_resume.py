import builtins
import sqlite3
import sys

import pytest

from src.runtime import (
    OptionalCheckpointDependencyError,
    RuntimeRunner,
    SQLiteRuntimeStore,
    SessionManager,
    TaskManager,
    TaskStatus,
    create_optional_sqlite_checkpointer,
)


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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase1C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class RecordingWorkflow:
    def __init__(self):
        self.calls = []

    def stream(self, state, config):
        self.calls.append({"state": state, "config": config})
        yield {
            "fake_node": {
                "thread_id": config["configurable"]["thread_id"],
                "jd_text": state["messages"][0].content,
            }
        }


class FailingWorkflow:
    def stream(self, state, config):
        raise RuntimeError("resume workflow failed")


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    sessions = SessionManager(store)
    tasks = TaskManager(store)
    return store, sessions, tasks


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def test_runner_passes_thread_id_config(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-config")
    workflow = RecordingWorkflow()
    runner = RuntimeRunner(store, graph_factory=lambda: workflow)

    runner.run_task(task.task_id)

    assert workflow.calls[0]["config"] == {"configurable": {"thread_id": "thread-config"}}


def test_resume_task_uses_original_thread_id_and_writes_events(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-resume")
    workflow = RecordingWorkflow()
    runner = RuntimeRunner(store, graph_factory=lambda: workflow)

    runner.resume_task(task.task_id)
    loaded = store.get_task(task.task_id)

    assert loaded.status == TaskStatus.COMPLETED
    assert workflow.calls[0]["config"]["configurable"]["thread_id"] == "thread-resume"
    assert event_types(store, task.task_id) == [
        "task_created",
        "task_resumed",
        "task_resumed_completed",
    ]


def test_resume_task_failure_marks_failed_and_saves_error(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: FailingWorkflow())

    failed = runner.resume_task(task.task_id)
    loaded = store.get_task(task.task_id)

    assert failed.status == TaskStatus.FAILED
    assert loaded.status == TaskStatus.FAILED
    assert loaded.error == "resume workflow failed"
    assert event_types(store, task.task_id) == [
        "task_created",
        "task_resumed",
        "task_failed",
    ]


def test_add_human_feedback_persists_feedback_and_event(tmp_path):
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: RecordingWorkflow())

    feedback = runner.add_human_feedback(task.task_id, "search_query_override", {"query": "PyTorch 3D"})
    loaded_feedback = store.list_human_feedback_by_task(task.task_id)

    assert feedback["feedback_id"]
    assert loaded_feedback[0]["feedback_type"] == "search_query_override"
    assert loaded_feedback[0]["payload"] == {"query": "PyTorch 3D"}
    assert event_types(store, task.task_id) == ["task_created", "feedback_added"]


def test_get_task_timeline_returns_lifecycle_events(tmp_path):
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: RecordingWorkflow())

    runner.add_human_feedback(task.task_id, "approval", {"approved": True})
    timeline = runner.get_task_timeline(task.task_id)

    assert [event.event_type for event in timeline] == ["task_created", "feedback_added"]


def test_checkpoint_metadata_feedback_and_events_survive_reinstantiation(tmp_path):
    db_path = tmp_path / "runtime.sqlite3"
    store = SQLiteRuntimeStore(db_path)
    sessions = SessionManager(store)
    tasks = TaskManager(store)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-checkpoint")
    checkpoint = store.record_graph_checkpoint(
        task.task_id,
        task.thread_id,
        checkpoint_ref="langgraph-checkpoint-ref",
        metadata={"node": "matcher_node"},
    )
    store.add_human_feedback(task.task_id, "approval", {"approved": True})

    reopened = SQLiteRuntimeStore(db_path)

    checkpoints = reopened.list_graph_checkpoints_by_task(task.task_id)
    feedback = reopened.list_human_feedback_by_task(task.task_id)
    events = reopened.list_events_by_task(task.task_id)

    assert checkpoints[0]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert checkpoints[0]["thread_id"] == "thread-checkpoint"
    assert checkpoints[0]["checkpoint_ref"] == "langgraph-checkpoint-ref"
    assert checkpoints[0]["metadata"] == {"node": "matcher_node"}
    assert feedback[0]["payload"] == {"approved": True}
    assert [event.event_type for event in events] == ["task_created", "feedback_added"]


def test_sqlite_store_initializes_phase1c_tables(tmp_path):
    db_path = tmp_path / "runtime.sqlite3"
    SQLiteRuntimeStore(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        table_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"graph_checkpoints", "human_feedback"}.issubset(table_names)


def test_optional_sqlite_checkpointer_has_clear_error_when_dependency_missing(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("langgraph.checkpoint.sqlite"):
            raise ImportError("blocked sqlite checkpointer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(OptionalCheckpointDependencyError) as exc_info:
        create_optional_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3")

    assert "langgraph-checkpoint-sqlite" in str(exc_info.value)


def test_phase1c_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store, sessions, tasks = make_runtime(tmp_path)
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: RecordingWorkflow())

    runner.resume_task(task.task_id)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
