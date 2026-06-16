import builtins
import sys

from src.runtime import InMemoryRuntimeStore, RuntimeRunner, SessionManager, TaskManager, TaskStatus


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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase1A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FakeWorkflow:
    def __init__(self):
        self.calls = []

    def stream(self, state, config):
        self.calls.append({"state": state, "config": config})
        yield {"fake_node": {"next_action": "end", "messages": []}}


class FailingWorkflow:
    def stream(self, state, config):
        raise RuntimeError("fake workflow failed")


def make_runtime():
    store = InMemoryRuntimeStore()
    sessions = SessionManager(store)
    tasks = TaskManager(store)
    return store, sessions, tasks


def event_types(store, task_id=None):
    return [event.event_type for event in store.list_events(task_id=task_id)]


def test_can_create_session():
    store, sessions, _ = make_runtime()

    session = sessions.create_session(metadata={"user": "tester"})

    assert session.session_id
    assert store.get_session(session.session_id) == session
    assert event_types(store) == ["session_created"]


def test_can_create_task_and_bind_thread_id():
    store, sessions, tasks = make_runtime()
    session = sessions.create_session()

    task = tasks.create_task(session.session_id, jd_text="招聘JD")

    assert task.task_id
    assert task.thread_id
    assert task.input["jd_text"] == "招聘JD"
    assert task.status == TaskStatus.CREATED
    assert event_types(store, task.task_id) == ["task_created"]


def test_task_can_use_explicit_thread_id():
    _, sessions, tasks = make_runtime()
    session = sessions.create_session()

    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-explicit")

    assert task.thread_id == "thread-explicit"


def test_runner_executes_fake_workflow_successfully(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime()
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD", thread_id="thread-success")
    workflow = FakeWorkflow()
    runner = RuntimeRunner(store, graph_factory=lambda: workflow)

    completed_task = runner.run_task(task.task_id)

    assert completed_task.status == TaskStatus.COMPLETED
    assert completed_task.status_history == [
        TaskStatus.CREATED,
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
    ]
    assert workflow.calls[0]["state"]["messages"][0].content == "招聘JD"
    assert workflow.calls[0]["config"]["configurable"]["thread_id"] == "thread-success"
    assert event_types(store, task.task_id) == [
        "task_created",
        "task_started",
        "task_completed",
    ]
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_runner_marks_task_failed_when_workflow_raises(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store, sessions, tasks = make_runtime()
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: FailingWorkflow())

    failed_task = runner.run_task(task.task_id)

    assert failed_task.status == TaskStatus.FAILED
    assert failed_task.status_history == [
        TaskStatus.CREATED,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    ]
    assert failed_task.error == "fake workflow failed"
    assert event_types(store, task.task_id) == [
        "task_created",
        "task_started",
        "task_failed",
    ]


def test_runtime_tests_do_not_import_real_retrieval_modules(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    store, sessions, tasks = make_runtime()
    session = sessions.create_session()
    task = tasks.create_task(session.session_id, jd_text="招聘JD")
    runner = RuntimeRunner(store, graph_factory=lambda: FakeWorkflow())

    runner.run_task(task.task_id)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
