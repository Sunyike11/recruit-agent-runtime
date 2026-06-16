from typing import Callable, Optional

from src.core.state import create_initial_state
from src.runtime.models import TaskStatus
from src.runtime.store import InMemoryRuntimeStore


class RuntimeRunner:
    """Phase1A graph runner wrapper.

    The runner accepts an injected graph factory so tests can execute without
    importing the real RetrieverAgent or retrieval dependencies.
    """

    def __init__(self, store: InMemoryRuntimeStore, graph_factory: Optional[Callable] = None):
        self.store = store
        self.graph_factory = graph_factory

    def run_task(self, task_id: str):
        return self._execute_task(
            task_id,
            start_event_type="task_started",
            completion_event_type="task_completed",
        )

    def resume_task(self, task_id: str):
        return self._execute_task(
            task_id,
            start_event_type="task_resumed",
            completion_event_type="task_resumed_completed",
        )

    def add_human_feedback(self, task_id: str, feedback_type: str, payload):
        if not hasattr(self.store, "add_human_feedback"):
            raise NotImplementedError("The configured runtime store does not support human feedback")
        return self.store.add_human_feedback(task_id, feedback_type, payload)

    def get_task_timeline(self, task_id: str):
        return self.store.list_events(task_id=task_id)

    def _execute_task(self, task_id: str, start_event_type: str, completion_event_type: str):
        task = self.store.get_task(task_id)
        self.store.update_task_status(task_id, TaskStatus.RUNNING)
        self.store.append_event(
            start_event_type,
            session_id=task.session_id,
            task_id=task.task_id,
            payload={"thread_id": task.thread_id},
        )

        try:
            result = self._run_workflow(task)
        except Exception as exc:
            self.store.update_task_status(task_id, TaskStatus.FAILED, error=str(exc))
            self.store.append_event(
                "task_failed",
                session_id=task.session_id,
                task_id=task.task_id,
                payload={"error": str(exc), "thread_id": task.thread_id},
            )
            return self.store.get_task(task_id)

        self.store.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
        self.store.append_event(
            completion_event_type,
            session_id=task.session_id,
            task_id=task.task_id,
            payload={"thread_id": task.thread_id},
        )
        return self.store.get_task(task_id)

    def _run_workflow(self, task):
        workflow = self._create_workflow()
        jd_text = task.input.get("jd_text", "")
        config = {"configurable": {"thread_id": task.thread_id}}
        state = create_initial_state(jd_text)

        if hasattr(workflow, "stream"):
            events = list(workflow.stream(state, config))
            return {"events": events}

        if callable(workflow):
            return workflow(state, config)

        raise TypeError("Runtime workflow must provide stream(state, config) or be callable")

    def _create_workflow(self):
        if self.graph_factory is None:
            from src.core.graph import create_recruit_graph

            return create_recruit_graph()
        return self.graph_factory()
