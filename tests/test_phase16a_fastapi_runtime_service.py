import json
import time

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.schemas import CreateMatchingTaskRequest
from src.runtime.entry import RuntimeEntryResult
from src.runtime.event_envelope import build_runtime_event_payload
from src.runtime.models import TaskStatus
from src.runtime.sqlite_store import SQLiteRuntimeStore


TENANT = {"X-Tenant-ID": "tenant_a", "Idempotency-Key": "idem-1"}


def make_submitter(store, *, status="completed", delay=0.0, fallback=False, fail=False):
    calls = {"count": 0, "tenant_ids": [], "candidate_sources": []}

    def submit(request: CreateMatchingTaskRequest, tenant_id: str):
        calls["count"] += 1
        calls["tenant_ids"].append(tenant_id)
        calls["candidate_sources"].append(request.candidate_source)
        if delay:
            time.sleep(delay)
        session = store.create_session(metadata={"tenant_id": tenant_id, "api_test": True})
        task = store.create_task(
            session.session_id,
            input_payload={
                "jd_text": request.jd_text,
                "metadata": {"tenant_id": tenant_id, "candidate_source": request.candidate_source, "summary_only": True},
            },
        )
        store.update_task_status(task.task_id, TaskStatus.RUNNING)
        for event_type in [
            "task_started",
            "graph_started",
            "skill_started",
            "tool_started",
            "tool_completed",
            "skill_completed",
        ]:
            store.append_event(
                event_type,
                session_id=session.session_id,
                task_id=task.task_id,
                payload=build_runtime_event_payload(
                    session_id=session.session_id,
                    task_id=task.task_id,
                    thread_id=task.thread_id,
                    graph_mode="skill",
                    runner_name="production_skill_graph",
                    skill_name="resume_retrieve" if event_type.startswith("tool") else "candidate_match",
                    status="completed" if event_type.endswith("completed") else "started",
                    extra={
                        "tool_name": "search_candidates" if event_type.startswith("tool") else "",
                        "mcp_server_name": "candidate_mcp" if event_type.startswith("tool") else "",
                        "transport": "stdio" if event_type.startswith("tool") else "",
                        "result_count": 3 if event_type == "tool_completed" else 0,
                        "summary_only": True,
                    },
                ),
            )
        output = {
            "status": "failed" if fail else "ok",
            "selected_graph_mode": "skill",
            "candidate_source": request.candidate_source,
            "candidate_count": 3,
            "report_count": 3,
            "candidate_preview_version": "v2",
            "claim_verification_enabled": True,
            "skill_names": ["planner_extract", "resume_retrieve", "candidate_match", "claim_verify"],
            "tool_success_count": 1,
            "fallback_attempted": fallback,
            "fallback_succeeded": fallback,
            "summary_only": True,
        }
        if fail:
            output["error_type"] = "InjectedFailure"
            store.update_task_status(task.task_id, TaskStatus.FAILED, result=output, error="InjectedFailure")
            task_status = "failed"
            status_value = "failed"
        elif fallback:
            store.update_task_status(task.task_id, TaskStatus.COMPLETED_WITH_FALLBACK, result=output)
            task_status = "completed_with_fallback"
            status_value = "completed_with_fallback"
        else:
            store.update_task_status(task.task_id, TaskStatus.COMPLETED, result=output)
            task_status = "completed"
            status_value = "ok"
        store.append_event(
            "task_completed" if not fail else "task_failed",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode="skill",
                runner_name="production_skill_graph",
                status=task_status,
            ),
        )
        return RuntimeEntryResult(
            status=status_value,
            session_id=session.session_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            runner_used="production_skill_graph",
            task_status=task_status,
            event_count=len(store.list_events(task_id=task.task_id)),
            output_summary=output,
            summary_only=True,
        )

    submit.calls = calls
    return submit


def build_client(tmp_path, *, submitter=None, worker_count=1, queue_max_size=10, task_timeout_seconds=5.0):
    store = SQLiteRuntimeStore(tmp_path / "api.sqlite")
    app = create_app(
        store=store,
        runtime_submitter=submitter or make_submitter(store),
        worker_count=worker_count,
        queue_max_size=queue_max_size,
        task_timeout_seconds=task_timeout_seconds,
    )
    return TestClient(app), store, app


def create_task(client, *, tenant="tenant_a", idem="idem-1", jd="招聘 Python RAG 工程师", candidate_source="mcp"):
    return client.post(
        "/matching/tasks",
        headers={"X-Tenant-ID": tenant, "Idempotency-Key": idem},
        json={"jd_text": jd, "candidate_source": candidate_source, "metadata": {"request_source": "test"}},
    )


def wait_for_status(client, task_id, expected=None, *, tenant="tenant_a", timeout=3.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = client.get(f"/tasks/{task_id}", headers={"X-Tenant-ID": tenant})
        last = response.json()
        if expected is None and last["status"] != "queued":
            return last
        if expected and last["status"] in expected:
            return last
        time.sleep(0.05)
    return last


def test_app_import_and_lifespan_health_ready(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        assert client.get("/healthz").json()["status"] == "ok"
        ready = client.get("/readyz").json()
        assert ready["status"] == "ready"
        assert ready["graph_factory_default_mode"] == "skill"


def test_create_task_requires_valid_tenant_and_returns_queued(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        assert client.post("/matching/tasks", json={"jd_text": "JD"}).status_code == 400
        assert client.post("/matching/tasks", headers={"X-Tenant-ID": "!"}, json={"jd_text": "JD"}).status_code == 400
        response = create_task(client)
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "queued"
        assert payload["created"] is True


def test_background_worker_invokes_runtime_default_skill_and_mcp_source(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "api.sqlite")
    submitter = make_submitter(store)
    app = create_app(store=store, runtime_submitter=submitter)
    with TestClient(app) as client:
        task_id = create_task(client, candidate_source="mcp").json()["task_id"]
        summary = wait_for_status(client, task_id, {"completed"})

    assert submitter.calls["count"] == 1
    assert submitter.calls["tenant_ids"] == ["tenant_a"]
    assert submitter.calls["candidate_sources"] == ["mcp"]
    assert summary["status"] == "completed"
    assert summary["graph_mode"] == "skill"
    assert summary["candidate_source"] == "mcp"
    assert summary["candidate_count"] == 3


def test_task_failed_and_completed_with_fallback_summaries(tmp_path):
    fail_client, _store, _app = build_client(tmp_path / "fail", submitter=make_submitter(SQLiteRuntimeStore(tmp_path / "fail.sqlite"), fail=True))
    with fail_client:
        task_id = create_task(fail_client, idem="fail").json()["task_id"]
        assert wait_for_status(fail_client, task_id, {"failed"})["error_type"] == "InjectedFailure"

    fallback_store = SQLiteRuntimeStore(tmp_path / "fallback.sqlite")
    fallback_client, _store, _app = build_client(tmp_path / "fallback", submitter=make_submitter(fallback_store, fallback=True))
    with fallback_client:
        task_id = create_task(fallback_client, idem="fallback").json()["task_id"]
        summary = wait_for_status(fallback_client, task_id, {"completed_with_fallback"})
        assert summary["fallback_succeeded"] is True


def test_cross_tenant_task_events_feedback_are_denied(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        task_id = create_task(client).json()["task_id"]
        wait_for_status(client, task_id, {"completed"})
        assert client.get(f"/tasks/{task_id}", headers={"X-Tenant-ID": "tenant_b"}).status_code == 403
        assert client.get(f"/tasks/{task_id}/events", headers={"X-Tenant-ID": "tenant_b"}).status_code == 403
        assert client.post(
            f"/tasks/{task_id}/feedback",
            headers={"X-Tenant-ID": "tenant_b"},
            json={"feedback_type": "comment", "comment": "ok"},
        ).status_code == 403


def test_events_and_sse_are_summary_only_and_ordered(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        task_id = create_task(client).json()["task_id"]
        wait_for_status(client, task_id, {"completed"})
        events = client.get(f"/tasks/{task_id}/events", headers={"X-Tenant-ID": "tenant_a"}).json()["events"]
        rendered = json.dumps(events, ensure_ascii=False)
        assert "resume_text" not in rendered
        assert "reasoning" not in rendered
        assert any(event["event_type"] == "tool_completed" for event in events)
        cursor = events[0]["event_id"]
        after = client.get(
            f"/tasks/{task_id}/events",
            headers={"X-Tenant-ID": "tenant_a"},
            params={"after_event_id": cursor},
        ).json()["events"]
        assert all(event["event_id"] != cursor for event in after)
        with client.stream("GET", f"/tasks/{task_id}/stream", headers={"X-Tenant-ID": "tenant_a"}) as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            body = next(stream.iter_text())
            assert "task_queued" in body


def test_feedback_persists_runtime_event(tmp_path):
    client, store, _app = build_client(tmp_path)
    with client:
        task_id = create_task(client).json()["task_id"]
        wait_for_status(client, task_id, {"completed"})
        response = client.post(
            f"/tasks/{task_id}/feedback",
            headers={"X-Tenant-ID": "tenant_a"},
            json={"feedback_type": "approve", "rating": 5, "comment": "good", "candidate_id": "candidate_001"},
        )
        assert response.status_code == 200
        assert response.json()["feedback_id"]
        runtime_task_id = client.get(f"/tasks/{task_id}", headers={"X-Tenant-ID": "tenant_a"}).json()["runtime_task_id"]
        assert store.list_human_feedback_by_task(runtime_task_id)


def test_cancel_queued_running_and_terminal_states(tmp_path):
    queued_client, _store, _app = build_client(tmp_path / "queued", worker_count=0)
    with queued_client:
        task_id = create_task(queued_client).json()["task_id"]
        cancelled = queued_client.post(f"/tasks/{task_id}/cancel", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert cancelled["status"] == "cancelled"

    slow_store = SQLiteRuntimeStore(tmp_path / "slow.sqlite")
    slow_client, _store, _app = build_client(tmp_path / "slow", submitter=make_submitter(slow_store, delay=0.4))
    with slow_client:
        task_id = create_task(slow_client, idem="slow").json()["task_id"]
        running = wait_for_status(slow_client, task_id, {"running", "cancel_requested"})
        response = slow_client.post(f"/tasks/{task_id}/cancel", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert response["status"] in {"cancel_requested", "completed"}
        final = wait_for_status(slow_client, task_id, {"completed", "cancel_requested"})
        terminal_cancel = slow_client.post(f"/tasks/{task_id}/cancel", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert terminal_cancel["already_terminal"] is (final["status"] in {"completed", "completed_with_fallback", "failed", "cancelled"})


def test_idempotency_replay_conflict_and_tenant_isolation(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        first = create_task(client, idem="same", jd="JD A").json()
        second = create_task(client, idem="same", jd="JD A").json()
        assert second["task_id"] == first["task_id"]
        assert second["idempotency_replayed"] is True
        conflict = create_task(client, idem="same", jd="JD B")
        assert conflict.status_code == 409
        other = create_task(client, tenant="tenant_b", idem="same", jd="JD B")
        assert other.status_code == 200
        assert other.json()["task_id"] != first["task_id"]


def test_queue_full_timeout_metadata_limit_and_metrics(tmp_path):
    client, _store, _app = build_client(tmp_path / "queue", worker_count=0, queue_max_size=1)
    with client:
        assert create_task(client, idem="one").status_code == 200
        assert create_task(client, idem="two").status_code == 503

    timeout_store = SQLiteRuntimeStore(tmp_path / "timeout.sqlite")
    timeout_client, _store, _app = build_client(
        tmp_path / "timeout",
        submitter=make_submitter(timeout_store, delay=0.2),
        task_timeout_seconds=0.01,
    )
    with timeout_client:
        task_id = create_task(timeout_client, idem="timeout").json()["task_id"]
        assert wait_for_status(timeout_client, task_id, {"failed"})["error_type"] == "TaskTimeout"
        metrics = timeout_client.get("/metrics/summary").json()
        assert metrics["task_failure_count"] >= 1

    client2, _store, _app = build_client(tmp_path / "metadata")
    with client2:
        bad = client2.post(
            "/matching/tasks",
            headers={"X-Tenant-ID": "tenant_a", "Idempotency-Key": "badmeta"},
            json={"jd_text": "JD", "metadata": {"graph_state": "no"}},
        )
        assert bad.status_code == 422


def test_error_response_has_no_traceback(tmp_path):
    client, _store, _app = build_client(tmp_path)
    with client:
        response = client.get("/tasks/nope", headers={"X-Tenant-ID": "tenant_a"})
        data = response.json()
        assert response.status_code == 404
        assert data["error"]["type"] == "TaskNotFound"
        assert "Traceback" not in json.dumps(data)
