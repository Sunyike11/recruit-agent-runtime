import json
import time

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.schemas import CreateMatchingTaskRequest
from src.memory.review_loop import ReviewMemoryStore, validate_correction_payload
from src.runtime.entry import RuntimeEntryResult
from src.runtime.event_envelope import build_runtime_event_payload
from src.runtime.memory_influence import MemoryInfluenceEvaluator
from src.runtime.models import TaskStatus
from src.runtime.sqlite_store import SQLiteRuntimeStore


def make_submitter(store):
    calls = {"memory_modes": [], "candidate_sources": []}

    def submit(request: CreateMatchingTaskRequest, tenant_id: str):
        calls["memory_modes"].append(request.memory_mode)
        calls["candidate_sources"].append(request.candidate_source)
        session = store.create_session(metadata={"tenant_id": tenant_id, "api_test": True})
        task = store.create_task(
            session.session_id,
            input_payload={
                "jd_text": request.jd_text,
                "metadata": {
                    "tenant_id": tenant_id,
                    "candidate_source": request.candidate_source,
                    "memory_mode": request.memory_mode,
                    "summary_only": True,
                },
            },
        )
        store.update_task_status(task.task_id, TaskStatus.RUNNING)
        for event_type in ["task_started", "graph_started", "skill_started", "skill_completed"]:
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
                    skill_name="candidate_match" if event_type.startswith("skill") else "",
                    status="completed" if event_type.endswith("completed") else "started",
                ),
            )
        memory_provided = request.memory_mode == "governed"
        output = {
            "status": "ok",
            "selected_graph_mode": "skill",
            "candidate_source": request.candidate_source,
            "candidate_count": 2,
            "report_count": 2,
            "candidate_preview_version": "v2",
            "claim_verification_enabled": True,
            "claim_verification_status": "review_required",
            "critical_unsupported_claim_count": 1,
            "human_review_status": "pending" if request.memory_mode == "off" else "not_required",
            "effective_result_status": "original",
            "memory_mode": request.memory_mode,
            "memory_context_requested": memory_provided,
            "memory_context_provided": memory_provided,
            "memory_records_seen": 1 if memory_provided else 0,
            "memory_context_eligible_count": 1 if memory_provided else 0,
            "memory_context_denied_count": 0,
            "memory_context_rendered_char_count": 80 if memory_provided else 0,
            "memory_context_governance_applied": memory_provided,
            "memory_ids_used": ["memory_test"] if memory_provided else [],
            "memory_versions_used": [1] if memory_provided else [],
            "summary_only": True,
        }
        store.update_task_status(task.task_id, TaskStatus.COMPLETED, result=output)
        store.append_event(
            "task_completed",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode="skill",
                runner_name="production_skill_graph",
                status="completed",
            ),
        )
        return RuntimeEntryResult(
            status="ok",
            session_id=session.session_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            runner_used="production_skill_graph",
            task_status="completed",
            event_count=len(store.list_events(task_id=task.task_id)),
            output_summary=output,
            summary_only=True,
        )

    submit.calls = calls
    return submit


def build_client(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "api.sqlite")
    submitter = make_submitter(store)
    app = create_app(
        store=store,
        runtime_submitter=submitter,
        db_path=str(tmp_path / "api.sqlite"),
        worker_count=1,
        queue_max_size=10,
        task_timeout_seconds=5,
    )
    return TestClient(app), store, submitter


def headers(tenant="tenant_a", key="idem-1"):
    values = {"X-Tenant-ID": tenant}
    if key:
        values["Idempotency-Key"] = key
    return values


def create_task(client, *, tenant="tenant_a", idem="idem-1", memory_mode="off"):
    response = client.post(
        "/matching/tasks",
        headers=headers(tenant, idem),
        json={
            "jd_text": "招聘 Python RAG 工程师",
            "candidate_source": "mcp",
            "memory_mode": memory_mode,
            "metadata": {"request_source": "test"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["task_id"]


def wait_task(client, task_id, tenant="tenant_a"):
    for _ in range(80):
        payload = client.get(f"/tasks/{task_id}", headers=headers(tenant, "")).json()
        if payload["status"] in {"completed", "completed_with_fallback", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_feedback_review_decision_memory_activation_revoke_and_governed_use(tmp_path):
    client, _store, submitter = build_client(tmp_path)
    with client:
        task_id = create_task(client)
        summary = wait_task(client, task_id)
        assert summary["human_review_status"] == "pending"
        response = client.post(
            f"/tasks/{task_id}/feedback",
            headers=headers("tenant_a", ""),
            json={
                "feedback_type": "correction",
                "rating": 2,
                "comment": "候选人没有已发表顶会论文",
                "candidate_id": "candidate_001",
                "report_id": "report_001",
                "resume_version_id": "rv1",
                "profile_version_id": "pv1",
                "claim_ids": ["claim_publication_1"],
                "correction": {"publication_status": "under_review"},
                "request_review": True,
            },
        )
        assert response.status_code == 200, response.text
        feedback_payload = response.json()
        assert feedback_payload["review_id"]
        assert feedback_payload["status"] == "review_created"

        feedback = client.get(f"/tasks/{task_id}/feedback", headers=headers("tenant_a", "")).json()["feedback"]
        assert feedback[0]["comment_length"] > 0
        assert "候选人" not in json.dumps(feedback, ensure_ascii=False)
        assert feedback[0]["resume_version_id"] == "rv1"

        reviews = client.get("/reviews", headers=headers("tenant_a", "")).json()["reviews"]
        assert reviews[0]["status"] == "pending"
        assert reviews[0]["review_type"] == "feedback_correction_review"
        review_id = reviews[0]["review_id"]

        decision = client.post(
            f"/reviews/{review_id}/decision",
            headers=headers("tenant_a", ""),
            json={
                "decision": "correct",
                "reason": "人工确认论文是在投，不是已发表。",
                "correction": {"publication_status": "under_review"},
                "promote_to_memory": True,
                "memory_candidate_type": "matching_rule",
            },
        )
        assert decision.status_code == 200, decision.text
        decided = decision.json()
        assert decided["review"]["status"] == "corrected"
        memory_candidate_id = decided["memory_candidate"]["memory_candidate_id"]
        assert decided["memory_candidate"]["status"] == "pending_review"

        approved = client.post(
            f"/memory-candidates/{memory_candidate_id}/approve",
            headers=headers("tenant_a", ""),
        )
        assert approved.status_code == 200, approved.text
        memory = approved.json()["memory"]
        assert memory["status"] == "active"
        assert memory["version"] == 1

        governed_task_id = create_task(client, idem="idem-governed", memory_mode="governed")
        governed = wait_task(client, governed_task_id)
        assert governed["memory_mode"] == "governed"
        assert governed["memory_context_requested"] is True
        assert governed["memory_context_provided"] is True
        assert governed["memory_eligible_count"] >= 1
        assert submitter.calls["memory_modes"][-1] == "governed"

        revoked = client.post(f"/memories/{memory['memory_id']}/revoke", headers=headers("tenant_a", ""))
        assert revoked.status_code == 200
        assert revoked.json()["memory"]["status"] == "revoked"
        assert client.get("/memories", headers=headers("tenant_a", ""), params={"status": "active"}).json()["memories"] == []


def test_comment_does_not_create_high_priority_review_or_memory(tmp_path):
    client, _store, _submitter = build_client(tmp_path)
    with client:
        task_id = create_task(client)
        wait_task(client, task_id)
        response = client.post(
            f"/tasks/{task_id}/feedback",
            headers=headers("tenant_a", ""),
            json={"feedback_type": "comment", "comment": "普通备注"},
        )
        assert response.status_code == 200
        assert response.json()["review_id"] == ""
        assert client.get("/reviews", headers=headers("tenant_a", "")).json()["reviews"] == []


def test_tenant_isolation_for_feedback_reviews_and_memories(tmp_path):
    client, _store, _submitter = build_client(tmp_path)
    with client:
        task_id = create_task(client, tenant="tenant_a")
        wait_task(client, task_id, tenant="tenant_a")
        response = client.post(
            f"/tasks/{task_id}/feedback",
            headers=headers("tenant_a", ""),
            json={"feedback_type": "reject", "comment": "wrong", "request_review": True},
        )
        review_id = response.json()["review_id"]
        assert client.get(f"/tasks/{task_id}/feedback", headers=headers("tenant_b", "")).status_code == 403
        assert client.get(f"/reviews/{review_id}", headers=headers("tenant_b", "")).status_code == 403
        assert client.post(
            f"/reviews/{review_id}/decision",
            headers=headers("tenant_b", ""),
            json={"decision": "reject"},
        ).status_code == 403


def test_review_store_guards_immutability_double_decision_and_promotion_rules(tmp_path):
    store = ReviewMemoryStore(tmp_path / "review.sqlite")
    feedback = store.create_feedback(
        tenant_id="tenant_a",
        task_id="task_1",
        feedback_type="correction",
        comment="x" * 20,
        correction={"education": "硕士待确认"},
        candidate_id="candidate_1",
    )
    review = store.create_review_from_feedback(feedback)
    assert review is not None
    decision, updated, candidate = store.decide_review(
        tenant_id="tenant_a",
        review_id=review.review_id,
        decision="correct",
        correction={"education": "硕士待确认"},
        promote_to_memory=True,
        memory_candidate_type="candidate_constraint",
    )
    assert updated.status == "corrected"
    assert candidate is not None
    try:
        store.decide_review(tenant_id="tenant_a", review_id=review.review_id, decision="approve")
        raise AssertionError("double decision should fail")
    except ValueError as exc:
        assert "terminal" in str(exc)
    comment = store.create_feedback(tenant_id="tenant_a", task_id="task_1", feedback_type="comment", comment="just note")
    comment_review = store.create_review(
        tenant_id="tenant_a",
        review_type="match_report_review",
        task_id="task_1",
        feedback_id=comment.feedback_id,
    )
    try:
        store.decide_review(
            tenant_id="tenant_a",
            review_id=comment_review.review_id,
            decision="approve",
            promote_to_memory=True,
            memory_candidate_type="tenant_preference",
        )
        raise AssertionError("pure comment promotion should fail")
    except ValueError as exc:
        assert "promotable" in str(exc)
    try:
        validate_correction_payload({"graph_state": "bad"})
        raise AssertionError("invalid correction field should fail")
    except ValueError:
        pass


def test_events_metrics_summary_only_and_memory_influence(tmp_path):
    client, _store, _submitter = build_client(tmp_path)
    with client:
        task_id = create_task(client)
        wait_task(client, task_id)
        response = client.post(
            f"/tasks/{task_id}/feedback",
            headers=headers("tenant_a", ""),
            json={
                "feedback_type": "evidence_missing",
                "comment": "PRIVATE-FEEDBACK-CONTENT-MUST-NOT-LEAK",
                "request_review": True,
            },
        )
        review_id = response.json()["review_id"]
        client.post(
            f"/reviews/{review_id}/decision",
            headers=headers("tenant_a", ""),
            json={"decision": "approve", "promote_to_memory": True, "memory_candidate_type": "tenant_preference"},
        )
        events = client.get(f"/tasks/{task_id}/events", headers=headers("tenant_a", "")).json()["events"]
        rendered = json.dumps(events, ensure_ascii=False)
        assert "PRIVATE-FEEDBACK-CONTENT-MUST-NOT-LEAK" not in rendered
        assert any(event["event_type"] == "feedback_submitted" for event in events)
        assert any(event["event_type"] == "review_decided" for event in events)
        metrics = client.get("/metrics/summary").json()
        assert metrics["feedback_submitted_count"] >= 1
        assert metrics["review_created_count"] >= 1
        assert metrics["human_intervention_rate"] >= 0

    evaluator = MemoryInfluenceEvaluator()
    delta = evaluator.compare_summaries(
        {"status": "ok", "candidate_count": 2, "report_count": 2, "candidate_ids": ["a", "b"]},
        {
            "status": "ok",
            "candidate_count": 2,
            "report_count": 2,
            "candidate_ids": ["b", "a"],
            "memory_context_provided": True,
            "memory_context_eligible_count": 1,
        },
    )
    assert delta.memory_context_used is True
    assert delta.decision == "warning"


def test_phase16b2_required_tests_do_not_need_network_or_real_llm(tmp_path):
    client, _store, _submitter = build_client(tmp_path)
    with client:
        assert client.get("/healthz").json()["status"] == "ok"
