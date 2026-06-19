import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.runtime.entry import RuntimeEntryResult
from src.domain.candidate_management import CandidateSQLiteStore, ResumeBlobStore
from src.mcp.candidate_provider import ManagedCandidateDataProvider
from src.skills.execution import SkillExecutionRecorder, SkillExecutor
from src.skills.registry import SkillRegistry
from src.skills.resume_ingestion import EvidenceExtractSkill, ResumeParseSkill


def client(tmp_path):
    db_path = tmp_path / "runtime.sqlite"

    def fake_matching_submitter(request, tenant_id):
        provider = ManagedCandidateDataProvider(db_path=db_path)
        found = provider.search_candidates(
            query=request.jd_text,
            top_k=5,
            tenant_id=tenant_id,
            access_scope="managed_candidates",
        )
        return RuntimeEntryResult(
            status="ok",
            session_id="runtime_session_fake",
            task_id="runtime_task_fake",
            thread_id="runtime_thread_fake",
            runner_used="production_skill_graph",
            task_status="completed",
            event_count=0,
            output_summary={
                "selected_graph_mode": "skill",
                "candidate_source": "mcp",
                "candidate_count": found["result_count"],
                "report_count": found["result_count"],
                "tool_success_count": 1,
                "summary_only": True,
            },
        )

    app = create_app(db_path=str(db_path), runtime_submitter=fake_matching_submitter, worker_count=1, queue_max_size=5, task_timeout_seconds=20)
    with TestClient(app) as test_client:
        yield test_client


def headers(key="k1", tenant="tenant_a"):
    return {"X-Tenant-ID": tenant, "Idempotency-Key": key}


def wait_task(c, task_id, tenant="tenant_a"):
    for _ in range(80):
        payload = c.get(f"/tasks/{task_id}", headers={"X-Tenant-ID": tenant}).json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def create_candidate(c, key="candidate-1", tenant="tenant_a"):
    response = c.post("/candidates", headers=headers(key, tenant), json={"external_ref": "demo", "metadata": {"request_source": "test"}})
    assert response.status_code == 200, response.text
    return response.json()["candidate_id"]


def upload(c, candidate_id, content, filename="resume.txt", key="upload-1", tenant="tenant_a", content_type="text/plain"):
    response = c.post(
        f"/candidates/{candidate_id}/resume-versions",
        headers=headers(key, tenant),
        files={"file": (filename, content, content_type)},
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_candidate_create_upload_ingest_profile_mcp_and_match(tmp_path):
    with next(client(tmp_path)) as c:
        candidate_id = create_candidate(c)
        body = """
姓名: 匿名工程师
教育经历: 计算机本科
技能: Python RAG LangGraph FastAPI
项目经历: 使用 LangGraph 和 RAG 构建招聘匹配系统，负责 Retriever、Matcher 与 Candidate MCP 接入。
工作经历: 三年 Python 后端与 Agent 系统开发经验。
""".encode()
        uploaded = upload(c, candidate_id, body)
        assert uploaded["version_number"] == 1
        assert uploaded["content_hash_prefix"]
        summary = wait_task(c, uploaded["ingestion_task_id"])
        assert summary["status"] == "completed"
        assert summary["task_type"] == "candidate_ingestion"
        events = c.get(f"/tasks/{uploaded['ingestion_task_id']}/events", headers={"X-Tenant-ID": "tenant_a"}).json()["events"]
        names = {event.get("event_type") for event in events}
        assert {"workflow_started", "skill_started", "skill_completed", "index_completed", "workflow_completed"}.issubset(names)
        assert "resume_text" not in json.dumps(events, ensure_ascii=False)

        candidate = c.get(f"/candidates/{candidate_id}", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert candidate["active_resume_version_id"] == uploaded["resume_version_id"]
        profile = c.get(f"/candidates/{candidate_id}/profile", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert profile["schema_version"] == "candidate_profile_preview_v2"
        assert profile["profile"]["candidate_id"] == candidate_id

        provider = ManagedCandidateDataProvider(db_path=tmp_path / "runtime.sqlite")
        found = provider.search_candidates(query="Python RAG LangGraph", top_k=5, tenant_id="tenant_a", access_scope="managed_candidates")
        assert found["results"][0]["candidate_id"] == candidate_id
        mcp_profile = provider.get_candidate_profile(candidate_id=candidate_id, tenant_id="tenant_a", access_scope="managed_candidates")
        assert mcp_profile["resume_version_id"] == uploaded["resume_version_id"]
        evidence = provider.get_resume_evidence(candidate_id=candidate_id, tenant_id="tenant_a", access_scope="managed_candidates")
        assert evidence["evidence_count"] > 0
        assert evidence["evidence"][0]["provenance"]["resume_version_id"] == uploaded["resume_version_id"]

        match = c.post(
            "/matching/tasks",
            headers=headers("match-1", "tenant_a"),
            json={"jd_text": "招聘 Python RAG LangGraph Agent 工程师", "candidate_source": "mcp"},
        ).json()
        match_summary = wait_task(c, match["task_id"])
        assert match_summary["status"] in {"completed", "completed_with_fallback"}
        assert match_summary["candidate_count"] >= 1


def test_duplicate_content_and_new_version_active_switch(tmp_path):
    with next(client(tmp_path)) as c:
        candidate_id = create_candidate(c)
        first = upload(c, candidate_id, "姓名: A\n技能: Python RAG\n项目: RAG 系统".encode(), key="up1")
        wait_task(c, first["ingestion_task_id"])
        duplicate = upload(c, candidate_id, "姓名: A\n技能: Python RAG\n项目: RAG 系统".encode(), key="up2")
        assert duplicate["created"] is False
        assert duplicate["duplicate_content"] is True
        assert duplicate["resume_version_id"] == first["resume_version_id"]
        second = upload(c, candidate_id, "姓名: A\n技能: Java Redis\n项目: 后端系统".encode(), key="up3")
        assert second["created"] is True
        assert second["version_number"] == 2
        wait_task(c, second["ingestion_task_id"])
        versions = c.get(f"/candidates/{candidate_id}/resume-versions", headers={"X-Tenant-ID": "tenant_a"}).json()["resume_versions"]
        assert [item["version_number"] for item in versions] == [2, 1]
        candidate = c.get(f"/candidates/{candidate_id}", headers={"X-Tenant-ID": "tenant_a"}).json()
        assert candidate["active_resume_version_id"] == second["resume_version_id"]


def test_upload_pdf_docx_unsupported_empty_and_tenant_isolation(tmp_path):
    with next(client(tmp_path)) as c:
        candidate_a = create_candidate(c, tenant="tenant_a")
        pdf = upload(c, candidate_a, "姓名: PDF\n技能: Python".encode(), filename="resume.pdf", content_type="application/pdf", key="pdf")
        docx = upload(c, candidate_a, "姓名: DOCX\n技能: Java".encode(), filename="resume.docx", content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="docx")
        assert pdf["resume_version_id"] != docx["resume_version_id"]
        assert c.post(
            f"/candidates/{candidate_a}/resume-versions",
            headers=headers("bad"),
            files={"file": ("bad.exe", b"x", "application/octet-stream")},
        ).status_code == 415
        assert c.post(
            f"/candidates/{candidate_a}/resume-versions",
            headers=headers("empty"),
            files={"file": ("empty.txt", b"", "text/plain")},
        ).status_code == 400
        assert c.get(f"/candidates/{candidate_a}", headers={"X-Tenant-ID": "tenant_b"}).status_code == 403
        provider = ManagedCandidateDataProvider(db_path=tmp_path / "runtime.sqlite")
        assert provider.search_candidates(query="Python", top_k=5, tenant_id="tenant_b", access_scope="managed_candidates")["result_count"] == 0


def test_resume_parse_and_evidence_extract_skills_execute(tmp_path):
    registry = SkillRegistry()
    registry.register(ResumeParseSkill())
    registry.register(EvidenceExtractSkill())
    executor = SkillExecutor(registry)
    parsed = executor.execute(
        "resume_parse",
        {"candidate_id": "candidate_x", "resume_version_id": "rv1", "filename": "a.txt", "media_type": "text/plain", "content_bytes": "姓名: X\n技能: Python\n项目: RAG 系统".encode()},
    )
    assert parsed.success
    extracted = executor.execute("evidence_extract", {**parsed.output, "tenant_id": "tenant_a"})
    assert extracted.success
    assert extracted.output["profile"]["preview_version"] == "v2"
    assert extracted.output["evidence_count"] > 0
