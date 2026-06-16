import json

from scripts.run_production_ab_baseline import run_cli
from src.integration.production_ab_baseline import (
    ProductionABBaselineConfig,
    ProductionABBaselineRunner,
    build_identity_key,
    load_baseline_manifest,
)


FULL_PRIVATE_CHUNK = "FULL PRIVATE RESUME TEXT SHOULD NOT LEAK. Python RAG LangGraph 项目细节很多。"


def legacy_ok():
    return {
        "status": "ok",
        "candidate_count": 2,
        "report_count": 2,
        "candidate_pool": [
            {
                "metadata": {
                    "candidate_id": "cand-a",
                    "document_id": "doc-a",
                    "file_name": "/tmp/private/张三_简历.pdf",
                },
                "text": FULL_PRIVATE_CHUNK,
            },
            {
                "metadata": {
                    "candidate_id": "cand-b",
                    "document_id": "doc-b",
                    "file_name": "李四_CV.pdf",
                }
            },
        ],
        "final_reports": [
            {"candidate_id": "cand-a", "candidate_name": "张三", "total_score": 88, "reasoning": "PRIVATE"},
            {"candidate_id": "cand-b", "candidate_name": "李四", "total_score": 72},
        ],
        "candidate_name_resolved_count": 2,
        "project_evidence_present_count": 2,
        "education_evidence_present_count": 2,
        "duration_ms": 100,
        "event_count": 5,
    }


def skill_ok():
    return {
        "status": "ok",
        "task_status": "completed",
        "candidate_count": 2,
        "candidate_profile_preview_count": 2,
        "report_count": 2,
        "candidate_previews": [
            {
                "candidate_id": "cand-a",
                "candidate_name": "张三",
                "source_document_id": "doc-a",
                "source_file_name": "张三_简历.pdf",
                "project_keywords": ["项目"],
                "education_keywords": ["本科"],
                "evidence_summary": "Python RAG",
            },
            {
                "candidate_id": "cand-b",
                "candidate_name": "",
                "source_document_id": "doc-b",
                "source_file_name": "第二份简历.pdf",
                "project_keywords": [],
                "education_keywords": [],
                "evidence_summary": "LangGraph",
            },
        ],
        "match_reports": [
            {"candidate_id": "cand-a", "candidate_name": "张三", "total_score": 82},
            {"candidate_id": "cand-b", "candidate_name": "未提供", "total_score": 65},
        ],
        "candidate_preview_audit": {
            "candidate_name_resolved_count": 1,
            "project_keywords_present_count": 1,
            "education_keywords_present_count": 1,
            "evidence_summary_present_count": 2,
        },
        "skill_execution_count": 4,
        "duration_ms": 150,
        "planner_fallback_used": True,
        "fallback_used": True,
    }


def skill_completed_with_limit():
    data = skill_ok()
    data.update(
        {
            "status": "failed",
            "error_type": "max_refine_loops",
            "error_hint": "max_loop_exceeded",
            "loop_count": 1,
        }
    )
    return data


def run_baseline(legacy_summary=None, skill_summary=None, **config):
    calls = {"legacy": 0, "skill": 0}

    def legacy_runner(_jd):
        calls["legacy"] += 1
        return legacy_summary or legacy_ok()

    def skill_runner(_jd):
        calls["skill"] += 1
        return skill_summary or skill_ok()

    result = ProductionABBaselineRunner(
        ProductionABBaselineConfig(enabled=True, **config)
    ).run("招聘 Python RAG LangGraph 工程师 WITH SECRET JD", legacy_runner=legacy_runner, skill_runner=skill_runner)
    return result, calls


def test_config_default_disabled():
    result = ProductionABBaselineRunner().run("招聘 Python", legacy_runner=lambda _jd: {}, skill_runner=lambda _jd: {})

    assert ProductionABBaselineConfig().enabled is False
    assert result["status"] == "skipped"


def test_legacy_and_skill_runner_each_execute_once_and_succeed():
    result, calls = run_baseline()

    assert calls == {"legacy": 1, "skill": 1}
    assert result["status"] == "ok"
    assert result["comparison"]["both_succeeded"] is True
    assert result["legacy_summary"]["candidate_count"] == 2
    assert result["skill_summary"]["skill_execution_count"] == 4


def test_legacy_success_skill_failure_recommends_high_risk_rollback():
    result, _calls = run_baseline(skill_summary={"status": "failed", "error_type": "MatcherSkillFailed"})
    comparison = result["comparison"]

    assert comparison["rollback_recommended"] is True
    assert comparison["risk_level"] == "high"
    assert comparison["decision"] == "rollback"


def test_candidate_identity_alignment_and_placeholder_names():
    assert build_identity_key({"name": "未知"}) == ""
    assert build_identity_key({"file_name": "我的简历.pdf"}) == "我的简历"
    assert build_identity_key({"metadata": {"source": "/tmp/private/张三_简历.pdf"}}) == "张三_简历"
    result, _calls = run_baseline()

    alignment = result["comparison"]["candidate_identity_alignment"]
    assert alignment["aligned_candidate_ids"] == ["cand-a", "cand-b"]
    assert alignment["unresolved_identity"] is False


def test_candidate_overlap_top_k_ranking_and_score_delta():
    result, _calls = run_baseline(top_k=2)
    comparison = result["comparison"]

    assert comparison["candidate_overlap_count"] == 2
    assert comparison["candidate_union_count"] == 2
    assert comparison["candidate_overlap_rate"] == 1.0
    assert comparison["top_k_overlap_count"] == 2
    assert comparison["top_k_overlap_rate"] == 1.0
    assert comparison["ranking_alignment"] == 1.0
    assert comparison["score_deltas"][0]["candidate_identity"] == "cand-a"
    assert comparison["score_deltas"][0]["absolute_delta"] == 6.0


def test_max_loop_with_reports_is_successful_baseline_observation():
    result, _calls = run_baseline(skill_summary=skill_completed_with_limit())
    comparison = result["comparison"]

    assert result["skill_summary"]["status"] == "completed_with_limit"
    assert result["skill_summary"]["error_type"] == ""
    assert result["skill_summary"]["error_hint"] == ""
    assert comparison["both_succeeded"] is True
    assert comparison["rollback_recommended"] is False
    assert comparison["risk_level"] == "medium"
    assert comparison["rollback_reason"] != "legacy succeeded while production skill graph failed"


def test_report_name_project_education_and_latency_deltas():
    result, _calls = run_baseline()
    comparison = result["comparison"]

    assert comparison["report_count_delta"] == 0
    assert comparison["name_resolution_delta"] == -1
    assert comparison["project_evidence_delta"] == -1
    assert comparison["education_evidence_delta"] == -1
    assert comparison["latency_delta_ms"] == 50
    assert comparison["risk_level"] == "medium"
    assert comparison["decision"] == "review"


def test_skill_zero_candidates_or_reports_triggers_rollback():
    zero_candidates, _ = run_baseline(skill_summary={"status": "ok", "candidate_count": 0, "report_count": 0})
    zero_reports, _ = run_baseline(skill_summary={"status": "ok", "candidate_count": 2, "candidate_ids": ["cand-a"], "report_count": 0})

    assert zero_candidates["comparison"]["rollback_recommended"] is True
    assert zero_candidates["comparison"]["risk_level"] == "high"
    assert zero_reports["comparison"]["rollback_recommended"] is True


def test_legacy_final_reports_and_candidate_pool_extract_summary_fields():
    summary = ProductionABBaselineRunner().summarize_legacy_result(
        {
            "status": "ok",
            "candidate_pool": [
                {"metadata": {"file_name": "/private/我的简历.pdf"}},
                {"metadata": {"source": "/private/第二份简历.pdf"}},
            ],
            "final_reports": [
                {"candidate_name": "孙一可", "total_score": 80, "reasoning": "PRIVATE"},
                {"candidate_name": "郭玉泽", "total_score": 55},
            ],
        }
    )

    assert summary.status == "ok"
    assert summary.ranking == ["孙一可", "郭玉泽"]
    assert summary.top_scores == [80.0, 55.0]
    assert summary.candidate_name_resolved_count == 2
    assert summary.document_ids == ["我的简历", "第二份简历"]


def test_skill_generated_candidate_ids_can_align_by_source_document():
    result, _calls = run_baseline(
        legacy_summary={
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "candidate_pool": [{"metadata": {"file_name": "我的简历.pdf"}}],
            "final_reports": [{"candidate_name": "孙一可", "total_score": 80}],
        },
        skill_summary={
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "candidate_previews": [
                {
                    "candidate_id": "candidate_preview_abc123",
                    "source_document_id": "我的简历.pdf",
                    "source_file_name": "我的简历.pdf",
                }
            ],
            "match_reports": [{"candidate_id": "candidate_preview_abc123", "total_score": 40}],
        },
    )

    comparison = result["comparison"]
    assert result["skill_summary"]["candidate_ids"] == ["我的简历"]
    assert result["skill_summary"]["document_ids"] == ["我的简历"]
    assert comparison["candidate_overlap_count"] == 1
    assert comparison["top_k_overlap_rate"] == 1.0
    assert comparison["score_deltas"][0]["absolute_delta"] == 40.0


def test_unavailable_ranking_is_not_faked():
    result, _calls = run_baseline(
        legacy_summary={
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "candidate_ids": ["cand-a"],
            "top_scores": [80],
        },
        skill_summary={
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "candidate_ids": ["cand-a"],
            "top_scores": [81],
        },
    )

    assert result["comparison"]["ranking_alignment"] == "unavailable"
    assert result["comparison"]["risk_level"] == "medium"


def test_json_summary_does_not_leak_full_jd_chunk_or_reasoning():
    result, _calls = run_baseline()
    payload = json.dumps(result, ensure_ascii=False)

    assert "WITH SECRET JD" not in payload
    assert "FULL PRIVATE RESUME TEXT SHOULD NOT LEAK" not in payload
    assert "PRIVATE" not in payload
    assert result["metadata"]["summary_only"] is True


def test_baseline_manifest_can_load():
    manifest = load_baseline_manifest("config/evaluation_baseline.json")

    assert manifest["baseline_version"] == "phase11b-v1"
    assert manifest["legacy_runner"] == "legacy_default_graph"
    assert manifest["skill_runner"] == "production_skill_graph"
    assert manifest["memory_enabled"] is False
    assert manifest["mcp_enabled"] is False


def test_cli_json_uses_fake_injected_runners_and_default_graph_not_replaced(capsys):
    exit_code = run_cli(
        ["--jd", "招聘 Python RAG", "--json"],
        legacy_runner=lambda _jd: legacy_ok(),
        skill_runner=lambda _jd: skill_ok(),
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["metadata"]["production_graph_replaced"] is False
    assert output["metadata"]["memory_enabled"] is False
    assert output["metadata"]["mcp_enabled"] is False
    assert output["comparison"]["top_k_overlap_rate"] == 1.0


def test_strict_cli_returns_nonzero_on_rollback(capsys):
    exit_code = run_cli(
        ["--jd", "招聘 Python", "--json", "--strict"],
        legacy_runner=lambda _jd: legacy_ok(),
        skill_runner=lambda _jd: {"status": "failed", "error_type": "MatcherSkillFailed"},
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["rollback_recommended"] is True
