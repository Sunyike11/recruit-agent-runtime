import json
from pathlib import Path

from scripts.run_recruit_runtime import run_cli
from src.core.graph_factory import RecruitGraphMode, resolve_recruit_graph_factory_config
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.inspect import RuntimeInspector
from src.runtime.models import TaskStatus
from src.runtime.store import InMemoryRuntimeStore


def legacy_ok(raw_jd, metadata=None):
    return {
        "status": "ok",
        "candidate_count": 2,
        "report_count": 2,
        "top_score_present": True,
        "output_keys": ["candidate_pool", "final_reports"],
        "metadata": {"summary_only": True, "metadata_seen": bool(metadata is not None)},
    }


def legacy_failed(raw_jd, metadata=None):
    return {
        "status": "failed",
        "candidate_count": 0,
        "report_count": 0,
        "error_type": "LegacyGraphFailed",
        "error_hint": "legacy_failed",
        "metadata": {"summary_only": True},
    }


def skill_ok(raw_jd, memory_context=None, metadata=None):
    return {
        "status": "ok",
        "candidate_count": 2,
        "candidate_profile_preview_count": 2,
        "candidate_preview_version": "v2",
        "report_count": 2,
        "top_score_present": True,
        "skill_names": ["planner_extract", "resume_retrieve", "candidate_match", "candidate_match"],
        "skill_execution_count": 4,
        "skill_event_count": 4,
        "production_skill_graph_enabled": True,
        "legacy_graph_invoked": False,
        "rollback_recommended": False,
        "metadata": {"summary_only": True},
    }


def skill_failed(error_type="PlannerSkillFailed", error_hint="planner_failed"):
    def run(raw_jd, memory_context=None, metadata=None):
        return {
            "status": "failed",
            "candidate_count": 0,
            "report_count": 0,
            "top_score_present": False,
            "error_type": error_type,
            "error_hint": error_hint,
            "skill_names": ["planner_extract"],
            "skill_execution_count": 1,
            "production_skill_graph_enabled": True,
            "legacy_graph_invoked": False,
            "metadata": {"summary_only": True},
        }

    return run


def skill_empty(candidate_count=0, report_count=0):
    def run(raw_jd, memory_context=None, metadata=None):
        return {
            "status": "ok",
            "candidate_count": candidate_count,
            "report_count": report_count,
            "top_score_present": False,
            "candidate_preview_version": "v2",
            "production_skill_graph_enabled": True,
            "metadata": {"summary_only": True},
        }

    return run


def skill_degraded(raw_jd, memory_context=None, metadata=None):
    return {
        "status": "completed_with_limit",
        "candidate_count": 2,
        "report_count": 2,
        "top_score_present": True,
        "candidate_preview_version": "v2",
        "rollback_recommended": True,
        "metadata": {"summary_only": True},
    }


def test_factory_default_mode_is_skill():
    config = resolve_recruit_graph_factory_config(env={})

    assert config.mode == RecruitGraphMode.SKILL
    assert config.default_graph_mode == RecruitGraphMode.SKILL
    assert config.skill_default_used is True
    assert config.selection_source == "default"


def test_cli_default_runs_skill_with_injected_runner(capsys):
    code = run_cli(["--jd", "招聘JD", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["runner_used"] == "production_skill_graph"
    assert payload["output_summary"]["selected_graph_mode"] == "skill"
    assert payload["output_summary"]["default_graph_mode"] == "skill"
    assert payload["output_summary"]["skill_default_used"] is True
    assert payload["output_summary"]["fallback_attempted"] is False


def test_cli_explicit_legacy_and_env_legacy(monkeypatch, capsys):
    run_cli(["--jd", "招聘JD", "--graph-mode", "legacy", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner_used"] == "default_graph"
    assert payload["output_summary"]["legacy_explicitly_requested"] is True

    monkeypatch.setenv("RECRUIT_GRAPH_MODE", "legacy")
    run_cli(["--jd", "招聘JD", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner_used"] == "default_graph"
    assert payload["output_summary"]["selection_source"] == "environment"


def test_cli_precedence_and_invalid_env(monkeypatch, capsys):
    monkeypatch.setenv("RECRUIT_GRAPH_MODE", "legacy")
    run_cli(["--jd", "招聘JD", "--graph-mode", "skill", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner_used"] == "production_skill_graph"
    assert payload["output_summary"]["selection_source"] == "cli"

    monkeypatch.setenv("RECRUIT_GRAPH_MODE", "bogus")
    run_cli(["--jd", "招聘JD", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["output_summary"]["error_hint"] == "invalid_env_graph_mode"


def test_legacy_alias_is_noop_skill_alias(capsys):
    run_cli(["--jd", "招聘JD", "--use-production-skill-graph", "--json"], default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = json.loads(capsys.readouterr().out)

    assert payload["runner_used"] == "production_skill_graph"
    assert payload["output_summary"]["legacy_alias_used"] is True
    assert payload["output_summary"]["selection_reason"] == "deprecated_alias_use_production_skill_graph"


def test_skill_hard_failure_fallback_succeeds_same_task_thread():
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run(
        "敏感 JD 不应泄露",
        default_runner=legacy_ok,
        production_skill_graph_runner=skill_failed("RetrieverSkillFailed", "retriever_failed"),
        store=store,
        config=RuntimeEntryConfig(),
    )
    inspection = RuntimeInspector().inspect_task(result.task_id, store)
    rendered = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.status == "completed_with_fallback"
    assert result.task_status == TaskStatus.COMPLETED_WITH_FALLBACK.value
    assert result.output_summary["primary_graph_mode"] == "skill"
    assert result.output_summary["fallback_attempted"] is True
    assert result.output_summary["fallback_graph_mode"] == "legacy"
    assert result.output_summary["fallback_succeeded"] is True
    assert result.output_summary["final_runner_used"] == "default_graph"
    assert result.output_summary["graph_health_status"] == "degraded"
    assert "敏感 JD" not in rendered
    assert "graph_fallback_requested" in inspection.event_types
    assert "graph_fallback_started" in inspection.event_types
    assert "graph_fallback_completed" in inspection.event_types
    assert inspection.session_id == result.session_id
    assert inspection.thread_id == result.thread_id
    assert result.output_summary["primary_attempt_id"] != result.output_summary["fallback_attempt_id"]


def test_fallback_covers_planner_matcher_schema_empty_candidates_and_empty_reports():
    cases = [
        skill_failed("PlannerSkillFailed", "planner_failed"),
        skill_failed("MatcherSkillFailed", "matcher_failed"),
        skill_failed("SchemaInvalid", "schema_invalid"),
        skill_empty(candidate_count=0, report_count=0),
        skill_empty(candidate_count=2, report_count=0),
    ]
    for runner in cases:
        result = RuntimeEntryHarness().run("招聘JD", default_runner=legacy_ok, production_skill_graph_runner=runner)
        assert result.status == "completed_with_fallback"
        assert result.output_summary["fallback_succeeded"] is True


def test_quality_warning_and_completed_with_limit_do_not_fallback():
    result = RuntimeEntryHarness().run("招聘JD", default_runner=legacy_ok, production_skill_graph_runner=skill_degraded)

    assert result.status == "ok"
    assert result.output_summary["graph_health_status"] == "degraded"
    assert result.output_summary["fallback_attempted"] is False
    assert result.output_summary["selected_graph_mode"] == "skill"


def test_disable_legacy_fallback_and_fallback_failure():
    no_fallback = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=legacy_ok,
        production_skill_graph_runner=skill_failed(),
        config=RuntimeEntryConfig(legacy_fallback_enabled=False),
    )
    assert no_fallback.status == "failed"
    assert no_fallback.output_summary["fallback_attempted"] is False

    fallback_failed = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=legacy_failed,
        production_skill_graph_runner=skill_failed(),
    )
    assert fallback_failed.status == "failed"
    assert fallback_failed.output_summary["fallback_attempted"] is True
    assert fallback_failed.output_summary["fallback_succeeded"] is False
    assert fallback_failed.output_summary["graph_health_status"] == "critical"


def test_explicit_legacy_does_not_fallback_to_skill():
    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=legacy_failed,
        production_skill_graph_runner=skill_ok,
        config=RuntimeEntryConfig(graph_mode="legacy"),
    )

    assert result.status == "failed"
    assert result.output_summary["selected_graph_mode"] == "legacy"
    assert result.output_summary["fallback_attempted"] is False
    assert result.output_summary["skill_graph_invoked"] is False


def test_main_py_defaults_to_runtime_skill_entry():
    source = Path("main.py").read_text(encoding="utf-8")

    assert "RuntimeEntryHarness" in source
    assert "resolve_recruit_graph_factory_config" in source
    assert "create_recruit_graph()" not in source


def test_summary_only_memory_mcp_and_preview_v2_preserved():
    result = RuntimeEntryHarness().run("招聘JD", default_runner=legacy_ok, production_skill_graph_runner=skill_ok)
    payload = result.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False)

    assert payload["output_summary"]["candidate_preview_version"] == "v2"
    assert payload["output_summary"]["memory_context_provided"] is False
    assert "mcp" not in rendered.lower()
    assert "招聘JD" not in rendered
