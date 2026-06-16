import json
from pathlib import Path

from scripts.run_recruit_runtime import run_cli
from src.core.graph_factory import (
    RecruitGraphFactory,
    RecruitGraphMode,
    resolve_recruit_graph_factory_config,
)
from src.integration.production_skill_graph import ProductionSkillGraphConfig, ProductionSkillGraphRunner
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.inspect import RuntimeInspector
from src.runtime.store import InMemoryRuntimeStore


def legacy_runner(raw_jd, metadata=None):
    return {
        "status": "ok",
        "candidate_count": 1,
        "report_count": 1,
        "top_score_present": True,
        "output_keys": ["candidate_pool", "final_reports"],
        "metadata": {
            "summary_only": True,
            "metadata_seen": bool(metadata is not None),
        },
    }


def skill_runner(raw_jd, memory_context=None, metadata=None):
    return {
        "status": "ok",
        "candidate_count": 2,
        "candidate_profile_preview_count": 2,
        "report_count": 2,
        "top_score_present": True,
        "skill_names": ["planner_extract", "resume_retrieve", "candidate_match", "candidate_match"],
        "skill_event_count": 4,
        "skill_execution_count": 4,
        "production_skill_graph_enabled": True,
        "legacy_graph_invoked": False,
        "rollback_recommended": False,
        "rollback_target": "legacy_default_graph",
        "candidate_preview_version": "v2",
        "metadata": {
            "summary_only": True,
            "metadata_graph_mode": (metadata or {}).get("graph_mode"),
            "memory_context_seen": bool(memory_context is not None),
        },
    }


def planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {"source": "FakePlanner", "search_query": "Python RAG LangGraph"},
        }
    }


def retriever(_input, _context=None):
    return {
        "resume_documents": [
            {
                "text": "张三 本科 计算机，负责 Python RAG LangGraph Agent 平台项目，完成检索、匹配和评估。",
                "metadata": {
                    "candidate_id": "candidate-a",
                    "candidate_name": "张三",
                    "document_id": "doc-a",
                    "file_name": "zhangsan.pdf",
                },
            },
            {
                "text": "李四 硕士 软件工程，参与 Python FastAPI RAG 检索系统和自动化测试。",
                "metadata": {
                    "candidate_id": "candidate-b",
                    "candidate_name": "李四",
                    "document_id": "doc-b",
                    "file_name": "lisi.pdf",
                },
            },
        ],
        "metadata": {"summary_only": True},
    }


def matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 88 if candidate["candidate_id"] == "candidate-a" else 76,
        "match_report": {
            "candidate_id": candidate["candidate_id"],
            "total_score": 88 if candidate["candidate_id"] == "candidate-a" else 76,
            "metadata": {"source": "FakeMatcher"},
        },
        "metadata": {"source": "FakeMatcher"},
    }


def refiner(_input, _context=None):
    return {"refined_query": "Python RAG LangGraph refined"}


def build_fake_production_runner():
    return ProductionSkillGraphRunner(
        ProductionSkillGraphConfig(enabled=True, allow_planner_fallback=True, max_refine_loops=1),
        planner_extract_callable=planner,
        retrieve_callable=retriever,
        match_callable=matcher,
        refine_callable=refiner,
    ).run


def test_graph_factory_defaults_to_skill_after_phase14b():
    config = resolve_recruit_graph_factory_config(env={})
    selection = RecruitGraphFactory(legacy_runner=legacy_runner, skill_runner=skill_runner, config=config).create_runner()

    assert config.mode == RecruitGraphMode.SKILL
    assert selection.runner_name == "production_skill_graph"
    assert selection.to_dict()["selected_graph_mode"] == "skill"
    assert selection.to_dict()["default_graph_mode"] == "skill"
    assert selection.to_dict()["skill_default_used"] is True
    assert selection.to_dict()["rollback_target"] == "legacy"


def test_graph_factory_selects_skill_explicitly():
    config = resolve_recruit_graph_factory_config(requested_graph_mode="skill", env={})
    selection = RecruitGraphFactory(legacy_runner=legacy_runner, skill_runner=skill_runner, config=config).create_runner()

    assert selection.runner is skill_runner
    assert selection.runner_name == "production_skill_graph"
    assert selection.to_dict()["selection_reason"] == "explicit_cli_graph_mode"


def test_old_production_skill_flag_maps_to_skill_alias():
    config = resolve_recruit_graph_factory_config(use_production_skill_graph_alias=True, env={})

    assert config.mode == RecruitGraphMode.SKILL
    assert config.legacy_alias_used is True
    assert config.selection_reason == "deprecated_alias_use_production_skill_graph"


def test_conflicting_graph_mode_flags_are_rejected(capsys):
    code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--use-production-skill-graph", "--json"],
        default_runner=legacy_runner,
        production_skill_graph_runner=skill_runner,
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "failed"
    assert payload["output_summary"]["error_type"] == "RuntimeConfigError"
    assert payload["output_summary"]["error_hint"] == "conflicting_graph_mode_flags"


def test_environment_graph_mode_and_cli_precedence(monkeypatch, capsys):
    monkeypatch.setenv("RECRUIT_GRAPH_MODE", "skill")
    run_cli(["--jd", "招聘JD", "--json"], default_runner=legacy_runner, production_skill_graph_runner=skill_runner)
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner_used"] == "production_skill_graph"
    assert payload["output_summary"]["selected_graph_mode"] == "skill"

    run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=legacy_runner,
        production_skill_graph_runner=skill_runner,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner_used"] == "default_graph"
    assert payload["output_summary"]["selected_graph_mode"] == "legacy"


def test_runtime_entry_uses_unified_entry_for_legacy_and_skill():
    legacy_store = InMemoryRuntimeStore()
    legacy_result = RuntimeEntryHarness().run(
        "敏感 JD 不应输出",
        default_runner=legacy_runner,
        store=legacy_store,
        config=RuntimeEntryConfig(graph_mode="legacy"),
    )
    skill_store = InMemoryRuntimeStore()
    skill_result = RuntimeEntryHarness().run(
        "敏感 JD 不应输出",
        default_runner=legacy_runner,
        production_skill_graph_runner=skill_runner,
        store=skill_store,
        config=RuntimeEntryConfig(graph_mode="skill"),
    )

    assert legacy_result.runner_used == "default_graph"
    assert skill_result.runner_used == "production_skill_graph"
    assert legacy_result.output_summary["legacy_graph_invoked"] is True
    assert skill_result.output_summary["skill_graph_invoked"] is True
    assert legacy_result.session_id and legacy_result.task_id and legacy_result.thread_id
    assert skill_result.session_id and skill_result.task_id and skill_result.thread_id
    assert "敏感 JD" not in json.dumps(legacy_result.to_dict(), ensure_ascii=False)


def test_skill_production_path_uses_skill_executor_and_unified_timeline():
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run(
        "招聘 Python RAG LangGraph 工程师",
        default_runner=legacy_runner,
        production_skill_graph_runner=build_fake_production_runner(),
        store=store,
        config=RuntimeEntryConfig(graph_mode="skill", allow_planner_fallback=True),
    )
    inspection = RuntimeInspector().inspect_task(result.task_id, store)
    event_types = inspection.event_types

    assert result.status == "ok"
    assert result.output_summary["candidate_preview_version"] == "v2"
    assert result.output_summary["skill_names"] == [
        "planner_extract",
        "resume_retrieve",
        "candidate_match",
        "claim_verify",
        "candidate_match",
        "claim_verify",
    ]
    assert result.output_summary["skill_execution_count"] == 6
    assert "graph_started" in event_types
    assert "node_started" in event_types
    assert "skill_started" in event_types
    assert "skill_completed" in event_types
    assert "node_completed" in event_types
    assert "graph_completed" in event_types
    assert inspection.event_count > result.output_summary["skill_execution_count"]
    skill_events = [item for item in inspection.timeline_summary if item["event_type"] == "skill_started"]
    assert skill_events
    assert skill_events[0]["graph_mode"] == "skill"
    assert skill_events[0]["runner_name"] == "production_skill_graph"
    assert skill_events[0]["skill_name"] == "planner_extract"


def test_skill_failure_falls_back_to_rollback_baseline():
    def failing_planner(_input, _context=None):
        raise RuntimeError("provider secret body should not leak")

    runner = ProductionSkillGraphRunner(
        ProductionSkillGraphConfig(enabled=True, allow_planner_fallback=False),
        planner_extract_callable=failing_planner,
        retrieve_callable=retriever,
        match_callable=matcher,
    ).run
    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=legacy_runner,
        production_skill_graph_runner=runner,
        config=RuntimeEntryConfig(graph_mode="skill"),
    )
    rendered = json.dumps(result.to_dict(), ensure_ascii=False)

    assert result.status == "completed_with_fallback"
    assert result.output_summary["fallback_attempted"] is True
    assert result.output_summary["fallback_succeeded"] is True
    assert result.output_summary["primary_error_type"] == "PlannerSkillFailed"
    assert result.output_summary["rollback_baseline"] == "legacy_default_graph"
    assert "provider secret body" not in rendered


def test_main_py_uses_runtime_entry_not_direct_core_harness():
    source = Path("main.py").read_text(encoding="utf-8")

    assert "RuntimeEntryHarness" in source
    assert "run_interactive_flow" not in source
    assert "create_recruit_graph()" not in source


def test_default_legacy_cli_still_runs_with_fake_runner(capsys):
    code = run_cli(["--jd", "招聘JD", "--graph-mode", "legacy", "--json"], default_runner=legacy_runner)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["runner_used"] == "default_graph"
    assert payload["output_summary"]["selected_graph_mode"] == "legacy"
    assert payload["output_summary"]["rollback_target"] == "legacy"


def test_skill_cli_uses_graph_mode_not_manual_variant(capsys):
    code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "skill", "--json"],
        default_runner=legacy_runner,
        production_skill_graph_runner=skill_runner,
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "ok"
    assert payload["runner_used"] == "production_skill_graph"
    assert payload["output_summary"]["selected_graph_mode"] == "skill"
    assert payload["output_summary"]["legacy_graph_invoked"] is False
    assert payload["output_summary"]["memory_context_provided"] is False
