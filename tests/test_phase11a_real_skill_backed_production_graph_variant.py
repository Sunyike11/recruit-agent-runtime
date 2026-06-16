import builtins
import json
import sys

from scripts.run_recruit_runtime import run_cli
from src.integration.production_skill_graph import (
    ProductionSkillGraphConfig,
    ProductionSkillGraphRunner,
    compare_legacy_and_skill_output_shape,
    legacy_state_to_skill_graph_input,
    skill_graph_result_to_legacy_compatible_summary,
)
from src.runtime.candidate_preview import (
    build_candidate_preview_quality_audit,
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
)
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.store import InMemoryRuntimeStore
from src.skills.agent_adapters import CandidateMatchSkill


FULL_CHUNK = (
    "FULL PRIVATE RESUME TEXT SHOULD NOT LEAK. 张三本科计算机，参与 Python RAG LangGraph Agent 检索匹配平台项目，"
    "负责 Docker 部署、自动化评估和工程测试。本段足够长用于验证 summary-only。"
)


def block_real_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked = (
            "llama_index",
            "chromadb",
            "src.agents.planner",
            "src.agents.matcher",
            "src.agents.refiner",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked):
            raise ModuleNotFoundError(f"blocked real dependency import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "tech_stack": ["Python", "RAG", "LangGraph"],
            "metadata": {"source": "FakePlanner", "search_query": "Python RAG LangGraph"},
        },
        "metadata": {"source": "FakePlanner"},
    }


def fallback_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG"],
            "metadata": {
                "source": "deterministic_planner_fallback",
                "planner_fallback_used": True,
                "planner_fallback_type": "deterministic",
                "real_planner_invoked": True,
                "real_planner_failed": True,
                "fallback_not_real_planner_success": True,
                "search_query": "Python RAG",
            },
        },
        "metadata": {
            "planner_fallback_used": True,
            "planner_fallback_type": "deterministic",
            "real_planner_failed": True,
        },
    }


def retriever(_input, _context=None):
    return {
        "resume_documents": [
            {
                "text": FULL_CHUNK,
                "metadata": {
                    "candidate_id": "cand-a",
                    "candidate_name": "张三",
                    "document_id": "doc-a",
                    "file_name": "zhangsan.pdf",
                },
                "score": 0.91,
            }
        ],
        "evidence": [],
        "metadata": {"source": "FakeRetriever", "summary_only": True},
    }


def two_candidate_retriever(_input, _context=None):
    return {
        "resume_documents": [
            {
                "text": FULL_CHUNK,
                "metadata": {
                    "candidate_id": "cand-a",
                    "candidate_name": "张三",
                    "document_id": "doc-a",
                    "file_name": "zhangsan.pdf",
                },
                "score": 0.91,
            },
            {
                "text": "李四本科计算机，熟悉 Python RAG LangGraph FastAPI 工程化。",
                "metadata": {
                    "candidate_id": "cand-b",
                    "candidate_name": "李四",
                    "document_id": "doc-b",
                    "file_name": "lisi.pdf",
                },
                "score": 0.88,
            },
        ],
        "evidence": [],
        "metadata": {"source": "FakeRetriever", "summary_only": True},
    }


def matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    assert isinstance(candidate.get("education"), str)
    return {
        "total_score": 88,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": candidate["candidate_id"],
            "total_score": 88,
            "metadata": {
                "source": "FakeMatcher",
                "candidate_profile_preview": candidate["metadata"]["candidate_profile_preview"],
            },
        },
        "metadata": {"source": "FakeMatcher"},
    }


def low_matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 30,
        "recommendation": "weak",
        "match_report": {"candidate_id": candidate["candidate_id"], "total_score": 30},
        "metadata": {"source": "FakeMatcher"},
    }


def refiner(input_data, _context=None):
    return {"refined_query": input_data["query"] + " refined"}


def build_runner(**kwargs):
    planner_callable = kwargs.pop("planner_callable", planner)
    retrieve_callable = kwargs.pop("retrieve_callable", retriever)
    match_callable = kwargs.pop("match_callable", matcher)
    refine_callable = kwargs.pop("refine_callable", refiner)
    max_refine_loops = kwargs.pop("max_refine_loops", 1)
    return ProductionSkillGraphRunner(
        ProductionSkillGraphConfig(enabled=True, max_refine_loops=max_refine_loops, **kwargs),
        planner_extract_callable=planner_callable,
        retrieve_callable=retrieve_callable,
        match_callable=match_callable,
        refine_callable=refine_callable,
    )


def test_config_default_disabled():
    config = ProductionSkillGraphConfig()
    runner = ProductionSkillGraphRunner(config, planner_extract_callable=planner, retrieve_callable=retriever)

    assert config.enabled is False
    assert runner.run("招聘 Python")["status"] == "skipped"


def test_enabled_runner_executes_all_core_skills_through_executor(monkeypatch):
    block_real_imports(monkeypatch)
    summary = build_runner().run("招聘 Python RAG LangGraph 工程师")
    payload = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "ok"
    assert summary["candidate_count"] == 1
    assert summary["candidate_profile_preview_count"] == 1
    assert summary["match_report_count"] == 1
    assert summary["enhanced_candidate_preview_used"] is True
    assert summary["skill_names"] == ["planner_extract", "resume_retrieve", "candidate_match", "claim_verify"]
    assert summary["claim_verification_enabled"] is True
    assert summary["legacy_graph_invoked"] is False
    assert summary["production_graph_replaced"] is False
    assert FULL_CHUNK not in payload
    assert "src.agents.retriever" not in sys.modules


def test_two_candidates_execute_matcher_twice_and_audit_matches_preview_count(monkeypatch):
    block_real_imports(monkeypatch)
    seen = []

    def recording_matcher(input_data, context=None):
        candidate = input_data["candidate_profile"]
        seen.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": candidate.get("name"),
                "candidate_name": candidate.get("candidate_name"),
            }
        )
        return matcher(input_data, context)

    summary = build_runner(retrieve_callable=two_candidate_retriever).run("招聘 Python RAG LangGraph 工程师")

    assert summary["status"] == "ok"
    assert summary["candidate_count"] == 2
    assert summary["candidate_profile_preview_count"] == 2
    assert summary["candidate_preview_audit"]["candidate_profile_preview_count"] == 2
    assert summary["candidate_preview_audit"]["candidate_id_present"] == 2
    assert summary["report_count"] == 2
    assert summary["top_score_present"] is True
    assert summary["skill_names"] == [
        "planner_extract",
        "resume_retrieve",
        "candidate_match",
        "claim_verify",
        "candidate_match",
        "claim_verify",
    ]
    assert summary["skill_execution_count"] == 6
    assert summary["matcher_input_source"] == "candidate_profile_preview"
    assert summary["matcher_source"] == "FakeMatcher"

    summary = build_runner(retrieve_callable=two_candidate_retriever, match_callable=recording_matcher).run(
        "招聘 Python RAG LangGraph 工程师"
    )
    assert summary["status"] == "ok"
    assert seen == [
        {"candidate_id": "cand-a", "name": "张三", "candidate_name": "张三"},
        {"candidate_id": "cand-b", "name": "李四", "candidate_name": "李四"},
    ]


def test_candidate_name_contract_and_placeholder_audit():
    build_result = build_candidate_profile_previews_from_retrieval_results(
        {
            "resume_documents": [
                {
                    "text": "王五 本科 计算机 Python RAG 项目经验。",
                    "metadata": {
                        "candidate_id": "cand-w",
                        "candidate_name": "王五",
                        "document_id": "doc-w",
                        "file_name": "我的简历.pdf",
                    },
                },
                {
                    "text": "候选人熟悉 Python LangGraph。",
                    "metadata": {"document_id": "doc-placeholder", "file_name": "第二份简历.pdf"},
                },
            ]
        },
        raw_jd="招聘 Python RAG LangGraph 工程师",
    )
    previews = [preview.to_dict() for preview in build_result.previews]
    by_id = {preview["candidate_id"]: preview for preview in previews}
    resolved_input = candidate_profile_preview_to_matcher_input(by_id["cand-w"])
    placeholder_preview = next(preview for preview in previews if preview["candidate_id"] != "cand-w")
    placeholder_input = candidate_profile_preview_to_matcher_input(placeholder_preview)
    audit = build_candidate_preview_quality_audit(previews)

    assert resolved_input["name"] == "王五"
    assert resolved_input["candidate_name"] == "王五"
    assert resolved_input["candidate_name_resolved"] is True
    assert placeholder_preview["candidate_name"] == ""
    assert placeholder_input["name"] == ""
    assert placeholder_input["candidate_name_resolved"] is False
    assert audit["candidate_name_field_present"] == 2
    assert audit["candidate_name_resolved_count"] == 1
    assert audit["candidate_name_placeholder_count"] == 0


def test_candidate_match_skill_preserves_candidate_id_and_resolved_name():
    skill = CandidateMatchSkill(
        match_callable=lambda _input, _context=None: {
            "total_score": 76,
            "recommendation": "possible_match",
            "match_report": {
                "candidate_name": "未提供",
                "total_score": 76,
            },
            "metadata": {"source": "FakeMatcher"},
        }
    )
    result = skill.run(
        {
            "job_requirement": {"required_skills": ["Python"]},
            "candidate_profile": {
                "candidate_id": "cand-name",
                "name": "赵六",
                "candidate_name": "赵六",
                "skills": ["Python"],
                "education": "本科",
                "experience": [],
                "projects": [],
            },
        }
    )

    report = result.output["match_report"]
    assert report["candidate_id"] == "cand-name"
    assert report["candidate_name"] == "赵六"
    assert report["metadata"]["candidate_id"] == "cand-name"
    assert report["metadata"]["candidate_name"] == "赵六"


def test_matcher_failure_is_fail_fast_with_sanitized_type_and_diagnostics(monkeypatch):
    block_real_imports(monkeypatch)
    calls = {"matcher": 0}

    def failing_matcher(input_data, _context=None):
        calls["matcher"] += 1
        raise TypeError("can only concatenate str (not \"list\") to str with PRIVATE REASONING")

    summary = build_runner(
        retrieve_callable=two_candidate_retriever,
        match_callable=failing_matcher,
    ).run("招聘 Python RAG LangGraph 工程师")
    payload = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "matcher_failed"
    assert summary["error_type"] == "MatcherSkillFailed"
    assert summary["error_type"] != "can"
    assert calls["matcher"] == 1
    assert summary["skill_names"] == ["planner_extract", "resume_retrieve", "candidate_match"]
    assert summary["matcher_invocation_stage"] == "skill_execute"
    assert summary["matcher_input_keys"] == ["candidate_profile", "evidence", "job_requirement", "metadata"]
    assert summary["matcher_candidate_id"] == "cand-a"
    assert summary["matcher_candidate_name_present"] is True
    assert summary["matcher_skills_count"] > 0
    assert summary["matcher_adapter_error_hint"] == "matcher_wrapper_failed"
    assert summary["matcher_provider_error_type"] == "TypeError"
    assert "PRIVATE REASONING" not in payload


def test_skill_events_are_recorded_to_runtime_timeline(monkeypatch):
    block_real_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run(
        "招聘 Python RAG LangGraph 工程师",
        default_runner=lambda _jd: {"status": "ok"},
        production_skill_graph_runner=build_runner().run,
        store=store,
        config=RuntimeEntryConfig(use_production_skill_graph=True),
    )
    events = [event.event_type for event in store.list_events(task_id=result.task_id)]

    assert result.status == "ok"
    assert result.runner_used == "production_skill_graph"
    assert "graph_started" in events
    assert "skill_started" in events
    assert "skill_completed" in events
    assert "graph_completed" in events
    assert result.output_summary["skill_event_count"] == 4
    assert "claim_verify" in result.output_summary["skill_names"]


def test_allow_planner_fallback_is_persisted_in_task_input(monkeypatch):
    block_real_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run(
        "招聘 Python RAG LangGraph 工程师",
        default_runner=lambda _jd: {"status": "ok"},
        production_skill_graph_runner=build_runner().run,
        store=store,
        config=RuntimeEntryConfig(
            use_production_skill_graph=True,
            allow_planner_fallback=True,
            metadata={"allow_planner_fallback": True},
        ),
    )
    task = store.get_task(result.task_id)

    assert result.status == "ok"
    assert task.input["metadata"]["use_production_skill_graph"] is True
    assert task.input["metadata"]["allow_planner_fallback"] is True


def test_provider_failure_without_fallback_keeps_planner_diagnostics(monkeypatch):
    block_real_imports(monkeypatch)
    import src.skills.agent_adapters as agent_adapters

    calls = {"retriever": 0}

    def failing_planner(**_kwargs):
        raise RuntimeError(
            "planner_wrapper_failed "
            "planner_invocation_stage=invoke_planner_agent "
            "planner_input_keys=candidate_pool,extracted_jd,final_reports,messages "
            "planner_has_messages=true "
            "planner_raw_text_length=19 "
            "planner_output_keys= "
            "provider_error_type=APIConnectionError "
            "planner_provider_openai_api_key=set "
            "planner_provider_openai_api_base=set "
            "planner_provider_llm_model=deepseek-chat "
            "planner_provider_planner_agent_class=PlannerAgent "
            "planner_provider_invocation_method=__call__"
        )

    def counted_retriever(input_data, context=None):
        calls["retriever"] += 1
        return retriever(input_data, context)

    monkeypatch.setattr(agent_adapters, "invoke_planner_agent_for_skill", failing_planner)
    summary = build_runner(
        planner_callable=None,
        retrieve_callable=counted_retriever,
        allow_planner_fallback=False,
    ).run("招聘 Python RAG LangGraph 工程师")
    payload = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_type"] == "PlannerSkillFailed"
    assert summary["error_hint"] == "planner_failed"
    assert summary["real_planner_invoked"] is True
    assert summary["real_planner_failed"] is True
    assert summary["planner_fallback_used"] is False
    assert summary["planner_invocation_stage"] == "invoke_planner_agent"
    assert summary["provider_error_type"] == "APIConnectionError"
    assert summary["planner_provider_diagnostics"]["openai_api_key"] == "set"
    assert summary["rollback_recommended"] is True
    assert calls["retriever"] == 0
    assert "招聘 Python RAG LangGraph 工程师" not in payload


def test_provider_failure_with_explicit_fallback_reaches_retriever(monkeypatch):
    block_real_imports(monkeypatch)
    import src.integration.production_skill_graph as production_skill_graph
    from src.skills.agent_adapters import invoke_planner_agent_for_skill

    calls = {"retriever": 0}

    def fallback_invoke(*, raw_text, metadata=None, allow_deterministic_fallback=False, **_kwargs):
        return invoke_planner_agent_for_skill(
            raw_text=raw_text,
            metadata=metadata,
            planner_factory=lambda: (_ for _ in ()).throw(ValueError("SENSITIVE PROVIDER DETAIL")),
            allow_deterministic_fallback=allow_deterministic_fallback,
        )

    def counted_retriever(input_data, context=None):
        calls["retriever"] += 1
        return retriever(input_data, context)

    monkeypatch.setattr(production_skill_graph, "invoke_planner_agent_for_skill", fallback_invoke)
    summary = build_runner(
        planner_callable=None,
        retrieve_callable=counted_retriever,
        allow_planner_fallback=True,
    ).run("招聘 Python RAG LangGraph 工程师")

    assert summary["status"] == "ok"
    assert summary["planner_source"] == "deterministic_fallback"
    assert summary["real_planner_invoked"] is True
    assert summary["real_planner_failed"] is True
    assert summary["planner_fallback_used"] is True
    assert summary["planner_fallback_type"] == "deterministic"
    assert summary["fallback_not_real_planner_success"] is True
    assert "resume_retrieve" in summary["skill_names"]
    assert calls["retriever"] == 1


def test_planner_success_provenance_is_self_consistent(monkeypatch):
    block_real_imports(monkeypatch)

    def real_like_planner(_input, _context=None):
        return {
            "job_requirement": {
                "required_skills": ["Python", "RAG"],
                "metadata": {
                    "source": "PlannerAgent",
                    "real_planner_invoked": True,
                    "real_planner_failed": False,
                    "planner_fallback_used": False,
                    "planner_invocation_stage": "completed",
                    "planner_raw_text_length": 42,
                    "search_query": "Python RAG",
                },
            },
            "metadata": {"source": "PlannerAgent"},
        }

    summary = build_runner(planner_callable=real_like_planner).run("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert summary["planner_source"] == "PlannerAgent"
    assert summary["real_planner_invoked"] is True
    assert summary["real_planner_failed"] is False
    assert summary["planner_fallback_used"] is False
    assert summary["planner_input_shape"]["raw_text_length"] == 42


def test_no_candidate_triggers_refiner_loop(monkeypatch):
    block_real_imports(monkeypatch)
    calls = {"retriever": 0}

    def staged_retriever(input_data, _context=None):
        calls["retriever"] += 1
        if "refined" not in input_data.get("query", ""):
            return {"resume_documents": [], "evidence": [], "metadata": {"source": "FakeRetriever"}}
        return retriever(input_data, _context)

    summary = build_runner(retrieve_callable=staged_retriever, max_refine_loops=1).run("招聘 Python RAG")

    assert summary["status"] == "ok"
    assert summary["refined_query_present"] is True
    assert summary["loop_count"] == 1
    assert "query_refine" in summary["skill_names"]
    assert calls["retriever"] == 2


def test_max_refine_loops_recommends_rollback(monkeypatch):
    block_real_imports(monkeypatch)
    summary = build_runner(
        retrieve_callable=lambda _input, _context=None: {"resume_documents": [], "evidence": []},
        max_refine_loops=0,
    ).run("招聘 Python RAG")

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "max_loop_exceeded"
    assert summary["rollback_recommended"] is True
    assert summary["rollback_target"] == "legacy_default_graph"


def test_max_refine_loops_with_reports_completes_with_limit(monkeypatch):
    block_real_imports(monkeypatch)
    summary = build_runner(
        retrieve_callable=two_candidate_retriever,
        match_callable=low_matcher,
        max_refine_loops=0,
    ).run("招聘 Python RAG")

    assert summary["status"] == "completed_with_limit"
    assert summary["report_count"] == 2
    assert summary["max_refine_loops_reached"] is True
    assert summary["termination_reason"] == "max_refine_loops_reached"
    assert summary["error_type"] == ""
    assert summary["error_hint"] == ""
    assert summary["rollback_recommended"] is False


def test_planner_fallback_default_closed_and_explicit_marker(monkeypatch):
    block_real_imports(monkeypatch)
    assert ProductionSkillGraphConfig().allow_planner_fallback is False

    summary = build_runner(
        allow_planner_fallback=True,
        planner_callable=fallback_planner,
    ).run("招聘 Python RAG")

    assert summary["status"] == "ok"
    assert summary["fallback_used"] is True
    assert summary["provenance"]["planner_source"] == "deterministic_planner_fallback"


def test_graph_failure_summary_and_rollback(monkeypatch):
    block_real_imports(monkeypatch)

    def failing_matcher(_input, _context=None):
        raise RuntimeError("FULL MATCH REASONING SHOULD NOT LEAK")

    summary = build_runner(match_callable=failing_matcher).run("招聘 Python RAG")
    payload = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "matcher_failed"
    assert summary["rollback_recommended"] is True
    assert "FULL MATCH REASONING" not in payload


def test_runtime_cli_uses_production_skill_graph(monkeypatch, capsys):
    block_real_imports(monkeypatch)
    exit_code = run_cli(
        ["--jd", "招聘 Python RAG LangGraph 工程师", "--use-production-skill-graph", "--json"],
        default_runner=lambda _jd: {"status": "ok", "candidate_count": 0},
        production_skill_graph_runner=build_runner().run,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "production_skill_graph"
    assert output["output_summary"]["status"] == "ok"
    assert output["output_summary"]["skill_event_count"] == 4
    assert "claim_verify" in output["output_summary"]["skill_names"]
    assert output["metadata"]["use_production_skill_graph"] is True


def test_cli_allow_planner_fallback_reaches_runner_metadata(monkeypatch, capsys):
    block_real_imports(monkeypatch)
    captured = {}

    def production_runner(raw_jd, memory_context=None, metadata=None):
        captured["metadata"] = dict(metadata or {})
        return {
            "status": "ok",
            "candidate_count": 0,
            "report_count": 0,
            "production_skill_graph_enabled": True,
            "legacy_graph_invoked": False,
            "production_graph_replaced": False,
            "summary_only": True,
        }

    exit_code = run_cli(
        [
            "--jd",
            "招聘 Python",
            "--use-production-skill-graph",
            "--allow-planner-fallback",
            "--json",
        ],
        default_runner=lambda _jd: {"status": "ok"},
        production_skill_graph_runner=production_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["metadata"]["allow_planner_fallback"] is True
    assert captured["metadata"]["allow_planner_fallback"] is True


def test_conflicting_cli_flags_are_rejected(monkeypatch, capsys):
    block_real_imports(monkeypatch)
    exit_code = run_cli(
        ["--jd", "招聘 Python", "--use-production-skill-graph", "--use-skill-backed-variant", "--json"],
        default_runner=lambda _jd: {"status": "ok"},
        production_skill_graph_runner=build_runner().run,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "failed"
    assert output["output_summary"]["error_hint"] == "conflicting_runtime_runner_flags"


def test_default_runtime_cli_uses_skill_after_phase14b(monkeypatch, capsys):
    block_real_imports(monkeypatch)
    exit_code = run_cli(
        ["--jd", "招聘 Python", "--json"],
        default_runner=lambda _jd: {"status": "ok", "candidate_count": 1, "report_count": 1},
        production_skill_graph_runner=build_runner().run,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "production_skill_graph"
    assert output["output_summary"]["selected_graph_mode"] == "skill"
    assert output["output_summary"]["skill_default_used"] is True
    assert output["metadata"]["use_production_skill_graph"] is False
    assert output["output_summary"]["candidate_count"] >= 1


def test_state_adapter_helpers_are_summary_only():
    legacy_input = legacy_state_to_skill_graph_input({"messages": [{"content": "招聘 Python"}]})
    summary = skill_graph_result_to_legacy_compatible_summary({"status": "ok", "candidate_count": 2, "report_count": 1})
    compare = compare_legacy_and_skill_output_shape(summary, {"status": "ok", "candidate_count": 2, "report_count": 1})

    assert legacy_input["raw_jd"] == "招聘 Python"
    assert summary["candidate_pool_count"] == 2
    assert compare["status_present"] is True
    assert compare["summary_only"] is True


def test_memory_mcp_and_default_graph_are_not_used(monkeypatch):
    block_real_imports(monkeypatch)
    summary = build_runner().run("招聘 Python RAG")
    import src.core.graph as graph

    assert summary["status"] == "ok"
    assert "MemorySQLiteStore" not in json.dumps(summary, ensure_ascii=False)
    assert "MCP" not in json.dumps(summary, ensure_ascii=False)
    assert "ProductionSkillGraphRunner" not in graph.create_recruit_graph.__code__.co_names
