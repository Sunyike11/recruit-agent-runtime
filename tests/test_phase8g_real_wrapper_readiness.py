import builtins
import json
import types

from scripts.run_recruit_runtime import run_cli
from src.runtime.variant_runner import (
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    check_llm_env_readiness,
    load_project_dotenv_for_real_wrappers,
)


SENSITIVE_KEY = "TEST_OPENAI_KEY_PHASE8G_SHOULD_NOT_LEAK"
SENSITIVE_ERROR = "FULL-WRAPPER-ERROR-SHOULD-NOT-LEAK"


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8G test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_import_module_with_dotenv(_name):
    def load_dotenv(*_args, **_kwargs):
        return True

    return types.SimpleNamespace(load_dotenv=load_dotenv)


def fake_import_module_sets_env(monkeypatch):
    def fake_import(_name):
        def load_dotenv(*_args, **_kwargs):
            monkeypatch.setenv("OPENAI_API_KEY", SENSITIVE_KEY)
            monkeypatch.setenv("OPENAI_API_BASE", "https://phase8g.example.invalid")
            return True

        return types.SimpleNamespace(load_dotenv=load_dotenv)

    return fake_import


def fake_planner(input_data, _context=None):
    return {
        "job_requirement": {
            "job_id": "phase8g_job",
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {"search_query": "Python RAG LangGraph"},
        },
        "extracted_keywords": ["Python", "RAG", "LangGraph"],
    }


def fake_retriever(input_data, _context=None):
    return {
        "candidates": [
            {
                "candidate_id": "candidate_8g",
                "name": "Candidate",
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "evidence": [{"candidate_id": "candidate_8g", "matched_skill_count": 3}],
        "metadata": {"summary_only": True},
    }


def fake_matcher(input_data, _context=None):
    candidate_id = input_data["candidate_profile"]["candidate_id"]
    return {
        "total_score": 91,
        "recommendation": "strong_match",
        "match_report": {"candidate_id": candidate_id, "total_score": 91},
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"], "reason": "not needed"}


def test_load_project_dotenv_reports_only_status(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = load_project_dotenv_for_real_wrappers(import_module=fake_import_module_with_dotenv)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["dotenv_loaded"] is True
    assert result["summary_only"] is True
    assert SENSITIVE_KEY not in serialized


def test_missing_llm_env_returns_safe_skip_diagnostics(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def missing_readiness(**_kwargs):
        return {
            "dotenv_loaded": "skip",
            "dotenv_path_present": False,
            "dotenv_error_type": "",
            "openai_api_key": "missing",
            "openai_api_base": "missing",
            "summary_only": True,
        }

    runner = build_real_skill_wrapper_variant_runner(
        retrieve_callable=fake_retriever,
        llm_readiness_checker=missing_readiness,
    )
    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "skipped"
    assert summary["error_hint"] == "llm_env_not_detected_for_lazy_wrappers"
    assert summary["env_readiness"]["openai_api_key"] == "missing"
    assert summary["env_readiness"]["openai_api_base"] == "missing"
    assert SENSITIVE_KEY not in serialized


def test_llm_readiness_set_key_and_base_reports_ready_without_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SENSITIVE_KEY)
    monkeypatch.setenv("OPENAI_API_BASE", "https://phase8g.example.invalid")

    readiness = check_llm_env_readiness(load_dotenv=False)
    serialized = json.dumps(readiness, ensure_ascii=False)

    assert readiness["openai_api_key"] == "set"
    assert readiness["openai_api_base"] == "set"
    assert SENSITIVE_KEY not in serialized
    assert "phase8g.example" not in serialized


def test_check_llm_env_readiness_can_load_project_dotenv_without_leaking(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    readiness = check_llm_env_readiness(import_module=fake_import_module_sets_env(monkeypatch))
    serialized = json.dumps(readiness, ensure_ascii=False)

    assert readiness["dotenv_loaded"] is True
    assert readiness["openai_api_key"] == "set"
    assert readiness["openai_api_base"] == "set"
    assert SENSITIVE_KEY not in serialized
    assert "phase8g.example" not in serialized


def test_real_wrapper_variant_fake_env_ready_and_injected_callables_runs(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", SENSITIVE_KEY)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert summary["real_skill_wrapper_mode"] is True
    assert summary["candidate_count"] == 1
    assert summary["match_count"] == 1
    assert summary["metadata"]["production_graph_invoked"] is False


def test_wrapper_failure_returns_sanitized_error_type_and_hint(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    def failing_matcher(_input_data, _context=None):
        raise RuntimeError(SENSITIVE_ERROR)

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=fake_retriever,
        match_callable=failing_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘JD")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_type"] == "SkillWorkflowFailed"
    assert summary["error_hint"] == "matcher_wrapper_failed"
    assert SENSITIVE_ERROR not in serialized


def test_default_cli_path_does_not_use_real_wrapper_readiness(monkeypatch, capsys):
    block_real_retrieval_imports(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = {"default": 0}

    def default_runner(_raw_jd):
        calls["default"] += 1
        return {"status": "ok", "candidate_count": 1, "report_count": 1}

    exit_code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=default_runner,
        real_retriever_callable=build_real_retriever_callable(search_runner=lambda *_args: []),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["runner_used"] == "default_graph"
    assert calls["default"] == 1
    assert payload["output_summary"]["env_readiness"] == {}


def test_required_path_does_not_import_real_retriever_dependencies(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=fake_retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘JD")

    assert summary["status"] == "ok"
