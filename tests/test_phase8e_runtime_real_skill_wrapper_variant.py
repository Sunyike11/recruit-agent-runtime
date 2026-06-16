import builtins
import json
import sys

from scripts.run_recruit_runtime import run_cli
from src.runtime import RuntimeEntryConfig, RuntimeEntryHarness, TaskStatus
from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner


def block_real_retrieval_and_llm_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.planner",
            "src.agents.matcher",
            "src.agents.refiner",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real dependency import in Phase8E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_default_runner(_raw_jd):
    return {
        "status": "ok",
        "candidate_count": 1,
        "report_count": 1,
        "top_score_present": True,
        "metadata": {"source": "fake_default"},
    }


def fake_planner(input_data, _context=None):
    raw_text = input_data["raw_text"]
    return {
        "job_requirement": {
            "job_id": "real_wrapper_test_job",
            "title": "AI Agent Engineer",
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {"search_query": "Python RAG LangGraph"},
            "raw_text_length": len(raw_text),
        },
        "extracted_keywords": ["Python", "RAG", "LangGraph"],
    }


def fake_retriever(_input_data, _context=None):
    return {
        "candidates": [
            {
                "candidate_id": "candidate_real_wrapper_1",
                "name": "Candidate One",
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "evidence": [{"candidate_id": "candidate_real_wrapper_1", "source": "fake"}],
    }


def fake_matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 96,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": candidate["candidate_id"],
            "total_score": 96,
            "recommendation": "strong_match",
        },
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"], "reason": "not needed"}


def build_injected_runner(**overrides):
    kwargs = {
        "planner_extract_callable": fake_planner,
        "retrieve_callable": fake_retriever,
        "match_callable": fake_matcher,
        "refine_callable": fake_refiner,
    }
    kwargs.update(overrides)
    return build_real_skill_wrapper_variant_runner(**kwargs)


def test_real_skill_wrapper_variant_runner_with_injected_callables(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_injected_runner()

    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert summary["workflow_status"] == "completed"
    assert summary["real_skill_wrapper_mode"] is True
    assert summary["candidate_count"] == 1
    assert summary["match_count"] == 1
    assert summary["top_score_present"] is True
    assert summary["metadata"]["runner_type"] == "real_skill_wrapper_variant"
    assert summary["metadata"]["deterministic_variant"] is False
    assert summary["metadata"]["real_skill_wrapper_mode"] is True
    assert summary["metadata"]["production_graph_invoked"] is False
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_real_skill_wrapper_variant_skips_when_retriever_callable_missing(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=None,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("招聘JD")

    assert summary["status"] == "skipped"
    assert summary["real_skill_wrapper_mode"] is True
    assert summary["error_hint"] == "retriever_callable_required"
    assert summary["metadata"]["production_graph_invoked"] is False


def test_real_skill_wrapper_variant_failure_is_sanitized(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)

    def failing_matcher(_input_data, _context=None):
        raise RuntimeError("完整敏感 matcher failure 不应输出")

    runner = build_injected_runner(match_callable=failing_matcher)
    summary = runner("招聘JD")

    assert summary["status"] == "failed"
    assert summary["error_type"] == "SkillWorkflowFailed"
    assert summary["error_hint"] == "matcher_wrapper_failed"
    assert "完整敏感 matcher failure" not in json.dumps(summary, ensure_ascii=False)


def test_runtime_entry_can_run_real_skill_wrapper_variant_runner(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_injected_runner()

    result = RuntimeEntryHarness().run(
        "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
        default_runner=fake_default_runner,
        variant_runner=runner,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    assert result.status == "ok"
    assert result.runner_used == "skill_backed_variant"
    assert result.task_status == TaskStatus.COMPLETED.value
    assert result.output_summary["candidate_count"] == 1
    assert result.output_summary["report_count"] == 1
    assert result.output_summary["top_score_present"] is True
    assert set(result.output_summary["metadata"]["keys"]) >= {
        "deterministic_variant",
        "env_readiness",
        "memory_context_provided",
        "production_graph_invoked",
        "production_graph_replaced",
        "raw_jd_length",
        "real_skill_wrapper_mode",
        "runner_type",
        "summary_only",
    }


def test_cli_real_skill_wrapper_flag_with_injected_runner(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)

    exit_code = run_cli(
        [
            "--jd",
            "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
            "--use-skill-backed-variant",
            "--use-real-skill-wrappers",
            "--json",
        ],
        default_runner=fake_default_runner,
        real_skill_wrapper_runner=build_injected_runner(),
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "skill_backed_variant"
    assert output["status"] == "ok"
    assert output["output_summary"]["candidate_count"] == 1
    assert output["metadata"]["use_skill_backed_variant"] is True
    assert output["summary_only"] is True


def test_cli_real_skill_wrapper_default_skips_without_retriever_callable(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)

    exit_code = run_cli(
        [
            "--jd",
            "招聘JD",
            "--use-skill-backed-variant",
            "--use-real-skill-wrappers",
            "--json",
        ],
        default_runner=fake_default_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "skill_backed_variant"
    assert output["output_summary"]["status"] == "skipped"
    assert output["output_summary"]["error_hint"] == "retriever_callable_required"


def test_cli_strict_returns_nonzero_on_real_wrapper_skip(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)

    exit_code = run_cli(
        [
            "--jd",
            "招聘JD",
            "--use-real-skill-wrappers",
            "--strict",
            "--json",
        ],
        default_runner=fake_default_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["output_summary"]["status"] == "skipped"


def test_default_cli_path_does_not_use_real_skill_wrapper_runner(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)
    calls = {"real": 0}

    def real_runner(_raw_jd, **_kwargs):
        calls["real"] += 1
        return {"status": "ok", "candidate_count": 9, "report_count": 9}

    exit_code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=fake_default_runner,
        real_skill_wrapper_runner=real_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "default_graph"
    assert output["output_summary"]["candidate_count"] == 1
    assert calls["real"] == 0


def test_allow_memory_context_does_not_read_or_write_memory(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_injected_runner()

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        variant_runner=runner,
        memory_context=None,
        config=RuntimeEntryConfig(use_skill_backed_variant=True, allow_memory_context=True),
    )

    assert result.runner_used == "skill_backed_variant"
    assert result.metadata["allow_memory_context"] is True
    assert result.output_summary["metadata"]["keys"]
    assert "MemorySQLiteStore" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_default_create_recruit_graph_behavior_not_modified(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    summary = build_injected_runner()("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
