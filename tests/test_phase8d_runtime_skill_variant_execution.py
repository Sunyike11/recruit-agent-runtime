import builtins
import json
import sys

from scripts.run_recruit_runtime import run_cli
from src.runtime import RuntimeEntryConfig, RuntimeEntryHarness, TaskStatus
from src.runtime.store import InMemoryRuntimeStore
from src.runtime.variant_runner import build_skill_backed_variant_runner


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
            raise ModuleNotFoundError(f"blocked real dependency import in Phase8D test: {name}")
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


def test_build_skill_backed_variant_runner_executes_deterministic_workflow(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_skill_backed_variant_runner(top_k=2)

    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert summary["workflow_status"] == "completed"
    assert summary["candidate_count"] == 2
    assert summary["match_count"] == 2
    assert summary["top_score_present"] is True
    assert summary["metadata"]["runner_type"] == "skill_backed_variant"
    assert summary["metadata"]["deterministic_variant"] is True
    assert summary["metadata"]["production_graph_invoked"] is False
    assert summary["metadata"]["summary_only"] is True
    assert "planner_extract" in summary["skill_names"]
    assert "resume_retrieve" in summary["skill_names"]
    assert "candidate_match" in summary["skill_names"]
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_variant_runner_summary_is_summary_only(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_skill_backed_variant_runner()

    summary = runner("包含完整敏感JD的招聘描述")
    payload = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "ok"
    assert "包含完整敏感JD" not in payload
    assert "raw_jd_length" in summary["metadata"]
    assert summary["metadata"]["production_graph_replaced"] is False


def test_runtime_entry_uses_skill_backed_variant_runner(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    runner = build_skill_backed_variant_runner()

    result = RuntimeEntryHarness().run(
        "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
        default_runner=fake_default_runner,
        variant_runner=runner,
        store=store,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    task = store.get_task(result.task_id)
    timeline = store.list_events(task_id=result.task_id)

    assert result.status == "ok"
    assert result.runner_used == "skill_backed_variant"
    assert result.task_status == TaskStatus.COMPLETED.value
    assert result.output_summary["candidate_count"] == 2
    assert result.output_summary["report_count"] == 2
    assert result.output_summary["top_score_present"] is True
    assert task.status == TaskStatus.COMPLETED
    assert [event.event_type for event in timeline] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]


def test_runtime_entry_variant_failure_marks_task_failed(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    def failing_variant(_raw_jd, **_kwargs):
        return {
            "status": "failed",
            "error_type": "SkillWorkflowFailed",
            "candidate_count": 0,
            "report_count": 0,
            "metadata": {"sensitive_payload": "完整错误上下文不应输出"},
        }

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        variant_runner=failing_variant,
        store=store,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    task = store.get_task(result.task_id)

    assert result.status == "failed"
    assert result.runner_used == "skill_backed_variant"
    assert result.task_status == TaskStatus.FAILED.value
    assert result.error_type == "SkillWorkflowFailed"
    assert task.status == TaskStatus.FAILED
    assert "完整错误上下文" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_allow_memory_context_does_not_auto_read_or_write_memory(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    runner = build_skill_backed_variant_runner()

    result = RuntimeEntryHarness().run(
        "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
        default_runner=fake_default_runner,
        variant_runner=runner,
        memory_context=None,
        config=RuntimeEntryConfig(use_skill_backed_variant=True, allow_memory_context=True),
    )

    assert result.runner_used == "skill_backed_variant"
    assert result.metadata["allow_memory_context"] is True
    assert result.output_summary["metadata"]["keys"]
    assert result.output_summary["metadata"]["summary_only"] is True
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


def test_cli_use_skill_backed_variant_runs_deterministic_variant(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)

    exit_code = run_cli(
        [
            "--jd",
            "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
            "--use-skill-backed-variant",
            "--json",
        ],
        default_runner=fake_default_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["runner_used"] == "skill_backed_variant"
    assert output["output_summary"]["candidate_count"] == 2
    assert output["output_summary"]["report_count"] == 2
    assert output["metadata"]["production_graph_replaced"] is False
    assert output["metadata"]["use_skill_backed_variant"] is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_default_cli_without_variant_flag_does_not_call_variant(monkeypatch, capsys):
    block_real_retrieval_and_llm_imports(monkeypatch)

    exit_code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=fake_default_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "default_graph"
    assert output["metadata"]["use_skill_backed_variant"] is False
    assert output["output_summary"]["candidate_count"] == 1


def test_default_create_recruit_graph_behavior_not_modified(monkeypatch):
    block_real_retrieval_and_llm_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    runner = build_skill_backed_variant_runner()
    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
