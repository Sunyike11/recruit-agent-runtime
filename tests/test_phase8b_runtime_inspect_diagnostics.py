import builtins
import json
import sys

from scripts.inspect_runtime_task import main as inspect_cli_main, run_cli as inspect_cli_run
from src.runtime import (
    RuntimeEntryHarness,
    RuntimeInspector,
    TaskStatus,
    build_default_graph_initial_state,
)
from src.runtime.sqlite_store import SQLiteRuntimeStore
from src.runtime.store import InMemoryRuntimeStore


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase8B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_graph_success_runner(raw_jd):
    state = build_default_graph_initial_state(raw_jd)
    return {
        "status": "ok",
        "candidate_count": 2,
        "report_count": 1,
        "top_score_present": True,
        "output_keys": ["candidate_pool", "final_reports"],
        "graph_input_keys": sorted(state.keys()),
        "graph_input_shape": {
            "messages": {"type": "list", "count": len(state["messages"])},
            "candidate_pool": {"type": "list", "count": 0},
        },
        "graph_config_has_thread_id": True,
        "graph_result_keys": ["candidate_pool", "final_reports"],
        "metadata": {"summary_only": True, "sensitive": "完整内容不应出现"},
    }


def fake_graph_failed_summary_runner(raw_jd):
    state = build_default_graph_initial_state(raw_jd)
    return {
        "status": "failed",
        "candidate_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "output_keys": ["error_type", "status"],
        "error_type": "ValueError",
        "runner_error_type": "ValueError",
        "runner_error_stage": "graph_stream",
        "error_hint": "unknown_value_error",
        "graph_input_keys": sorted(state.keys()),
        "graph_input_shape": {"messages": {"type": "list", "count": 1}},
        "graph_config_has_thread_id": True,
        "graph_result_keys": [],
        "metadata": {"summary_only": True},
    }


def test_build_default_graph_initial_state_matches_core_harness_shape(monkeypatch):
    block_retrieval_imports(monkeypatch)

    state = build_default_graph_initial_state("招聘JD")

    assert sorted(state.keys()) == [
        "candidate_pool",
        "extracted_jd",
        "final_reports",
        "human_feedback",
        "loop_count",
        "messages",
        "next_action",
        "refinement_advice",
    ]
    assert state["messages"][0].content == "招聘JD"
    assert state["candidate_pool"] == []
    assert state["final_reports"] == []


def test_runtime_entry_fake_default_graph_success_completes_task(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_graph_success_runner,
        store=store,
    )

    task = store.get_task(result.task_id)
    assert result.status == "ok"
    assert task.status == TaskStatus.COMPLETED
    assert result.output_summary["graph_config_has_thread_id"] is True
    assert "messages" in result.output_summary["graph_input_keys"]


def test_runtime_entry_runner_reported_failure_marks_task_failed_with_diagnostics(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_graph_failed_summary_runner,
        store=store,
    )

    task = store.get_task(result.task_id)
    assert result.status == "failed"
    assert task.status == TaskStatus.FAILED
    assert result.error_type == "ValueError"
    assert result.output_summary["runner_error_type"] == "ValueError"
    assert result.output_summary["error_hint"] == "unknown_value_error"
    assert result.output_summary["runner_error_stage"] == "graph_stream"


def test_runtime_entry_exception_failure_has_sanitized_diagnostics(monkeypatch):
    block_retrieval_imports(monkeypatch)

    def exploding_runner(_raw_jd):
        raise ValueError("完整异常正文不应暴露")

    result = RuntimeEntryHarness().run("招聘JD", default_runner=exploding_runner)

    assert result.status == "failed"
    assert result.output_summary["runner_error_type"] == "ValueError"
    assert result.output_summary["error_hint"] == "graph_invoke_error"
    assert "完整异常正文" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_runtime_inspector_can_inspect_successful_task(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run("招聘JD", default_runner=fake_graph_success_runner, store=store)

    inspection = RuntimeInspector().inspect_task(result.task_id, store)

    assert inspection.task_status == "completed"
    assert inspection.runner_used == "default_graph"
    assert inspection.event_count == 5
    assert inspection.event_types == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]
    assert inspection.output_summary["candidate_count"] == 2


def test_runtime_inspector_can_inspect_failed_task(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run("招聘JD", default_runner=fake_graph_failed_summary_runner, store=store)

    inspection = RuntimeInspector().inspect_task(result.task_id, store)

    assert inspection.task_status == "failed"
    assert inspection.error_type == "ValueError"
    assert inspection.error_hint == "unknown_value_error"
    assert inspection.event_types == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_failed",
        "task_failed",
    ]


def test_runtime_inspection_is_summary_only(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    result = RuntimeEntryHarness().run(
        "这是一段完整敏感 JD 内容",
        default_runner=fake_graph_success_runner,
        store=store,
    )

    payload = RuntimeInspector().inspect_task(result.task_id, store).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False)

    assert payload["input_summary"]["jd_length"] == len("这是一段完整敏感 JD 内容")
    assert "这是一段完整敏感 JD 内容" not in rendered
    assert "完整内容不应出现" not in rendered


def test_inspect_latest_task_works(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()
    RuntimeEntryHarness().run("第一条JD", default_runner=fake_graph_success_runner, store=store)
    latest_result = RuntimeEntryHarness().run("第二条JD", default_runner=fake_graph_success_runner, store=store)

    inspection = RuntimeInspector().inspect_latest_task(store)

    assert inspection.task_id == latest_result.task_id


def test_inspect_runtime_task_help_executes(capsys):
    try:
        inspect_cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert "--db-path" in output
    assert "--latest" in output


def test_inspect_runtime_task_json_output_with_sqlite_store(tmp_path, monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)
    db_path = tmp_path / "runtime.sqlite"
    store = SQLiteRuntimeStore(db_path)
    RuntimeEntryHarness().run("招聘JD", default_runner=fake_graph_success_runner, store=store)

    exit_code = inspect_cli_run(["--db-path", str(db_path), "--latest", "--json", "--events"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["task_status"] == "completed"
    assert payload["event_count"] == 5
    assert payload["event_types"] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]
    assert payload["timeline_summary"][1]["runner_used"] == "default_graph"


def test_inspect_runtime_task_json_can_omit_events(tmp_path, monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)
    db_path = tmp_path / "runtime.sqlite"
    store = SQLiteRuntimeStore(db_path)
    RuntimeEntryHarness().run("招聘JD", default_runner=fake_graph_success_runner, store=store)

    inspect_cli_run(["--db-path", str(db_path), "--latest", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["timeline_summary"] == []
    assert payload["event_count"] == 5


def test_phase8b_does_not_import_real_retrieval_or_modify_default_graph(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    RuntimeEntryHarness().run("招聘JD", default_runner=fake_graph_success_runner)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
