import builtins
import json
import sys

from scripts.run_recruit_runtime import main as runtime_cli_main, run_cli
from src.runtime import RuntimeEntryConfig, RuntimeEntryHarness, TaskStatus
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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase8A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_default_runner(raw_jd):
    return {
        "status": "ok",
        "candidate_count": 2,
        "report_count": 1,
        "top_score_present": True,
        "raw_jd": raw_jd,
        "metadata": {"sensitive_payload": "完整简历内容不应输出"},
    }


def fake_demo_runner(raw_jd):
    return {
        "status": "ok",
        "candidate_count": 3,
        "report_count": 2,
        "top_score_present": True,
        "metadata": {"demo": True},
    }


def test_runtime_entry_config_defaults_disable_demo_mode():
    config = RuntimeEntryConfig()

    assert config.use_demo_mode is False
    assert config.demo_mode_enabled is False
    assert config.use_skill_backed_variant is False
    assert config.allow_memory_context is False
    assert config.summary_only is True


def test_runtime_entry_harness_creates_session_task_and_events_with_fake_default(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        store=store,
    )

    task = store.get_task(result.task_id)
    timeline = store.list_events(task_id=result.task_id)

    assert result.status == "ok"
    assert result.runner_used == "default_graph"
    assert result.task_status == TaskStatus.COMPLETED.value
    assert result.session_id == task.session_id
    assert result.thread_id == task.thread_id
    assert result.event_count == 5
    assert task.result["candidate_count"] == 2
    assert [event.event_type for event in timeline] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_runtime_entry_harness_marks_task_failed_with_sanitized_error(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    def failing_runner(_raw_jd):
        raise RuntimeError("完整敏感异常正文不应保存")

    result = RuntimeEntryHarness().run(
        "包含敏感内容的 JD",
        default_runner=failing_runner,
        store=store,
    )

    task = store.get_task(result.task_id)
    timeline = store.list_events(task_id=result.task_id)

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.output_summary["error_type"] == "RuntimeError"
    assert task.status == TaskStatus.FAILED
    assert task.error == "RuntimeError"
    assert "完整敏感异常正文" not in json.dumps(result.to_dict(), ensure_ascii=False)
    assert [event.event_type for event in timeline] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_failed",
        "task_failed",
    ]


def test_demo_runner_not_called_when_use_demo_mode_false(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"demo": 0}

    def demo_runner(_raw_jd):
        calls["demo"] += 1
        return fake_demo_runner(_raw_jd)

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        demo_runner=demo_runner,
        config=RuntimeEntryConfig(use_demo_mode=False, demo_mode_enabled=True),
    )

    assert result.runner_used == "default_graph"
    assert calls["demo"] == 0


def test_demo_runner_called_only_when_demo_mode_enabled(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "demo": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def demo_runner(raw_jd):
        calls["demo"] += 1
        return fake_demo_runner(raw_jd)

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        demo_runner=demo_runner,
        config=RuntimeEntryConfig(use_demo_mode=True, demo_mode_enabled=True),
    )

    assert result.runner_used == "demo_mode"
    assert result.output_summary["candidate_count"] == 3
    assert calls == {"default": 0, "demo": 1}


def test_demo_mode_disabled_falls_back_to_default_runner(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "demo": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def demo_runner(raw_jd):
        calls["demo"] += 1
        return fake_demo_runner(raw_jd)

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        demo_runner=demo_runner,
        config=RuntimeEntryConfig(use_demo_mode=True, demo_mode_enabled=False),
    )

    assert result.runner_used == "default_graph"
    assert calls == {"default": 1, "demo": 0}


def test_sqlite_runtime_store_persists_task_and_timeline(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    db_path = tmp_path / "runtime_entry.sqlite"
    store = SQLiteRuntimeStore(db_path)

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        store=store,
    )

    restored_store = SQLiteRuntimeStore(db_path)
    task = restored_store.get_task(result.task_id)
    timeline = restored_store.list_events(task_id=result.task_id)

    assert task.status == TaskStatus.COMPLETED
    assert task.result["report_count"] == 1
    assert [event.event_type for event in timeline] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]


def test_cli_help_executes(capsys):
    try:
        runtime_cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out

    assert "--jd" in output
    assert "--demo-mode" in output


def test_cli_json_output_uses_injected_fake_runner(monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)

    exit_code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=fake_default_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["runner_used"] == "default_graph"
    assert output["summary_only"] is True
    assert output["metadata"]["jd_length"] == 4
    assert output["output_summary"]["candidate_count"] == 2
    assert "完整简历内容" not in json.dumps(output, ensure_ascii=False)
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_cli_json_output_can_use_injected_demo_runner(monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)

    exit_code = run_cli(
        ["--jd", "招聘JD", "--demo-mode", "--enable-demo-mode", "--json"],
        default_runner=fake_default_runner,
        demo_runner=fake_demo_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "demo_mode"
    assert output["output_summary"]["candidate_count"] == 3


def test_default_create_recruit_graph_behavior_not_modified(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    RuntimeEntryHarness().run("招聘JD", default_runner=fake_default_runner)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
