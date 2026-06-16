import builtins
import json
import sys

from scripts.run_recruit_runtime import main as runtime_cli_main, run_cli
from src.runtime import RuntimeEntryConfig, RuntimeEntryHarness, TaskStatus
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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase8C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_default_runner(raw_jd):
    return {
        "status": "ok",
        "candidate_count": 1,
        "report_count": 1,
        "top_score_present": True,
        "raw_jd": raw_jd,
        "metadata": {"sensitive_payload": "完整敏感JD不应输出"},
    }


def test_runtime_entry_config_defaults_disable_demo_variant_and_memory():
    config = RuntimeEntryConfig()

    assert config.use_demo_mode is False
    assert config.demo_mode_enabled is False
    assert config.use_skill_backed_variant is False
    assert config.allow_memory_context is False
    assert config.require_ab_smoke_pass is True
    assert config.rollback_on_variant_failure is True


def test_default_run_only_calls_default_runner(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "demo": 0, "variant": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def demo_runner(raw_jd, **_kwargs):
        calls["demo"] += 1
        return {"status": "ok", "candidate_count": 2, "report_count": 2}

    def variant_runner(raw_jd, **_kwargs):
        calls["variant"] += 1
        return {"status": "ok", "candidate_count": 3, "report_count": 3}

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        demo_runner=demo_runner,
        variant_runner=variant_runner,
    )

    assert result.runner_used == "default_graph"
    assert result.output_summary["candidate_count"] == 1
    assert calls == {"default": 1, "demo": 0, "variant": 0}


def test_demo_requested_but_disabled_falls_back_to_default(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "demo": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def demo_runner(raw_jd, **_kwargs):
        calls["demo"] += 1
        return {"status": "ok", "candidate_count": 2, "report_count": 2}

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        demo_runner=demo_runner,
        config=RuntimeEntryConfig(use_demo_mode=True, demo_mode_enabled=False),
    )

    assert result.runner_used == "default_graph"
    assert result.metadata["demo_mode_requested_but_disabled"] is True
    assert result.metadata["runner_selection_reason"] == "demo_mode_disabled_default_fallback"
    assert calls == {"default": 1, "demo": 0}


def test_demo_enabled_calls_demo_runner(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "demo": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def demo_runner(raw_jd, **_kwargs):
        calls["demo"] += 1
        return {"status": "ok", "candidate_count": 4, "report_count": 2}

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        demo_runner=demo_runner,
        config=RuntimeEntryConfig(use_demo_mode=True, demo_mode_enabled=True),
    )

    assert result.runner_used == "demo_mode"
    assert result.output_summary["candidate_count"] == 4
    assert calls == {"default": 0, "demo": 1}


def test_skill_backed_variant_requested_calls_variant_runner(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0, "variant": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    def variant_runner(raw_jd, **_kwargs):
        calls["variant"] += 1
        return {
            "status": "ok",
            "candidate_count": 5,
            "report_count": 3,
            "top_score_present": True,
            "metadata": {"variant": True},
        }

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        variant_runner=variant_runner,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    assert result.runner_used == "skill_backed_variant"
    assert result.output_summary["candidate_count"] == 5
    assert result.metadata["skill_backed_variant_requested"] is True
    assert calls == {"default": 0, "variant": 1}


def test_variant_requested_without_runner_safely_falls_back_to_default(monkeypatch):
    block_retrieval_imports(monkeypatch)
    calls = {"default": 0}

    def default_runner(raw_jd):
        calls["default"] += 1
        return fake_default_runner(raw_jd)

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=default_runner,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    assert result.runner_used == "default_graph"
    assert result.metadata["variant_requested_but_unavailable"] is True
    assert result.metadata["runner_selection_reason"] == "variant_runner_missing_default_fallback"
    assert calls == {"default": 1}


def test_memory_context_not_passed_when_flag_disabled(monkeypatch):
    block_retrieval_imports(monkeypatch)
    seen = {"memory_context": "unset"}

    def variant_runner(raw_jd, memory_context=None, metadata=None):
        seen["memory_context"] = memory_context
        seen["metadata"] = metadata
        return {"status": "ok", "candidate_count": 1, "report_count": 1}

    RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        variant_runner=variant_runner,
        memory_context="只读 memory preview",
        config=RuntimeEntryConfig(use_skill_backed_variant=True, allow_memory_context=False),
    )

    assert seen["memory_context"] is None
    assert seen["metadata"]["allow_memory_context"] is False


def test_memory_context_passed_as_preview_when_flag_enabled(monkeypatch):
    block_retrieval_imports(monkeypatch)
    seen = {"memory_context": None}

    def variant_runner(raw_jd, memory_context=None, metadata=None):
        seen["memory_context"] = memory_context
        return {
            "status": "ok",
            "candidate_count": 1,
            "report_count": 1,
            "metadata": {
                "memory_context_seen": bool(memory_context),
                "sensitive_payload": "完整 memory 内容不应输出",
            },
        }

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        variant_runner=variant_runner,
        memory_context="read-only preview",
        config=RuntimeEntryConfig(use_skill_backed_variant=True, allow_memory_context=True),
    )

    assert seen["memory_context"] == "read-only preview"
    assert result.runner_used == "skill_backed_variant"
    assert result.output_summary["metadata"]["keys"] == ["memory_context_seen", "sensitive_payload"]
    assert "完整 memory 内容" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_demo_and_variant_paths_create_runtime_events(monkeypatch):
    block_retrieval_imports(monkeypatch)
    store = InMemoryRuntimeStore()

    def demo_runner(raw_jd, **_kwargs):
        return {
            "status": "ok",
            "candidate_count": 2,
            "report_count": 2,
            "rollback_to_default": False,
            "ab_smoke_summary": {"case_id": "demo", "risk_level": "low"},
        }

    result = RuntimeEntryHarness().run(
        "招聘JD",
        default_runner=fake_default_runner,
        demo_runner=demo_runner,
        store=store,
        config=RuntimeEntryConfig(use_demo_mode=True, demo_mode_enabled=True),
    )
    events = store.list_events(task_id=result.task_id)

    assert result.task_status == TaskStatus.COMPLETED.value
    assert result.event_count == 5
    assert [event.event_type for event in events] == [
        "task_created",
        "task_started",
        "graph_primary_started",
        "graph_primary_completed",
        "task_completed",
    ]
    assert result.output_summary["ab_smoke_summary"]["case_id"] == "demo"


def test_cli_help_includes_demo_variant_flags(capsys):
    try:
        runtime_cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out

    assert "--enable-demo-mode" in output
    assert "--use-skill-backed-variant" in output
    assert "--allow-memory-context" in output


def test_cli_json_with_fake_demo_runner_is_parseable(monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)

    def demo_runner(raw_jd, memory_context=None, metadata=None):
        return {
            "status": "ok",
            "candidate_count": 2,
            "report_count": 1,
            "metadata": {
                "memory_context_seen": bool(memory_context),
                "sensitive_payload": "完整敏感内容不应输出",
            },
        }

    exit_code = run_cli(
        [
            "--jd",
            "招聘JD",
            "--demo-mode",
            "--enable-demo-mode",
            "--allow-memory-context",
            "--json",
        ],
        default_runner=fake_default_runner,
        demo_runner=demo_runner,
        memory_context="read-only preview",
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "demo_mode"
    assert output["metadata"]["allow_memory_context"] is True
    assert output["summary_only"] is True
    assert "完整敏感内容" not in json.dumps(output, ensure_ascii=False)
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_cli_json_with_fake_variant_runner_is_parseable(monkeypatch, capsys):
    block_retrieval_imports(monkeypatch)

    def variant_runner(raw_jd, memory_context=None, metadata=None):
        return {
            "status": "ok",
            "candidate_count": 3,
            "report_count": 2,
            "top_score_present": True,
            "metadata": {"variant": True},
        }

    exit_code = run_cli(
        ["--jd", "招聘JD", "--use-skill-backed-variant", "--json"],
        default_runner=fake_default_runner,
        variant_runner=variant_runner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "skill_backed_variant"
    assert output["output_summary"]["candidate_count"] == 3
    assert output["metadata"]["skill_backed_variant_requested"] is True


def test_default_create_recruit_graph_behavior_not_modified(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    RuntimeEntryHarness().run("招聘JD", default_runner=fake_default_runner)

    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
