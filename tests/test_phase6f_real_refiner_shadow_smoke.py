import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_real_refiner_shadow_compare.py"
SENSITIVE_TEXT = "FULL-QUERY-LLM-SECRET-MUST-NOT-ENTER-REFINER-SMOKE"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_real_refiner_shadow_compare", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeReadinessResult:
    def __init__(self, name, status):
        self.name = name
        self.status = status


def readiness_ok():
    return [
        FakeReadinessResult("refiner_agent_import", "OK"),
        FakeReadinessResult("openai_api_key", "OK"),
        FakeReadinessResult("refiner_agent_init", "OK"),
    ]


def readiness_missing():
    return [
        FakeReadinessResult("refiner_agent_import", "OK"),
        FakeReadinessResult("openai_api_key", "FAIL"),
    ]


def fake_real_refiner(query, context):
    return {"extracted_jd": {"search_query": "python langgraph remote"}}


def fake_shadow_refiner(query, context):
    return {"refined_query": "python langgraph remote"}


def test_optional_real_refiner_smoke_script_exists_and_help_executes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert SCRIPT_PATH.exists()
    assert result.returncode == 0
    assert "--query" in result.stdout
    assert "--context" in result.stdout
    assert "--json" in result.stdout
    assert "--strict" in result.stdout


def test_missing_readiness_default_mode_skips_without_invoking_runners():
    module = load_module()
    invoked = []

    summary = module.run_smoke(
        query="Need Python agent",
        readiness_runner=readiness_missing,
        real_refiner_runner=lambda query, context: invoked.append("real"),
        shadow_runner=lambda query, context: invoked.append("shadow"),
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] == 0
    assert summary["real_refiner_invoked"] is False
    assert summary["shadow_invoked"] is False
    assert summary["readiness"]["missing"] == [{"name": "openai_api_key", "status": "FAIL"}]
    assert invoked == []


def test_missing_readiness_strict_mode_returns_nonzero():
    module = load_module()

    summary = module.run_smoke(
        query="Need Python agent",
        strict=True,
        readiness_runner=readiness_missing,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=fake_shadow_refiner,
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] != 0


def test_fake_real_and_shadow_refiner_outputs_generate_ok_summary_and_decision():
    module = load_module()

    summary = module.run_smoke(
        query="Need Python agent",
        context="broaden",
        readiness_runner=readiness_ok,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=fake_shadow_refiner,
    )

    assert summary["status"] == "ok"
    assert summary["real_refiner_invoked"] is True
    assert summary["shadow_invoked"] is True
    assert summary["decision_status"] == "match"
    assert summary["risk_level"] == "low"
    assert summary["refined_query_length"] == len("python langgraph remote")
    assert summary["refined_query_preview"] == "<present; redacted>"


def test_json_output_is_parseable_with_fake_injected_runners(capsys):
    module = load_module()

    exit_code = module.main(
        ["--json", "--query", "Need Python agent", "--context", "broaden"],
        readiness_runner=readiness_ok,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=fake_shadow_refiner,
    )
    parsed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert parsed["status"] == "ok"
    assert parsed["decision_status"] == "match"
    assert parsed["summary_only"] is True
    assert "exit_code" not in parsed


def test_mismatch_is_observed_without_treating_optional_smoke_as_execution_failure():
    module = load_module()

    summary = module.run_smoke(
        query="Need Python agent",
        readiness_runner=readiness_ok,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=lambda query, context: {"refined_query": "python backend"},
    )

    assert summary["status"] == "ok"
    assert summary["decision_status"] == "mismatch"
    assert summary["risk_level"] == "high"
    assert summary["exit_code"] == 0


def test_failed_real_refiner_path_keeps_only_sanitized_error_type():
    module = load_module()

    def fail(query, context):
        print(SENSITIVE_TEXT)
        raise RuntimeError(f"provider failure: {SENSITIVE_TEXT}")

    summary = module.run_smoke(
        query=SENSITIVE_TEXT,
        readiness_runner=readiness_ok,
        real_refiner_runner=fail,
        shadow_runner=fake_shadow_refiner,
    )
    serialized = json.dumps(summary)

    assert summary["status"] == "failed"
    assert summary["decision_status"] == "skipped"
    assert summary["error_type"] == "RuntimeError"
    assert summary["real_refiner_invoked"] is True
    assert SENSITIVE_TEXT not in serialized


def test_script_summary_does_not_emit_complete_real_or_shadow_output(capsys):
    module = load_module()

    summary = module.run_smoke(
        query=SENSITIVE_TEXT,
        readiness_runner=readiness_ok,
        real_refiner_runner=lambda query, context: {
            "extracted_jd": {"search_query": SENSITIVE_TEXT}
        },
        shadow_runner=lambda query, context: {"refined_query": SENSITIVE_TEXT},
    )
    module.print_text_summary(summary)
    output = capsys.readouterr().out

    assert summary["decision_status"] == "match"
    assert SENSITIVE_TEXT not in str(summary)
    assert SENSITIVE_TEXT not in output
    assert "<present; redacted>" in output


def test_structural_test_path_does_not_import_real_agents_retrieval_or_external_dependencies(monkeypatch):
    module = load_module()
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            ("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")
        ):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase6F test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    summary = module.run_smoke(
        query="Need Python agent",
        readiness_runner=readiness_ok,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=fake_shadow_refiner,
    )

    assert summary["status"] == "ok"
    assert blocked == []


def test_optional_refiner_smoke_does_not_modify_or_invoke_production_graph():
    graph_source = (PROJECT_ROOT / "src" / "core" / "graph.py").read_text(encoding="utf-8")
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = load_module()

    summary = module.run_smoke(
        query="Need Python agent",
        readiness_runner=readiness_ok,
        real_refiner_runner=fake_real_refiner,
        shadow_runner=fake_shadow_refiner,
    )

    assert summary["production_graph_invoked"] is False
    assert "smoke_real_refiner_shadow_compare" not in graph_source
    assert "create_recruit_graph" not in script_source
    assert "src.core.graph" not in script_source
    assert "RetrieverAgent" not in script_source
