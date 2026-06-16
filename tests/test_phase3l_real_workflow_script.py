import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_real_workflow.py"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_real_workflow", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeReadinessResult:
    def __init__(self, name, status, detail=""):
        self.name = name
        self.status = status
        self.detail = detail


def readiness_ok():
    return [
        FakeReadinessResult("config_loads", "OK", "ok"),
        FakeReadinessResult("openai_api_key", "OK", "OPENAI_API_KEY=set"),
    ]


def readiness_missing():
    return [
        FakeReadinessResult("config_loads", "OK", "ok"),
        FakeReadinessResult("openai_api_key", "FAIL", "OPENAI_API_KEY is not set"),
    ]


def fake_workflow_runner(jd_text, max_candidates):
    return {
        "status": "ok",
        "jd_length": len(jd_text),
        "graph_invoked": True,
        "final_state_keys": ["candidate_pool", "final_reports"],
        "retrieved_count": 1,
        "candidate_count": 1,
        "candidate_summaries": [{"source": "fixture", "text_preview": "short"}],
        "match_count": 1,
        "top_scores": [88.0],
        "need_refine": False,
        "refined_query": "Python LangGraph RAG",
        "error": "",
    }


def test_smoke_real_workflow_script_file_exists():
    assert SCRIPT_PATH.exists()


def test_smoke_real_workflow_script_can_import_main_and_helpers():
    module = load_module()

    assert hasattr(module, "main")
    assert hasattr(module, "run_smoke")
    assert hasattr(module, "run_production_workflow")


def test_smoke_real_workflow_help_executes_without_real_llm():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--jd" in result.stdout
    assert "--json" in result.stdout


def test_readiness_not_green_default_mode_gracefully_skips_and_exits_zero():
    module = load_module()

    summary = module.run_smoke(
        jd_text="Need Python agent",
        readiness_runner=readiness_missing,
        workflow_runner=fake_workflow_runner,
    )

    assert summary["status"] == "skipped"
    assert summary["graph_invoked"] is False
    assert summary["exit_code"] == 0
    assert summary["readiness"]["missing"][0]["name"] == "openai_api_key"


def test_readiness_not_green_strict_mode_returns_nonzero():
    module = load_module()

    summary = module.run_smoke(
        jd_text="Need Python agent",
        strict=True,
        readiness_runner=readiness_missing,
        workflow_runner=fake_workflow_runner,
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] != 0


def test_json_output_mode_returns_parseable_json_without_real_llm(capsys):
    module = load_module()

    exit_code = module.main(
        ["--json", "--jd", "Need Python agent"],
        readiness_runner=readiness_ok,
        workflow_runner=fake_workflow_runner,
    )
    output = capsys.readouterr().out
    parsed = json.loads(output)

    assert exit_code == 0
    assert parsed["status"] == "ok"
    assert parsed["graph_invoked"] is True
    assert parsed["match_count"] == 1


def test_no_readiness_mode_can_run_fake_workflow_runner():
    module = load_module()

    summary = module.run_smoke(
        jd_text="Need Python agent",
        no_readiness=True,
        readiness_runner=readiness_missing,
        workflow_runner=fake_workflow_runner,
    )

    assert summary["status"] == "ok"
    assert summary["graph_invoked"] is True


def test_failed_workflow_default_mode_returns_failure_summary_without_raising():
    module = load_module()

    def failing_runner(jd_text, max_candidates):
        raise RuntimeError("real workflow failed")

    summary = module.run_smoke(
        jd_text="Need Python agent",
        readiness_runner=readiness_ok,
        workflow_runner=failing_runner,
    )

    assert summary["status"] == "failed"
    assert summary["exit_code"] == 0
    assert summary["error"] == "real workflow failed"


def test_phase3l_does_not_modify_production_graph():
    graph_source = (PROJECT_ROOT / "src" / "core" / "graph.py").read_text(encoding="utf-8")

    assert "smoke_real_workflow" not in graph_source
    assert "SkillRegistry" not in graph_source
    assert "RecruitmentSkillWorkflow" not in graph_source
