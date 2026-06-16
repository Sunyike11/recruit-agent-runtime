import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_real_matcher_shadow_compare.py"
SENSITIVE_TEXT = "FULL-CANDIDATE-RESUME-SECRET-MUST-NOT-ENTER-MATCHER-SMOKE"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_real_matcher_shadow_compare", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeReadinessResult:
    def __init__(self, name, status):
        self.name = name
        self.status = status


def readiness_ok():
    return [
        FakeReadinessResult("matcher_agent_import", "OK"),
        FakeReadinessResult("openai_api_key", "OK"),
        FakeReadinessResult("matcher_agent_init", "OK"),
    ]


def readiness_missing():
    return [
        FakeReadinessResult("matcher_agent_import", "OK"),
        FakeReadinessResult("openai_api_key", "FAIL"),
    ]


def fake_candidate():
    return {
        "candidate_id": "candidate_smoke_1",
        "name": "Candidate",
        "skills": ["Python", "RAG", "LangGraph"],
    }


def fake_real_matcher(job_requirement, candidate_profile):
    return {
        "final_reports": [
            {
                "candidate_id": candidate_profile["candidate_id"],
                "total_score": 91,
                "final_verdict": "QUALIFIED",
            }
        ]
    }


def fake_shadow_matcher(job_requirement, candidate_profile):
    return {
        "match_report": {
            "candidate_id": candidate_profile["candidate_id"],
            "total_score": 84,
        },
        "total_score": 84,
    }


def test_optional_real_matcher_smoke_script_exists_and_help_executes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert SCRIPT_PATH.exists()
    assert result.returncode == 0
    assert "--jd" in result.stdout
    assert "--candidate-json" in result.stdout
    assert "--compare-exact-scores" in result.stdout
    assert "--json" in result.stdout
    assert "--strict" in result.stdout


def test_missing_readiness_default_mode_skips_without_invoking_runners():
    module = load_module()
    invoked = []

    summary = module.run_smoke(
        jd_text="Need Python agent",
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_missing,
        real_matcher_runner=lambda job, candidate: invoked.append("real"),
        shadow_runner=lambda job, candidate: invoked.append("shadow"),
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] == 0
    assert summary["real_matcher_invoked"] is False
    assert summary["shadow_invoked"] is False
    assert summary["readiness"]["missing"] == [{"name": "openai_api_key", "status": "FAIL"}]
    assert invoked == []


def test_missing_readiness_strict_mode_returns_nonzero():
    module = load_module()

    summary = module.run_smoke(
        jd_text="Need Python agent",
        candidate_profile=fake_candidate(),
        strict=True,
        readiness_runner=readiness_missing,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] != 0


def test_fake_real_and_shadow_matcher_outputs_generate_summary_and_decision():
    module = load_module()

    summary = module.run_smoke(
        jd_text="Need Python LangGraph engineer",
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )

    assert summary["status"] == "ok"
    assert summary["candidate_id"] == "candidate_smoke_1"
    assert summary["real_matcher_invoked"] is True
    assert summary["shadow_invoked"] is True
    assert summary["decision_status"] == "match"
    assert summary["risk_level"] == "low"
    assert summary["score_present"] is True
    assert summary["report_keys"] == ["candidate_id", "final_verdict", "total_score"]


def test_score_difference_is_not_failure_by_default_but_can_be_compared_exactly():
    module = load_module()

    default_summary = module.run_smoke(
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )
    exact_summary = module.run_smoke(
        candidate_profile=fake_candidate(),
        compare_exact_scores=True,
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )

    assert default_summary["decision_status"] == "match"
    assert default_summary["status"] == "ok"
    assert exact_summary["decision_status"] == "mismatch"
    assert exact_summary["risk_level"] == "high"
    assert exact_summary["status"] == "ok"


def test_json_output_is_parseable_with_fake_injected_runners(capsys):
    module = load_module()

    exit_code = module.main(
        ["--json", "--jd", "Need Python agent"],
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )
    parsed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert parsed["status"] == "ok"
    assert parsed["decision_status"] == "match"
    assert parsed["summary_only"] is True
    assert "exit_code" not in parsed


def test_candidate_json_input_is_supported_without_emitting_profile_content(tmp_path, capsys):
    module = load_module()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate_file_1",
                "name": SENSITIVE_TEXT,
                "skills": ["Python"],
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        ["--json", "--candidate-json", str(candidate_path)],
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )
    output = capsys.readouterr().out
    parsed = json.loads(output)

    assert exit_code == 0
    assert parsed["candidate_id"] == "candidate_file_1"
    assert SENSITIVE_TEXT not in output


def test_failed_real_matcher_path_keeps_only_sanitized_error_type():
    module = load_module()

    def fail(job_requirement, candidate_profile):
        print(SENSITIVE_TEXT)
        raise RuntimeError(f"provider failure: {SENSITIVE_TEXT}")

    summary = module.run_smoke(
        jd_text=SENSITIVE_TEXT,
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_ok,
        real_matcher_runner=fail,
        shadow_runner=fake_shadow_matcher,
    )
    serialized = json.dumps(summary)

    assert summary["status"] == "failed"
    assert summary["decision_status"] == "skipped"
    assert summary["error_type"] == "RuntimeError"
    assert summary["real_matcher_invoked"] is True
    assert SENSITIVE_TEXT not in serialized


def test_summary_does_not_emit_complete_jd_candidate_or_match_report(capsys):
    module = load_module()

    summary = module.run_smoke(
        jd_text=SENSITIVE_TEXT,
        candidate_profile={**fake_candidate(), "name": SENSITIVE_TEXT},
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )
    module.print_text_summary(summary)
    output = capsys.readouterr().out

    assert SENSITIVE_TEXT not in str(summary)
    assert SENSITIVE_TEXT not in output
    assert "final_verdict" in output
    assert "QUALIFIED" not in output


def test_structural_path_does_not_import_real_agents_retrieval_or_external_dependencies(monkeypatch):
    module = load_module()
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            ("src.agents", "src.services.retriever", "llama_index", "chromadb", "mcp")
        ):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase6G test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    summary = module.run_smoke(
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )

    assert summary["status"] == "ok"
    assert blocked == []


def test_optional_matcher_smoke_does_not_modify_or_invoke_production_graph():
    graph_source = (PROJECT_ROOT / "src" / "core" / "graph.py").read_text(encoding="utf-8")
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = load_module()

    summary = module.run_smoke(
        candidate_profile=fake_candidate(),
        readiness_runner=readiness_ok,
        real_matcher_runner=fake_real_matcher,
        shadow_runner=fake_shadow_matcher,
    )

    assert summary["production_graph_invoked"] is False
    assert "smoke_real_matcher_shadow_compare" not in graph_source
    assert "create_recruit_graph" not in script_source
    assert "src.core.graph" not in script_source
    assert "RetrieverAgent" not in script_source
