import builtins
import json
from pathlib import Path

from scripts.eval_memory_influence import run_cli
from src.runtime.memory_influence import MemoryInfluenceEvalCase, MemoryInfluenceEvaluator
from src.runtime.memory_influence_export import (
    export_memory_influence_result_json,
    export_memory_influence_result_text,
    sanitize_memory_influence_report,
)


SENSITIVE_JD = "PRIVATE-JD-MUST-NOT-LEAK-PHASE9D"
SENSITIVE_MEMORY = "PRIVATE-MEMORY-MUST-NOT-LEAK-PHASE9D"


def fake_summary(*, memory=False, status="ok", candidate_count=1, report_count=1):
    return {
        "status": status,
        "runner_used": "fake",
        "candidate_count": candidate_count,
        "report_count": report_count,
        "top_score_present": report_count > 0,
        "top_scores": [88.0],
        "candidate_ids": [f"candidate_{idx}" for idx in range(candidate_count)],
        "candidate_profile_preview_count": candidate_count,
        "memory_context_provided": memory,
        "memory_context_eligible_count": 1 if memory else 0,
        "memory_context_rendered_char_count": 80 if memory else 0,
        "output_keys": ["candidate_count", "report_count"],
        "metadata": {"memory": SENSITIVE_MEMORY},
    }


def build_result(*, with_memory=None, expected_effect=None):
    evaluator = MemoryInfluenceEvaluator()
    case = MemoryInfluenceEvalCase(
        "case_export",
        SENSITIVE_JD,
        memory_source="demo",
        memory_config={"memory_content": SENSITIVE_MEMORY},
        expected_effect=expected_effect,
    )
    with_memory_summary = with_memory or fake_summary(memory=True)
    return evaluator.run_case(
        case,
        no_memory_runner=lambda _jd: fake_summary(),
        with_memory_runner=lambda _jd, **_kwargs: with_memory_summary,
        memory_context=object(),
    )


def test_json_export_is_summary_only_and_redacts_sensitive_payloads():
    exported = export_memory_influence_result_json(build_result())
    payload = json.loads(exported)

    assert payload["case_id"] == "case_export"
    assert payload["summary_only"] is True
    assert "decision" in payload
    assert SENSITIVE_JD not in exported
    assert SENSITIVE_MEMORY not in exported


def test_text_export_contains_decision_risk_and_counts():
    text = export_memory_influence_result_text(
        build_result(with_memory=fake_summary(memory=True, candidate_count=2), expected_effect="candidate_count")
    )

    assert "Memory Influence Report" in text
    assert "decision: expected_effect" in text
    assert "risk_level: medium" in text
    assert "candidate_count: 1" in text
    assert "candidate_count: 2" in text


def test_sanitize_report_exposes_only_summary_fields():
    report = sanitize_memory_influence_report(build_result())

    assert report["summary_only"] is True
    assert "no_memory_summary" in report
    assert "with_memory_summary" in report
    assert "top_score_summary" in report["no_memory_summary"]
    assert "candidate_ids" not in report["no_memory_summary"]


def test_cli_json_is_parseable_for_demo_memory(capsys):
    code = run_cli(
        [
            "--case-id",
            "cli_demo",
            "--jd",
            SENSITIVE_JD,
            "--memory-source",
            "demo",
            "--json",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert code == 0
    assert payload["case_id"] == "cli_demo"
    assert payload["decision"] == "expected_effect"
    assert payload["risk_level"] == "medium"
    assert SENSITIVE_JD not in output


def test_cli_text_outputs_report(capsys):
    code = run_cli(["--case-id", "cli_text", "--memory-source", "demo", "--text"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Memory Influence Report" in output
    assert "case_id: cli_text" in output
    assert "Delta:" in output


def test_cli_no_memory_source_is_skipped_or_warning(capsys):
    code = run_cli(["--case-id", "cli_none", "--memory-source", "none", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["decision"] == "skipped"
    assert payload["risk_level"] == "medium"
    assert payload["memory_context_used"] is False


def test_with_memory_failure_exports_regression_risk():
    result = build_result(with_memory=fake_summary(memory=True, status="failed", report_count=0))
    payload = json.loads(export_memory_influence_result_json(result))

    assert payload["decision"] == "regression_risk"
    assert payload["risk_level"] == "high"


def test_cli_runtime_variant_path_is_safe_skipped(capsys):
    code = run_cli(["--use-runtime-variant", "--memory-source", "demo", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["decision"] == "skipped"
    assert payload["with_memory_summary"]["runner_used"] == "runtime_variant_not_enabled"


def test_cli_sqlite_nonexistent_path_is_fake_and_summary_only(capsys, tmp_path):
    missing = tmp_path / "missing.sqlite3"
    code = run_cli(
        [
            "--memory-source",
            "sqlite",
            "--memory-db-path",
            str(missing),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["summary_only"] is True
    assert missing.exists() is False


def test_phase9d_does_not_import_real_llm_chroma_hf_or_mcp(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "sentence_transformers", "mcp", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase9D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    exported = export_memory_influence_result_json(build_result())

    assert json.loads(exported)["summary_only"] is True
    assert blocked == []


def test_default_graph_source_is_not_modified_for_memory_influence_cli():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "eval_memory_influence" not in graph_source
    assert "MemoryInfluenceEvaluator" not in graph_source
    assert "memory influence" not in graph_source.lower()
