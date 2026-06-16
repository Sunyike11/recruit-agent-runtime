import builtins
from pathlib import Path

from src.integration.ab_smoke import (
    ABSmokeCase,
    ABSmokeHarness,
    ABSmokeReport,
    ABSmokeResult,
)


SENSITIVE_JD = "PRIVATE-JD-CONTENT-MUST-NOT-APPEAR-IN-AB-SMOKE-RESULT"


def default_runner(candidate_count=1, report_count=1, status="ok"):
    def run(raw_jd):
        return {
            "status": status,
            "candidate_count": candidate_count,
            "report_count": report_count,
            "top_score_present": report_count > 0,
            "output_keys": ["candidate_pool", "final_reports"],
            "metadata": {"runner": "default", "raw_jd": raw_jd},
        }

    return run


def variant_runner(candidate_count=1, report_count=1, status="ok", error_type="", risk_level=""):
    def run(raw_jd, enable_memory_context=False, metadata=None):
        return {
            "status": status,
            "candidate_count": candidate_count,
            "report_count": report_count,
            "top_score_present": report_count > 0,
            "error_type": error_type,
            "risk_level": risk_level,
            "output_keys": ["candidate_pool_preview", "final_reports_preview"],
            "metadata": {
                "runner": "variant",
                "memory_context_requested": enable_memory_context,
                "raw_jd": raw_jd,
                **dict(metadata or {}),
            },
        }

    return run


def test_ab_smoke_case_can_create_and_redacts_raw_jd():
    case = ABSmokeCase(
        case_id="case_1",
        raw_jd=SENSITIVE_JD,
        enable_variant=True,
        metadata={"tag": "smoke"},
    )

    data = case.to_dict()

    assert data["case_id"] == "case_1"
    assert data["raw_jd"] == "<present; redacted>"
    assert data["raw_jd_length"] == len(SENSITIVE_JD)
    assert SENSITIVE_JD not in str(data)


def test_ab_smoke_harness_runs_aligned_fake_runners_without_rollback():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_aligned", "Need Python", enable_variant=True),
        default_runner=default_runner(),
        variant_runner=variant_runner(),
    )

    assert isinstance(result, ABSmokeResult)
    assert result.default_status == "ok"
    assert result.variant_status == "ok"
    assert result.rollback_to_default is False
    assert result.risk_level == "low"
    assert result.comparison_summary["count_difference"] is False


def test_variant_failed_triggers_high_risk_rollback():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_failed", "Need Python", enable_variant=True),
        default_runner=default_runner(),
        variant_runner=variant_runner(status="failed", error_type="RuntimeError"),
    )

    assert result.rollback_to_default is True
    assert result.risk_level == "high"
    assert "variant status" in result.rollback_reason


def test_variant_exception_is_sanitized_and_triggers_rollback():
    def failing_variant(raw_jd, **kwargs):
        raise RuntimeError("full sensitive exception text should not be copied")

    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_exception", "Need Python", enable_variant=True),
        default_runner=default_runner(),
        variant_runner=failing_variant,
    )

    assert result.rollback_to_default is True
    assert result.variant_summary["error_type"] == "RuntimeError"
    assert "full sensitive exception" not in str(result.to_dict())


def test_variant_empty_candidates_when_default_has_candidates_triggers_rollback():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_empty_candidates", "Need Python", enable_variant=True),
        default_runner=default_runner(candidate_count=2, report_count=1),
        variant_runner=variant_runner(candidate_count=0, report_count=0),
    )

    assert result.rollback_to_default is True
    assert result.risk_level == "high"
    assert "no candidates" in result.rollback_reason


def test_variant_report_count_difference_is_medium_risk_without_rollback():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_count_diff", "Need Python", enable_variant=True),
        default_runner=default_runner(candidate_count=2, report_count=2),
        variant_runner=variant_runner(candidate_count=2, report_count=1),
    )

    assert result.rollback_to_default is False
    assert result.risk_level == "medium"
    assert result.comparison_summary["report_count_delta"] == -1


def test_variant_high_risk_metadata_triggers_rollback():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_high_risk", "Need Python", enable_variant=True),
        default_runner=default_runner(),
        variant_runner=variant_runner(risk_level="high"),
    )

    assert result.rollback_to_default is True
    assert result.risk_level == "high"
    assert "high risk" in result.rollback_reason


def test_run_cases_generates_ab_smoke_report_counts():
    report = ABSmokeHarness().run_cases(
        [
            ABSmokeCase("case_ok", "Need Python", enable_variant=True),
            ABSmokeCase("case_rollback", "Need Python", enable_variant=True),
        ],
        default_runner=default_runner(),
        variant_runner=lambda raw_jd, **kwargs: (
            variant_runner()(raw_jd, **kwargs)
            if kwargs["metadata"]["case_id"] == "case_ok"
            else variant_runner(status="failed", error_type="ValueError")(raw_jd, **kwargs)
        ),
    )

    assert isinstance(report, ABSmokeReport)
    assert report.total_cases == 2
    assert report.default_success_count == 2
    assert report.variant_success_count == 1
    assert report.rollback_count == 1
    assert report.high_risk_count == 1


def test_result_summary_does_not_contain_full_sensitive_payload():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_sensitive", SENSITIVE_JD, enable_variant=True),
        default_runner=default_runner(),
        variant_runner=variant_runner(),
    )

    payload = str(result.to_dict())

    assert SENSITIVE_JD not in payload
    assert result.metadata["raw_jd_length"] == len(SENSITIVE_JD)
    assert result.default_summary["metadata"]["keys"] == ["raw_jd", "runner"]


def test_memory_context_enabled_case_only_marks_variant_metadata():
    result = ABSmokeHarness().run_case(
        ABSmokeCase(
            "case_memory",
            "Need Python",
            enable_variant=True,
            enable_memory_context=True,
        ),
        default_runner=default_runner(),
        variant_runner=variant_runner(),
    )

    assert result.metadata["memory_context_requested"] is True
    assert result.variant_summary["metadata"]["keys"] == [
        "case_id",
        "memory_context_requested",
        "raw_jd",
        "runner",
        "summary_only",
    ]
    assert "memory_context_requested" not in result.default_summary["metadata"]["keys"]


def test_disabled_variant_is_skipped_and_rolls_back_to_default():
    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_disabled", "Need Python", enable_variant=False),
        default_runner=default_runner(),
        variant_runner=variant_runner(),
    )

    assert result.variant_status == "skipped"
    assert result.rollback_to_default is True
    assert result.risk_level == "high"


def test_default_production_graph_source_is_not_modified():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "ABSmokeHarness" not in graph_source
    assert "ABSmokeCase" not in graph_source
    assert "rollback_to_default" not in graph_source


def test_phase7d_does_not_import_real_llm_retriever_chroma_or_llamaindex(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase7D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = ABSmokeHarness().run_case(
        ABSmokeCase("case_import_guard", "Need Python", enable_variant=True),
        default_runner=default_runner(),
        variant_runner=variant_runner(),
    )

    assert result.risk_level == "low"
    assert blocked == []
