import builtins
from pathlib import Path

from src.integration.ab_smoke import ABSmokeHarness
from src.integration.demo_mode import (
    DemoModeConfig,
    DemoModeResult,
    LimitedProductionDemoHarness,
)


SENSITIVE_JD = "PRIVATE-JD-CONTENT-MUST-NOT-APPEAR-IN-DEMO-RESULT"


def default_runner(status="ok", candidate_count=1, report_count=1):
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


def variant_runner(status="ok", candidate_count=1, report_count=1, error_type="", risk_level=""):
    def run(raw_jd, memory_context=None, metadata=None, enable_memory_context=False):
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
                "memory_context_seen": memory_context is not None,
                "raw_jd": raw_jd,
                **dict(metadata or {}),
            },
        }

    return run


def enabled_config(**kwargs):
    values = {
        "enabled": True,
        "use_skill_backed_variant": True,
        "require_ab_smoke_pass": True,
    }
    values.update(kwargs)
    return DemoModeConfig(**values)


def test_demo_mode_config_defaults_disabled():
    config = DemoModeConfig()

    assert config.enabled is False
    assert config.use_skill_backed_variant is False
    assert config.allow_memory_context is False
    assert config.require_ab_smoke_pass is True
    assert config.rollback_on_variant_failure is True
    assert config.summary_only is True


def test_disabled_config_does_not_run_variant_or_default():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=DemoModeConfig(enabled=False),
    )

    assert result.status == "disabled"
    assert result.default_invoked is False
    assert result.variant_invoked is False
    assert result.output_summary == {}


def test_enabled_without_skill_backed_variant_runs_default_only():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=DemoModeConfig(enabled=True, use_skill_backed_variant=False),
    )

    assert result.status == "default_only"
    assert result.default_invoked is True
    assert result.variant_invoked is False
    assert result.output_summary["status"] == "ok"


def test_enabled_variant_with_ab_pass_uses_variant():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=enabled_config(),
    )

    assert isinstance(result, DemoModeResult)
    assert result.status == "variant_used"
    assert result.default_invoked is True
    assert result.variant_invoked is True
    assert result.rollback_to_default is False
    assert result.risk_level == "low"
    assert result.output_summary["metadata"]["keys"] == [
        "memory_context_seen",
        "raw_jd",
        "runner",
        "summary_only",
    ]


def test_enabled_variant_with_ab_rollback_uses_default_summary():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(candidate_count=1, report_count=1),
        variant_runner=variant_runner(candidate_count=0, report_count=0),
        config=enabled_config(),
    )

    assert result.status == "rolled_back"
    assert result.rollback_to_default is True
    assert result.risk_level == "high"
    assert result.output_summary["output_keys"] == ["candidate_pool", "final_reports"]
    assert result.ab_smoke_summary["rollback_to_default"] is True


def test_variant_failure_with_rollback_enabled_falls_back_to_default():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(status="failed", error_type="RuntimeError"),
        config=enabled_config(require_ab_smoke_pass=False, rollback_on_variant_failure=True),
    )

    assert result.status == "rolled_back"
    assert result.default_invoked is True
    assert result.variant_invoked is True
    assert result.rollback_to_default is True
    assert result.output_summary["status"] == "ok"


def test_variant_failure_with_rollback_disabled_returns_failed():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(status="failed", error_type="RuntimeError"),
        config=enabled_config(require_ab_smoke_pass=False, rollback_on_variant_failure=False),
    )

    assert result.status == "failed"
    assert result.default_invoked is False
    assert result.variant_invoked is True
    assert result.rollback_to_default is False
    assert result.output_summary["error_type"] == "RuntimeError"


def test_allow_memory_context_false_does_not_pass_memory_context_to_variant():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=enabled_config(
            require_ab_smoke_pass=False,
            allow_memory_context=False,
        ),
        memory_context="readonly preview",
    )

    assert result.status == "variant_used"
    assert result.metadata["memory_context_used"] is False
    assert "memory_context_seen" in result.output_summary["metadata"]["keys"]


def test_allow_memory_context_true_passes_readonly_preview_to_variant_metadata():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=enabled_config(
            require_ab_smoke_pass=False,
            allow_memory_context=True,
        ),
        memory_context="readonly preview",
    )

    assert result.status == "variant_used"
    assert result.metadata["memory_context_used"] is True
    assert "memory_context_seen" in result.output_summary["metadata"]["keys"]


def test_result_summary_does_not_contain_full_sensitive_payload():
    result = LimitedProductionDemoHarness().run(
        SENSITIVE_JD,
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=enabled_config(),
    )

    payload = str(result.to_dict())

    assert SENSITIVE_JD not in payload
    assert result.metadata["raw_jd_length"] == len(SENSITIVE_JD)
    assert result.output_summary["metadata"]["keys"]


def test_missing_variant_runner_rolls_back_when_enabled():
    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=None,
        config=enabled_config(),
    )

    assert result.status == "rolled_back"
    assert result.rollback_to_default is True
    assert result.rollback_reason == "variant runner missing"
    assert result.default_invoked is True


def test_run_with_ab_smoke_can_use_injected_ab_harness():
    result = LimitedProductionDemoHarness().run_with_ab_smoke(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        ab_harness=ABSmokeHarness(),
        config=enabled_config(),
    )

    assert result.status == "variant_used"
    assert result.ab_smoke_summary["risk_level"] == "low"


def test_default_production_graph_source_is_not_modified():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "LimitedProductionDemoHarness" not in graph_source
    assert "DemoModeConfig" not in graph_source
    assert "DemoModeResult" not in graph_source


def test_phase7e_does_not_import_real_llm_retriever_chroma_or_llamaindex(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase7E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = LimitedProductionDemoHarness().run(
        "Need Python",
        default_runner=default_runner(),
        variant_runner=variant_runner(),
        config=enabled_config(),
    )

    assert result.status == "variant_used"
    assert blocked == []
