from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from src.integration.ab_smoke import ABSmokeCase, ABSmokeHarness, ABSmokeResult


@dataclass
class DemoModeConfig:
    enabled: bool = False
    use_skill_backed_variant: bool = False
    allow_memory_context: bool = False
    require_ab_smoke_pass: bool = True
    rollback_on_variant_failure: bool = True
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DemoModeResult:
    status: str
    default_invoked: bool
    variant_invoked: bool
    rollback_to_default: bool
    rollback_reason: str
    risk_level: str
    output_summary: Dict[str, Any]
    ab_smoke_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LimitedProductionDemoHarness:
    """Controlled demo wrapper; never routes the default production graph by itself."""

    def run(
        self,
        raw_jd: str,
        default_runner: Callable[[str], Mapping[str, Any]],
        variant_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
        config: Optional[DemoModeConfig] = None,
        memory_context: Any = None,
    ) -> DemoModeResult:
        demo_config = config or DemoModeConfig()
        if not demo_config.enabled:
            return DemoModeResult(
                status="disabled",
                default_invoked=False,
                variant_invoked=False,
                rollback_to_default=False,
                rollback_reason="demo mode disabled",
                risk_level="low",
                output_summary={},
                metadata=_metadata(raw_jd, demo_config, memory_context_used=False),
            )

        if not demo_config.use_skill_backed_variant:
            default_summary = _safe_run_default(default_runner, raw_jd)
            return DemoModeResult(
                status="default_only",
                default_invoked=True,
                variant_invoked=False,
                rollback_to_default=False,
                rollback_reason="skill-backed variant disabled",
                risk_level="low" if default_summary["status"] == "ok" else "high",
                output_summary=default_summary,
                metadata=_metadata(raw_jd, demo_config, memory_context_used=False),
            )

        if variant_runner is None:
            default_summary = _safe_run_default(default_runner, raw_jd)
            return DemoModeResult(
                status="rolled_back" if demo_config.rollback_on_variant_failure else "failed",
                default_invoked=True,
                variant_invoked=False,
                rollback_to_default=bool(demo_config.rollback_on_variant_failure),
                rollback_reason="variant runner missing",
                risk_level="high",
                output_summary=default_summary if demo_config.rollback_on_variant_failure else _failed_summary("ValueError"),
                metadata=_metadata(raw_jd, demo_config, memory_context_used=False),
            )

        if demo_config.require_ab_smoke_pass:
            return self.run_with_ab_smoke(
                raw_jd,
                default_runner=default_runner,
                variant_runner=variant_runner,
                ab_harness=ABSmokeHarness(),
                config=demo_config,
                memory_context=memory_context,
            )

        variant_summary, error_type = _safe_run_variant(
            variant_runner,
            raw_jd,
            allow_memory_context=demo_config.allow_memory_context,
            memory_context=memory_context,
        )
        if variant_summary["status"] == "ok" and not variant_summary["error_type"]:
            return DemoModeResult(
                status="variant_used",
                default_invoked=False,
                variant_invoked=True,
                rollback_to_default=False,
                rollback_reason="variant completed",
                risk_level=variant_summary.get("risk_level") or "low",
                output_summary=variant_summary,
                metadata=_metadata(
                    raw_jd,
                    demo_config,
                    memory_context_used=bool(demo_config.allow_memory_context and memory_context is not None),
                ),
            )

        if demo_config.rollback_on_variant_failure:
            default_summary = _safe_run_default(default_runner, raw_jd)
            return DemoModeResult(
                status="rolled_back",
                default_invoked=True,
                variant_invoked=True,
                rollback_to_default=True,
                rollback_reason="variant failed",
                risk_level="high",
                output_summary=default_summary,
                metadata=_metadata(raw_jd, demo_config, memory_context_used=False),
            )
        return DemoModeResult(
            status="failed",
            default_invoked=False,
            variant_invoked=True,
            rollback_to_default=False,
            rollback_reason="variant failed and rollback disabled",
            risk_level="high",
            output_summary=variant_summary or _failed_summary(error_type),
            metadata=_metadata(raw_jd, demo_config, memory_context_used=False),
        )

    def run_with_ab_smoke(
        self,
        raw_jd: str,
        default_runner: Callable[[str], Mapping[str, Any]],
        variant_runner: Callable[..., Mapping[str, Any]],
        ab_harness: ABSmokeHarness,
        config: DemoModeConfig,
        memory_context: Any = None,
    ) -> DemoModeResult:
        case = ABSmokeCase(
            case_id="limited_demo_ab_smoke",
            raw_jd=raw_jd,
            enable_variant=bool(config.use_skill_backed_variant),
            enable_memory_context=bool(config.allow_memory_context and memory_context is not None),
            metadata={"demo_mode": True},
        )
        ab_result = ab_harness.run_case(
            case,
            default_runner=default_runner,
            variant_runner=variant_runner,
        )
        ab_summary = _ab_summary(ab_result)
        if ab_result.rollback_to_default:
            return DemoModeResult(
                status="rolled_back",
                default_invoked=True,
                variant_invoked=True,
                rollback_to_default=True,
                rollback_reason=ab_result.rollback_reason,
                risk_level=ab_result.risk_level,
                output_summary=ab_result.default_summary,
                ab_smoke_summary=ab_summary,
                metadata=_metadata(raw_jd, config, memory_context_used=False),
            )

        variant_summary, error_type = _safe_run_variant(
            variant_runner,
            raw_jd,
            allow_memory_context=config.allow_memory_context,
            memory_context=memory_context,
        )
        if variant_summary["status"] == "ok" and not variant_summary["error_type"]:
            return DemoModeResult(
                status="variant_used",
                default_invoked=True,
                variant_invoked=True,
                rollback_to_default=False,
                rollback_reason="ab smoke passed",
                risk_level=ab_result.risk_level,
                output_summary=variant_summary,
                ab_smoke_summary=ab_summary,
                metadata=_metadata(
                    raw_jd,
                    config,
                    memory_context_used=bool(config.allow_memory_context and memory_context is not None),
                ),
            )

        if config.rollback_on_variant_failure:
            return DemoModeResult(
                status="rolled_back",
                default_invoked=True,
                variant_invoked=True,
                rollback_to_default=True,
                rollback_reason="variant failed after ab smoke",
                risk_level="high",
                output_summary=ab_result.default_summary,
                ab_smoke_summary=ab_summary,
                metadata=_metadata(raw_jd, config, memory_context_used=False),
            )
        return DemoModeResult(
            status="failed",
            default_invoked=True,
            variant_invoked=True,
            rollback_to_default=False,
            rollback_reason="variant failed after ab smoke and rollback disabled",
            risk_level="high",
            output_summary=variant_summary or _failed_summary(error_type),
            ab_smoke_summary=ab_summary,
            metadata=_metadata(raw_jd, config, memory_context_used=False),
        )


def _safe_run_default(runner, raw_jd: str) -> Dict[str, Any]:
    try:
        return _normalize_output(runner(raw_jd))
    except Exception as exc:
        return _failed_summary(type(exc).__name__)


def _safe_run_variant(
    runner,
    raw_jd: str,
    *,
    allow_memory_context: bool,
    memory_context: Any,
):
    try:
        try:
            output = runner(
                raw_jd,
                memory_context=memory_context if allow_memory_context else None,
                metadata={"summary_only": True},
            )
        except TypeError:
            output = runner(raw_jd)
        return _normalize_output(output), ""
    except Exception as exc:
        return _failed_summary(type(exc).__name__), type(exc).__name__


def _normalize_output(output: Any) -> Dict[str, Any]:
    data = dict(output or {}) if isinstance(output, Mapping) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    output_keys = data.get("output_keys")
    if output_keys is None:
        output_keys = [key for key in data.keys() if key != "metadata"]
    return {
        "status": str(data.get("status") or "unknown"),
        "candidate_count": _safe_int(data.get("candidate_count")),
        "report_count": _safe_int(data.get("report_count")),
        "top_score_present": bool(data.get("top_score_present", data.get("score_present", False))),
        "error_type": str(data.get("error_type") or ""),
        "risk_level": str(data.get("risk_level") or ""),
        "output_keys": sorted(str(key) for key in output_keys),
        "metadata": {
            "keys": sorted(str(key) for key in metadata.keys()),
            "summary_only": True,
        },
    }


def _failed_summary(error_type: str) -> Dict[str, Any]:
    return {
        "status": "failed",
        "candidate_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "error_type": str(error_type or "Error"),
        "risk_level": "high",
        "output_keys": ["error_type", "status"],
        "metadata": {"keys": [], "summary_only": True},
    }


def _metadata(raw_jd: str, config: DemoModeConfig, memory_context_used: bool) -> Dict[str, Any]:
    return {
        "mode": "limited_production_demo_contract",
        "raw_jd_length": len(raw_jd),
        "enabled": bool(config.enabled),
        "use_skill_backed_variant": bool(config.use_skill_backed_variant),
        "allow_memory_context": bool(config.allow_memory_context),
        "memory_context_used": bool(memory_context_used),
        "default_production_graph_replaced": False,
        "summary_only": bool(config.summary_only),
        "metadata_keys": sorted(str(key) for key in config.metadata.keys()),
    }


def _ab_summary(result: ABSmokeResult) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "default_status": result.default_status,
        "variant_status": result.variant_status,
        "rollback_to_default": result.rollback_to_default,
        "rollback_reason": result.rollback_reason,
        "risk_level": result.risk_level,
        "summary_only": True,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
