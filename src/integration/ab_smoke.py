from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


RISK_LEVELS = {"low", "medium", "high"}


@dataclass
class ABSmokeCase:
    case_id: str
    raw_jd: str
    enable_variant: bool = False
    enable_memory_context: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_jd"] = _redact_text(self.raw_jd)
        data["raw_jd_length"] = len(self.raw_jd)
        return data


@dataclass
class ABSmokeResult:
    case_id: str
    default_status: str
    variant_status: str
    default_summary: Dict[str, Any]
    variant_summary: Dict[str, Any]
    comparison_summary: Dict[str, Any]
    rollback_to_default: bool
    rollback_reason: str
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ABSmokeReport:
    total_cases: int
    default_success_count: int
    variant_success_count: int
    rollback_count: int
    high_risk_count: int
    results: List[ABSmokeResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "default_success_count": self.default_success_count,
            "variant_success_count": self.variant_success_count,
            "rollback_count": self.rollback_count,
            "high_risk_count": self.high_risk_count,
            "results": [result.to_dict() for result in self.results],
            "metadata": dict(self.metadata),
        }


class ABSmokeHarness:
    """Run dependency-injected default/variant summaries and decide rollback."""

    def run_case(
        self,
        case: ABSmokeCase,
        default_runner: Callable[[str], Mapping[str, Any]],
        variant_runner: Callable[..., Mapping[str, Any]],
    ) -> ABSmokeResult:
        default_summary, default_error = self._run_default(case, default_runner)
        if case.enable_variant:
            variant_summary, variant_error = self._run_variant(case, variant_runner)
        else:
            variant_summary = _normalize_summary({"status": "skipped", "metadata": {}})
            variant_error = ""

        rollback_to_default, rollback_reason, risk_level, comparison = self.decide_rollback(
            default_summary,
            variant_summary,
            errors={
                "default": default_error,
                "variant": variant_error,
            },
        )
        return ABSmokeResult(
            case_id=case.case_id,
            default_status=default_summary["status"],
            variant_status=variant_summary["status"],
            default_summary=default_summary,
            variant_summary=variant_summary,
            comparison_summary=comparison,
            rollback_to_default=rollback_to_default,
            rollback_reason=rollback_reason,
            risk_level=risk_level,
            metadata={
                "mode": "summary_only_ab_smoke",
                "raw_jd_length": len(case.raw_jd),
                "variant_enabled": bool(case.enable_variant),
                "memory_context_requested": bool(case.enable_memory_context),
                "default_production_graph_replaced": False,
                "summary_only": True,
                "metadata_keys": sorted(str(key) for key in case.metadata.keys()),
            },
        )

    def run_cases(
        self,
        cases: Iterable[ABSmokeCase],
        default_runner: Callable[[str], Mapping[str, Any]],
        variant_runner: Callable[..., Mapping[str, Any]],
    ) -> ABSmokeReport:
        results = [
            self.run_case(case, default_runner=default_runner, variant_runner=variant_runner)
            for case in cases
        ]
        return ABSmokeReport(
            total_cases=len(results),
            default_success_count=sum(result.default_status == "ok" for result in results),
            variant_success_count=sum(result.variant_status == "ok" for result in results),
            rollback_count=sum(result.rollback_to_default for result in results),
            high_risk_count=sum(result.risk_level == "high" for result in results),
            results=results,
            metadata={
                "mode": "summary_only_ab_smoke_report",
                "default_production_graph_replaced": False,
                "summary_only": True,
            },
        )

    def decide_rollback(
        self,
        default_summary: Mapping[str, Any],
        variant_summary: Mapping[str, Any],
        errors: Optional[Mapping[str, str]] = None,
    ):
        default_data = _normalize_summary(default_summary)
        variant_data = _normalize_summary(variant_summary)
        error_map = dict(errors or {})
        comparison = _comparison_summary(default_data, variant_data)

        if variant_data["status"] != "ok":
            return True, "variant status is not ok", "high", comparison
        if variant_data.get("error_type"):
            return True, "variant reported an error type", "high", comparison
        if error_map.get("variant"):
            return True, "variant runner failed", "high", comparison
        if variant_data.get("risk_level") == "high":
            return True, "variant reported high risk", "high", comparison
        if (
            variant_data["candidate_count"] == 0
            and default_data["candidate_count"] > 0
        ):
            return True, "variant returned no candidates while default did", "high", comparison
        if (
            variant_data["report_count"] == 0
            and default_data["report_count"] > 0
        ):
            return True, "variant returned no reports while default did", "high", comparison
        if comparison["critical_mismatch"]:
            return True, "critical output summary mismatch", "high", comparison
        if comparison["count_difference"]:
            return False, "summary counts differ and require review", "medium", comparison
        return False, "default and variant summaries are shape-aligned", "low", comparison

    @staticmethod
    def _run_default(case: ABSmokeCase, runner):
        try:
            return _normalize_summary(runner(case.raw_jd)), ""
        except Exception as exc:
            return _normalize_summary({"status": "failed", "error_type": type(exc).__name__}), type(exc).__name__

    @staticmethod
    def _run_variant(case: ABSmokeCase, runner):
        try:
            output = runner(
                case.raw_jd,
                enable_memory_context=case.enable_memory_context,
                metadata={
                    "case_id": case.case_id,
                    "summary_only": True,
                },
            )
            summary = _normalize_summary(output)
            summary["metadata"]["memory_context_requested"] = bool(case.enable_memory_context)
            return summary, ""
        except TypeError:
            try:
                output = runner(case.raw_jd)
                summary = _normalize_summary(output)
                summary["metadata"]["memory_context_requested"] = bool(case.enable_memory_context)
                return summary, ""
            except Exception as exc:
                return _normalize_summary({"status": "failed", "error_type": type(exc).__name__}), type(exc).__name__
        except Exception as exc:
            return _normalize_summary({"status": "failed", "error_type": type(exc).__name__}), type(exc).__name__


def _normalize_summary(summary: Any) -> Dict[str, Any]:
    data = dict(summary or {}) if isinstance(summary, Mapping) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    output_keys = data.get("output_keys")
    if output_keys is None:
        output_keys = [key for key in data.keys() if key not in {"metadata"}]
    return {
        "status": str(data.get("status") or "unknown"),
        "candidate_count": _safe_int(data.get("candidate_count")),
        "report_count": _safe_int(data.get("report_count")),
        "top_score_present": bool(data.get("top_score_present", data.get("score_present", False))),
        "error_type": str(data.get("error_type") or ""),
        "output_keys": sorted(str(key) for key in output_keys),
        "risk_level": str(data.get("risk_level") or ""),
        "metadata": _safe_metadata(metadata),
    }


def _comparison_summary(
    default_summary: Mapping[str, Any],
    variant_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_delta = variant_summary["candidate_count"] - default_summary["candidate_count"]
    report_delta = variant_summary["report_count"] - default_summary["report_count"]
    count_difference = bool(candidate_delta or report_delta)
    critical_mismatch = bool(
        variant_summary["status"] == "ok"
        and default_summary["status"] == "ok"
        and variant_summary["report_count"] == 0
        and default_summary["report_count"] > 0
    )
    return {
        "default_candidate_count": default_summary["candidate_count"],
        "variant_candidate_count": variant_summary["candidate_count"],
        "candidate_count_delta": candidate_delta,
        "default_report_count": default_summary["report_count"],
        "variant_report_count": variant_summary["report_count"],
        "report_count_delta": report_delta,
        "default_top_score_present": bool(default_summary.get("top_score_present")),
        "variant_top_score_present": bool(variant_summary.get("top_score_present")),
        "count_difference": count_difference,
        "critical_mismatch": critical_mismatch,
        "summary_only": True,
    }


def _safe_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "keys": sorted(str(key) for key in metadata.keys()),
        "summary_only": True,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _redact_text(text: str) -> str:
    return "<present; redacted>" if text else ""
