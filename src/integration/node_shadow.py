from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from src.integration.parity import ProductionShadowParityFixture, ProductionShadowParityReport
from src.integration.shadow_compare import (
    ShadowCompareDecision,
    ShadowCompareObservation,
    ShadowCompareObserver,
)


@dataclass
class SingleNodeShadowCompareCase:
    case_id: str
    node_name: str
    node_type: str
    input_data: Dict[str, Any]
    production_callable: Callable[[Dict[str, Any]], Any]
    shadow_callable: Optional[Callable[[Dict[str, Any]], Any]] = None
    shadow_skill_name: Optional[str] = None
    skill_executor: Any = None
    expected_alignment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleNodeShadowCompareResult:
    case_id: str
    node_name: str
    node_type: str
    production_output_summary: Dict[str, Any]
    shadow_output_summary: Dict[str, Any]
    parity_report: ProductionShadowParityReport
    observation: ShadowCompareObservation
    decision: ShadowCompareDecision
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "node_name": self.node_name,
            "node_type": self.node_type,
            "production_output_summary": dict(self.production_output_summary),
            "shadow_output_summary": dict(self.shadow_output_summary),
            "parity_report": self.parity_report.to_dict(),
            "observation": self.observation.to_dict(),
            "decision": self.decision.to_dict(),
            "success": self.success,
            "metadata": dict(self.metadata),
        }


class SingleNodeShadowCompareHarness:
    """Execute only injected fake node capabilities and emit safe comparison summaries."""

    def __init__(self, observer: Optional[ShadowCompareObserver] = None):
        self.observer = observer or ShadowCompareObserver()

    def run_case(self, case: SingleNodeShadowCompareCase) -> SingleNodeShadowCompareResult:
        production_output, production_error = _invoke_callable(case.production_callable, case.input_data)
        shadow_output, shadow_error = self._run_shadow(case)

        if production_error or shadow_error:
            return self._failed_execution_result(
                case,
                production_output,
                shadow_output,
                production_error,
                shadow_error,
            )

        parity = _node_parity_report(case, production_output, shadow_output)
        observation = _observation_for_case(
            self.observer,
            case,
            production_output,
            shadow_output,
            parity,
        )
        decision = self.observer.decide(observation)
        return SingleNodeShadowCompareResult(
            case_id=case.case_id,
            node_name=case.node_name,
            node_type=case.node_type,
            production_output_summary=_output_summary(case.node_type, production_output),
            shadow_output_summary=_output_summary(case.node_type, shadow_output),
            parity_report=parity,
            observation=observation,
            decision=decision,
            success=decision.status in {"match", "warning"},
            metadata=_result_metadata(case, "completed", "completed"),
        )

    def run_cases(self, cases: Iterable[SingleNodeShadowCompareCase]) -> List[SingleNodeShadowCompareResult]:
        return [self.run_case(case) for case in cases]

    def _run_shadow(self, case: SingleNodeShadowCompareCase):
        if case.shadow_callable is not None:
            return _invoke_callable(case.shadow_callable, case.input_data)
        if case.skill_executor is not None and case.shadow_skill_name:
            try:
                result = case.skill_executor.execute(case.shadow_skill_name, dict(case.input_data))
            except Exception as exc:
                return {}, type(exc).__name__
            return _unwrap_shadow_result(result)
        return {}, "ShadowCallableMissing"

    def _failed_execution_result(
        self,
        case: SingleNodeShadowCompareCase,
        production_output: Mapping[str, Any],
        shadow_output: Mapping[str, Any],
        production_error: str,
        shadow_error: str,
    ) -> SingleNodeShadowCompareResult:
        observation = self.observer.observe(
            observation_id=f"observation_{case.case_id}",
            target_name=case.node_name,
            target_type="node",
            raw_jd="",
            production_snapshot=None if production_error else {},
            shadow_snapshot=None if shadow_error else {},
            metadata=case.metadata,
        )
        decision = self.observer.decide(observation)
        if production_error:
            decision.reason = "Injected production callable failed; comparison was skipped."
            decision.recommended_action = "Review the injected production callable failure before comparison."
        elif shadow_error:
            decision.reason = "Injected shadow callable failed; comparison was skipped."
            decision.recommended_action = "Review the injected shadow callable failure before comparison."
        return SingleNodeShadowCompareResult(
            case_id=case.case_id,
            node_name=case.node_name,
            node_type=case.node_type,
            production_output_summary=_output_summary(case.node_type, production_output),
            shadow_output_summary=_output_summary(case.node_type, shadow_output),
            parity_report=observation.parity_report,
            observation=observation,
            decision=decision,
            success=False,
            metadata=_result_metadata(
                case,
                "failed" if production_error else "completed",
                "failed" if shadow_error else "completed",
                production_error,
                shadow_error,
            ),
        )


def _invoke_callable(callable_value: Callable[[Dict[str, Any]], Any], input_data: Dict[str, Any]):
    try:
        return _as_output_mapping(callable_value(dict(input_data))), ""
    except Exception as exc:
        return {}, type(exc).__name__


def _unwrap_shadow_result(result: Any):
    if hasattr(result, "success"):
        if not result.success:
            return {}, "SkillExecutionFailed"
        return _as_output_mapping(getattr(result, "output", {})), ""
    return _as_output_mapping(result), ""


def _as_output_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "output") and isinstance(value.output, Mapping):
        return dict(value.output)
    raise ValueError("Injected callable output must be a mapping.")


def _node_parity_report(
    case: SingleNodeShadowCompareCase,
    production: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> ProductionShadowParityReport:
    node_type = case.node_type
    if node_type == "refiner":
        return _refiner_parity(case, production, shadow)
    if node_type == "matcher":
        return _matcher_parity(case, production, shadow)
    return _generic_parity(case, production, shadow)


def _refiner_parity(
    case: SingleNodeShadowCompareCase,
    production: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> ProductionShadowParityReport:
    production_query = _production_refined_query(production)
    shadow_query = shadow.get("refined_query")
    missing: List[str] = []
    mismatch: List[str] = []
    aligned: List[str] = []
    if not isinstance(production_query, str) or not production_query:
        missing.append("production.refined_query")
    if not isinstance(shadow_query, str) or not shadow_query:
        missing.append("shadow.refined_query")
    if not missing:
        if production_query == shadow_query:
            aligned.append("refined_query")
        else:
            mismatch.append("refined_query")
    return _report(case, aligned, mismatch, missing)


def _matcher_parity(
    case: SingleNodeShadowCompareCase,
    production: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> ProductionShadowParityReport:
    production_report = _first_report(production, "final_reports", "match_report")
    shadow_report = _first_report(shadow, "match_reports", "match_report")
    missing: List[str] = []
    mismatch: List[str] = []
    aligned: List[str] = []
    if not production_report:
        missing.append("production.match_report")
    if not shadow_report:
        missing.append("shadow.match_report")
    if not missing:
        production_id = production_report.get("candidate_id")
        shadow_id = shadow_report.get("candidate_id")
        if production_id is None or shadow_id is None:
            missing.append("match_report.candidate_id")
        elif production_id == shadow_id:
            aligned.append("match_report.candidate_id")
        else:
            mismatch.append("match_report.candidate_id")

        production_score = _score(production, production_report)
        shadow_score = _score(shadow, shadow_report)
        if production_score is None or shadow_score is None:
            missing.append("match_report.total_score")
        elif case.expected_alignment.get("compare_exact_scores"):
            if production_score == shadow_score:
                aligned.append("match_report.total_score")
            else:
                mismatch.append("match_report.total_score")
        else:
            aligned.append("match_report.score_presence")
    return _report(
        case,
        aligned,
        mismatch,
        missing,
        exact_scores=bool(case.expected_alignment.get("compare_exact_scores")),
    )


def _generic_parity(
    case: SingleNodeShadowCompareCase,
    production: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> ProductionShadowParityReport:
    production_keys = sorted(str(key) for key in production.keys())
    shadow_keys = sorted(str(key) for key in shadow.keys())
    if production_keys == shadow_keys:
        return _report(case, ["output_keys"], [], [])
    return _report(case, [], ["output_keys"], [])


def _report(
    case: SingleNodeShadowCompareCase,
    aligned: List[str],
    mismatch: List[str],
    missing: List[str],
    exact_scores: bool = False,
) -> ProductionShadowParityReport:
    return ProductionShadowParityReport(
        fixture_id=case.case_id,
        passed=not mismatch and not missing,
        aligned_fields=aligned,
        mismatched_fields=mismatch,
        missing_fields=missing,
        metadata={
            "mode": "deterministic_single_node_parity",
            "node_name": case.node_name,
            "node_type": case.node_type,
            "exact_scores_compared": exact_scores,
            "real_production_graph_invoked": False,
            "summary_only": True,
        },
    )


def _observation_for_case(
    observer: ShadowCompareObserver,
    case: SingleNodeShadowCompareCase,
    production_output: Mapping[str, Any],
    shadow_output: Mapping[str, Any],
    parity: ProductionShadowParityReport,
) -> ShadowCompareObservation:
    fixture = ProductionShadowParityFixture(
        fixture_id=case.case_id,
        raw_jd="<single-node-input>",
        production_state=_node_snapshot(case.node_type, production_output, True),
        shadow_result=_node_snapshot(case.node_type, shadow_output, False),
        expected_alignment={},
        metadata={},
    )
    observation = observer.observe(
        fixture,
        observation_id=f"observation_{case.case_id}",
        target_name=case.node_name,
        target_type="node",
        metadata=case.metadata,
    )
    observation.parity_report = parity
    observation.metadata["node_type"] = case.node_type
    return observation


def _node_snapshot(node_type: str, output: Mapping[str, Any], production: bool) -> Dict[str, Any]:
    if node_type == "refiner":
        query = _production_refined_query(output) if production else output.get("refined_query")
        return {"refined_query": query}
    if node_type == "matcher":
        report = _first_report(output, "final_reports", "match_report")
        return {"final_reports" if production else "match_reports": [report] if report else []}
    return {"keys": sorted(str(key) for key in output.keys())}


def _output_summary(node_type: str, output: Mapping[str, Any]) -> Dict[str, Any]:
    summary = {
        "keys": sorted(str(key) for key in output.keys()),
        "node_type": node_type,
    }
    if node_type == "refiner":
        value = _production_refined_query(output) or output.get("refined_query")
        summary.update(
            {
                "refined_query_present": isinstance(value, str) and bool(value),
                "refined_query_length": len(value) if isinstance(value, str) else 0,
            }
        )
    if node_type == "matcher":
        report = _first_report(output, "final_reports", "match_report")
        score = _score(output, report)
        summary.update(
            {
                "report_present": bool(report),
                "candidate_id_present": bool(report and report.get("candidate_id")),
                "score_present": score is not None,
                "score_bucket": _score_bucket(score),
            }
        )
    return summary


def _production_refined_query(output: Mapping[str, Any]):
    direct = output.get("refined_query")
    if isinstance(direct, str):
        return direct
    extracted = output.get("extracted_jd")
    return extracted.get("search_query") if isinstance(extracted, Mapping) else None


def _first_report(output: Mapping[str, Any], list_key: str, dict_key: str) -> Dict[str, Any]:
    reports = output.get(list_key)
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        return dict(reports[0])
    report = output.get(dict_key)
    return dict(report) if isinstance(report, Mapping) else {}


def _score(output: Mapping[str, Any], report: Mapping[str, Any]):
    value = output.get("total_score", report.get("total_score"))
    return value if isinstance(value, (int, float)) else None


def _score_bucket(score: Any) -> str:
    if not isinstance(score, (int, float)):
        return "missing"
    if score >= 80:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _result_metadata(
    case: SingleNodeShadowCompareCase,
    production_status: str,
    shadow_status: str,
    production_error: str = "",
    shadow_error: str = "",
) -> Dict[str, Any]:
    return {
        "mode": "deterministic_single_node_shadow_compare",
        "production_execution_status": production_status,
        "shadow_execution_status": shadow_status,
        "production_error_type": production_error,
        "shadow_error_type": shadow_error,
        "metadata_keys": sorted(str(key) for key in case.metadata.keys()),
        "real_production_graph_invoked": False,
        "summary_only": True,
    }
