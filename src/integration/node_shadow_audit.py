import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from src.integration.node_shadow import SingleNodeShadowCompareResult


@dataclass
class NodeShadowCompareAuditReport:
    total_cases: int
    match_count: int
    warning_count: int
    mismatch_count: int
    skipped_count: int
    high_risk_count: int
    node_types: List[str] = field(default_factory=list)
    case_summaries: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "match_count": self.match_count,
            "warning_count": self.warning_count,
            "mismatch_count": self.mismatch_count,
            "skipped_count": self.skipped_count,
            "high_risk_count": self.high_risk_count,
            "node_types": list(self.node_types),
            "case_summaries": [dict(summary) for summary in self.case_summaries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeShadowCompareAuditReport":
        return cls(
            total_cases=int(data.get("total_cases", 0)),
            match_count=int(data.get("match_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            mismatch_count=int(data.get("mismatch_count", 0)),
            skipped_count=int(data.get("skipped_count", 0)),
            high_risk_count=int(data.get("high_risk_count", 0)),
            node_types=[str(value) for value in data.get("node_types", [])],
            case_summaries=[dict(value) for value in data.get("case_summaries", [])],
            metadata=dict(data.get("metadata") or {}),
        )


class NodeShadowCompareAuditor:
    """Project safe node comparison results into a read-only audit summary."""

    def build_report(
        self,
        results: Iterable[SingleNodeShadowCompareResult],
    ) -> NodeShadowCompareAuditReport:
        collected = list(results)
        summaries = [_case_summary(result) for result in collected]
        return NodeShadowCompareAuditReport(
            total_cases=len(collected),
            match_count=_count_status(collected, "match"),
            warning_count=_count_status(collected, "warning"),
            mismatch_count=_count_status(collected, "mismatch"),
            skipped_count=_count_status(collected, "skipped"),
            high_risk_count=sum(result.decision.risk_level == "high" for result in collected),
            node_types=sorted({result.node_type for result in collected}),
            case_summaries=summaries,
            metadata={
                "mode": "summary_only_node_shadow_compare_audit",
                "real_production_graph_invoked": False,
                "real_production_node_invoked": False,
                "input_output_payloads_included": False,
                "summary_only": True,
            },
        )


class NodeShadowCompareAuditExporter:
    @staticmethod
    def export_json(report: NodeShadowCompareAuditReport) -> str:
        return json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True)

    @staticmethod
    def export_text(report: NodeShadowCompareAuditReport) -> str:
        lines = [
            "Node Shadow Compare Audit Report",
            (
                f"total_cases={report.total_cases} match={report.match_count} "
                f"warning={report.warning_count} mismatch={report.mismatch_count} "
                f"skipped={report.skipped_count} high_risk={report.high_risk_count}"
            ),
            f"node_types={','.join(report.node_types) or '-'}",
        ]
        for summary in report.case_summaries:
            mismatches = ",".join(summary["mismatched_fields"]) or "-"
            missing = ",".join(summary["missing_fields"]) or "-"
            lines.append(
                f"case_id={summary['case_id']} node_type={summary['node_type']} "
                f"status={summary['decision_status']} risk={summary['risk_level']} "
                f"mismatched_fields={mismatches} missing_fields={missing}"
            )
        return "\n".join(lines)


def _case_summary(result: SingleNodeShadowCompareResult) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "node_type": result.node_type,
        "decision_status": result.decision.status,
        "risk_level": result.decision.risk_level,
        "mismatched_fields": list(result.parity_report.mismatched_fields),
        "missing_fields": list(result.parity_report.missing_fields),
    }


def _count_status(results: List[SingleNodeShadowCompareResult], status: str) -> int:
    return sum(result.decision.status == status for result in results)
