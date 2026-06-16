from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.integration.parity import (
    ProductionShadowParityComparator,
    ProductionShadowParityFixture,
    ProductionShadowParityReport,
)


@dataclass
class ShadowCompareObservation:
    observation_id: str
    target_name: str
    target_type: str
    input_summary: Dict[str, Any]
    production_output_summary: Dict[str, Any]
    shadow_output_summary: Dict[str, Any]
    parity_report: ProductionShadowParityReport
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parity_report"] = self.parity_report.to_dict()
        return data


@dataclass
class ShadowCompareDecision:
    observation_id: str
    status: str
    risk_level: str
    reason: str
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowCompareReport:
    total_observations: int
    match_count: int
    mismatch_count: int
    warning_count: int
    skipped_count: int
    high_risk_count: int
    decisions: List[ShadowCompareDecision] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "match_count": self.match_count,
            "mismatch_count": self.mismatch_count,
            "warning_count": self.warning_count,
            "skipped_count": self.skipped_count,
            "high_risk_count": self.high_risk_count,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "metadata": dict(self.metadata),
        }


class ShadowCompareObserver:
    """Build summary-only comparison observations from supplied fake snapshots."""

    def __init__(self, comparator: Optional[ProductionShadowParityComparator] = None):
        self.comparator = comparator or ProductionShadowParityComparator()

    def observe(
        self,
        fixture: Optional[ProductionShadowParityFixture] = None,
        *,
        observation_id: Optional[str] = None,
        target_name: Optional[str] = None,
        target_type: str = "workflow",
        raw_jd: str = "",
        production_snapshot: Optional[Mapping[str, Any]] = None,
        shadow_snapshot: Optional[Mapping[str, Any]] = None,
        expected_alignment: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ShadowCompareObservation:
        supplied_metadata = dict(metadata or {})
        if fixture is None:
            fixture_id = observation_id or "snapshot_observation"
            missing_snapshot = production_snapshot is None or shadow_snapshot is None
            fixture = ProductionShadowParityFixture(
                fixture_id=fixture_id,
                raw_jd=raw_jd,
                production_state=dict(production_snapshot or {}),
                shadow_result=dict(shadow_snapshot or {}),
                expected_alignment=dict(expected_alignment or {}),
            )
        else:
            fixture_id = fixture.fixture_id
            missing_snapshot = False

        safe_observation_id = observation_id or f"observation_{fixture_id}"
        if missing_snapshot:
            parity_report = _skipped_parity_report(fixture_id, production_snapshot, shadow_snapshot)
        else:
            parity_report = self.comparator.compare(fixture)

        production = fixture.production_state
        shadow = fixture.shadow_result
        return ShadowCompareObservation(
            observation_id=safe_observation_id,
            target_name=target_name or fixture_id,
            target_type=target_type,
            input_summary=_input_summary(fixture.raw_jd, production),
            production_output_summary=_production_summary(production),
            shadow_output_summary=_shadow_summary(shadow),
            parity_report=parity_report,
            metadata={
                "mode": "summary_only_shadow_compare_observation",
                "source_fixture_id": fixture_id,
                "missing_snapshot": missing_snapshot,
                "preview_only": True,
                "real_production_graph_invoked": False,
                "summary_only": True,
                "tag_keys": sorted(str(key) for key in supplied_metadata.keys()),
            },
        )

    def decide(self, observation: ShadowCompareObservation) -> ShadowCompareDecision:
        parity = observation.parity_report
        base_metadata = {
            "target_name": observation.target_name,
            "target_type": observation.target_type,
            "preview_only": True,
            "real_production_graph_invoked": False,
        }
        if observation.metadata.get("missing_snapshot"):
            return ShadowCompareDecision(
                observation_id=observation.observation_id,
                status="skipped",
                risk_level="medium",
                reason="Production or shadow snapshot was not supplied.",
                recommended_action="Supply both summary snapshots before comparison.",
                metadata=base_metadata,
            )
        if not parity.passed:
            finding_count = len(parity.missing_fields) + len(parity.mismatched_fields)
            return ShadowCompareDecision(
                observation_id=observation.observation_id,
                status="mismatch",
                risk_level="high",
                reason=f"Parity reported {finding_count} missing or mismatched critical field(s).",
                recommended_action="Resolve structural mismatches before extending shadow comparison.",
                metadata=base_metadata,
            )
        if parity.warnings or parity.preview_only_fields:
            has_comparison_warning = bool(parity.warnings)
            return ShadowCompareDecision(
                observation_id=observation.observation_id,
                status="warning",
                risk_level="medium" if has_comparison_warning else "low",
                reason="Parity passed with warning or preview-only fields requiring review.",
                recommended_action="Review warning and preview-only signals before any integration experiment.",
                metadata=base_metadata,
            )
        return ShadowCompareDecision(
            observation_id=observation.observation_id,
            status="match",
            risk_level="low",
            reason="Compared summary fields matched without observation warnings.",
            recommended_action="Retain as parity evidence; do not treat it as migration approval.",
            metadata=base_metadata,
        )

    def observe_many(
        self,
        fixtures: Iterable[Any],
        target_type: str = "workflow",
    ) -> ShadowCompareReport:
        observations = [
            item
            if isinstance(item, ShadowCompareObservation)
            else self.observe(item, target_type=target_type)
            for item in fixtures
        ]
        decisions = [self.decide(observation) for observation in observations]
        return ShadowCompareReport(
            total_observations=len(decisions),
            match_count=_count_status(decisions, "match"),
            mismatch_count=_count_status(decisions, "mismatch"),
            warning_count=_count_status(decisions, "warning"),
            skipped_count=_count_status(decisions, "skipped"),
            high_risk_count=sum(decision.risk_level == "high" for decision in decisions),
            decisions=decisions,
            metadata={
                "mode": "summary_only_shadow_compare_report",
                "target_type": target_type,
                "real_production_graph_invoked": False,
                "summary_only": True,
            },
        )


def _skipped_parity_report(
    fixture_id: str,
    production_snapshot: Optional[Mapping[str, Any]],
    shadow_snapshot: Optional[Mapping[str, Any]],
) -> ProductionShadowParityReport:
    missing = []
    if production_snapshot is None:
        missing.append("production_snapshot")
    if shadow_snapshot is None:
        missing.append("shadow_snapshot")
    return ProductionShadowParityReport(
        fixture_id=fixture_id,
        passed=False,
        missing_fields=missing,
        metadata={
            "mode": "snapshot_missing_no_comparison",
            "observation_skipped": True,
            "real_production_graph_invoked": False,
            "summary_only": True,
        },
    )


def _input_summary(raw_jd: str, production: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "raw_jd_present": bool(raw_jd),
        "raw_jd_length": len(raw_jd),
        "production_input_keys": sorted(str(key) for key in production.keys()),
    }


def _production_summary(production: Mapping[str, Any]) -> Dict[str, Any]:
    extracted = production.get("extracted_jd")
    candidates = production.get("candidate_pool")
    reports = production.get("final_reports")
    return {
        "keys": sorted(str(key) for key in production.keys()),
        "extracted_jd_keys": _mapping_keys(extracted),
        "candidate_count": _list_count(candidates),
        "candidate_ids": _candidate_ids(candidates),
        "report_count": _list_count(reports),
        "report_candidate_ids": _candidate_ids(reports),
        "refined_query_present": bool(production.get("refined_query")),
    }


def _shadow_summary(shadow: Mapping[str, Any]) -> Dict[str, Any]:
    requirement = shadow.get("job_requirement")
    candidates = shadow.get("retrieved_candidates")
    reports = shadow.get("match_reports")
    return {
        "keys": sorted(str(key) for key in shadow.keys()),
        "job_requirement_keys": _mapping_keys(requirement),
        "candidate_count": _list_count(candidates),
        "candidate_ids": _candidate_ids(candidates),
        "report_count": _list_count(reports),
        "report_candidate_ids": _candidate_ids(reports),
        "refined_query_present": bool(shadow.get("refined_query")),
        "memory_context_present": "memory_context" in shadow,
    }


def _mapping_keys(value: Any) -> List[str]:
    return sorted(str(key) for key in value.keys()) if isinstance(value, Mapping) else []


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _candidate_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item["candidate_id"])
        for item in value
        if isinstance(item, Mapping) and "candidate_id" in item
    ]


def _count_status(decisions: List[ShadowCompareDecision], status: str) -> int:
    return sum(decision.status == status for decision in decisions)
