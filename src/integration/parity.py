import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.integration.compatibility import ProductionStateAdapter


PREVIEW_ONLY_FIELDS = [
    "memory_context",
    "reflection_metadata",
    "closed_loop_memory_preview",
]


@dataclass
class ProductionShadowParityFixture:
    fixture_id: str
    raw_jd: str
    production_state: Dict[str, Any]
    shadow_result: Dict[str, Any]
    expected_alignment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProductionShadowParityFixture":
        return cls(
            fixture_id=str(data["fixture_id"]),
            raw_jd=str(data["raw_jd"]),
            production_state=dict(data.get("production_state") or {}),
            shadow_result=dict(data.get("shadow_result") or {}),
            expected_alignment=dict(data.get("expected_alignment") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ProductionShadowParityReport:
    fixture_id: str
    passed: bool
    aligned_fields: List[str] = field(default_factory=list)
    mismatched_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    preview_only_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionShadowParityBatchReport:
    total_fixtures: int
    passed_fixtures: int
    failed_fixtures: int
    reports: List[ProductionShadowParityReport] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fixtures": self.total_fixtures,
            "passed_fixtures": self.passed_fixtures,
            "failed_fixtures": self.failed_fixtures,
            "reports": [report.to_dict() for report in self.reports],
            "metadata": dict(self.metadata),
        }


class ProductionShadowParityComparator:
    """Compare deterministic state/result fixtures without executing either workflow."""

    def compare(self, fixture: ProductionShadowParityFixture) -> ProductionShadowParityReport:
        production = fixture.production_state
        shadow = fixture.shadow_result
        expected = fixture.expected_alignment
        aligned: List[str] = []
        mismatched: List[str] = []
        missing: List[str] = []
        warnings: List[str] = []

        if _enabled(expected, "raw_jd"):
            production_raw = ProductionStateAdapter.production_state_to_shadow_input(production)["raw_jd"]
            if not production_raw:
                missing.append("production.raw_jd/messages[-1].content")
            elif production_raw == fixture.raw_jd:
                aligned.append("raw_jd")
            else:
                mismatched.append("raw_jd")

        if _enabled(expected, "query"):
            production_query = _production_query(production)
            shadow_query = _shadow_query(shadow)
            if production_query or shadow_query:
                if not production_query:
                    missing.append("production.extracted_jd.search_query")
                elif not shadow_query:
                    missing.append("shadow_result.query/job_requirement.metadata.search_query")
                elif production_query == shadow_query:
                    aligned.append("query")
                else:
                    mismatched.append("query")

        if _enabled(expected, "job_requirement"):
            self._compare_job_requirement(production, shadow, expected, aligned, mismatched, missing)

        if _enabled(expected, "retrieved_candidates"):
            _compare_id_list(
                "retrieved_candidates/candidate_pool",
                production.get("candidate_pool"),
                shadow.get("retrieved_candidates"),
                aligned,
                mismatched,
                missing,
            )

        if _enabled(expected, "match_reports"):
            _compare_id_list(
                "match_reports/final_reports",
                production.get("final_reports"),
                shadow.get("match_reports"),
                aligned,
                mismatched,
                missing,
            )
            if expected.get("compare_exact_scores"):
                if _score_list(production.get("final_reports")) == _score_list(shadow.get("match_reports")):
                    aligned.append("match_reports.exact_scores")
                else:
                    mismatched.append("match_reports.exact_scores")
            else:
                warnings.append("match report scores are excluded from parity unless explicitly requested")

        if _enabled(expected, "refined_query") and (
            "refined_query" in shadow or "refined_query" in production
        ):
            production_refined = production.get("refined_query")
            shadow_refined = shadow.get("refined_query")
            if production_refined == shadow_refined:
                aligned.append("refined_query")
            else:
                mismatched.append("refined_query")

        preview_only = _find_preview_only_fields(production, shadow, fixture.metadata)
        passed = not missing and not mismatched
        return ProductionShadowParityReport(
            fixture_id=fixture.fixture_id,
            passed=passed,
            aligned_fields=_dedupe(aligned),
            mismatched_fields=_dedupe(mismatched),
            missing_fields=_dedupe(missing),
            preview_only_fields=preview_only,
            warnings=_dedupe(warnings),
            metadata={
                "mode": "deterministic_fixture_parity",
                "preview_only": True,
                "real_production_graph_invoked": False,
                "exact_scores_compared": bool(expected.get("compare_exact_scores")),
                "summary_only": True,
            },
        )

    def compare_many(
        self,
        fixtures: Iterable[ProductionShadowParityFixture],
    ) -> ProductionShadowParityBatchReport:
        reports = [self.compare(fixture) for fixture in fixtures]
        passed = sum(report.passed for report in reports)
        return ProductionShadowParityBatchReport(
            total_fixtures=len(reports),
            passed_fixtures=passed,
            failed_fixtures=len(reports) - passed,
            reports=reports,
            metadata={
                "mode": "deterministic_fixture_parity_batch",
                "real_production_graph_invoked": False,
                "summary_only": True,
            },
        )

    @staticmethod
    def _compare_job_requirement(
        production: Mapping[str, Any],
        shadow: Mapping[str, Any],
        expected: Mapping[str, Any],
        aligned: List[str],
        mismatched: List[str],
        missing: List[str],
    ) -> None:
        extracted = production.get("extracted_jd")
        requirement = shadow.get("job_requirement")
        if not isinstance(extracted, Mapping):
            missing.append("production.extracted_jd")
            return
        if not isinstance(requirement, Mapping):
            missing.append("shadow_result.job_requirement")
            return
        keys = list(expected.get("job_requirement_keys") or ["title", "required_skills"])
        compared = False
        for key in keys:
            if key not in extracted or key not in requirement:
                missing.append(f"job_requirement/extracted_jd.{key}")
                continue
            compared = True
            field_name = f"job_requirement/extracted_jd.{key}"
            if extracted[key] == requirement[key]:
                aligned.append(field_name)
            else:
                mismatched.append(field_name)
        if not keys and not compared:
            aligned.append("job_requirement/extracted_jd.shape")


def load_production_shadow_parity_fixtures(
    path: Any,
) -> List[ProductionShadowParityFixture]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_fixtures = data.get("fixtures", data) if isinstance(data, Mapping) else data
    if not isinstance(raw_fixtures, list):
        raise ValueError("Parity fixture file must contain a fixture list.")
    return [ProductionShadowParityFixture.from_dict(item) for item in raw_fixtures]


def _enabled(expected: Mapping[str, Any], key: str) -> bool:
    return expected.get(key, True) is not False


def _production_query(production: Mapping[str, Any]) -> str:
    extracted = production.get("extracted_jd")
    if isinstance(extracted, Mapping):
        value = extracted.get("search_query")
        return value if isinstance(value, str) else ""
    return ""


def _shadow_query(shadow: Mapping[str, Any]) -> str:
    query = shadow.get("query")
    if isinstance(query, str):
        return query
    requirement = shadow.get("job_requirement")
    if not isinstance(requirement, Mapping):
        return ""
    direct = requirement.get("search_query")
    if isinstance(direct, str):
        return direct
    metadata = requirement.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("search_query"), str):
        return metadata["search_query"]
    return ""


def _compare_id_list(
    field_name: str,
    production_value: Any,
    shadow_value: Any,
    aligned: List[str],
    mismatched: List[str],
    missing: List[str],
) -> None:
    if not isinstance(production_value, list):
        missing.append(f"production.{field_name.split('/')[-1]}")
        return
    if not isinstance(shadow_value, list):
        missing.append(f"shadow_result.{field_name.split('/')[0]}")
        return
    if len(production_value) != len(shadow_value):
        mismatched.append(f"{field_name}.count")
        return
    production_ids = _ids(production_value)
    shadow_ids = _ids(shadow_value)
    if production_ids and shadow_ids:
        if production_ids == shadow_ids:
            aligned.append(f"{field_name}.ids")
        else:
            mismatched.append(f"{field_name}.ids")
        return
    aligned.append(f"{field_name}.count")


def _ids(items: List[Any]) -> List[str]:
    values: List[str] = []
    for item in items:
        if not isinstance(item, Mapping) or "candidate_id" not in item:
            return []
        values.append(str(item["candidate_id"]))
    return values


def _score_list(value: Any) -> List[Any]:
    if not isinstance(value, list):
        return []
    return [item.get("total_score") if isinstance(item, Mapping) else None for item in value]


def _find_preview_only_fields(*sources: Mapping[str, Any]) -> List[str]:
    found: List[str] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        metadata = source.get("metadata")
        for name in PREVIEW_ONLY_FIELDS:
            if name in source or (isinstance(metadata, Mapping) and name in metadata):
                found.append(name)
    return _dedupe(found)


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))
