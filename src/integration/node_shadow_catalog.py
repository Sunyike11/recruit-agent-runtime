import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.integration.node_shadow import (
    SingleNodeShadowCompareCase,
    SingleNodeShadowCompareHarness,
    SingleNodeShadowCompareResult,
)


@dataclass
class NodeShadowCompareFixture:
    """Declarative fake node snapshots for deterministic shadow comparison."""

    case_id: str
    node_name: str
    node_type: str
    input_data: Dict[str, Any]
    production_output: Dict[str, Any]
    shadow_output: Dict[str, Any]
    expected_decision: Optional[str] = None
    compare_exact_scores: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeShadowCompareFixture":
        required = ("case_id", "node_name", "node_type", "production_output", "shadow_output")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Node shadow fixture missing required fields: {', '.join(missing)}")
        return cls(
            case_id=str(data["case_id"]),
            node_name=str(data["node_name"]),
            node_type=str(data["node_type"]),
            input_data=dict(data.get("input_data") or {}),
            production_output=dict(data.get("production_output") or {}),
            shadow_output=dict(data.get("shadow_output") or {}),
            expected_decision=(
                str(data["expected_decision"]) if data.get("expected_decision") is not None else None
            ),
            compare_exact_scores=bool(data.get("compare_exact_scores", False)),
            tags=[str(tag) for tag in data.get("tags", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_compare_case(self) -> SingleNodeShadowCompareCase:
        production_output = copy.deepcopy(self.production_output)
        shadow_output = copy.deepcopy(self.shadow_output)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "fixture_source": "static_snapshot",
                "fixture_tags": list(self.tags),
                "expected_decision": self.expected_decision,
            }
        )

        return SingleNodeShadowCompareCase(
            case_id=self.case_id,
            node_name=self.node_name,
            node_type=self.node_type,
            input_data=copy.deepcopy(self.input_data),
            production_callable=lambda input_data: copy.deepcopy(production_output),
            shadow_callable=lambda input_data: copy.deepcopy(shadow_output),
            expected_alignment={"compare_exact_scores": self.compare_exact_scores},
            metadata=metadata,
        )


class NodeShadowCompareFixtureCatalog:
    """Load and select fake single-node compare fixtures without running real nodes."""

    def __init__(self, fixtures: Iterable[NodeShadowCompareFixture]):
        self._fixtures: Dict[str, NodeShadowCompareFixture] = {}
        for fixture in fixtures:
            if fixture.case_id in self._fixtures:
                raise ValueError(f"Duplicate node shadow compare case_id: {fixture.case_id}")
            self._fixtures[fixture.case_id] = fixture

    @classmethod
    def from_dict(cls, data: Any) -> "NodeShadowCompareFixtureCatalog":
        raw_cases = data.get("cases", data) if isinstance(data, Mapping) else data
        if not isinstance(raw_cases, list):
            raise ValueError("Node shadow compare catalog must contain a case list.")
        return cls(NodeShadowCompareFixture.from_dict(item) for item in raw_cases)

    @classmethod
    def from_json_file(cls, path: Any) -> "NodeShadowCompareFixtureCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def list_cases(self) -> List[NodeShadowCompareFixture]:
        return list(self._fixtures.values())

    def get_case(self, case_id: str) -> NodeShadowCompareFixture:
        if case_id not in self._fixtures:
            raise KeyError(f"Unknown node shadow compare case_id: {case_id}")
        return self._fixtures[case_id]

    def filter_by_node_type(self, node_type: str) -> List[NodeShadowCompareFixture]:
        return [fixture for fixture in self.list_cases() if fixture.node_type == node_type]

    def filter_by_tag(self, tag: str) -> List[NodeShadowCompareFixture]:
        return [fixture for fixture in self.list_cases() if tag in fixture.tags]

    def run_cases(
        self,
        harness: Optional[SingleNodeShadowCompareHarness] = None,
    ) -> List[SingleNodeShadowCompareResult]:
        runner = harness or SingleNodeShadowCompareHarness()
        return runner.run_cases(fixture.to_compare_case() for fixture in self.list_cases())
