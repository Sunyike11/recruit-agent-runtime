from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


PRODUCTION_REQUIRED_KEYS = ["messages"]
PRODUCTION_OPTIONAL_KEYS = [
    "extracted_jd",
    "candidate_pool",
    "final_reports",
    "loop_count",
    "refinement_advice",
    "next_action",
    "human_feedback",
]
SHADOW_REQUIRED_KEYS = ["status", "success"]
SHADOW_OPTIONAL_KEYS = [
    "job_requirement",
    "retrieved_candidates",
    "resume_documents",
    "evidence",
    "match_reports",
    "refined_query",
    "metadata",
]


@dataclass
class ProductionStateShape:
    required_keys: List[str] = field(default_factory=lambda: list(PRODUCTION_REQUIRED_KEYS))
    optional_keys: List[str] = field(default_factory=lambda: list(PRODUCTION_OPTIONAL_KEYS))
    node_outputs: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "planner_node": ["extracted_jd", "next_action", "messages"],
            "retriever_node": ["candidate_pool", "next_action", "messages"],
            "matcher_node": [
                "final_reports",
                "next_action",
                "refinement_advice",
                "loop_count",
                "messages",
            ],
            "refiner_node": ["extracted_jd", "messages"],
        }
    )
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "source": "RecruitState",
            "input_location": "messages[-1].content",
            "memory_context_supported": False,
            "summary_only": True,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowWorkflowShape:
    required_keys: List[str] = field(default_factory=lambda: list(SHADOW_REQUIRED_KEYS))
    optional_keys: List[str] = field(default_factory=lambda: list(SHADOW_OPTIONAL_KEYS))
    outputs: Dict[str, str] = field(
        default_factory=lambda: {
            "job_requirement": "structured JD-like output",
            "retrieved_candidates": "candidate profile-like list",
            "match_reports": "match report-like list",
            "refined_query": "optional refined query",
            "memory_context": "optional shadow preview input only",
        }
    )
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "source": "RecruitmentSkillWorkflow",
            "input_location": "raw_jd argument",
            "memory_context_mode": "optional_preview_only",
            "production_graph_used": False,
            "summary_only": True,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionIntegrationCompatibilityReport:
    compatible: bool
    missing_keys: List[str] = field(default_factory=list)
    extra_keys: List[str] = field(default_factory=list)
    incompatible_fields: List[str] = field(default_factory=list)
    migration_risks: List[str] = field(default_factory=list)
    recommended_steps: List[str] = field(default_factory=list)
    rollback_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionStateAdapter:
    """Read-only shape adapter; returned production mappings are previews only."""

    @staticmethod
    def production_state_to_shadow_input(
        production_state: Optional[Mapping[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        state = production_state if isinstance(production_state, Mapping) else {}
        raw_jd = _extract_raw_jd(state)
        query = _extract_query(state)
        return {
            "raw_jd": raw_jd,
            "top_k": top_k,
            "metadata": {
                "integration_mode": "preview_only",
                "source": "production_state_shape",
                "query_available": bool(query),
                "missing_raw_jd": not bool(raw_jd),
                "memory_context_required": False,
            },
        }

    @staticmethod
    def shadow_result_to_production_update(
        shadow_result: Any,
    ) -> Dict[str, Any]:
        result = _as_mapping(shadow_result)
        return {
            "extracted_jd_preview": result.get("job_requirement"),
            "candidate_pool_preview": list(result.get("retrieved_candidates") or []),
            "final_reports_preview": list(result.get("match_reports") or []),
            "refined_query_preview": result.get("refined_query"),
            "preview_metadata": {
                "integration_mode": "preview_only",
                "apply_to_graph": False,
                "memory_context_applied": False,
                "summary_only": True,
            },
        }

    @staticmethod
    def validate_production_state(
        production_state: Optional[Mapping[str, Any]],
    ) -> ProductionIntegrationCompatibilityReport:
        return compare_production_and_shadow_shapes(production_state, None)

    @staticmethod
    def validate_shadow_result(
        shadow_result: Any,
    ) -> ProductionIntegrationCompatibilityReport:
        return compare_production_and_shadow_shapes(
            {"messages": [{"content": "<preview input>"}]},
            shadow_result,
        )


def compare_production_and_shadow_shapes(
    production_state: Optional[Mapping[str, Any]],
    shadow_result: Any = None,
    memory_context_requested: bool = False,
) -> ProductionIntegrationCompatibilityReport:
    state = production_state if isinstance(production_state, Mapping) else {}
    missing_keys = [key for key in PRODUCTION_REQUIRED_KEYS if key not in state]
    if not _extract_raw_jd(state):
        missing_keys.append("messages[-1].content/raw_jd")
    extra_keys = sorted(
        str(key)
        for key in state.keys()
        if key not in set(PRODUCTION_REQUIRED_KEYS + PRODUCTION_OPTIONAL_KEYS)
    )
    incompatible_fields = [
        "candidate_pool expects retrieved text chunks while shadow retrieved_candidates may be candidate profiles",
        "extracted_jd and shadow job_requirement require schema compatibility validation",
        "final_reports and shadow match_reports require parity validation",
    ]
    migration_risks = [
        "node replacement could alter refinement loop and interrupt behavior",
        "real retrieval and matching quality are not covered by shadow fixtures",
        "production checkpoint/resume compatibility is not yet established",
    ]

    if shadow_result is not None:
        result = _as_mapping(shadow_result)
        for key in SHADOW_REQUIRED_KEYS:
            if key not in result:
                missing_keys.append(f"shadow_result.{key}")
        if "match_reports" not in result:
            migration_risks.append("shadow result has no match_reports output for report parity review")
        if "job_requirement" not in result:
            migration_risks.append("shadow result has no job_requirement output for JD schema review")

    if memory_context_requested:
        incompatible_fields.append("memory_context is shadow preview-only and is not a production state field")
        migration_risks.append("production Agent memory consumption is not authorized by Phase6A")

    missing_keys = _dedupe(missing_keys)
    report = ProductionIntegrationCompatibilityReport(
        compatible=not missing_keys,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        incompatible_fields=incompatible_fields,
        migration_risks=_dedupe(migration_risks),
        recommended_steps=build_production_integration_plan(),
        rollback_notes=_rollback_notes(),
        metadata={
            "mode": "read_only_compatibility_analysis",
            "preview_only": True,
            "memory_context_required": False,
            "memory_context_requested": memory_context_requested,
            "production_graph_modified": False,
            "summary_only": True,
        },
    )
    return report


def build_production_integration_plan(
    report: Optional[ProductionIntegrationCompatibilityReport] = None,
) -> List[str]:
    steps = [
        "Freeze current production graph and state contract as rollback baseline",
        "Compare production and shadow fixture outputs through preview-only adapter mappings",
        "Define explicit candidate_pool and match report schema compatibility gates",
        "Validate refinement, interruption, and checkpoint behavior before any node migration",
        "Require explicit opt-in and governance checks before any memory-context experiment",
        "Run side-by-side evaluation before proposing a narrowly scoped production change",
    ]
    if report is not None and report.missing_keys:
        return ["Resolve missing compatibility fields before migration planning"] + steps
    return steps


def validate_safe_migration_boundary(
    production_state: Optional[Mapping[str, Any]] = None,
    shadow_result: Any = None,
    memory_context_requested: bool = False,
) -> ProductionIntegrationCompatibilityReport:
    report = compare_production_and_shadow_shapes(
        production_state or {"messages": [{"content": "<preview input>"}]},
        shadow_result,
        memory_context_requested=memory_context_requested,
    )
    if memory_context_requested:
        report.compatible = False
    report.metadata["safe_boundary"] = (
        "preview_only_no_graph_write_no_agent_memory_consumption"
    )
    return report


def _extract_raw_jd(state: Mapping[str, Any]) -> str:
    raw_jd = state.get("raw_jd")
    if isinstance(raw_jd, str) and raw_jd.strip():
        return raw_jd
    messages = state.get("messages") or []
    if not messages:
        return ""
    latest = messages[-1]
    if isinstance(latest, str):
        return latest
    if isinstance(latest, Mapping):
        content = latest.get("content", "")
    else:
        content = getattr(latest, "content", "")
    return content if isinstance(content, str) else ""


def _extract_query(state: Mapping[str, Any]) -> str:
    extracted = state.get("extracted_jd")
    if not isinstance(extracted, Mapping):
        return ""
    query = extracted.get("search_query", "")
    return query if isinstance(query, str) else ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        mapped = value.to_dict()
        return mapped if isinstance(mapped, Mapping) else {}
    return {}


def _rollback_notes() -> List[str]:
    return [
        "Keep create_recruit_graph() as the default execution path",
        "Keep all compatibility mappings preview-only until explicit migration approval",
        "Disable any future memory-context experiment independently of graph execution",
        "Retain current state/checkpoint contract until parity verification passes",
    ]


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))
