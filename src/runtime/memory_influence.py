from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


RISK_LEVELS = {"low", "medium", "high"}
DECISIONS = {"no_effect", "expected_effect", "warning", "regression_risk", "skipped"}


@dataclass
class MemoryInfluenceEvalCase:
    case_id: str
    raw_jd: str
    memory_source: str = "none"
    memory_config: Dict[str, Any] = field(default_factory=dict)
    expected_effect: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw_jd"] = "<present; redacted>" if self.raw_jd else ""
        data["raw_jd_length"] = len(self.raw_jd or "")
        data["metadata"] = _safe_metadata(self.metadata)
        data["memory_config"] = _safe_config(self.memory_config)
        return data


@dataclass
class MemoryInfluenceRunSummary:
    status: str = "unknown"
    runner_used: str = "skill_backed_variant"
    candidate_count: int = 0
    report_count: int = 0
    top_score_present: bool = False
    top_scores: List[float] = field(default_factory=list)
    candidate_ids: List[str] = field(default_factory=list)
    candidate_profile_preview_count: int = 0
    memory_context_provided: bool = False
    memory_context_eligible_count: int = 0
    memory_context_rendered_char_count: int = 0
    output_keys: List[str] = field(default_factory=list)
    error_type: str = ""
    summary_only: bool = True

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> "MemoryInfluenceRunSummary":
        data = dict(summary or {})
        return cls(
            status=str(data.get("status") or "unknown"),
            runner_used=str(data.get("runner_used") or data.get("metadata", {}).get("runner_type") or "skill_backed_variant"),
            candidate_count=_safe_int(data.get("candidate_count")),
            report_count=_safe_int(data.get("report_count")),
            top_score_present=bool(data.get("top_score_present", False)),
            top_scores=_safe_float_list(data.get("top_scores") or data.get("score_summary") or []),
            candidate_ids=_safe_string_list(data.get("candidate_ids") or data.get("candidate_id_list") or []),
            candidate_profile_preview_count=_safe_int(data.get("candidate_profile_preview_count")),
            memory_context_provided=bool(data.get("memory_context_provided", False)),
            memory_context_eligible_count=_safe_int(data.get("memory_context_eligible_count")),
            memory_context_rendered_char_count=_safe_int(data.get("memory_context_rendered_char_count")),
            output_keys=sorted(set(_safe_string_list(data.get("output_keys") or list(data.keys())))),
            error_type=str(data.get("error_type") or ""),
            summary_only=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryInfluenceDelta:
    candidate_count_changed: bool = False
    report_count_changed: bool = False
    top_score_changed: bool = False
    ranking_changed: bool = False
    candidate_ids_changed: bool = False
    memory_context_used: bool = False
    risk_level: str = "low"
    decision: str = "no_effect"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryInfluenceEvalResult:
    case_id: str
    no_memory_summary: MemoryInfluenceRunSummary
    with_memory_summary: MemoryInfluenceRunSummary
    delta: MemoryInfluenceDelta
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "no_memory_summary": self.no_memory_summary.to_dict(),
            "with_memory_summary": self.with_memory_summary.to_dict(),
            "delta": self.delta.to_dict(),
            "passed": self.passed,
            "metadata": _safe_metadata(self.metadata),
        }


class MemoryInfluenceEvaluator:
    """Compare no-memory and with-memory variant summaries without judging quality."""

    def run_case(
        self,
        case: MemoryInfluenceEvalCase,
        no_memory_runner: Callable[[str], Mapping[str, Any]],
        with_memory_runner: Callable[..., Mapping[str, Any]],
        memory_context: Any = None,
    ) -> MemoryInfluenceEvalResult:
        no_memory_summary = self._run_no_memory(case, no_memory_runner)
        with_memory_summary = self._run_with_memory(case, with_memory_runner, memory_context)
        delta = self.compare_summaries(
            no_memory_summary,
            with_memory_summary,
            expected_effect=case.expected_effect,
        )
        return MemoryInfluenceEvalResult(
            case_id=case.case_id,
            no_memory_summary=no_memory_summary,
            with_memory_summary=with_memory_summary,
            delta=delta,
            passed=delta.decision not in {"regression_risk"},
            metadata={
                "mode": "summary_only_memory_influence_eval",
                "raw_jd_length": len(case.raw_jd or ""),
                "memory_source": str(case.memory_source or "none"),
                "expected_effect": str(case.expected_effect or ""),
                "summary_only": True,
                "metadata_keys": sorted(str(key) for key in case.metadata.keys()),
            },
        )

    def compare_summaries(
        self,
        no_memory_summary: Mapping[str, Any] | MemoryInfluenceRunSummary,
        with_memory_summary: Mapping[str, Any] | MemoryInfluenceRunSummary,
        *,
        expected_effect: Optional[str] = None,
    ) -> MemoryInfluenceDelta:
        no_mem = _ensure_run_summary(no_memory_summary)
        with_mem = _ensure_run_summary(with_memory_summary)
        notes: List[str] = []

        if not with_mem.memory_context_provided:
            return MemoryInfluenceDelta(
                memory_context_used=False,
                risk_level="medium",
                decision="skipped",
                notes=["with-memory run did not receive memory context"],
            )
        if with_mem.status != "ok" and no_mem.status == "ok":
            return MemoryInfluenceDelta(
                memory_context_used=True,
                risk_level="high",
                decision="regression_risk",
                notes=["with-memory run failed while no-memory run succeeded"],
            )
        if with_mem.report_count == 0 and no_mem.report_count > 0:
            return MemoryInfluenceDelta(
                report_count_changed=True,
                memory_context_used=True,
                risk_level="high",
                decision="regression_risk",
                notes=["with-memory run returned no reports while no-memory run had reports"],
            )

        candidate_count_changed = no_mem.candidate_count != with_mem.candidate_count
        report_count_changed = no_mem.report_count != with_mem.report_count
        candidate_ids_changed = no_mem.candidate_ids != with_mem.candidate_ids
        ranking_changed = _ranking_changed(no_mem.candidate_ids, with_mem.candidate_ids)
        top_score_changed = _top_score_changed(no_mem.top_scores, with_mem.top_scores)

        if candidate_count_changed:
            notes.append("candidate count changed")
        if report_count_changed:
            notes.append("report count changed")
        if candidate_ids_changed:
            notes.append("candidate ids changed")
        if ranking_changed:
            notes.append("candidate ranking changed")
        if top_score_changed:
            notes.append("top score changed")

        any_change = any(
            [
                candidate_count_changed,
                report_count_changed,
                candidate_ids_changed,
                ranking_changed,
                top_score_changed,
            ]
        )
        if not any_change:
            return MemoryInfluenceDelta(
                memory_context_used=True,
                risk_level="low",
                decision="no_effect",
                notes=["summary fields are unchanged"],
            )

        expected = str(expected_effect or "")
        if expected and _expected_effect_matches(expected, notes):
            decision = "expected_effect"
        else:
            decision = "warning"
        return MemoryInfluenceDelta(
            candidate_count_changed=candidate_count_changed,
            report_count_changed=report_count_changed,
            top_score_changed=top_score_changed,
            ranking_changed=ranking_changed,
            candidate_ids_changed=candidate_ids_changed,
            memory_context_used=True,
            risk_level="medium",
            decision=decision,
            notes=notes,
        )

    @staticmethod
    def _run_no_memory(case: MemoryInfluenceEvalCase, runner) -> MemoryInfluenceRunSummary:
        try:
            return MemoryInfluenceRunSummary.from_summary(runner(case.raw_jd))
        except Exception as exc:
            return MemoryInfluenceRunSummary.from_summary(
                {"status": "failed", "error_type": type(exc).__name__}
            )

    @staticmethod
    def _run_with_memory(case: MemoryInfluenceEvalCase, runner, memory_context: Any) -> MemoryInfluenceRunSummary:
        try:
            return MemoryInfluenceRunSummary.from_summary(
                runner(
                    case.raw_jd,
                    memory_context=memory_context,
                    metadata={"case_id": case.case_id, "summary_only": True},
                )
            )
        except TypeError:
            try:
                return MemoryInfluenceRunSummary.from_summary(runner(case.raw_jd))
            except Exception as exc:
                return MemoryInfluenceRunSummary.from_summary(
                    {"status": "failed", "error_type": type(exc).__name__}
                )
        except Exception as exc:
            return MemoryInfluenceRunSummary.from_summary(
                {"status": "failed", "error_type": type(exc).__name__}
            )


def _ensure_run_summary(value: Mapping[str, Any] | MemoryInfluenceRunSummary) -> MemoryInfluenceRunSummary:
    if isinstance(value, MemoryInfluenceRunSummary):
        return value
    return MemoryInfluenceRunSummary.from_summary(value)


def _ranking_changed(left: Sequence[str], right: Sequence[str]) -> bool:
    if not left or not right:
        return False
    return list(left) != list(right) and set(left) == set(right)


def _top_score_changed(left: Sequence[float], right: Sequence[float]) -> bool:
    if not left or not right:
        return False
    return abs(float(left[0]) - float(right[0])) > 1e-9


def _expected_effect_matches(expected: str, notes: Sequence[str]) -> bool:
    normalized = expected.lower().replace("_", " ")
    return any(normalized in note for note in notes)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float_list(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    output = []
    for item in value[:5]:
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _safe_string_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in list(value)[:20] if item is not None]


def _safe_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    return {
        "summary_only": True,
        "metadata_keys": sorted(str(key) for key in data.keys()),
        "mode": str(data.get("mode") or ""),
        "raw_jd_length": _safe_int(data.get("raw_jd_length")),
        "memory_source": str(data.get("memory_source") or ""),
        "expected_effect": str(data.get("expected_effect") or ""),
    }


def _safe_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(config or {}) if isinstance(config, Mapping) else {}
    return {
        "summary_only": True,
        "keys": sorted(str(key) for key in data.keys()),
        "memory_source": str(data.get("memory_source") or data.get("source") or ""),
        "max_items": _safe_int(data.get("max_items")),
        "max_chars": _safe_int(data.get("max_chars")),
    }
