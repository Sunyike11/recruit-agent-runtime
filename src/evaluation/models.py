from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


EVAL_TARGET_TYPES = {
    "task",
    "skill_workflow",
    "tool_workflow",
    "runtime_timeline",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EvalCase:
    case_id: str
    target_type: str
    input_data: Any = None
    expected: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        if not isinstance(data, dict):
            raise ValueError("EvalCase fixture must be a dict")
        case = cls(
            case_id=data.get("case_id", ""),
            target_type=data.get("target_type", ""),
            input_data=data.get("input_data"),
            expected=dict(data.get("expected", {})),
            checks=list(data.get("checks", [])),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )
        return case.validate()

    def validate(self) -> "EvalCase":
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("EvalCase case_id must be non-empty")
        if self.target_type not in EVAL_TARGET_TYPES:
            raise ValueError(f"unsupported EvalCase target_type: {self.target_type}")
        if not isinstance(self.expected, dict):
            raise ValueError("EvalCase expected must be a dict")
        if not isinstance(self.checks, list) or not all(isinstance(check, dict) for check in self.checks):
            raise ValueError("EvalCase checks must be a list of dicts")
        if not isinstance(self.tags, list):
            raise ValueError("EvalCase tags must be a list")
        if not isinstance(self.metadata, dict):
            raise ValueError("EvalCase metadata must be a dict")
        return self


@dataclass
class EvalResult:
    case_id: str
    target_type: str
    passed: bool
    score: float
    checks: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target_type": self.target_type,
            "passed": self.passed,
            "score": self.score,
            "checks": [dict(check) for check in self.checks],
            "error": self.error,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            case_id=data["case_id"],
            target_type=data["target_type"],
            passed=bool(data["passed"]),
            score=float(data["score"]),
            checks=[dict(check) for check in data.get("checks", [])],
            error=data.get("error", ""),
            metadata=dict(data.get("metadata", {})),
            created_at=created_at or utc_now(),
        )


@dataclass
class EvalReport:
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_score: float
    results: List[EvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: List[EvalResult],
        metadata: Dict[str, Any] = None,
    ) -> "EvalReport":
        passed_cases = sum(1 for result in results if result.passed)
        average_score = 0.0
        if results:
            average_score = round(sum(result.score for result in results) / len(results), 4)
        return cls(
            total_cases=len(results),
            passed_cases=passed_cases,
            failed_cases=len(results) - passed_cases,
            average_score=average_score,
            results=list(results),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "average_score": self.average_score,
            "results": [result.to_dict() for result in self.results],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalReport":
        return cls(
            total_cases=int(data["total_cases"]),
            passed_cases=int(data["passed_cases"]),
            failed_cases=int(data["failed_cases"]),
            average_score=float(data["average_score"]),
            results=[EvalResult.from_dict(result) for result in data.get("results", [])],
            metadata=dict(data.get("metadata", {})),
        )
