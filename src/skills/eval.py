import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutor
from src.skills.models import SkillResult
from src.skills.registry import SkillRegistry


@dataclass
class SkillEvalCase:
    case_id: str
    skill_name: str
    input_data: Dict[str, Any]
    skill_version: Optional[str] = None
    expected_output: Optional[Dict[str, Any]] = None
    expected_success: Optional[bool] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEvalCase":
        return cls(
            case_id=data["case_id"],
            skill_name=data["skill_name"],
            skill_version=data.get("skill_version"),
            input_data=dict(data.get("input_data", {})),
            expected_output=data.get("expected_output"),
            expected_success=data.get("expected_success"),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SkillEvalResult:
    case_id: str
    skill_name: str
    skill_version: str
    success: bool
    passed: bool
    output: Any = None
    error: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillEvalReport:
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: List[SkillEvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillEvalRunner:
    """Deterministic skill evaluation runner for fixture-based checks."""

    def __init__(
        self,
        skill_source=None,
        registry: Optional[SkillRegistry] = None,
        executor: Optional[SkillExecutor] = None,
    ):
        self.executor = executor
        self.registry = registry

        if isinstance(skill_source, SkillExecutor):
            self.executor = skill_source
        elif isinstance(skill_source, SkillRegistry):
            self.registry = skill_source
        elif skill_source is not None:
            raise TypeError("skill_source must be a SkillRegistry or SkillExecutor")

        if self.executor is None and self.registry is None:
            raise ValueError("SkillEvalRunner requires a SkillRegistry or SkillExecutor")

    def run_case(
        self,
        eval_case: SkillEvalCase,
        context: Optional[SkillExecutionContext] = None,
    ) -> SkillEvalResult:
        started = time.perf_counter()
        result = self._execute(eval_case, context=context)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        checks = _evaluate_checks(eval_case, result)
        passed = all(check["passed"] for check in checks)

        return SkillEvalResult(
            case_id=eval_case.case_id,
            skill_name=result.skill_name,
            skill_version=result.version,
            success=result.success,
            passed=passed,
            output=result.output,
            error=result.error,
            checks=checks,
            duration_ms=duration_ms,
            metadata=dict(eval_case.metadata),
        )

    def run_cases(
        self,
        eval_cases: List[SkillEvalCase],
        context: Optional[SkillExecutionContext] = None,
    ) -> SkillEvalReport:
        results = [self.run_case(eval_case, context=context) for eval_case in eval_cases]
        passed_cases = sum(1 for result in results if result.passed)
        return SkillEvalReport(
            total_cases=len(results),
            passed_cases=passed_cases,
            failed_cases=len(results) - passed_cases,
            results=results,
            metadata={"runner": "deterministic_skill_eval"},
        )

    def _execute(
        self,
        eval_case: SkillEvalCase,
        context: Optional[SkillExecutionContext] = None,
    ) -> SkillResult:
        execution_context = context or SkillExecutionContext()
        if self.executor is not None:
            return self.executor.execute(
                eval_case.skill_name,
                eval_case.input_data,
                context=execution_context,
                version=eval_case.skill_version,
            )

        skill = self.registry.get(eval_case.skill_name, version=eval_case.skill_version)
        return skill.execute(eval_case.input_data, context=execution_context)


def replay_case_from_fixture(
    fixture,
    runner: SkillEvalRunner,
    context: Optional[SkillExecutionContext] = None,
) -> SkillEvalResult:
    """Replay a fixture-backed eval case with full original input data."""

    eval_case = fixture if isinstance(fixture, SkillEvalCase) else SkillEvalCase.from_dict(fixture)
    result = runner.run_case(eval_case, context=context)
    result.metadata.update(
        {
            "replay_mode": "fixture_full_replay",
            "full_replay": True,
        }
    )
    return result


def replay_case_from_skill_event_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a limited replay audit record from a runtime skill event summary.

    Runtime skill events intentionally store summaries, not complete inputs and
    outputs. This helper never claims full replay is possible from that payload.
    """

    return {
        "replay_mode": "event_summary_limited_replay",
        "full_replay": False,
        "can_execute_skill": False,
        "reason": "runtime skill event payload stores summaries only, not full input_data",
        "execution_id": payload.get("execution_id"),
        "skill_name": payload.get("skill_name"),
        "skill_version": payload.get("skill_version"),
        "status": payload.get("status"),
        "input_summary": payload.get("input_summary"),
        "output_summary": payload.get("output_summary"),
        "error": payload.get("error", ""),
    }


def _evaluate_checks(eval_case: SkillEvalCase, result: SkillResult) -> List[Dict[str, Any]]:
    expected_success = True if eval_case.expected_success is None else eval_case.expected_success
    checks = [
        {
            "name": "expected_success",
            "expected": expected_success,
            "actual": result.success,
            "passed": result.success is expected_success,
        }
    ]

    if expected_success is False:
        checks.append(
            {
                "name": "error_present",
                "expected": True,
                "actual": bool(result.error),
                "passed": bool(result.error),
            }
        )

    if eval_case.expected_output is not None:
        output_check = _check_expected_output_contains(eval_case.expected_output, result.output)
        checks.append(output_check)

    return checks


def _check_expected_output_contains(expected_output: Dict[str, Any], actual_output: Any) -> Dict[str, Any]:
    missing_or_mismatched = {}
    if not isinstance(actual_output, dict):
        return {
            "name": "expected_output_contains",
            "expected": expected_output,
            "actual": actual_output,
            "passed": False,
            "mismatches": {"output": "actual output is not a dict"},
        }

    for key, expected_value in expected_output.items():
        if actual_output.get(key) != expected_value:
            missing_or_mismatched[key] = {
                "expected": expected_value,
                "actual": actual_output.get(key),
            }

    return {
        "name": "expected_output_contains",
        "expected": expected_output,
        "actual_keys": sorted(str(key) for key in actual_output.keys()),
        "passed": not missing_or_mismatched,
        "mismatches": missing_or_mismatched,
    }
