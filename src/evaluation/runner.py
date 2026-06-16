from typing import Any, Callable, Iterable, Optional

from src.evaluation.checks import evaluate_checks
from src.evaluation.models import EvalCase, EvalReport, EvalResult


class EvalRunner:
    """General deterministic evaluation runner for summaries and timelines."""

    def run_case(
        self,
        eval_case: EvalCase,
        evaluator: Optional[Callable[[Any], Any]] = None,
    ) -> EvalResult:
        eval_case.validate()
        try:
            target = evaluator(eval_case.input_data) if evaluator is not None else eval_case.input_data
            checks = evaluate_checks(target, eval_case.checks, expected=eval_case.expected)
            score = _score(checks)
            passed = bool(checks) and all(check["passed"] for check in checks)
            error = "" if checks else "no deterministic checks configured"
        except Exception as exc:
            checks = []
            score = 0.0
            passed = False
            error = str(exc)
        return EvalResult(
            case_id=eval_case.case_id,
            target_type=eval_case.target_type,
            passed=passed,
            score=score,
            checks=checks,
            error=error,
            metadata=dict(eval_case.metadata),
        )

    def run_cases(
        self,
        eval_cases: Iterable[EvalCase],
        evaluator: Optional[Callable[[Any], Any]] = None,
    ) -> EvalReport:
        results = [self.run_case(eval_case, evaluator=evaluator) for eval_case in eval_cases]
        return EvalReport.from_results(
            results,
            metadata={"runner": "deterministic_general_eval"},
        )


def _score(checks) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for check in checks if check["passed"])
    return round(passed / len(checks), 4)
