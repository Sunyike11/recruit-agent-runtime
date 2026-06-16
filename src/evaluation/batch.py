from dataclasses import replace
from typing import Any, Callable, List, Optional, Tuple

from src.evaluation.catalog import EvaluationCatalog
from src.evaluation.models import EvalCase, EvalReport, EvalResult
from src.evaluation.runner import EvalRunner
from src.evaluation.store import EvalRecord, create_eval_record


TargetResolver = Callable[[EvalCase], Any]
TargetIdResolver = Callable[[EvalCase], str]


class BatchEvalRunner:
    """Run local deterministic evaluation cases against resolved summary targets."""

    def __init__(self, eval_runner: Optional[EvalRunner] = None):
        self.eval_runner = eval_runner or EvalRunner()

    def run_catalog(
        self,
        catalog: EvaluationCatalog,
        target_resolver: Optional[TargetResolver] = None,
    ) -> EvalReport:
        results = [
            self._run_resolved_case(eval_case, target_resolver)
            for eval_case in catalog.list_cases()
        ]
        return EvalReport.from_results(
            results,
            metadata={
                "runner": "deterministic_batch_eval",
                "catalog_metadata_keys": sorted(str(key) for key in catalog.metadata),
                "summary_only": True,
            },
        )

    def run_catalog_with_records(
        self,
        catalog: EvaluationCatalog,
        target_resolver: Optional[TargetResolver] = None,
        target_id_resolver: Optional[TargetIdResolver] = None,
    ) -> Tuple[EvalReport, List[EvalRecord]]:
        report = self.run_catalog(catalog, target_resolver=target_resolver)
        records = self.create_records(
            catalog,
            report,
            target_id_resolver=target_id_resolver,
        )
        return report, records

    @staticmethod
    def create_records(
        catalog: EvaluationCatalog,
        report: EvalReport,
        target_id_resolver: Optional[TargetIdResolver] = None,
    ) -> List[EvalRecord]:
        results_by_case = {result.case_id: result for result in report.results}
        records = []
        for eval_case in catalog.list_cases():
            result = results_by_case.get(eval_case.case_id)
            if result is None:
                raise ValueError(f"evaluation result missing for case: {eval_case.case_id}")
            target_id = (
                target_id_resolver(eval_case)
                if target_id_resolver is not None
                else eval_case.case_id
            )
            records.append(
                create_eval_record(
                    eval_case,
                    result,
                    target_id=target_id,
                    metadata={"batch_run": True, "summary_only": True},
                )
            )
        return records

    def _run_resolved_case(
        self,
        eval_case: EvalCase,
        target_resolver: Optional[TargetResolver],
    ) -> EvalResult:
        if target_resolver is None:
            return self.eval_runner.run_case(eval_case)
        try:
            target = target_resolver(eval_case)
        except Exception as exc:
            return _resolution_failure(eval_case, f"target resolution failed: {exc}")
        if target is None:
            return _resolution_failure(
                eval_case,
                f"target_resolver returned no target for case: {eval_case.case_id}",
            )
        return self.eval_runner.run_case(replace(eval_case, input_data=target))


def _resolution_failure(eval_case: EvalCase, error: str) -> EvalResult:
    return EvalResult(
        case_id=eval_case.case_id,
        target_type=eval_case.target_type,
        passed=False,
        score=0.0,
        error=error,
        metadata={**dict(eval_case.metadata), "target_resolution_failed": True},
    )
