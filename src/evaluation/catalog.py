import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from src.evaluation.models import EvalCase


class EvaluationCatalogError(ValueError):
    pass


class EvaluationCatalog:
    """Local deterministic evaluation fixture catalog."""

    def __init__(self, cases: Iterable[EvalCase], metadata: Dict[str, Any] = None):
        self.cases = list(cases)
        self.metadata = dict(metadata or {})
        self.validate()

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> "EvaluationCatalog":
        if isinstance(data, list):
            entries = data
            metadata = {}
        elif isinstance(data, dict):
            entries = data.get("cases")
            metadata = data.get("metadata", {})
        else:
            raise EvaluationCatalogError("evaluation catalog must be a dict or list")
        if not isinstance(entries, list):
            raise EvaluationCatalogError("evaluation catalog must contain a cases list")
        if not isinstance(metadata, dict):
            raise EvaluationCatalogError("evaluation catalog metadata must be a dict")
        cases = []
        for index, entry in enumerate(entries):
            try:
                cases.append(EvalCase.from_dict(entry))
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationCatalogError(
                    f"invalid evaluation case at index {index}: {exc}"
                ) from exc
        return cls(cases, metadata=metadata)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "EvaluationCatalog":
        try:
            with Path(path).open("r", encoding="utf-8") as catalog_file:
                return cls.from_dict(json.load(catalog_file))
        except json.JSONDecodeError as exc:
            raise EvaluationCatalogError(f"invalid evaluation catalog JSON: {exc}") from exc

    def validate(self) -> "EvaluationCatalog":
        seen = set()
        for eval_case in self.cases:
            eval_case.validate()
            if eval_case.case_id in seen:
                raise EvaluationCatalogError(f"duplicate evaluation case_id: {eval_case.case_id}")
            seen.add(eval_case.case_id)
        return self

    def list_cases(self) -> List[EvalCase]:
        return list(self.cases)

    def get_case(self, case_id: str) -> EvalCase:
        for eval_case in self.cases:
            if eval_case.case_id == case_id:
                return eval_case
        raise EvaluationCatalogError(f"evaluation case not found: {case_id}")

    def filter_by_tag(self, tag: str) -> List[EvalCase]:
        return [eval_case for eval_case in self.cases if tag in eval_case.tags]

    def filter_by_target_type(self, target_type: str) -> List[EvalCase]:
        return [eval_case for eval_case in self.cases if eval_case.target_type == target_type]
