import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.evaluation.models import EvalCase, EvalReport, EvalResult, utc_now


def new_eval_id() -> str:
    return f"eval_{uuid.uuid4()}"


@dataclass
class EvalRecord:
    eval_id: str = field(default_factory=new_eval_id)
    case_id: str = ""
    target_type: str = ""
    target_id: str = ""
    passed: bool = False
    score: float = 0.0
    report_json: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "case_id": self.case_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "passed": self.passed,
            "score": self.score,
            "report_json": dict(self.report_json),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalRecord":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            eval_id=data["eval_id"],
            case_id=data["case_id"],
            target_type=data["target_type"],
            target_id=data["target_id"],
            passed=bool(data["passed"]),
            score=float(data["score"]),
            report_json=dict(data.get("report_json", {})),
            created_at=created_at or utc_now(),
            metadata=dict(data.get("metadata", {})),
        )


class InMemoryEvalRecordStore:
    """Minimal non-durable evaluation record store for deterministic tests."""

    def __init__(self):
        self._records: Dict[str, EvalRecord] = {}

    def save_record(self, record: EvalRecord) -> EvalRecord:
        self._records[record.eval_id] = record
        return record

    def get_record(self, eval_id: str) -> EvalRecord:
        if eval_id not in self._records:
            raise KeyError(eval_id)
        return self._records[eval_id]

    def list_records(
        self,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> List[EvalRecord]:
        records = list(self._records.values())
        if target_type is not None:
            records = [record for record in records if record.target_type == target_type]
        if target_id is not None:
            records = [record for record in records if record.target_id == target_id]
        if case_id is not None:
            records = [record for record in records if record.case_id == case_id]
        return records


def create_eval_record(
    eval_case: EvalCase,
    eval_output: Union[EvalResult, EvalReport],
    target_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalRecord:
    eval_case.validate()
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be non-empty")
    if isinstance(eval_output, EvalResult):
        passed = eval_output.passed
        score = eval_output.score
        report_json = eval_output.to_dict()
        record_metadata = {"record_source": "eval_result"}
    elif isinstance(eval_output, EvalReport):
        passed = eval_output.failed_cases == 0
        score = eval_output.average_score
        report_json = eval_output.to_dict()
        record_metadata = {"record_source": "eval_report"}
    else:
        raise TypeError("eval_output must be an EvalResult or EvalReport")
    return EvalRecord(
        case_id=eval_case.case_id,
        target_type=eval_case.target_type,
        target_id=target_id,
        passed=passed,
        score=score,
        report_json=report_json,
        metadata={**record_metadata, **dict(metadata or {})},
    )
