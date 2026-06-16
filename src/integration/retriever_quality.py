from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class RetrieverQualityCase:
    case_id: str
    query: str
    top_k: int = 3
    expected_candidate_id: Optional[str] = None
    expected_source_contains: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrieverQualityCase":
        return cls(
            case_id=str(data["case_id"]),
            query=str(data["query"]),
            top_k=int(data.get("top_k", 3)),
            expected_candidate_id=data.get("expected_candidate_id"),
            expected_source_contains=data.get("expected_source_contains"),
            tags=list(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrieverQualityObservation:
    case_id: str
    query_length: int
    top_k: int
    result_count: int
    result_summary: List[Dict[str, Any]]
    source_keys: List[str]
    candidate_ids: List[str]
    document_ids: List[str]
    score_present: bool
    expected_candidate_found: Optional[bool] = None
    expected_source_found: Optional[bool] = None
    index_record_count: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrieverQualityReport:
    total_cases: int
    ok_count: int
    failed_count: int
    observations: List[RetrieverQualityObservation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "ok_count": self.ok_count,
            "failed_count": self.failed_count,
            "observations": [observation.to_dict() for observation in self.observations],
            "metadata": dict(self.metadata),
        }


def summarize_retrieval_results(
    results: Iterable[Mapping[str, Any]],
    *,
    query: str,
    top_k: int,
    expected_candidate_id: Optional[str] = None,
    expected_source_contains: Optional[str] = None,
    index_record_count: Optional[int] = None,
) -> RetrieverQualityObservation:
    result_list = [dict(result) for result in results]
    summaries: List[Dict[str, Any]] = []
    source_keys = set()
    candidate_ids: List[str] = []
    document_ids: List[str] = []
    score_present = False
    source_match = False

    for index, result in enumerate(result_list):
        text = result.get("text", "")
        metadata = dict(result.get("metadata") or {})
        score = result.get("score")
        score_present = score_present or isinstance(score, (int, float))
        source_keys.update(str(key) for key in metadata.keys())

        candidate_id = _first_string(metadata, ("candidate_id", "candidateId", "id"))
        if candidate_id:
            candidate_ids.append(_safe_identifier(candidate_id))

        document_id = _first_string(metadata, ("resume_id", "document_id", "doc_id", "file_name", "source"))
        if document_id:
            document_ids.append(_safe_identifier(document_id))

        if expected_source_contains:
            source_match = source_match or _metadata_contains(metadata, expected_source_contains)

        summaries.append(
            {
                "rank": index + 1,
                "text_length": len(text) if isinstance(text, str) else 0,
                "metadata_keys": sorted(str(key) for key in metadata.keys()),
                "score_present": isinstance(score, (int, float)),
            }
        )

    expected_candidate_found = None
    if expected_candidate_id is not None:
        expected_candidate_found = expected_candidate_id in set(candidate_ids)

    expected_source_found = None
    if expected_source_contains is not None:
        expected_source_found = source_match

    return RetrieverQualityObservation(
        case_id="adhoc_retriever_quality_smoke",
        query_length=len(query),
        top_k=top_k,
        result_count=len(result_list),
        result_summary=summaries,
        source_keys=sorted(source_keys),
        candidate_ids=_dedupe(candidate_ids),
        document_ids=_dedupe(document_ids),
        score_present=score_present,
        expected_candidate_found=expected_candidate_found,
        expected_source_found=expected_source_found,
        index_record_count=index_record_count,
        metadata={
            "mode": "summary_only_retriever_quality_observation",
            "summary_only": True,
            "production_graph_invoked": False,
        },
    )


def _first_string(metadata: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_identifier(value: str, max_length: int = 80) -> str:
    safe = value.replace("\\", "/").split("/")[-1].strip()
    if len(safe) <= max_length:
        return safe
    return safe[: max_length - 3] + "..."


def _metadata_contains(metadata: Mapping[str, Any], expected: str) -> bool:
    needle = expected.lower()
    for value in metadata.values():
        if isinstance(value, (str, int, float)) and needle in str(value).lower():
            return True
    return False


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
