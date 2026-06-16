import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.runtime.candidate_preview import (
    CandidatePreviewBuildConfig,
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
)


FORBIDDEN_PAYLOAD_KEYS = {
    "raw_text",
    "full_text",
    "full_resume",
    "raw_chunks",
    "embedding",
    "prompt",
    "llm_response",
    "reasoning",
    "api_key",
    "hf_token",
}

SCORE_WEIGHTS = {
    "grouping": 0.20,
    "identity_source": 0.20,
    "field_extraction": 0.35,
    "evidence_quality_flags": 0.15,
    "privacy_matcher": 0.10,
}

PASS_THRESHOLD = 0.75


@dataclass
class CandidatePreviewExpectedProfile:
    expected_candidate_id: str = ""
    expected_candidate_name: str = ""
    expected_source_document_id: str = ""
    expected_skills: List[str] = field(default_factory=list)
    expected_project_keywords: List[str] = field(default_factory=list)
    expected_education_keywords: List[str] = field(default_factory=list)
    expected_experience_keywords: List[str] = field(default_factory=list)
    expected_matched_query_terms: List[str] = field(default_factory=list)
    expected_quality_flags: List[str] = field(default_factory=list)
    forbidden_quality_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidatePreviewExpectedProfile":
        return cls(
            expected_candidate_id=str(data.get("expected_candidate_id") or ""),
            expected_candidate_name=str(data.get("expected_candidate_name") or ""),
            expected_source_document_id=str(data.get("expected_source_document_id") or ""),
            expected_skills=_strings(data.get("expected_skills")),
            expected_project_keywords=_strings(data.get("expected_project_keywords")),
            expected_education_keywords=_strings(data.get("expected_education_keywords")),
            expected_experience_keywords=_strings(data.get("expected_experience_keywords")),
            expected_matched_query_terms=_strings(data.get("expected_matched_query_terms")),
            expected_quality_flags=_strings(data.get("expected_quality_flags")),
            forbidden_quality_flags=_strings(data.get("forbidden_quality_flags")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_candidate_id": self.expected_candidate_id,
            "expected_candidate_name": self.expected_candidate_name,
            "expected_source_document_id": self.expected_source_document_id,
            "expected_skills": list(self.expected_skills),
            "expected_project_keywords": list(self.expected_project_keywords),
            "expected_education_keywords": list(self.expected_education_keywords),
            "expected_experience_keywords": list(self.expected_experience_keywords),
            "expected_matched_query_terms": list(self.expected_matched_query_terms),
            "expected_quality_flags": list(self.expected_quality_flags),
            "forbidden_quality_flags": list(self.forbidden_quality_flags),
            "metadata": _metadata_summary(self.metadata),
        }


@dataclass
class CandidatePreviewEvalCase:
    case_id: str
    raw_jd: str = ""
    query: str = ""
    retrieval_chunks: List[Dict[str, Any]] = field(default_factory=list)
    expected_profiles: List[CandidatePreviewExpectedProfile] = field(default_factory=list)
    build_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidatePreviewEvalCase":
        return cls(
            case_id=str(data.get("case_id") or ""),
            raw_jd=str(data.get("raw_jd") or data.get("query") or ""),
            query=str(data.get("query") or ""),
            retrieval_chunks=[dict(item) for item in data.get("retrieval_chunks") or [] if isinstance(item, Mapping)],
            expected_profiles=[
                CandidatePreviewExpectedProfile.from_dict(item)
                for item in data.get("expected_profiles") or []
                if isinstance(item, Mapping)
            ],
            build_config=dict(data.get("build_config") or {}),
            tags=_strings(data.get("tags")),
            metadata=dict(data.get("metadata") or {}),
        ).validate()

    def validate(self) -> "CandidatePreviewEvalCase":
        if not self.case_id:
            raise ValueError("CandidatePreviewEvalCase case_id must be non-empty")
        if not isinstance(self.retrieval_chunks, list):
            raise ValueError("CandidatePreviewEvalCase retrieval_chunks must be a list")
        if not self.expected_profiles:
            raise ValueError("CandidatePreviewEvalCase expected_profiles must be non-empty")
        return self


@dataclass
class CandidatePreviewFieldMetrics:
    expected_count: int = 0
    actual_count: int = 0
    true_positive_count: int = 0
    missing_count: int = 0
    unexpected_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "true_positive_count": self.true_positive_count,
            "missing_count": self.missing_count,
            "unexpected_count": self.unexpected_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class CandidatePreviewEvalResult:
    case_id: str
    status: str
    preview_count: int
    expected_profile_count: int
    grouping_correct: bool
    candidate_id_match_count: int
    candidate_name_match_count: int
    source_document_match_count: int
    skills_metrics: CandidatePreviewFieldMetrics
    project_keywords_metrics: CandidatePreviewFieldMetrics
    education_keywords_metrics: CandidatePreviewFieldMetrics
    experience_keywords_metrics: CandidatePreviewFieldMetrics
    matched_query_terms_metrics: CandidatePreviewFieldMetrics
    quality_flags_match: bool
    evidence_summary_present_count: int
    evidence_summary_truncated_count: int
    privacy_checks_passed: bool
    matcher_input_compatible_count: int
    score: float
    errors: List[str] = field(default_factory=list)
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "preview_count": self.preview_count,
            "expected_profile_count": self.expected_profile_count,
            "grouping_correct": self.grouping_correct,
            "candidate_id_match_count": self.candidate_id_match_count,
            "candidate_name_match_count": self.candidate_name_match_count,
            "source_document_match_count": self.source_document_match_count,
            "skills_metrics": self.skills_metrics.to_dict(),
            "project_keywords_metrics": self.project_keywords_metrics.to_dict(),
            "education_keywords_metrics": self.education_keywords_metrics.to_dict(),
            "experience_keywords_metrics": self.experience_keywords_metrics.to_dict(),
            "matched_query_terms_metrics": self.matched_query_terms_metrics.to_dict(),
            "quality_flags_match": self.quality_flags_match,
            "evidence_summary_present_count": self.evidence_summary_present_count,
            "evidence_summary_truncated_count": self.evidence_summary_truncated_count,
            "privacy_checks_passed": self.privacy_checks_passed,
            "matcher_input_compatible_count": self.matcher_input_compatible_count,
            "score": self.score,
            "errors": list(self.errors),
            "summary_only": True,
            "metadata": _metadata_summary(self.metadata),
        }


@dataclass
class CandidatePreviewEvalReport:
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_score: float
    grouping_accuracy: float
    candidate_name_accuracy: float
    source_document_accuracy: float
    average_skills_precision: float
    average_skills_recall: float
    average_skills_f1: float
    privacy_pass_rate: float
    matcher_compatibility_rate: float
    results: List[CandidatePreviewEvalResult] = field(default_factory=list)
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "average_score": self.average_score,
            "grouping_accuracy": self.grouping_accuracy,
            "candidate_name_accuracy": self.candidate_name_accuracy,
            "source_document_accuracy": self.source_document_accuracy,
            "average_skills_precision": self.average_skills_precision,
            "average_skills_recall": self.average_skills_recall,
            "average_skills_f1": self.average_skills_f1,
            "privacy_pass_rate": self.privacy_pass_rate,
            "matcher_compatibility_rate": self.matcher_compatibility_rate,
            "results": [result.to_dict() for result in self.results],
            "summary_only": True,
        }


class CandidatePreviewEvaluator:
    def run_case(self, eval_case: CandidatePreviewEvalCase | Mapping[str, Any]) -> CandidatePreviewEvalResult:
        case = eval_case if isinstance(eval_case, CandidatePreviewEvalCase) else CandidatePreviewEvalCase.from_dict(eval_case)
        try:
            config = CandidatePreviewBuildConfig(**dict(case.build_config or {}))
            build_result = build_candidate_profile_previews_from_retrieval_results(
                case.retrieval_chunks,
                raw_jd=case.raw_jd,
                query=case.query,
                config=config,
            )
            previews = [preview.to_dict() for preview in build_result.previews]
            matcher_inputs = [candidate_profile_preview_to_matcher_input(preview) for preview in build_result.previews]
            matched_pairs = _match_expected_profiles(case.expected_profiles, previews)
            result = self._evaluate_outputs(
                case=case,
                previews=previews,
                matcher_inputs=matcher_inputs,
                matched_pairs=matched_pairs,
                max_evidence_chars=config.max_evidence_chars,
            )
        except Exception as exc:
            result = _failed_result(case.case_id, str(type(exc).__name__))
        return result

    def run_cases(self, cases: Iterable[CandidatePreviewEvalCase | Mapping[str, Any]]) -> CandidatePreviewEvalReport:
        return self.build_report([self.run_case(case) for case in cases])

    def evaluate_profile(
        self,
        expected: CandidatePreviewExpectedProfile,
        preview: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "candidate_id_match": _optional_exact_match(expected.expected_candidate_id, preview.get("candidate_id")),
            "candidate_name_match": _optional_exact_match(expected.expected_candidate_name, preview.get("candidate_name")),
            "source_document_match": _optional_exact_match(
                expected.expected_source_document_id,
                preview.get("source_document_id"),
            ),
            "skills_metrics": self.compare_field_values(expected.expected_skills, preview.get("skills")).to_dict(),
            "summary_only": True,
        }

    def compare_field_values(self, expected: Iterable[Any], actual: Iterable[Any]) -> CandidatePreviewFieldMetrics:
        expected_set = {_normalize_value(item) for item in expected if _normalize_value(item)}
        actual_set = {_normalize_value(item) for item in actual if _normalize_value(item)}
        true_positive = expected_set & actual_set
        missing = expected_set - actual_set
        unexpected = actual_set - expected_set
        precision = len(true_positive) / len(actual_set) if actual_set else (1.0 if not expected_set else 0.0)
        recall = len(true_positive) / len(expected_set) if expected_set else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return CandidatePreviewFieldMetrics(
            expected_count=len(expected_set),
            actual_count=len(actual_set),
            true_positive_count=len(true_positive),
            missing_count=len(missing),
            unexpected_count=len(unexpected),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )

    def build_report(self, results: Sequence[CandidatePreviewEvalResult]) -> CandidatePreviewEvalReport:
        items = list(results)
        total = len(items)
        return CandidatePreviewEvalReport(
            total_cases=total,
            passed_cases=sum(1 for result in items if result.status == "passed"),
            failed_cases=sum(1 for result in items if result.status != "passed"),
            average_score=_average([result.score for result in items]),
            grouping_accuracy=_ratio(sum(1 for result in items if result.grouping_correct), total),
            candidate_name_accuracy=_ratio(
                sum(result.candidate_name_match_count for result in items),
                sum(result.expected_profile_count for result in items),
            ),
            source_document_accuracy=_ratio(
                sum(result.source_document_match_count for result in items),
                sum(result.expected_profile_count for result in items),
            ),
            average_skills_precision=_average([result.skills_metrics.precision for result in items]),
            average_skills_recall=_average([result.skills_metrics.recall for result in items]),
            average_skills_f1=_average([result.skills_metrics.f1 for result in items]),
            privacy_pass_rate=_ratio(sum(1 for result in items if result.privacy_checks_passed), total),
            matcher_compatibility_rate=_ratio(
                sum(result.matcher_input_compatible_count for result in items),
                sum(result.expected_profile_count for result in items),
            ),
            results=items,
            summary_only=True,
        )

    def _evaluate_outputs(
        self,
        *,
        case: CandidatePreviewEvalCase,
        previews: List[Dict[str, Any]],
        matcher_inputs: List[Dict[str, Any]],
        matched_pairs: List[tuple[CandidatePreviewExpectedProfile, Optional[Dict[str, Any]]]],
        max_evidence_chars: int,
    ) -> CandidatePreviewEvalResult:
        errors: List[str] = []
        grouping_correct = len(previews) == len(case.expected_profiles) and all(preview is not None for _expected, preview in matched_pairs)
        if not grouping_correct:
            errors.append("grouping_mismatch")
        candidate_id_match_count = sum(
            1
            for expected, preview in matched_pairs
            if preview is not None and _optional_exact_match(expected.expected_candidate_id, preview.get("candidate_id"))
        )
        candidate_name_match_count = sum(
            1
            for expected, preview in matched_pairs
            if preview is not None and _optional_exact_match(expected.expected_candidate_name, preview.get("candidate_name"))
        )
        source_document_match_count = sum(
            1
            for expected, preview in matched_pairs
            if preview is not None
            and _optional_exact_match(expected.expected_source_document_id, preview.get("source_document_id"))
        )
        skills_metrics = _aggregate_metrics(
            [self.compare_field_values(expected.expected_skills, (preview or {}).get("skills", [])) for expected, preview in matched_pairs]
        )
        project_metrics = _aggregate_metrics(
            [
                self.compare_field_values(expected.expected_project_keywords, (preview or {}).get("project_keywords", []))
                for expected, preview in matched_pairs
            ]
        )
        education_metrics = _aggregate_metrics(
            [
                self.compare_field_values(expected.expected_education_keywords, (preview or {}).get("education_keywords", []))
                for expected, preview in matched_pairs
            ]
        )
        experience_metrics = _aggregate_metrics(
            [
                self.compare_field_values(expected.expected_experience_keywords, (preview or {}).get("experience_keywords", []))
                for expected, preview in matched_pairs
            ]
        )
        query_metrics = _aggregate_metrics(
            [
                self.compare_field_values(expected.expected_matched_query_terms, (preview or {}).get("matched_query_terms", []))
                for expected, preview in matched_pairs
            ]
        )
        quality_flags_match = all(_quality_flags_match(expected, preview or {}) for expected, preview in matched_pairs)
        if not quality_flags_match:
            errors.append("quality_flags_mismatch")
        evidence_present = sum(1 for preview in previews if bool(preview.get("evidence_summary")))
        evidence_truncated = sum(
            1
            for preview in previews
            if "summary_truncated" in _strings(preview.get("preview_quality_flags"))
        )
        privacy = validate_candidate_preview_privacy(
            previews,
            retrieval_chunks=case.retrieval_chunks,
            max_evidence_chars=max_evidence_chars,
        )
        matcher_privacy = validate_matcher_input_privacy(matcher_inputs, retrieval_chunks=case.retrieval_chunks)
        privacy_passed = bool(privacy["passed"] and matcher_privacy["passed"])
        if not privacy_passed:
            errors.append("privacy_check_failed")
        matcher_compatible_count = sum(1 for candidate in matcher_inputs if _matcher_input_compatible(candidate))
        if matcher_compatible_count != len(case.expected_profiles):
            errors.append("matcher_input_incompatible")
        score = _score(
            grouping_correct=grouping_correct,
            identity_source_score=_identity_source_score(
                candidate_id_match_count,
                candidate_name_match_count,
                source_document_match_count,
                len(case.expected_profiles),
            ),
            field_score=_average(
                [skills_metrics.f1, project_metrics.f1, education_metrics.f1, experience_metrics.f1, query_metrics.f1]
            ),
            evidence_quality_score=_average([
                _ratio(evidence_present, len(case.expected_profiles)),
                1.0 if quality_flags_match else 0.0,
            ]),
            privacy_matcher_score=_average([
                1.0 if privacy_passed else 0.0,
                _ratio(matcher_compatible_count, len(case.expected_profiles)),
            ]),
        )
        passed = bool(score >= PASS_THRESHOLD and grouping_correct and privacy_passed)
        return CandidatePreviewEvalResult(
            case_id=case.case_id,
            status="passed" if passed else "failed",
            preview_count=len(previews),
            expected_profile_count=len(case.expected_profiles),
            grouping_correct=grouping_correct,
            candidate_id_match_count=candidate_id_match_count,
            candidate_name_match_count=candidate_name_match_count,
            source_document_match_count=source_document_match_count,
            skills_metrics=skills_metrics,
            project_keywords_metrics=project_metrics,
            education_keywords_metrics=education_metrics,
            experience_keywords_metrics=experience_metrics,
            matched_query_terms_metrics=query_metrics,
            quality_flags_match=quality_flags_match,
            evidence_summary_present_count=evidence_present,
            evidence_summary_truncated_count=evidence_truncated,
            privacy_checks_passed=privacy_passed,
            matcher_input_compatible_count=matcher_compatible_count,
            score=score,
            errors=errors,
            summary_only=True,
            metadata={
                "score_weights": dict(SCORE_WEIGHTS),
                "pass_threshold": PASS_THRESHOLD,
                "privacy_errors": privacy["errors"] + matcher_privacy["errors"],
                "tags": list(case.tags),
            },
        )


def build_candidate_preview_quality_gate(result: CandidatePreviewEvalResult | Mapping[str, Any]) -> Dict[str, Any]:
    data = result.to_dict() if isinstance(result, CandidatePreviewEvalResult) else dict(result)
    score = float(data.get("score") or 0.0)
    grouping = bool(data.get("grouping_correct", False))
    privacy = bool(data.get("privacy_checks_passed", False))
    matcher_count = int(data.get("matcher_input_compatible_count") or 0)
    expected_count = int(data.get("expected_profile_count") or 0)
    status = "pass" if score >= PASS_THRESHOLD and grouping and privacy else "fail"
    if status == "pass" and matcher_count < expected_count:
        status = "warning"
    return {
        "status": status,
        "score": round(score, 4),
        "grouping_correct": grouping,
        "identity_quality": _ratio(
            int(data.get("candidate_id_match_count") or 0) + int(data.get("source_document_match_count") or 0),
            max(1, expected_count * 2),
        ),
        "field_coverage_quality": float((data.get("skills_metrics") or {}).get("recall", 0.0)),
        "privacy_passed": privacy,
        "matcher_compatible": matcher_count == expected_count,
        "summary_only": True,
    }


def validate_candidate_preview_privacy(
    previews: Sequence[Mapping[str, Any]],
    *,
    retrieval_chunks: Sequence[Mapping[str, Any]],
    max_evidence_chars: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    full_texts = _full_texts(retrieval_chunks)
    for index, preview in enumerate(previews):
        _collect_forbidden_key_errors(preview, errors, prefix=f"preview[{index}]")
        evidence = str(preview.get("evidence_summary") or "")
        if len(evidence) > max_evidence_chars:
            errors.append("evidence_summary_too_long")
        if any(evidence and evidence == text for text in full_texts):
            errors.append("evidence_summary_equals_full_chunk")
        serialized = json.dumps(preview, ensure_ascii=False)
        for text in full_texts:
            if len(text) > 60 and text in serialized:
                errors.append("preview_contains_full_chunk")
        source_file = str(preview.get("source_file_name") or "")
        if "/" in source_file or "\\" in source_file:
            errors.append("source_file_name_contains_path")
    return {"passed": not errors, "errors": sorted(set(errors)), "summary_only": True}


def validate_matcher_input_privacy(
    matcher_inputs: Sequence[Mapping[str, Any]],
    *,
    retrieval_chunks: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    full_texts = _full_texts(retrieval_chunks)
    for index, candidate in enumerate(matcher_inputs):
        _collect_forbidden_key_errors(candidate, errors, prefix=f"matcher[{index}]")
        serialized = json.dumps(candidate, ensure_ascii=False)
        for text in full_texts:
            if len(text) > 60 and text in serialized:
                errors.append("matcher_input_contains_full_chunk")
    return {"passed": not errors, "errors": sorted(set(errors)), "summary_only": True}


def export_candidate_preview_eval_json(report: CandidatePreviewEvalReport | CandidatePreviewEvalResult) -> str:
    data = report.to_dict()
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def export_candidate_preview_eval_text(report: CandidatePreviewEvalReport | CandidatePreviewEvalResult) -> str:
    if isinstance(report, CandidatePreviewEvalResult):
        return "\n".join(
            [
                "Candidate Preview Evaluation Result",
                f"case_id: {report.case_id}",
                f"status: {report.status}",
                f"score: {report.score:.4f}",
                f"grouping_correct: {report.grouping_correct}",
                f"privacy_checks_passed: {report.privacy_checks_passed}",
                f"matcher_input_compatible_count: {report.matcher_input_compatible_count}",
            ]
        )
    return "\n".join(
        [
            "Candidate Preview Evaluation Report",
            f"total_cases: {report.total_cases}",
            f"passed_cases: {report.passed_cases}",
            f"failed_cases: {report.failed_cases}",
            f"average_score: {report.average_score:.4f}",
            f"grouping_accuracy: {report.grouping_accuracy:.4f}",
            f"privacy_pass_rate: {report.privacy_pass_rate:.4f}",
        ]
    )


def _failed_result(case_id: str, error_type: str) -> CandidatePreviewEvalResult:
    empty = CandidatePreviewFieldMetrics()
    return CandidatePreviewEvalResult(
        case_id=case_id,
        status="failed",
        preview_count=0,
        expected_profile_count=0,
        grouping_correct=False,
        candidate_id_match_count=0,
        candidate_name_match_count=0,
        source_document_match_count=0,
        skills_metrics=empty,
        project_keywords_metrics=empty,
        education_keywords_metrics=empty,
        experience_keywords_metrics=empty,
        matched_query_terms_metrics=empty,
        quality_flags_match=False,
        evidence_summary_present_count=0,
        evidence_summary_truncated_count=0,
        privacy_checks_passed=False,
        matcher_input_compatible_count=0,
        score=0.0,
        errors=[str(error_type or "evaluation_failed")],
        summary_only=True,
        metadata={"summary_only": True},
    )


def _match_expected_profiles(
    expected_profiles: Sequence[CandidatePreviewExpectedProfile],
    previews: Sequence[Mapping[str, Any]],
) -> List[tuple[CandidatePreviewExpectedProfile, Optional[Dict[str, Any]]]]:
    remaining = [dict(preview) for preview in previews]
    pairs = []
    for expected in expected_profiles:
        match_index = _find_preview_index(expected, remaining)
        if match_index is None:
            pairs.append((expected, None))
            continue
        pairs.append((expected, remaining.pop(match_index)))
    return pairs


def _find_preview_index(expected: CandidatePreviewExpectedProfile, previews: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for field_name, expected_value in (
        ("candidate_id", expected.expected_candidate_id),
        ("source_document_id", expected.expected_source_document_id),
        ("candidate_name", expected.expected_candidate_name),
    ):
        if not expected_value:
            continue
        for index, preview in enumerate(previews):
            if _normalize_value(preview.get(field_name)) == _normalize_value(expected_value):
                return index
    return 0 if previews else None


def _aggregate_metrics(items: Sequence[CandidatePreviewFieldMetrics]) -> CandidatePreviewFieldMetrics:
    expected_count = sum(item.expected_count for item in items)
    actual_count = sum(item.actual_count for item in items)
    true_positive = sum(item.true_positive_count for item in items)
    missing = sum(item.missing_count for item in items)
    unexpected = sum(item.unexpected_count for item in items)
    precision = true_positive / actual_count if actual_count else (1.0 if expected_count == 0 else 0.0)
    recall = true_positive / expected_count if expected_count else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return CandidatePreviewFieldMetrics(
        expected_count=expected_count,
        actual_count=actual_count,
        true_positive_count=true_positive,
        missing_count=missing,
        unexpected_count=unexpected,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def _quality_flags_match(expected: CandidatePreviewExpectedProfile, preview: Mapping[str, Any]) -> bool:
    actual = {_normalize_value(item) for item in _strings(preview.get("preview_quality_flags"))}
    required = {_normalize_value(item) for item in expected.expected_quality_flags}
    forbidden = {_normalize_value(item) for item in expected.forbidden_quality_flags}
    return required.issubset(actual) and not (forbidden & actual)


def _score(
    *,
    grouping_correct: bool,
    identity_source_score: float,
    field_score: float,
    evidence_quality_score: float,
    privacy_matcher_score: float,
) -> float:
    score = (
        (1.0 if grouping_correct else 0.0) * SCORE_WEIGHTS["grouping"]
        + identity_source_score * SCORE_WEIGHTS["identity_source"]
        + field_score * SCORE_WEIGHTS["field_extraction"]
        + evidence_quality_score * SCORE_WEIGHTS["evidence_quality_flags"]
        + privacy_matcher_score * SCORE_WEIGHTS["privacy_matcher"]
    )
    return round(score, 4)


def _identity_source_score(candidate_id_matches: int, name_matches: int, source_matches: int, expected_count: int) -> float:
    return _ratio(candidate_id_matches + name_matches + source_matches, max(1, expected_count * 3))


def _matcher_input_compatible(candidate: Mapping[str, Any]) -> bool:
    required = {"candidate_id", "name", "skills", "evidence_summary", "source_document_id", "metadata"}
    metadata = candidate.get("metadata")
    return required.issubset(set(candidate.keys())) and isinstance(metadata, Mapping) and bool(metadata.get("candidate_profile_preview"))


def _optional_exact_match(expected: Any, actual: Any) -> bool:
    expected_text = _normalize_value(expected)
    if not expected_text:
        return True
    return expected_text == _normalize_value(actual)


def _normalize_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _average(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _metadata_summary(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {"keys": sorted(str(key) for key in metadata.keys()), "size": len(metadata), "summary_only": True}


def _full_texts(chunks: Sequence[Mapping[str, Any]]) -> List[str]:
    output = []
    for chunk in chunks:
        for key in ("text", "content", "page_content"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                output.append(value)
    return output


def _collect_forbidden_key_errors(value: Any, errors: List[str], *, prefix: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_PAYLOAD_KEYS:
                errors.append(f"forbidden_key:{key_text}")
            _collect_forbidden_key_errors(nested, errors, prefix=f"{prefix}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_forbidden_key_errors(item, errors, prefix=f"{prefix}[{index}]")
