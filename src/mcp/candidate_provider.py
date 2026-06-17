import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from src.evaluation.dataset import RecruitmentCandidate, load_recruitment_eval_dataset
from src.runtime.candidate_preview import (
    build_candidate_profile_preview_v2,
    candidate_profile_preview_v2_to_matcher_input,
)


SERVER_NAME = "candidate_mcp"
SERVER_VERSION = "1.0.0"
DEFAULT_ACCESS_SCOPE = "evaluation_data_v1"
ALLOWED_PROFILE_FIELDS = {
    "identity",
    "education",
    "experience",
    "projects",
    "skills",
    "skill_evidence",
    "achievements",
    "safety",
    "provenance",
}


class CandidateDataProvider(Protocol):
    def search_candidates(
        self,
        *,
        query: str,
        top_k: int,
        required_skills: Optional[Sequence[str]] = None,
        excluded_candidate_ids: Optional[Sequence[str]] = None,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        ...

    def get_candidate_profile(
        self,
        *,
        candidate_id: str,
        requested_fields: Optional[Sequence[str]] = None,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        ...

    def get_resume_evidence(
        self,
        *,
        candidate_id: str,
        evidence_ids: Optional[Sequence[str]] = None,
        field_names: Optional[Sequence[str]] = None,
        max_items: int = 10,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        ...


@dataclass
class EvaluationDatasetCandidateProvider:
    dataset_dir: str | Path = "evaluation_data/v1"
    max_top_k: int = 20
    max_query_chars: int = 1200
    _candidates: Dict[str, RecruitmentCandidate] = field(init=False, default_factory=dict)
    _attack_types: Dict[str, str] = field(init=False, default_factory=dict)
    _dataset_version: str = field(init=False, default="")

    def __post_init__(self) -> None:
        dataset = load_recruitment_eval_dataset(self.dataset_dir)
        self._candidates = {candidate.candidate_id: candidate for candidate in dataset.candidates}
        self._attack_types = {case.candidate_id: case.attack_type for case in dataset.attack_cases}
        self._dataset_version = str(dataset.manifest.get("dataset_version") or "")

    def search_candidates(
        self,
        *,
        query: str,
        top_k: int,
        required_skills: Optional[Sequence[str]] = None,
        excluded_candidate_ids: Optional[Sequence[str]] = None,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        _require_scope(tenant_id=tenant_id, access_scope=access_scope)
        query_text = _bounded_text(query, self.max_query_chars, "query")
        top_k = _bounded_int(top_k, 1, self.max_top_k, "top_k")
        required = _safe_strings(required_skills, limit=12)
        excluded = set(_safe_strings(excluded_candidate_ids, limit=100))
        query_terms = _tokenize(" ".join([query_text, " ".join(required)]))
        scored = []
        for candidate in self._candidates.values():
            if candidate.candidate_id in excluded:
                continue
            score, matched_skills = _score_candidate(candidate, query_terms, required)
            scored.append((score, candidate.candidate_id, candidate, matched_skills))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for rank, (score, _cid, candidate, matched_skills) in enumerate(scored[:top_k], start=1):
            evidence_ids = _candidate_evidence_ids(candidate)
            results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "rank": rank,
                    "retrieval_score": round(float(score), 6),
                    "profile_summary": _safe_excerpt(candidate.summary, 180),
                    "matched_skills": matched_skills,
                    "evidence_ids": evidence_ids[:6],
                    "source_document_id": candidate.source_file_name or candidate.candidate_id,
                    "special_case_flags": _special_case_flags(candidate),
                    "suspicious_instruction_present": _has_suspicious_instruction(candidate.resume_text),
                    "instruction_treated_as_data": _has_suspicious_instruction(candidate.resume_text),
                    "summary_only": True,
                }
            )
        return {
            "server_name": SERVER_NAME,
            "server_version": SERVER_VERSION,
            "read_only": True,
            "results": results,
            "result_count": len(results),
            "dataset_version": self._dataset_version,
            "request_id": str(request_id or ""),
            "summary_only": True,
        }

    def get_candidate_profile(
        self,
        *,
        candidate_id: str,
        requested_fields: Optional[Sequence[str]] = None,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        _require_scope(tenant_id=tenant_id, access_scope=access_scope)
        candidate = self._candidate(candidate_id)
        fields = _requested_fields(requested_fields)
        preview = build_candidate_profile_preview_v2(_candidate_to_preview_input(candidate)).to_dict()
        profile: Dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "profile_version": "candidate_profile_preview_v2",
            "source_document_version": self._dataset_version,
            "source_document_id": candidate.source_file_name or candidate.candidate_id,
            "summary_only": True,
        }
        if "identity" in fields:
            profile["identity"] = {
                "candidate_id": candidate.candidate_id,
                "display_name": candidate.display_name,
                "candidate_name_resolved": bool(preview.get("candidate_name_resolved")),
                "source_file_name": _safe_basename(candidate.source_file_name),
                "summary_only": True,
            }
        if "education" in fields:
            profile["education"] = {
                "highest_degree": preview.get("highest_degree", ""),
                "majors": list(preview.get("majors") or []),
                "institutions_summary": preview.get("institutions_summary", ""),
                "evidence_summaries": list(preview.get("education_evidence_summaries") or []),
                "summary_only": True,
            }
        if "experience" in fields:
            profile["experience"] = {
                "total_years": preview.get("total_years"),
                "roles": list(preview.get("roles") or []),
                "domains": list(preview.get("domains") or []),
                "evidence_summaries": list(preview.get("experience_evidence_summaries") or []),
                "summary_only": True,
            }
        if "projects" in fields:
            profile["projects"] = list(preview.get("projects") or [])
        if "skills" in fields:
            profile["skills"] = list(preview.get("skills") or [])
        if "skill_evidence" in fields:
            profile["skill_evidence"] = dict(preview.get("skill_evidence") or {})
        if "achievements" in fields:
            profile["achievements"] = dict(preview.get("achievements") or {})
        if "safety" in fields:
            profile["safety"] = {
                "suspicious_instruction_present": bool(preview.get("suspicious_instruction_present")),
                "job_description_like_content": bool(preview.get("job_description_like_content")),
                "keyword_stuffing_signal": bool(preview.get("keyword_stuffing_signal")),
                "filename_injection_signal": bool(preview.get("filename_injection_signal")),
                "invalid_resume_structure_signal": bool(preview.get("invalid_resume_structure_signal")),
                "special_case_type": self._attack_types.get(candidate.candidate_id, candidate.special_case_type),
                "instruction_treated_as_data": bool(preview.get("suspicious_instruction_present")),
                "summary_only": True,
            }
        if "provenance" in fields:
            profile["field_provenance"] = dict(preview.get("field_provenance") or {})
        profile["request_id"] = str(request_id or "")
        return profile

    def get_resume_evidence(
        self,
        *,
        candidate_id: str,
        evidence_ids: Optional[Sequence[str]] = None,
        field_names: Optional[Sequence[str]] = None,
        max_items: int = 10,
        tenant_id: str = "",
        access_scope: str = "",
        request_id: str = "",
    ) -> Dict[str, Any]:
        _require_scope(tenant_id=tenant_id, access_scope=access_scope)
        candidate = self._candidate(candidate_id)
        max_items = _bounded_int(max_items, 1, 20, "max_items")
        requested_ids = set(_safe_strings(evidence_ids, limit=40))
        requested_fields = set(_safe_strings(field_names, limit=12))
        evidence_items = _build_evidence_items(candidate, self._dataset_version)
        if requested_ids:
            evidence_items = [item for item in evidence_items if item["evidence_id"] in requested_ids]
        if requested_fields:
            evidence_items = [item for item in evidence_items if item["field_name"] in requested_fields]
        evidence_items = evidence_items[:max_items]
        return {
            "server_name": SERVER_NAME,
            "candidate_id": candidate.candidate_id,
            "evidence": evidence_items,
            "evidence_count": len(evidence_items),
            "source_document_id": candidate.source_file_name or candidate.candidate_id,
            "document_version": self._dataset_version,
            "request_id": str(request_id or ""),
            "summary_only": True,
        }

    def matcher_profile_for_candidate(self, candidate_id: str) -> Dict[str, Any]:
        candidate = self._candidate(candidate_id)
        preview = build_candidate_profile_preview_v2(_candidate_to_preview_input(candidate))
        return candidate_profile_preview_v2_to_matcher_input(preview)

    def _candidate(self, candidate_id: str) -> RecruitmentCandidate:
        key = str(candidate_id or "").strip()
        if key not in self._candidates:
            raise KeyError("candidate_not_found")
        return self._candidates[key]


def _candidate_to_preview_input(candidate: RecruitmentCandidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "display_name": candidate.display_name,
        "education": candidate.education,
        "years_of_experience": candidate.years_of_experience,
        "skills": list(candidate.skills),
        "projects": list(candidate.projects),
        "work_experience": list(candidate.work_experience),
        "research_experience": list(candidate.research_experience),
        "certifications": list(candidate.certifications),
        "open_source": list(candidate.open_source),
        "awards": list(candidate.awards),
        "summary": candidate.summary,
        "resume_text": candidate.resume_text,
        "source_file_name": candidate.source_file_name,
        "source_document_id": candidate.source_file_name or candidate.candidate_id,
        "metadata": {
            "candidate_id": candidate.candidate_id,
            "candidate_name": candidate.display_name,
            "source_document_id": candidate.source_file_name or candidate.candidate_id,
            "file_name": candidate.source_file_name,
        },
    }


def _build_evidence_items(candidate: RecruitmentCandidate, dataset_version: str) -> List[Dict[str, Any]]:
    source_document_id = candidate.source_file_name or candidate.candidate_id
    raw_items: List[tuple[str, str, str]] = []
    if candidate.education:
        raw_items.append(("education", "education", candidate.education))
    for index, value in enumerate(candidate.work_experience, start=1):
        raw_items.append((f"experience_{index}", "experience", value))
    for index, value in enumerate(candidate.projects, start=1):
        raw_items.append((f"project_{index}", "projects", value))
    for index, value in enumerate(candidate.research_experience, start=1):
        raw_items.append((f"research_{index}", "research", value))
    for index, value in enumerate(candidate.open_source, start=1):
        raw_items.append((f"open_source_{index}", "open_source", value))
    for index, value in enumerate(candidate.awards, start=1):
        raw_items.append((f"award_{index}", "awards", value))
    if candidate.summary:
        raw_items.append(("summary_1", "summary", candidate.summary))
    return [
        {
            "evidence_id": f"{candidate.candidate_id}:{local_id}",
            "evidence_type": field_name,
            "field_name": field_name,
            "summary": _safe_excerpt(text, 220),
            "source_document_id": source_document_id,
            "document_version": dataset_version,
            "provenance": {
                "candidate_id": candidate.candidate_id,
                "source_field": field_name,
                "source_document_id": source_document_id,
                "evidence_present": bool(text),
                "summary_only": True,
            },
            "summary_only": True,
        }
        for local_id, field_name, text in raw_items
        if text
    ]


def _candidate_evidence_ids(candidate: RecruitmentCandidate) -> List[str]:
    return [item["evidence_id"] for item in _build_evidence_items(candidate, "")]


def _score_candidate(
    candidate: RecruitmentCandidate,
    query_terms: Sequence[str],
    required_skills: Sequence[str],
) -> tuple[float, List[str]]:
    text = " ".join(
        [
            candidate.summary,
            " ".join(candidate.skills),
            " ".join(candidate.projects),
            " ".join(candidate.work_experience),
            " ".join(candidate.research_experience),
            " ".join(candidate.tags),
        ]
    )
    text_terms = set(_tokenize(text))
    matched_query = sum(1 for term in query_terms if term in text_terms or term.lower() in text.lower())
    matched_skills = [
        skill for skill in required_skills or candidate.skills if skill and skill.lower() in text.lower()
    ]
    score = matched_query + (2.0 * len(matched_skills)) + math.log1p(max(candidate.years_of_experience, 0))
    if candidate.is_special_case:
        score *= 0.72
    return score, _dedupe(matched_skills)[:8]


def _special_case_flags(candidate: RecruitmentCandidate) -> Dict[str, Any]:
    return {
        "is_special_case": bool(candidate.is_special_case),
        "special_case_type": str(candidate.special_case_type or ""),
        "summary_only": True,
    }


def _requested_fields(requested_fields: Optional[Sequence[str]]) -> set[str]:
    fields = set(_safe_strings(requested_fields, limit=16)) if requested_fields else set(ALLOWED_PROFILE_FIELDS)
    invalid = sorted(field for field in fields if field not in ALLOWED_PROFILE_FIELDS)
    if invalid:
        raise ValueError("invalid_requested_fields")
    return fields


def _require_scope(*, tenant_id: str, access_scope: str) -> None:
    if not str(tenant_id or "").strip() and not str(access_scope or "").strip():
        raise PermissionError("access_scope_required")


def _bounded_text(value: Any, max_chars: int, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name}_required")
    if len(text) > max_chars:
        raise ValueError(f"{field_name}_too_long")
    return text


def _bounded_int(value: Any, lower: int, upper: int, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}_invalid") from exc
    if number < lower or number > upper:
        raise ValueError(f"{field_name}_out_of_range")
    return number


def _safe_strings(values: Optional[Sequence[Any]], *, limit: int) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    output = []
    for value in list(values)[:limit]:
        text = str(value or "").strip()
        if text:
            output.append(text[:120])
    return output


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.]*|[\u4e00-\u9fff]{2,}", text or "")
        if len(token.strip()) >= 2
    ]


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _safe_excerpt(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    compact = _remove_sensitive_instruction(compact)
    return compact[:max_chars]


def _remove_sensitive_instruction(text: str) -> str:
    if _has_suspicious_instruction(text):
        return "[suspicious instruction treated as data]"
    return text


def _has_suspicious_instruction(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in [
            "忽略之前",
            "忽略岗位要求",
            "给我满分",
            "total_score",
            "outstanding",
            "管理员权限",
            "调用其他工具",
        ]
    )


def _safe_basename(value: Any) -> str:
    return Path(str(value or "")).name

