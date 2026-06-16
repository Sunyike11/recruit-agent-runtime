import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


ALLOWED_RELEVANCE = {0, 1, 2}
REQUIRED_ATTACK_TYPES = {
    "jd_as_resume",
    "prompt_injection",
    "keyword_stuffing",
    "duplicate_resume",
    "missing_name",
    "missing_education",
    "oversized_noisy_resume",
    "filename_injection",
    "same_name_candidates",
    "non_technical_keyword_camouflage",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
CN_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
ABS_PATH_RE = re.compile(r"(^/|[A-Za-z]:\\|/Users/|/home/|/var/|/tmp/)")


@dataclass
class RecruitmentJob:
    job_id: str
    title: str
    level: str
    department: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    education_requirement: str = ""
    experience_requirement: str = ""
    responsibilities: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    soft_preferences: List[str] = field(default_factory=list)
    jd_text: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecruitmentJob":
        return cls(
            job_id=str(data.get("job_id") or ""),
            title=str(data.get("title") or ""),
            level=str(data.get("level") or ""),
            department=str(data.get("department") or ""),
            required_skills=_strings(data.get("required_skills")),
            preferred_skills=_strings(data.get("preferred_skills")),
            education_requirement=str(data.get("education_requirement") or ""),
            experience_requirement=str(data.get("experience_requirement") or ""),
            responsibilities=_strings(data.get("responsibilities")),
            hard_constraints=_strings(data.get("hard_constraints")),
            soft_preferences=_strings(data.get("soft_preferences")),
            jd_text=str(data.get("jd_text") or ""),
            tags=_strings(data.get("tags")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RecruitmentCandidate:
    candidate_id: str
    display_name: str
    education: str = ""
    years_of_experience: int = 0
    skills: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    work_experience: List[str] = field(default_factory=list)
    research_experience: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    open_source: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    summary: str = ""
    resume_text: str = ""
    source_file_name: str = ""
    tags: List[str] = field(default_factory=list)
    is_special_case: bool = False
    special_case_type: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecruitmentCandidate":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            display_name=str(data.get("display_name") or ""),
            education=str(data.get("education") or ""),
            years_of_experience=int(data.get("years_of_experience") or 0),
            skills=_strings(data.get("skills")),
            projects=_strings(data.get("projects")),
            work_experience=_strings(data.get("work_experience")),
            research_experience=_strings(data.get("research_experience")),
            certifications=_strings(data.get("certifications")),
            open_source=_strings(data.get("open_source")),
            awards=_strings(data.get("awards")),
            summary=str(data.get("summary") or ""),
            resume_text=str(data.get("resume_text") or ""),
            source_file_name=str(data.get("source_file_name") or ""),
            tags=_strings(data.get("tags")),
            is_special_case=bool(data.get("is_special_case")),
            special_case_type=str(data.get("special_case_type") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RecruitmentRelevanceLabel:
    job_id: str
    candidate_relevance: Dict[str, int] = field(default_factory=dict)
    ideal_ranking: List[str] = field(default_factory=list)
    label_reason_codes: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecruitmentRelevanceLabel":
        return cls(
            job_id=str(data.get("job_id") or ""),
            candidate_relevance={str(k): int(v) for k, v in dict(data.get("candidate_relevance") or {}).items()},
            ideal_ranking=_strings(data.get("ideal_ranking")),
            label_reason_codes={
                str(k): _strings(v) for k, v in dict(data.get("label_reason_codes") or {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "candidate_relevance": dict(self.candidate_relevance),
            "ideal_ranking": list(self.ideal_ranking),
            "label_reason_codes": {key: list(value) for key, value in self.label_reason_codes.items()},
        }


@dataclass
class RecruitmentAttackCase:
    case_id: str
    candidate_id: str
    attack_type: str
    attack_text_present: bool
    expected_retrieval_behavior: str = ""
    expected_match_behavior: str = ""
    expected_security_flags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecruitmentAttackCase":
        return cls(
            case_id=str(data.get("case_id") or ""),
            candidate_id=str(data.get("candidate_id") or ""),
            attack_type=str(data.get("attack_type") or ""),
            attack_text_present=bool(data.get("attack_text_present")),
            expected_retrieval_behavior=str(data.get("expected_retrieval_behavior") or ""),
            expected_match_behavior=str(data.get("expected_match_behavior") or ""),
            expected_security_flags=_strings(data.get("expected_security_flags")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RecruitmentEvalDataset:
    dataset_dir: Path
    manifest: Dict[str, Any]
    jobs: List[RecruitmentJob]
    candidates: List[RecruitmentCandidate]
    relevance_labels: List[RecruitmentRelevanceLabel]
    attack_cases: List[RecruitmentAttackCase]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.manifest.get("dataset_name"),
            "dataset_version": self.manifest.get("dataset_version"),
            "job_count": len(self.jobs),
            "candidate_count": len(self.candidates),
            "special_case_count": len([candidate for candidate in self.candidates if candidate.is_special_case]),
            "attack_case_count": len(self.attack_cases),
            "summary_only": True,
        }


@dataclass
class DatasetValidationResult:
    status: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    job_count: int = 0
    candidate_count: int = 0
    special_case_count: int = 0
    attack_case_count: int = 0
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "job_count": self.job_count,
            "candidate_count": self.candidate_count,
            "special_case_count": self.special_case_count,
            "attack_case_count": self.attack_case_count,
            "summary_only": True,
        }


def load_recruitment_eval_dataset(dataset_dir: str | Path) -> RecruitmentEvalDataset:
    root = Path(dataset_dir)
    manifest = _read_json(root / "manifest.json")
    jobs = [RecruitmentJob.from_dict(item) for item in _read_json(root / "jobs.json")]
    candidates = [RecruitmentCandidate.from_dict(item) for item in _read_json(root / "candidates.json")]
    relevance_labels = [
        RecruitmentRelevanceLabel.from_dict(item) for item in _read_json(root / "relevance_labels.json")
    ]
    attack_cases = [RecruitmentAttackCase.from_dict(item) for item in _read_json(root / "attack_cases.json")]
    return RecruitmentEvalDataset(
        dataset_dir=root,
        manifest=dict(manifest),
        jobs=jobs,
        candidates=candidates,
        relevance_labels=relevance_labels,
        attack_cases=attack_cases,
    )


def validate_recruitment_eval_dataset(dataset: RecruitmentEvalDataset) -> DatasetValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    job_ids = [job.job_id for job in dataset.jobs]
    candidate_ids = [candidate.candidate_id for candidate in dataset.candidates]
    job_id_set = set(job_ids)
    candidate_id_set = set(candidate_ids)

    _require(len(job_ids) == len(job_id_set), "job_id_not_unique", errors)
    _require(len(candidate_ids) == len(candidate_id_set), "candidate_id_not_unique", errors)
    _require(all(re.match(r"^candidate_\d{3}$", cid) for cid in candidate_ids), "candidate_id_scheme_invalid", errors)

    for job in dataset.jobs:
        _require(bool(job.job_id and job.title and job.jd_text), f"job_required_fields_missing:{job.job_id}", errors)

    special_candidates = [candidate for candidate in dataset.candidates if candidate.is_special_case]
    special_types = {candidate.special_case_type for candidate in special_candidates}
    _require(len(special_candidates) >= 10, "special_case_count_below_10", errors)
    for attack_type in REQUIRED_ATTACK_TYPES:
        _require(attack_type in special_types, f"missing_special_case:{attack_type}", errors)

    display_names: Dict[str, int] = {}
    for candidate in dataset.candidates:
        display_names[candidate.display_name] = display_names.get(candidate.display_name, 0) + 1
        _require(bool(candidate.resume_text.strip()), f"resume_text_empty:{candidate.candidate_id}", errors)
        _require(not _is_absolute_or_pathlike(candidate.source_file_name), f"source_file_name_pathlike:{candidate.candidate_id}", errors)
        _require(not _contains_sensitive_pattern(candidate.resume_text), f"sensitive_pattern:{candidate.candidate_id}", errors)
    _require(any(count > 1 for count in display_names.values()), "same_name_candidate_missing", errors)

    label_job_ids = {label.job_id for label in dataset.relevance_labels}
    _require(label_job_ids == job_id_set, "relevance_job_set_mismatch", errors)
    for label in dataset.relevance_labels:
        relevance_ids = set(label.candidate_relevance.keys())
        _require(relevance_ids == candidate_id_set, f"relevance_candidate_set_mismatch:{label.job_id}", errors)
        _require(set(label.candidate_relevance.values()).issubset(ALLOWED_RELEVANCE), f"invalid_relevance:{label.job_id}", errors)
        high_count = sum(1 for value in label.candidate_relevance.values() if value == 2)
        partial_count = sum(1 for value in label.candidate_relevance.values() if value == 1)
        _require(high_count >= 3, f"high_relevance_below_3:{label.job_id}", errors)
        _require(partial_count >= 3, f"partial_relevance_below_3:{label.job_id}", errors)
        _validate_ideal_ranking(label, candidate_id_set, errors)

    attack_candidate_ids = {case.candidate_id for case in dataset.attack_cases}
    attack_types = {case.attack_type for case in dataset.attack_cases}
    _require(len(dataset.attack_cases) >= 10, "attack_case_count_below_10", errors)
    _require(attack_candidate_ids.issubset(candidate_id_set), "attack_case_candidate_missing", errors)
    for attack_type in REQUIRED_ATTACK_TYPES:
        _require(attack_type in attack_types, f"missing_attack_case:{attack_type}", errors)

    manifest = dataset.manifest
    _require(int(manifest.get("job_count") or -1) == len(dataset.jobs), "manifest_job_count_mismatch", errors)
    _require(
        int(manifest.get("candidate_count") or -1) == len(dataset.candidates),
        "manifest_candidate_count_mismatch",
        errors,
    )
    _require(
        int(manifest.get("special_case_count") or -1) == len(special_candidates),
        "manifest_special_case_count_mismatch",
        errors,
    )
    _require(bool(manifest.get("synthetic_data")), "manifest_synthetic_data_false", errors)
    _require(str(manifest.get("privacy_mode")) == "synthetic_anonymized", "manifest_privacy_mode_invalid", errors)

    readme = dataset.dataset_dir / "README.md"
    _require(readme.exists(), "readme_missing", errors)
    valid = not errors
    return DatasetValidationResult(
        status="passed" if valid else "failed",
        valid=valid,
        errors=errors,
        warnings=warnings,
        job_count=len(dataset.jobs),
        candidate_count=len(dataset.candidates),
        special_case_count=len(special_candidates),
        attack_case_count=len(dataset.attack_cases),
    )


def _validate_ideal_ranking(label: RecruitmentRelevanceLabel, candidate_ids: set[str], errors: List[str]) -> None:
    seen_partial = False
    ranked = set(label.ideal_ranking)
    positive = {cid for cid, relevance in label.candidate_relevance.items() if relevance > 0}
    _require(ranked == positive, f"ideal_ranking_positive_set_mismatch:{label.job_id}", errors)
    for candidate_id in label.ideal_ranking:
        _require(candidate_id in candidate_ids, f"ideal_ranking_candidate_missing:{label.job_id}", errors)
        relevance = label.candidate_relevance.get(candidate_id, 0)
        _require(relevance > 0, f"ideal_ranking_contains_zero:{label.job_id}", errors)
        if relevance == 1:
            seen_partial = True
        if relevance == 2 and seen_partial:
            _require(False, f"ideal_ranking_order_invalid:{label.job_id}", errors)


def _strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, error: str, errors: List[str]) -> None:
    if not condition:
        errors.append(error)


def _contains_sensitive_pattern(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or CN_ID_RE.search(text) or ABS_PATH_RE.search(text))


def _is_absolute_or_pathlike(file_name: str) -> bool:
    return bool(ABS_PATH_RE.search(file_name) or "/" in file_name or "\\" in file_name)
