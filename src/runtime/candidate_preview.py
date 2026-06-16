import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SAFE_SKILL_KEYWORDS = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "Redis",
    "Docker",
    "Linux",
    "RAG",
    "LangGraph",
    "LangChain",
    "LlamaIndex",
    "Chroma",
    "Milvus",
    "FastAPI",
    "Flask",
    "Vue",
    "React",
    "PyTorch",
    "TensorFlow",
    "DeepSeek",
    "GLM",
    "Agent",
    "MCP",
    "Git",
]

PROJECT_KEYWORDS = ["项目", "系统", "平台", "Agent", "RAG", "推荐", "检索", "匹配", "自动化", "评估", "部署", "微服务", "高并发"]
EDUCATION_KEYWORDS = ["本科", "硕士", "研究生", "计算机", "通信", "信息", "软件", "人工智能"]
EXPERIENCE_KEYWORDS = ["实习", "项目", "开源", "论文", "比赛", "部署", "测试", "压测", "工程"]
PLACEHOLDER_NAME_TOKENS = {
    "",
    "简历",
    "个人",
    "个人简历",
    "我的",
    "我的简历",
    "第一份",
    "第二份",
    "第三份",
    "候选人",
    "未知",
    "未提供",
    "无名",
    "resume",
    "cv",
}


@dataclass
class CandidateProfilePreview:
    candidate_id: str
    candidate_name: str
    source_document_id: str
    source_file_name: str
    skills: List[str] = field(default_factory=list)
    project_keywords: List[str] = field(default_factory=list)
    education_keywords: List[str] = field(default_factory=list)
    experience_keywords: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    evidence_chunk_count: int = 0
    matched_query_terms: List[str] = field(default_factory=list)
    preview_quality_flags: List[str] = field(default_factory=list)
    evidence_text_length: int = 0
    evidence_metadata_keys: List[str] = field(default_factory=list)
    score_present: bool = False
    source: str = "retrieval_preview"
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "source_document_id": self.source_document_id,
            "source_file_name": self.source_file_name,
            "skills": list(self.skills),
            "project_keywords": list(self.project_keywords),
            "education_keywords": list(self.education_keywords),
            "experience_keywords": list(self.experience_keywords),
            "evidence_summary": self.evidence_summary,
            "evidence_chunk_count": int(self.evidence_chunk_count),
            "matched_query_terms": list(self.matched_query_terms),
            "preview_quality_flags": list(self.preview_quality_flags),
            "evidence_text_length": int(self.evidence_text_length),
            "evidence_metadata_keys": list(self.evidence_metadata_keys),
            "score_present": bool(self.score_present),
            "source": self.source,
            "summary_only": True,
        }


@dataclass
class CandidatePreviewBuildConfig:
    max_skills: int = 12
    max_project_keywords: int = 8
    max_education_keywords: int = 6
    max_experience_keywords: int = 6
    max_evidence_chars: int = 320
    max_matched_query_terms: int = 10
    redact_full_text: bool = True
    summary_only: bool = True


@dataclass
class CandidatePreviewBuildResult:
    previews: List[CandidateProfilePreview] = field(default_factory=list)
    candidate_profile_preview_count: int = 0
    grouped_document_count: int = 0
    skipped_chunk_count: int = 0
    quality_summary: Dict[str, Any] = field(default_factory=dict)
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previews": [preview.to_dict() for preview in self.previews],
            "candidate_profile_preview_count": int(self.candidate_profile_preview_count),
            "grouped_document_count": int(self.grouped_document_count),
            "skipped_chunk_count": int(self.skipped_chunk_count),
            "quality_summary": dict(self.quality_summary),
            "summary_only": True,
        }


@dataclass
class CandidateProfilePreviewV2:
    candidate_id: str
    candidate_name: str = ""
    candidate_name_resolved: bool = False
    source_document_id: str = ""
    source_file_name: str = ""
    highest_degree: str = ""
    majors: List[str] = field(default_factory=list)
    institutions_summary: str = ""
    graduation_status: str = ""
    education_evidence_summaries: List[str] = field(default_factory=list)
    total_years: Optional[int] = None
    roles: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    companies_summary: str = ""
    responsibility_summaries: List[str] = field(default_factory=list)
    experience_evidence_summaries: List[str] = field(default_factory=list)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    skill_evidence: Dict[str, List[str]] = field(default_factory=dict)
    achievements: Dict[str, List[str]] = field(default_factory=dict)
    suspicious_instruction_present: bool = False
    job_description_like_content: bool = False
    keyword_stuffing_signal: bool = False
    filename_injection_signal: bool = False
    invalid_resume_structure_signal: bool = False
    source_text_length: int = 0
    preview_rendered_length: int = 0
    compression_ratio: float = 0.0
    truncated_fields: List[str] = field(default_factory=list)
    field_counts: Dict[str, int] = field(default_factory=dict)
    field_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preview_version: str = "v2"
    source: str = "candidate_profile_preview_v2"
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "candidate_name_resolved": bool(self.candidate_name_resolved),
            "source_document_id": self.source_document_id,
            "source_file_name": self.source_file_name,
            "highest_degree": self.highest_degree,
            "majors": list(self.majors),
            "institutions_summary": self.institutions_summary,
            "graduation_status": self.graduation_status,
            "education_evidence_summaries": list(self.education_evidence_summaries),
            "total_years": self.total_years,
            "roles": list(self.roles),
            "domains": list(self.domains),
            "companies_summary": self.companies_summary,
            "responsibility_summaries": list(self.responsibility_summaries),
            "experience_evidence_summaries": list(self.experience_evidence_summaries),
            "projects": [dict(project) for project in self.projects],
            "skills": list(self.skills),
            "skill_evidence": {key: list(value) for key, value in self.skill_evidence.items()},
            "achievements": {key: list(value) for key, value in self.achievements.items()},
            "suspicious_instruction_present": bool(self.suspicious_instruction_present),
            "job_description_like_content": bool(self.job_description_like_content),
            "keyword_stuffing_signal": bool(self.keyword_stuffing_signal),
            "filename_injection_signal": bool(self.filename_injection_signal),
            "invalid_resume_structure_signal": bool(self.invalid_resume_structure_signal),
            "source_text_length": int(self.source_text_length),
            "preview_rendered_length": int(self.preview_rendered_length),
            "compression_ratio": float(self.compression_ratio),
            "truncated_fields": list(self.truncated_fields),
            "field_counts": dict(self.field_counts),
            "field_provenance": {key: dict(value) for key, value in self.field_provenance.items()},
            "preview_version": self.preview_version,
            "source": self.source,
            "summary_only": True,
        }


@dataclass
class CandidatePreviewV2BuildConfig:
    max_projects: int = 4
    max_project_chars: int = 220
    max_evidence_chars: int = 180
    max_skill_evidence_items: int = 2
    max_skills: int = 16
    max_achievements_per_type: int = 4
    max_preview_chars: int = 2400
    summary_only: bool = True


def normalize_retrieval_chunk(chunk: Any, rank: int = 0) -> Dict[str, Any]:
    """Normalize one retrieval chunk into a summary-only, grouping-friendly shape."""

    if not isinstance(chunk, Mapping):
        return {"valid": False, "summary_only": True}
    metadata = dict(chunk.get("metadata") or {})
    text = _chunk_text(chunk)
    file_name = _safe_basename(
        chunk.get("file_name") or metadata.get("file_name") or metadata.get("source") or chunk.get("source")
    )
    source_document_id = _safe_basename(
        chunk.get("source_document_id")
        or chunk.get("document_id")
        or metadata.get("document_id")
        or metadata.get("source_document_id")
        or file_name
        or chunk.get("id")
    )
    candidate_id = _safe_identifier(metadata.get("candidate_id") or chunk.get("candidate_id"))
    candidate_name = _candidate_name_from_metadata_or_file(metadata, file_name)
    metadata_keys = sorted(str(key) for key in metadata.keys())
    if not metadata_keys:
        metadata_keys = sorted(str(key) for key in chunk.get("metadata_keys") or [])
    score = chunk.get("score")
    excerpt = _safe_excerpt(text)
    return {
        "valid": True,
        "rank": int(rank),
        "chunk_id": _safe_identifier(chunk.get("id")) or f"chunk_{rank}",
        "group_key": _group_key(candidate_id, source_document_id, file_name, chunk.get("id"), rank),
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "source_document_id": source_document_id,
        "source_file_name": file_name,
        "text_length": len(text) if text else _safe_int(chunk.get("text_length")),
        "metadata_keys": metadata_keys,
        "score_present": isinstance(score, (int, float)) or bool(chunk.get("score_present")),
        "skills": _filter_safe_skills(_strings(chunk.get("skills")) + _extract_keywords(text, SAFE_SKILL_KEYWORDS)),
        "project_keywords": _extract_keywords(text, PROJECT_KEYWORDS),
        "education_keywords": _extract_keywords(text, EDUCATION_KEYWORDS),
        "experience_keywords": _extract_keywords(text, EXPERIENCE_KEYWORDS),
        "text_excerpt": excerpt,
        "text_truncated": bool(text and len(text) > len(excerpt)),
        "summary_only": True,
    }


def group_retrieval_chunks_by_candidate(chunks: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for index, chunk in enumerate(chunks):
        normalized = normalize_retrieval_chunk(chunk, rank=index + 1)
        if not normalized.get("valid"):
            continue
        grouped.setdefault(str(normalized["group_key"]), []).append(normalized)
    return grouped


def build_candidate_profile_previews_from_retrieval_results(
    retrieval_results: Any,
    *,
    raw_jd: str = "",
    query: str = "",
    config: Optional[CandidatePreviewBuildConfig] = None,
) -> CandidatePreviewBuildResult:
    cfg = config or CandidatePreviewBuildConfig()
    chunks = _coerce_retrieval_chunks(retrieval_results)
    grouped = group_retrieval_chunks_by_candidate(chunks)
    previews = [
        _build_preview_for_group(group_chunks, raw_jd=raw_jd, query=query, config=cfg)
        for group_chunks in grouped.values()
        if group_chunks
    ]
    quality_summary = build_candidate_preview_quality_audit(previews)
    return CandidatePreviewBuildResult(
        previews=previews,
        candidate_profile_preview_count=len(previews),
        grouped_document_count=len(grouped),
        skipped_chunk_count=max(0, len(chunks) - sum(len(items) for items in grouped.values())),
        quality_summary=quality_summary,
        summary_only=True,
    )


def candidate_profile_preview_to_matcher_input(preview: CandidateProfilePreview | Mapping[str, Any]) -> Dict[str, Any]:
    data = preview.to_dict() if isinstance(preview, CandidateProfilePreview) else dict(preview)
    evidence_text = str(data.get("evidence_summary") or "")
    metadata_keys = _strings(data.get("evidence_metadata_keys"))
    source_document_id = _safe_basename(data.get("source_document_id"))
    education_keywords = _strings(data.get("education_keywords"))
    candidate_name = _clean_candidate_name(data.get("candidate_name") or data.get("name") or "")
    candidate_name_resolved = _is_resolved_candidate_name(candidate_name)
    if not candidate_name_resolved:
        candidate_name = ""
    return {
        "candidate_id": str(data.get("candidate_id") or ""),
        "name": candidate_name,
        "candidate_name": candidate_name,
        "candidate_name_resolved": candidate_name_resolved,
        "skills": _strings(data.get("skills")),
        "projects": _strings(data.get("project_keywords")),
        "project_keywords": _strings(data.get("project_keywords")),
        "education": " ".join(education_keywords),
        "education_keywords": education_keywords,
        "experience": _strings(data.get("experience_keywords")),
        "experience_keywords": _strings(data.get("experience_keywords")),
        "evidence_text_summary": evidence_text,
        "evidence_summary": {
            "text_length": _safe_int(data.get("evidence_text_length")),
            "metadata_keys": metadata_keys,
            "score_present": bool(data.get("score_present")),
            "snippet_present": bool(evidence_text),
            "summary_only": True,
        },
        "evidence_chunk_count": _safe_int(data.get("evidence_chunk_count")),
        "matched_query_terms": _strings(data.get("matched_query_terms")),
        "preview_quality_flags": _strings(data.get("preview_quality_flags")),
        "source_document_id": source_document_id,
        "source_file_name": _safe_basename(data.get("source_file_name")),
        "candidate_profile_preview": True,
        "summary_only": True,
        "metadata": {
            "candidate_profile_preview": True,
            "source": "document_chunk_projection",
            "enhanced_candidate_preview": True,
            "summary_only": True,
        },
    }


def build_candidate_profile_preview_v2(
    candidate: Mapping[str, Any],
    *,
    raw_jd: str = "",
    config: Optional[CandidatePreviewV2BuildConfig] = None,
) -> CandidateProfilePreviewV2:
    cfg = config or CandidatePreviewV2BuildConfig()
    metadata = dict(candidate.get("metadata") or {})
    candidate_id = str(candidate.get("candidate_id") or metadata.get("candidate_id") or "")
    source_file_name = _safe_basename(candidate.get("source_file_name") or metadata.get("file_name") or "")
    source_document_id = _safe_basename(
        candidate.get("source_document_id") or metadata.get("source_document_id") or candidate_id or source_file_name
    )
    candidate_name = _clean_candidate_name(candidate.get("display_name") or metadata.get("candidate_name") or metadata.get("name") or "")
    if not _is_resolved_candidate_name(candidate_name):
        candidate_name = _candidate_name_from_metadata_or_file(metadata, source_file_name)
    candidate_name_resolved = _is_resolved_candidate_name(candidate_name)
    if not candidate_name_resolved:
        candidate_name = ""

    resume_text = str(candidate.get("resume_text") or candidate.get("text") or candidate.get("content") or "")
    source_text_length = len(resume_text)
    clean_lines = _resume_lines_without_attack_text(resume_text)
    clean_text = "\n".join(clean_lines)
    suspicious_instruction = _has_suspicious_instruction(resume_text)
    jd_like = _is_job_description_like(resume_text)
    keyword_stuffing = _has_keyword_stuffing(resume_text)
    filename_injection = _has_filename_injection(source_file_name)
    invalid_structure = jd_like or (not clean_text.strip()) or (not _strings(candidate.get("projects")) and "项目" not in clean_text)

    skills = _dedupe_limited(_strings(candidate.get("skills")) + _extract_keywords(clean_text, SAFE_SKILL_KEYWORDS), cfg.max_skills)
    projects, project_truncated = _extract_project_summaries(candidate, clean_text, source_document_id, cfg)
    education_evidence = _extract_evidence_lines(
        [str(candidate.get("education") or "")] + clean_lines,
        ["本科", "硕士", "博士", "研究生", "计算机", "软件", "人工智能", "信息"],
        cfg.max_evidence_chars,
        limit=3,
    )
    experience_evidence = _extract_evidence_lines(
        _strings(candidate.get("work_experience")) + clean_lines,
        ["工作", "实习", "经历", "负责", "开发", "上线", "部署"],
        cfg.max_evidence_chars,
        limit=4,
    )
    responsibility_summaries = _extract_evidence_lines(
        _strings(candidate.get("work_experience")) + _strings(candidate.get("projects")) + clean_lines,
        ["负责", "设计", "开发", "构建", "上线", "指标", "复盘", "测试"],
        cfg.max_evidence_chars,
        limit=4,
    )
    skill_evidence = _build_skill_evidence(skills, clean_lines + _strings(candidate.get("projects")) + _strings(candidate.get("work_experience")), cfg)
    achievements = _extract_achievements(candidate, clean_lines, cfg)
    majors = _extract_keywords(str(candidate.get("education") or "") + " " + clean_text, ["计算机", "软件", "人工智能", "信息", "通信", "数学", "自动化"])
    highest_degree = _highest_degree(str(candidate.get("education") or "") + " " + clean_text)
    roles = _extract_keywords(clean_text, ["后端工程师", "算法工程师", "测试开发", "平台工程师", "前端工程师", "实习生", "研究员", "工程师"])
    domains = _extract_keywords(clean_text + " " + raw_jd, PROJECT_KEYWORDS + ["视觉", "多模态", "DevOps", "运维", "前端"])
    total_years = _safe_int(candidate.get("years_of_experience")) or _extract_years(clean_text)
    field_provenance = {
        key: _provenance(key, source_document_id, bool(value))
        for key, value in {
            "identity": candidate_id or candidate_name,
            "education": education_evidence,
            "experience": experience_evidence,
            "projects": projects,
            "skills": skills,
            "achievements": sum((list(v) for v in achievements.values()), []),
            "safety": suspicious_instruction or jd_like or keyword_stuffing or filename_injection,
        }.items()
    }
    truncated_fields = []
    if project_truncated:
        truncated_fields.append("projects")
    rendered_length = _preview_visible_length(
        skills=skills,
        education_evidence=education_evidence,
        experience_evidence=experience_evidence,
        projects=projects,
        skill_evidence=skill_evidence,
        achievements=achievements,
    )
    if rendered_length > cfg.max_preview_chars:
        truncated_fields.append("preview")
        rendered_length = cfg.max_preview_chars
    if source_text_length and rendered_length >= source_text_length:
        truncated_fields.append("preview")
        rendered_length = max(0, source_text_length - 1)
    compression_ratio = round(rendered_length / source_text_length, 6) if source_text_length else 0.0
    field_counts = {
        "skills": len(skills),
        "projects": len(projects),
        "education_evidence": len(education_evidence),
        "experience_evidence": len(experience_evidence),
        "achievements": sum(len(value) for value in achievements.values()),
    }
    return CandidateProfilePreviewV2(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        candidate_name_resolved=candidate_name_resolved,
        source_document_id=source_document_id,
        source_file_name=source_file_name,
        highest_degree=highest_degree,
        majors=majors,
        institutions_summary=_safe_excerpt(str(candidate.get("education") or ""), max_chars=80),
        graduation_status=_extract_graduation_status(clean_text),
        education_evidence_summaries=education_evidence,
        total_years=total_years if total_years else None,
        roles=roles,
        domains=domains,
        companies_summary=_companies_summary(_strings(candidate.get("work_experience"))),
        responsibility_summaries=responsibility_summaries,
        experience_evidence_summaries=experience_evidence,
        projects=projects,
        skills=skills,
        skill_evidence=skill_evidence,
        achievements=achievements,
        suspicious_instruction_present=suspicious_instruction,
        job_description_like_content=jd_like,
        keyword_stuffing_signal=keyword_stuffing,
        filename_injection_signal=filename_injection,
        invalid_resume_structure_signal=invalid_structure,
        source_text_length=source_text_length,
        preview_rendered_length=rendered_length,
        compression_ratio=compression_ratio,
        truncated_fields=truncated_fields,
        field_counts=field_counts,
        field_provenance=field_provenance,
        summary_only=True,
    )


def candidate_profile_preview_v2_to_matcher_input(preview: CandidateProfilePreviewV2 | Mapping[str, Any]) -> Dict[str, Any]:
    data = preview.to_dict() if isinstance(preview, CandidateProfilePreviewV2) else dict(preview)
    projects = [dict(item) for item in data.get("projects") or [] if isinstance(item, Mapping)]
    project_texts = [
        "；".join(
            str(value)
            for value in [
                project.get("project_name"),
                ", ".join(_strings(project.get("technologies"))),
                project.get("task"),
                project.get("candidate_contribution"),
                project.get("result"),
                project.get("evidence_summary"),
            ]
            if value
        )
        for project in projects
    ]
    return {
        "candidate_id": str(data.get("candidate_id") or ""),
        "name": str(data.get("candidate_name") or ""),
        "candidate_name": str(data.get("candidate_name") or ""),
        "candidate_name_resolved": bool(data.get("candidate_name_resolved")),
        "skills": _strings(data.get("skills")),
        "skill_evidence": dict(data.get("skill_evidence") or {}),
        "education": "；".join(
            item
            for item in [
                str(data.get("highest_degree") or ""),
                "、".join(_strings(data.get("majors"))),
                "；".join(_strings(data.get("education_evidence_summaries"))),
            ]
            if item
        ),
        "education_evidence": _strings(data.get("education_evidence_summaries")),
        "experience": _strings(data.get("experience_evidence_summaries")) + _strings(data.get("responsibility_summaries")),
        "projects": project_texts,
        "project_evidence": project_texts,
        "achievements": dict(data.get("achievements") or {}),
        "safety_signals": {
            "suspicious_instruction_present": bool(data.get("suspicious_instruction_present")),
            "job_description_like_content": bool(data.get("job_description_like_content")),
            "keyword_stuffing_signal": bool(data.get("keyword_stuffing_signal")),
            "filename_injection_signal": bool(data.get("filename_injection_signal")),
            "invalid_resume_structure_signal": bool(data.get("invalid_resume_structure_signal")),
            "summary_only": True,
        },
        "source_document_id": _safe_basename(data.get("source_document_id")),
        "source_file_name": _safe_basename(data.get("source_file_name")),
        "field_provenance": dict(data.get("field_provenance") or {}),
        "preview_version": "v2",
        "candidate_profile_preview": True,
        "summary_only": True,
        "metadata": {
            "candidate_profile_preview": True,
            "source": "candidate_profile_preview_v2",
            "preview_version": "v2",
            "summary_only": True,
        },
    }


def build_candidate_preview_quality_audit(previews: Sequence[CandidateProfilePreview | Mapping[str, Any]]) -> Dict[str, Any]:
    items = [preview.to_dict() if isinstance(preview, CandidateProfilePreview) else dict(preview) for preview in previews]
    names = [_clean_candidate_name(item.get("candidate_name") or item.get("name") or "") for item in items]
    resolved_names = [name for name in names if _is_resolved_candidate_name(name)]
    placeholder_names = [name for name in names if name and not _is_resolved_candidate_name(name)]
    return {
        "candidate_profile_preview_count": len(items),
        "candidate_id_present": sum(1 for item in items if bool(item.get("candidate_id"))),
        "candidate_name_present": len(resolved_names),
        "candidate_name_present_count": len(resolved_names),
        "candidate_name_field_present": sum(1 for item in items if "candidate_name" in item or "name" in item),
        "candidate_name_resolved_count": len(resolved_names),
        "candidate_name_placeholder_count": len(placeholder_names),
        "skills_count": sum(len(_strings(item.get("skills"))) for item in items),
        "skills_present_count": sum(1 for item in items if _strings(item.get("skills"))),
        "project_keywords_present_count": sum(1 for item in items if _strings(item.get("project_keywords") or item.get("projects"))),
        "education_keywords_present_count": sum(1 for item in items if _strings(item.get("education_keywords") or item.get("education"))),
        "experience_keywords_present_count": sum(1 for item in items if _strings(item.get("experience_keywords") or item.get("experience"))),
        "evidence_summary_present": sum(1 for item in items if bool(item.get("evidence_summary"))),
        "evidence_summary_present_count": sum(1 for item in items if bool(item.get("evidence_summary") or item.get("evidence_text_summary"))),
        "source_document_id_present": sum(1 for item in items if bool(item.get("source_document_id"))),
        "source_document_id_present_count": sum(1 for item in items if bool(item.get("source_document_id"))),
        "low_evidence_chunk_count": sum(
            1
            for item in items
            if "low_evidence_chunk_count" in _strings(item.get("preview_quality_flags"))
        ),
        "summary_only": True,
    }


def _build_preview_for_group(
    chunks: List[Dict[str, Any]],
    *,
    raw_jd: str,
    query: str,
    config: CandidatePreviewBuildConfig,
) -> CandidateProfilePreview:
    first = chunks[0]
    source_document_id = _first_present(chunks, "source_document_id")
    source_file_name = _first_present(chunks, "source_file_name")
    candidate_name = _first_present(chunks, "candidate_name")
    candidate_id = _first_present(chunks, "candidate_id")
    if not candidate_id:
        candidate_id = _stable_candidate_id(source_document_id or source_file_name or first.get("group_key") or "candidate")
    skills = _dedupe_limited(_flatten(chunks, "skills"), config.max_skills)
    project_keywords = _dedupe_limited(_flatten(chunks, "project_keywords"), config.max_project_keywords)
    education_keywords = _dedupe_limited(_flatten(chunks, "education_keywords"), config.max_education_keywords)
    experience_keywords = _dedupe_limited(_flatten(chunks, "experience_keywords"), config.max_experience_keywords)
    evidence_summary, truncated = _build_evidence_summary(chunks, max_chars=config.max_evidence_chars)
    matched_terms = _matched_query_terms(
        " ".join([raw_jd or "", query or ""]),
        chunks,
        max_terms=config.max_matched_query_terms,
    )
    flags = _quality_flags(
        candidate_name=candidate_name,
        skills=skills,
        project_keywords=project_keywords,
        education_keywords=education_keywords,
        source_document_id=source_document_id,
        evidence_chunk_count=len(chunks),
        summary_truncated=truncated or any(bool(chunk.get("text_truncated")) for chunk in chunks),
    )
    return CandidateProfilePreview(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        source_document_id=source_document_id,
        source_file_name=source_file_name,
        skills=skills,
        project_keywords=project_keywords,
        education_keywords=education_keywords,
        experience_keywords=experience_keywords,
        evidence_summary=evidence_summary,
        evidence_chunk_count=len(chunks),
        matched_query_terms=matched_terms,
        preview_quality_flags=flags,
        evidence_text_length=sum(_safe_int(chunk.get("text_length")) for chunk in chunks),
        evidence_metadata_keys=sorted(set(_flatten(chunks, "metadata_keys"))),
        score_present=any(bool(chunk.get("score_present")) for chunk in chunks),
        summary_only=True,
    )


def _coerce_retrieval_chunks(value: Any) -> List[Any]:
    if isinstance(value, Mapping):
        documents = list(value.get("resume_documents") or [])
        evidence = list(value.get("evidence") or [])
        if documents:
            return [_merge_document_and_evidence(document, evidence[index] if index < len(evidence) else {}) for index, document in enumerate(documents)]
        if evidence:
            return evidence
        candidates = value.get("candidates")
        if isinstance(candidates, list):
            return candidates
        results = value.get("results") or value.get("matches")
        if isinstance(results, list):
            return results
        return []
    if isinstance(value, list):
        return list(value)
    return []


def _resume_lines_without_attack_text(text: str) -> List[str]:
    lines = [line.strip() for line in re.split(r"[\n。；;]+", text or "") if line.strip()]
    return [line for line in lines if not _has_suspicious_instruction(line)]


def _has_suspicious_instruction(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in [
            "忽略岗位要求",
            "忽略之前",
            "total_score",
            "outstanding",
            "给我满分",
            "设置为 100",
            "设置为100",
        ]
    )


def _is_job_description_like(text: str) -> bool:
    compact = text or ""
    return ("岗位职责" in compact or "任职要求" in compact) and not any(
        token in compact for token in ["教育", "项目", "工作经历", "实习", "毕业"]
    )


def _has_keyword_stuffing(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.]*|[\u4e00-\u9fff]{2,}", text or "")
    if not tokens:
        return False
    counts: Dict[str, int] = {}
    for token in tokens:
        key = token.lower()
        counts[key] = counts.get(key, 0) + 1
    return any(count >= 6 for count in counts.values())


def _has_filename_injection(file_name: str) -> bool:
    lowered = (file_name or "").lower()
    return any(token in lowered for token in ["满分", "必须录用", "100分", "100_score", "hire"])


def _extract_project_summaries(
    candidate: Mapping[str, Any],
    clean_text: str,
    source_document_id: str,
    config: CandidatePreviewV2BuildConfig,
) -> tuple[List[Dict[str, Any]], bool]:
    raw_projects = _strings(candidate.get("projects"))
    if not raw_projects:
        raw_projects = _extract_evidence_lines(clean_text.splitlines() or re.split(r"[。；;]", clean_text), ["项目", "系统", "平台"], config.max_project_chars, limit=config.max_projects)
    projects = []
    truncated = False
    for index, raw in enumerate(raw_projects[: config.max_projects], start=1):
        text = str(raw)
        safe_text = _truncate(text, config.max_project_chars)
        truncated = truncated or len(text) > len(safe_text)
        projects.append(
            {
                "project_name": _infer_project_name(safe_text, index),
                "technologies": _extract_keywords(safe_text, SAFE_SKILL_KEYWORDS),
                "task": _extract_phrase(safe_text, ["问题", "任务", "需求", "构建", "负责"]),
                "candidate_contribution": _extract_phrase(safe_text, ["负责", "设计", "开发", "实现", "搭建"]),
                "result": _extract_phrase(safe_text, ["提升", "降低", "上线", "指标", "论文", "部署"]),
                "evidence_summary": safe_text,
                "provenance": _provenance("projects", source_document_id, bool(safe_text)),
                "summary_only": True,
            }
        )
    return projects, truncated or len(raw_projects) > len(projects)


def _extract_evidence_lines(lines: Sequence[str], keywords: Sequence[str], max_chars: int, *, limit: int) -> List[str]:
    output = []
    for line in lines:
        text = str(line).strip()
        if not text or _has_suspicious_instruction(text):
            continue
        if any(keyword.lower() in text.lower() for keyword in keywords):
            output.append(_truncate(text, max_chars))
        if len(output) >= limit:
            break
    return output


def _build_skill_evidence(skills: Sequence[str], lines: Sequence[str], config: CandidatePreviewV2BuildConfig) -> Dict[str, List[str]]:
    evidence: Dict[str, List[str]] = {}
    for skill in skills:
        matched = []
        for line in lines:
            text = str(line)
            if _has_suspicious_instruction(text):
                continue
            if skill.lower() in text.lower():
                matched.append(_truncate(text, config.max_evidence_chars))
            if len(matched) >= config.max_skill_evidence_items:
                break
        if matched:
            evidence[skill] = matched
    return evidence


def _extract_achievements(candidate: Mapping[str, Any], lines: Sequence[str], config: CandidatePreviewV2BuildConfig) -> Dict[str, List[str]]:
    buckets = {
        "research_publications": _strings(candidate.get("research_experience")),
        "open_source": _strings(candidate.get("open_source")),
        "awards": _strings(candidate.get("awards")),
        "certifications": _strings(candidate.get("certifications")),
        "internships": [],
    }
    buckets["research_publications"].extend(_extract_evidence_lines(lines, ["论文", "顶会", "发表", "CVPR", "ICCV"], config.max_evidence_chars, limit=2))
    buckets["open_source"].extend(_extract_evidence_lines(lines, ["开源", "GitHub"], config.max_evidence_chars, limit=2))
    buckets["awards"].extend(_extract_evidence_lines(lines, ["奖", "比赛", "竞赛"], config.max_evidence_chars, limit=2))
    buckets["internships"].extend(_extract_evidence_lines(lines, ["实习"], config.max_evidence_chars, limit=2))
    return {
        key: _dedupe_limited((_truncate(item, config.max_evidence_chars) for item in values), config.max_achievements_per_type)
        for key, values in buckets.items()
        if values
    }


def _preview_visible_length(
    *,
    skills: Sequence[str],
    education_evidence: Sequence[str],
    experience_evidence: Sequence[str],
    projects: Sequence[Mapping[str, Any]],
    skill_evidence: Mapping[str, Sequence[str]],
    achievements: Mapping[str, Sequence[str]],
) -> int:
    parts: List[str] = []
    parts.extend(skills)
    parts.extend(education_evidence)
    parts.extend(experience_evidence)
    for project in projects:
        parts.append(str(project.get("evidence_summary") or ""))
    for values in skill_evidence.values():
        parts.extend(str(value) for value in values)
    for values in achievements.values():
        parts.extend(str(value) for value in values)
    return len(" ".join(part for part in parts if part))


def _highest_degree(text: str) -> str:
    for degree in ["博士", "硕士", "研究生", "本科", "大专"]:
        if degree in (text or ""):
            return "硕士" if degree == "研究生" else degree
    return ""


def _extract_years(text: str) -> int:
    match = re.search(r"(\d{1,2})\s*年", text or "")
    return int(match.group(1)) if match else 0


def _extract_graduation_status(text: str) -> str:
    match = re.search(r"(20\d{2})\s*年", text or "")
    if match:
        return f"{match.group(1)}年"
    if "在读" in (text or ""):
        return "在读"
    return ""


def _companies_summary(work_experience: Sequence[str]) -> str:
    if not work_experience:
        return ""
    return f"experience_item_count={len(work_experience)}"


def _provenance(source_field: str, source_document_id: str, evidence_present: bool) -> Dict[str, Any]:
    return {
        "source_field": source_field,
        "source_document_id": _safe_basename(source_document_id),
        "evidence_present": bool(evidence_present),
        "summary_only": True,
    }


def _infer_project_name(text: str, index: int) -> str:
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]+(?:系统|平台|项目|Agent|服务|工具))", text or "")
    return match.group(1)[:40] if match else f"project_{index}"


def _extract_phrase(text: str, keywords: Sequence[str]) -> str:
    if not text:
        return ""
    if any(keyword in text for keyword in keywords):
        return _truncate(text, 96)
    return ""


def _truncate(text: Any, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def _merge_document_and_evidence(document: Any, evidence: Any) -> Dict[str, Any]:
    merged = dict(document) if isinstance(document, Mapping) else {}
    if isinstance(evidence, Mapping):
        for key in ("metadata_keys", "score_present", "skills", "text_length", "file_name", "source", "source_document_id"):
            if key not in merged and key in evidence:
                merged[key] = evidence[key]
    return merged


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    for key in ("text", "content", "page_content"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def _safe_identifier(value: Any, max_length: int = 96) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/").split("/")[-1]
    if not text:
        return ""
    return text[:max_length]


def _safe_basename(value: Any, max_length: int = 96) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    name = Path(text.replace("\\", "/")).name
    return name[:max_length]


def _candidate_name_from_metadata_or_file(metadata: Mapping[str, Any], file_name: str) -> str:
    for key in ("candidate_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = _clean_candidate_name(value)
            if _is_resolved_candidate_name(cleaned):
                return cleaned
    cleaned_from_file = _clean_candidate_name(file_name)
    return cleaned_from_file if _is_resolved_candidate_name(cleaned_from_file) else ""


def _clean_candidate_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    name = Path(value.replace("\\", "/")).name
    name = re.sub(r"\.(pdf|docx?|txt|md)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(个人)?简历|resume|cv", "", name, flags=re.IGNORECASE)
    name = name.strip(" _-()[]")
    return name[:64]


def _is_resolved_candidate_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    normalized = re.sub(r"[\s_\-()（）\[\]]+", "", text).lower()
    if normalized in PLACEHOLDER_NAME_TOKENS:
        return False
    if re.fullmatch(r"\d+", normalized):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", normalized))


def _group_key(candidate_id: str, source_document_id: str, file_name: str, chunk_id: Any, rank: int) -> str:
    return candidate_id or source_document_id or file_name or _safe_identifier(chunk_id) or f"chunk_{rank}"


def _stable_candidate_id(seed: Any) -> str:
    text = str(seed or "candidate")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"candidate_preview_{digest}"


def _extract_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    lowered = (text or "").lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def _matched_query_terms(query: str, chunks: Sequence[Mapping[str, Any]], *, max_terms: int) -> List[str]:
    query_terms = _query_terms(query)
    evidence = " ".join(
        [str(chunk.get("text_excerpt") or "") for chunk in chunks]
        + _flatten(chunks, "skills")
        + _flatten(chunks, "project_keywords")
    ).lower()
    matched = [term for term in query_terms if term.lower() in evidence]
    return _dedupe_limited(matched, max_terms)


def _query_terms(text: str) -> List[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,}|[\u4e00-\u9fff]{2,}", text or "")
    return [term for term in terms if len(term.strip()) >= 2]


def _safe_excerpt(text: str, max_chars: int = 80) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars]


def _build_evidence_summary(chunks: Sequence[Mapping[str, Any]], *, max_chars: int) -> tuple[str, bool]:
    excerpts = [str(chunk.get("text_excerpt") or "") for chunk in chunks if chunk.get("text_excerpt")]
    joined = " | ".join(excerpts)
    if not joined:
        total_length = sum(_safe_int(chunk.get("text_length")) for chunk in chunks)
        return (f"text_length={total_length}; chunk_count={len(chunks)}" if total_length else "", False)
    truncated = len(joined) > max_chars
    if truncated:
        return joined[: max(0, max_chars - 3)] + "...", True
    return joined, False


def _quality_flags(
    *,
    candidate_name: str,
    skills: Sequence[str],
    project_keywords: Sequence[str],
    education_keywords: Sequence[str],
    source_document_id: str,
    evidence_chunk_count: int,
    summary_truncated: bool,
) -> List[str]:
    flags: List[str] = []
    if not candidate_name:
        flags.append("candidate_name_missing")
    if not skills:
        flags.append("skills_missing")
    if not project_keywords:
        flags.append("project_evidence_missing")
    if not education_keywords:
        flags.append("education_missing")
    if evidence_chunk_count < 2:
        flags.append("low_evidence_chunk_count")
    if not source_document_id:
        flags.append("source_document_missing")
    if summary_truncated:
        flags.append("summary_truncated")
    return flags


def _first_present(chunks: Sequence[Mapping[str, Any]], key: str) -> str:
    for chunk in chunks:
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _flatten(chunks: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    values: List[str] = []
    for chunk in chunks:
        values.extend(_strings(chunk.get(key)))
    return values


def _strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_limited(values: Iterable[str], limit: int) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _filter_safe_skills(values: Iterable[str]) -> List[str]:
    lowered = {str(value).lower() for value in values if str(value)}
    return [keyword for keyword in SAFE_SKILL_KEYWORDS if keyword.lower() in lowered]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0
