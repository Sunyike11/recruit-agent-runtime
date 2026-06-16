import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.domain.serialization import dataclass_from_dict, dataclass_to_dict


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


@dataclass
class SerializableDomainModel:
    def to_dict(self) -> dict:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return dataclass_from_dict(cls, data)


@dataclass
class JobRequirement(SerializableDomainModel):
    job_id: str = field(default_factory=lambda: new_id("job"))
    raw_text: str = ""
    title: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    education: str = ""
    experience_years: Optional[int] = None
    location: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateProfile(SerializableDomainModel):
    candidate_id: str = field(default_factory=lambda: new_id("candidate"))
    name: str = ""
    skills: List[str] = field(default_factory=list)
    education: str = ""
    experience: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    source_resume_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeDocument(SerializableDomainModel):
    resume_id: str = field(default_factory=lambda: new_id("resume"))
    candidate_id: str = ""
    source_path: str = ""
    raw_text: str = ""
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchReport(SerializableDomainModel):
    match_id: str = field(default_factory=lambda: new_id("match"))
    job_id: str = ""
    candidate_id: str = ""
    total_score: float = 0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchAttempt(SerializableDomainModel):
    search_id: str = field(default_factory=lambda: new_id("search"))
    job_id: str = ""
    query: str = ""
    retrieved_candidate_ids: List[str] = field(default_factory=list)
    retrieved_resume_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedback(SerializableDomainModel):
    feedback_id: str = field(default_factory=lambda: new_id("feedback"))
    task_id: str = ""
    target_type: str = ""
    target_id: str = ""
    feedback_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
