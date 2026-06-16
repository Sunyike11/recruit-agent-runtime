import re
from typing import List, Optional, Tuple

from src.domain.models import CandidateProfile, ResumeDocument


DEFAULT_SKILL_KEYWORDS = [
    "Python",
    "PyTorch",
    "TensorFlow",
    "LangChain",
    "LangGraph",
    "LlamaIndex",
    "RAG",
    "Chroma",
    "FAISS",
    "Diffusion",
    "Stable Diffusion",
    "3DGS",
    "3D生成",
    "AIGC",
    "LLM",
    "Agent",
    "Java",
    "SQL",
]


def split_resume_chunks(raw_text: str) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", raw_text) if chunk.strip()]
    if chunks:
        return chunks
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines or ([raw_text.strip()] if raw_text.strip() else [])


class DeterministicResumeParser:
    """Small heuristic parser for deterministic Phase2B ingestion tests."""

    def __init__(self, skill_keywords: Optional[List[str]] = None):
        self.skill_keywords = skill_keywords or DEFAULT_SKILL_KEYWORDS

    def parse(self, raw_text: str) -> dict:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return {
            "name": self.extract_name(lines),
            "skills": self.extract_skills(raw_text),
            "education": self.extract_education(lines),
            "experience": self.extract_lines(lines, ["实习", "工作", "经历", "公司", "工程师"]),
            "projects": self.extract_lines(lines, ["项目", "系统", "平台", "研究", "算法"]),
        }

    def extract_name(self, lines: List[str]) -> str:
        for line in lines:
            match = re.search(r"(?:姓名|Name)\s*[:：]\s*([\u4e00-\u9fa5A-Za-z\s·.-]{2,40})", line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        for line in lines[:3]:
            if len(line) <= 20 and not any(token in line for token in ["教育", "项目", "技能", "经历", "电话", "邮箱"]):
                return line
        return ""

    def extract_skills(self, raw_text: str) -> List[str]:
        lowered = raw_text.lower()
        found = []
        for skill in self.skill_keywords:
            if skill.lower() in lowered and skill not in found:
                found.append(skill)
        return found

    def extract_education(self, lines: List[str]) -> str:
        education_keywords = ["博士", "硕士", "研究生", "本科", "学士", "大学", "学院"]
        for line in lines:
            if any(keyword in line for keyword in education_keywords):
                return line
        return ""

    def extract_lines(self, lines: List[str], keywords: List[str]) -> List[str]:
        results = []
        for line in lines:
            if any(keyword in line for keyword in keywords):
                results.append(line)
        return results


class ResumeIngestionPipeline:
    def __init__(self, store=None, parser: Optional[DeterministicResumeParser] = None):
        self.store = store
        self.parser = parser or DeterministicResumeParser()

    def ingest_text(
        self,
        raw_text: str,
        source_path: str = "",
        candidate_id: Optional[str] = None,
        save: bool = True,
    ) -> Tuple[ResumeDocument, CandidateProfile]:
        profile_data = self.parser.parse(raw_text)
        candidate = CandidateProfile(
            candidate_id=candidate_id or CandidateProfile().candidate_id,
            name=profile_data["name"],
            skills=profile_data["skills"],
            education=profile_data["education"],
            experience=profile_data["experience"],
            projects=profile_data["projects"],
            metadata={"parser": "deterministic"},
        )
        resume = ResumeDocument(
            candidate_id=candidate.candidate_id,
            source_path=source_path,
            raw_text=raw_text,
            chunks=split_resume_chunks(raw_text),
            metadata={"parser": "deterministic"},
        )
        candidate.source_resume_id = resume.resume_id

        if save and self.store is not None:
            self.store.save_resume_document(resume)
            self.store.save_candidate_profile(candidate)

        return resume, candidate


def ingest_resume_text(raw_text: str, source_path: str = "", candidate_id: Optional[str] = None, store=None):
    return ResumeIngestionPipeline(store=store).ingest_text(
        raw_text=raw_text,
        source_path=source_path,
        candidate_id=candidate_id,
        save=store is not None,
    )
