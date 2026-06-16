from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.skills.context import SkillExecutionContext
from src.skills.models import SkillResult, SkillSpec


class BaseSkill(ABC):
    spec: SkillSpec

    def __init__(self, spec: Optional[SkillSpec] = None):
        if spec is not None:
            self.spec = spec
        if not hasattr(self, "spec"):
            raise ValueError("Skill must define a SkillSpec")

    @abstractmethod
    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        raise NotImplementedError

    def execute(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        try:
            return self.run(input_data, context=context)
        except Exception as exc:
            return SkillResult(
                skill_name=self.spec.name,
                version=self.spec.version,
                success=False,
                output=None,
                error=str(exc),
            )

    def __call__(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        return self.execute(input_data, context=context)


class EchoSkill(BaseSkill):
    spec = SkillSpec(
        name="echo",
        version="v1",
        description="Return input data unchanged.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        tags=["demo", "deterministic"],
    )

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        metadata = {}
        if context is not None:
            metadata = {
                "task_id": context.task_id,
                "session_id": context.session_id,
                "thread_id": context.thread_id,
                "has_memory_context": context.memory_context is not None,
            }
        return SkillResult(
            skill_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output=input_data,
            metadata=metadata,
        )


class KeywordExtractSkill(BaseSkill):
    DEFAULT_KEYWORDS = [
        "Python",
        "PyTorch",
        "LangGraph",
        "RAG",
        "LLM",
        "Agent",
        "Chroma",
        "LlamaIndex",
        "3DGS",
        "AIGC",
    ]

    spec = SkillSpec(
        name="keyword_extract_stub",
        version="v1",
        description="Extract configured keywords from text deterministically.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"keywords": {"type": "array"}}},
        tags=["demo", "deterministic", "keyword"],
    )

    def __init__(self, keywords=None, spec: Optional[SkillSpec] = None):
        super().__init__(spec=spec)
        self.keywords = keywords or self.DEFAULT_KEYWORDS

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        text = input_data.get("text", "")
        lowered = text.lower()
        keywords = [keyword for keyword in self.keywords if keyword.lower() in lowered]
        return SkillResult(
            skill_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"keywords": keywords},
        )


class CandidateMatchStubSkill(BaseSkill):
    spec = SkillSpec(
        name="candidate_match_stub",
        version="v1",
        description="Score candidate skill overlap deterministically.",
        input_schema={
            "type": "object",
            "properties": {
                "required_skills": {"type": "array"},
                "candidate_skills": {"type": "array"},
            },
        },
        output_schema={"type": "object", "properties": {"score": {"type": "number"}}},
        tags=["demo", "deterministic", "match"],
    )

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        required = set(input_data.get("required_skills", []))
        candidate = set(input_data.get("candidate_skills", []))
        matched = sorted(required.intersection(candidate))
        score = 0 if not required else round((len(matched) / len(required)) * 100, 2)
        return SkillResult(
            skill_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"matched_skills": matched, "score": score},
        )
