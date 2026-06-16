from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.tools.models import ToolExecutionContext, ToolResult, ToolSpec


class BaseTool(ABC):
    spec: ToolSpec

    def __init__(self, spec: Optional[ToolSpec] = None):
        if spec is not None:
            self.spec = spec
        if not hasattr(self, "spec"):
            raise ValueError("Tool must define a ToolSpec")

    @abstractmethod
    def run(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        raise NotImplementedError

    def execute(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        try:
            return self.run(input_data, context=context)
        except Exception as exc:
            return ToolResult(
                tool_name=self.spec.name,
                version=self.spec.version,
                success=False,
                output=None,
                error=str(exc),
            )

    def __call__(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        return self.execute(input_data, context=context)


class EchoTool(BaseTool):
    spec = ToolSpec(
        name="echo_tool",
        version="v1",
        description="Return input data unchanged.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        category="demo",
        side_effects="none",
        metadata={"deterministic": True},
    )

    def run(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        metadata = {}
        if context is not None:
            metadata = {
                "caller_type": context.caller_type,
                "caller_name": context.caller_name,
                "permissions": list(context.permissions),
            }
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output=dict(input_data),
            metadata=metadata,
        )


class CandidateLookupFakeTool(BaseTool):
    spec = ToolSpec(
        name="candidate_lookup_fake",
        version="v1",
        description="Return deterministic fake candidate records by skill.",
        input_schema={"type": "object", "properties": {"skill": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"candidates": {"type": "array"}}},
        category="candidate",
        side_effects="read",
        permissions_required=["candidate:read"],
        metadata={"deterministic": True},
    )

    def __init__(self, candidates=None, spec: Optional[ToolSpec] = None):
        super().__init__(spec=spec)
        self.candidates = candidates or [
            {"candidate_id": "candidate_fake_1", "name": "Alice", "skills": ["Python", "LangGraph"]},
            {"candidate_id": "candidate_fake_2", "name": "Bob", "skills": ["SQL"]},
        ]

    def run(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        skill = str(input_data.get("skill", "")).lower()
        candidates = [
            candidate
            for candidate in self.candidates
            if not skill or skill in " ".join(candidate.get("skills", [])).lower()
        ]
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"candidates": candidates},
        )


class ResumeTextParseFakeTool(BaseTool):
    spec = ToolSpec(
        name="resume_text_parse_fake",
        version="v1",
        description="Parse resume text into deterministic lightweight fields.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        output_schema={
            "type": "object",
            "properties": {
                "text_length": {"type": "integer"},
                "keywords": {"type": "array"},
            },
        },
        category="document",
        side_effects="none",
        metadata={"deterministic": True},
    )

    DEFAULT_KEYWORDS = ["Python", "LangGraph", "RAG", "Chroma", "LLM"]

    def run(self, input_data: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        text = str(input_data.get("text", ""))
        lowered = text.lower()
        keywords = [keyword for keyword in self.DEFAULT_KEYWORDS if keyword.lower() in lowered]
        return ToolResult(
            tool_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output={"text_length": len(text), "keywords": keywords},
        )
