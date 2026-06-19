from typing import Any, Dict, Optional

from src.domain.candidate_management import (
    build_evidence_from_preview,
    extract_text_from_resume_bytes,
)
from src.runtime.candidate_preview import build_candidate_profile_preview_v2
from src.skills.base import BaseSkill
from src.skills.context import SkillExecutionContext
from src.skills.models import SkillResult, SkillSpec


class ResumeParseSkill(BaseSkill):
    spec = SkillSpec(
        name="resume_parse",
        version="1.0.0",
        description="Parse an immutable ResumeVersion into summary-only structured resume sections.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        tags=["recruitment", "ingestion", "deterministic"],
    )

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        raw_bytes = input_data.get("content_bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return SkillResult(self.spec.name, self.spec.version, False, error="ResumeParseFailed: missing_content")
        try:
            text = extract_text_from_resume_bytes(
                bytes(raw_bytes),
                media_type=str(input_data.get("media_type") or ""),
                filename=str(input_data.get("filename") or ""),
            )
        except Exception as exc:
            return SkillResult(self.spec.name, self.spec.version, False, error=f"{type(exc).__name__}: {exc}")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        output = {
            "candidate_id": str(input_data.get("candidate_id") or ""),
            "resume_version_id": str(input_data.get("resume_version_id") or ""),
            "source_file_name": str(input_data.get("filename") or ""),
            "resume_text": text,
            "safe_text_statistics": {
                "char_count": len(text),
                "line_count": len(lines),
                "summary_only": True,
            },
            "summary_only": True,
        }
        return SkillResult(self.spec.name, self.spec.version, True, output=output)


class EvidenceExtractSkill(BaseSkill):
    spec = SkillSpec(
        name="evidence_extract",
        version="1.0.0",
        description="Build CandidateProfilePreview v2 and evidence records from parsed resume text.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        tags=["recruitment", "ingestion", "deterministic", "candidate_profile_preview_v2"],
    )

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        text = str(input_data.get("resume_text") or "")
        candidate_id = str(input_data.get("candidate_id") or "")
        resume_version_id = str(input_data.get("resume_version_id") or "")
        tenant_id = str(input_data.get("tenant_id") or "")
        if not text.strip() or not candidate_id or not resume_version_id:
            return SkillResult(self.spec.name, self.spec.version, False, error="EvidenceExtractionFailed: missing_input")
        preview = build_candidate_profile_preview_v2(
            {
                "candidate_id": candidate_id,
                "resume_text": text,
                "source_file_name": input_data.get("source_file_name") or "",
                "source_document_id": resume_version_id,
                "metadata": {
                    "candidate_id": candidate_id,
                    "source_document_id": resume_version_id,
                    "file_name": input_data.get("source_file_name") or "",
                },
            }
        ).to_dict()
        preview["resume_version_id"] = resume_version_id
        evidence = build_evidence_from_preview(
            tenant_id=tenant_id,
            candidate_id=candidate_id,
            resume_version_id=resume_version_id,
            preview=preview,
        )
        return SkillResult(
            self.spec.name,
            self.spec.version,
            True,
            output={
                "candidate_id": candidate_id,
                "resume_version_id": resume_version_id,
                "profile": preview,
                "evidence": [item.to_dict() for item in evidence],
                "evidence_count": len(evidence),
                "summary_only": True,
            },
        )
