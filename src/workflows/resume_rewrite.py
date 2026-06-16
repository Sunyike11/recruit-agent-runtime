import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.skills.claim_verify import (
    ClaimVerifySkill,
    build_claim_evidence_from_candidate_preview,
    summarize_claim_verification_result,
)
from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutor
from src.skills.registry import SkillRegistry


@dataclass
class ResumeRewriteInput:
    candidate_id: str
    original_candidate_profile: Dict[str, Any]
    rewrite_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeRewriteVerificationResult:
    status: str
    rewrite_accepted: bool
    review_required: bool
    unsupported_new_claim_count: int
    critical_unsupported_claim_count: int
    claim_verification_summary: Dict[str, Any]
    claim_count: int
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rewrite_accepted": bool(self.rewrite_accepted),
            "review_required": bool(self.review_required),
            "unsupported_new_claim_count": int(self.unsupported_new_claim_count),
            "critical_unsupported_claim_count": int(self.critical_unsupported_claim_count),
            "claim_verification_summary": dict(self.claim_verification_summary),
            "claim_count": int(self.claim_count),
            "summary_only": True,
        }


class MinimalResumeRewriteWorkflow:
    def __init__(
        self,
        *,
        rewrite_callable: Optional[Callable[[ResumeRewriteInput], str]] = None,
        executor: Optional[SkillExecutor] = None,
    ):
        self.rewrite_callable = rewrite_callable
        self.executor = executor or self._default_executor()

    def run(
        self,
        input_data: ResumeRewriteInput | Mapping[str, Any],
        *,
        context: Optional[SkillExecutionContext] = None,
    ) -> ResumeRewriteVerificationResult:
        payload = _coerce_input(input_data)
        rewrite_text = payload.rewrite_text
        if self.rewrite_callable is not None:
            rewrite_text = str(self.rewrite_callable(payload) or "")
        claims = extract_rewrite_claims(rewrite_text, payload.original_candidate_profile)
        evidence = build_claim_evidence_from_candidate_preview(payload.original_candidate_profile)
        verification = self.executor.execute(
            "claim_verify",
            {
                "claims": claims,
                "evidence": evidence,
                "policy": {"summary_only": True},
                "metadata": {
                    "source_component": "minimal_resume_rewrite",
                    "candidate_id": payload.candidate_id,
                    "summary_only": True,
                },
            },
            context=context or SkillExecutionContext(metadata={"workflow": "minimal_resume_rewrite"}),
        )
        output = dict(verification.output or {})
        summary = summarize_claim_verification_result(output)
        unsupported = int(output.get("unsupported_count") or 0) + int(output.get("unverifiable_count") or 0)
        critical = int(output.get("critical_unsupported_count") or 0)
        accepted = verification.success and output.get("status") == "passed"
        review = verification.success and output.get("status") == "review_required"
        status = "accepted" if accepted else ("review_required" if review else "rejected")
        return ResumeRewriteVerificationResult(
            status=status,
            rewrite_accepted=accepted,
            review_required=review,
            unsupported_new_claim_count=unsupported,
            critical_unsupported_claim_count=critical,
            claim_verification_summary=summary,
            claim_count=int(output.get("claim_count") or len(claims)),
        )

    @staticmethod
    def _default_executor() -> SkillExecutor:
        registry = SkillRegistry()
        registry.register(ClaimVerifySkill())
        return SkillExecutor(registry)


def extract_rewrite_claims(rewrite_text: str, original_profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    text = str(rewrite_text or "")
    claims: List[Dict[str, Any]] = []
    if not text.strip():
        return claims
    if _contains_new(text, original_profile, ["博士", "硕士", "本科", "学士"]):
        claims.append(_claim("rewrite_education", "改写内容声明候选人具有相应学历", "education", "critical"))
    if any(token in text for token in ["发表", "顶会", "顶刊", "SCI", "CVPR", "ICCV"]):
        claims.append(_claim("rewrite_publication", "改写内容声明候选人具有发表论文或顶会成果", "publication", "critical"))
    if re.search(r"[三四五六七八九十\d]+\s*年", text) and _contains_new(text, original_profile, ["年"]):
        claims.append(_claim("rewrite_experience", "改写内容声明候选人具有多年全职工作经验", "experience", "critical"))
    if any(token in text for token in ["负责", "构建", "开发", "上线", "项目"]):
        claims.append(_claim("rewrite_project", "改写内容声明候选人具有相关项目贡献", "project", "critical"))
    if any(token in text.lower() for token in ["忽略之前", "给我满分", "ignore previous"]):
        claims.append(_claim("rewrite_prompt_injection", text[:120], "other", "critical"))
    if not claims:
        claims.append(_claim("rewrite_no_new_fact", "改写内容没有新增关键事实", "other", "normal"))
    return claims


def _contains_new(text: str, profile: Mapping[str, Any], tokens: List[str]) -> bool:
    original = " ".join(str(value) for value in profile.values() if not isinstance(value, (dict, list)))
    original += " " + " ".join(str(item) for value in profile.values() if isinstance(value, list) for item in value)
    return any(token in text and token not in original for token in tokens)


def _claim(claim_id: str, text: str, claim_type: str, importance: str) -> Dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "claim_type": claim_type,
        "source_component": "minimal_resume_rewrite",
        "importance": importance,
        "summary_only": True,
    }


def _coerce_input(input_data: ResumeRewriteInput | Mapping[str, Any]) -> ResumeRewriteInput:
    if isinstance(input_data, ResumeRewriteInput):
        return input_data
    return ResumeRewriteInput(
        candidate_id=str(input_data.get("candidate_id") or ""),
        original_candidate_profile=dict(input_data.get("original_candidate_profile") or {}),
        rewrite_text=str(input_data.get("rewrite_text") or ""),
        metadata=dict(input_data.get("metadata") or {}),
    )
