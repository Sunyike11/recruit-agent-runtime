import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.skills.base import BaseSkill
from src.skills.context import SkillExecutionContext
from src.skills.models import SkillResult, SkillSpec


CLAIM_TYPES = {
    "identity",
    "education",
    "skill",
    "project",
    "experience",
    "research",
    "publication",
    "open_source",
    "award",
    "score_reason",
    "comparison",
    "other",
}

SUSPICIOUS_INSTRUCTIONS = [
    "忽略之前",
    "忽略岗位",
    "给我满分",
    "设置为100",
    "设置为 100",
    "标记为 supported",
    "mark as supported",
    "ignore previous",
    "output outstanding",
]

FIELD_MAP = {
    "identity": {"identity", "candidate", "profile"},
    "education": {"education"},
    "skill": {"skills", "skill_evidence", "projects", "experience"},
    "project": {"projects", "project_evidence", "experience"},
    "experience": {"experience", "work_experience", "projects"},
    "research": {"achievements", "research", "publication"},
    "publication": {"achievements", "publication", "research"},
    "open_source": {"achievements", "open_source"},
    "award": {"achievements", "award", "certification"},
    "score_reason": {"skills", "skill_evidence", "projects", "education", "experience", "achievements"},
    "comparison": {"skills", "skill_evidence", "projects", "education", "experience"},
    "other": {"skills", "skill_evidence", "projects", "education", "experience", "achievements"},
}


@dataclass
class ClaimItem:
    claim_id: str
    claim_text: str
    claim_type: str = "other"
    source_component: str = ""
    importance: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ClaimItem":
        claim_type = str(data.get("claim_type") or "other")
        if claim_type not in CLAIM_TYPES:
            claim_type = "other"
        return cls(
            claim_id=str(data.get("claim_id") or ""),
            claim_text=_truncate(str(data.get("claim_text") or ""), 400),
            claim_type=claim_type,
            source_component=str(data.get("source_component") or ""),
            importance=str(data.get("importance") or "normal"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EvidenceItem:
    evidence_id: str
    evidence_type: str = "other"
    field_name: str = ""
    summary: str = ""
    source_document_id: str = ""
    candidate_id: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        return cls(
            evidence_id=str(data.get("evidence_id") or ""),
            evidence_type=str(data.get("evidence_type") or data.get("field_name") or "other"),
            field_name=str(data.get("field_name") or data.get("evidence_type") or "other"),
            summary=_truncate(str(data.get("summary") or ""), 500),
            source_document_id=str(data.get("source_document_id") or ""),
            candidate_id=str(data.get("candidate_id") or ""),
            provenance=dict(data.get("provenance") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ClaimVerificationPolicy:
    minimum_support_score: float = 0.72
    partial_support_threshold: float = 0.38
    reject_unsupported_critical_claims: bool = True
    allow_semantic_verifier: bool = False
    max_claims: int = 20
    max_evidence_items: int = 80
    max_claim_chars: int = 400
    max_evidence_chars: int = 500
    summary_only: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ClaimVerificationPolicy":
        raw = dict(data or {})
        return cls(
            minimum_support_score=float(raw.get("minimum_support_score", 0.72)),
            partial_support_threshold=float(raw.get("partial_support_threshold", 0.38)),
            reject_unsupported_critical_claims=bool(raw.get("reject_unsupported_critical_claims", True)),
            allow_semantic_verifier=bool(raw.get("allow_semantic_verifier", False)),
            max_claims=int(raw.get("max_claims", 20)),
            max_evidence_items=int(raw.get("max_evidence_items", 80)),
            max_claim_chars=int(raw.get("max_claim_chars", 400)),
            max_evidence_chars=int(raw.get("max_evidence_chars", 500)),
            summary_only=bool(raw.get("summary_only", True)),
        )


@dataclass
class ClaimVerificationItemResult:
    claim_id: str
    status: str
    support_score: float
    matched_evidence_ids: List[str] = field(default_factory=list)
    reason_code: str = ""
    critical: bool = False
    suspicious_instruction_present: bool = False
    instruction_ignored: bool = False
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    status: str
    claim_count: int
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    unverifiable_count: int
    critical_unsupported_count: int
    support_rate: float
    evidence_coverage_rate: float
    item_results: List[ClaimVerificationItemResult] = field(default_factory=list)
    suspicious_instruction_present: bool = False
    instruction_ignored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["item_results"] = [item.to_dict() for item in self.item_results]
        data["summary_only"] = True
        return data


class ClaimVerifySkill(BaseSkill):
    spec = SkillSpec(
        name="claim_verify",
        version="1.0.0",
        description="Verify recruitment-domain claims against structured evidence summaries.",
        input_schema={
            "type": "object",
            "properties": {
                "claims": {"type": "array"},
                "evidence": {"type": "array"},
                "policy": {"type": "object"},
                "metadata": {"type": "object"},
            },
            "required": ["claims", "evidence"],
        },
        output_schema={"type": "object"},
        tags=["recruitment", "verification", "deterministic", "summary-only"],
    )

    def __init__(
        self,
        semantic_verifier: Optional[Callable[[ClaimItem, EvidenceItem], float]] = None,
        spec: Optional[SkillSpec] = None,
    ):
        super().__init__(spec=spec)
        self.semantic_verifier = semantic_verifier

    def run(self, input_data: Dict[str, Any], context: Optional[SkillExecutionContext] = None) -> SkillResult:
        policy = ClaimVerificationPolicy.from_mapping(input_data.get("policy") if isinstance(input_data, Mapping) else {})
        claims = [
            ClaimItem.from_mapping(item)
            for item in list(input_data.get("claims") or [])[: policy.max_claims]
            if isinstance(item, Mapping)
        ]
        evidence = [
            EvidenceItem.from_mapping(item)
            for item in list(input_data.get("evidence") or [])[: policy.max_evidence_items]
            if isinstance(item, Mapping)
        ]

        item_results = [self._verify_claim(claim, evidence, policy) for claim in claims]
        supported = sum(1 for item in item_results if item.status == "supported")
        partial = sum(1 for item in item_results if item.status == "partially_supported")
        unsupported = sum(1 for item in item_results if item.status == "unsupported")
        unverifiable = sum(1 for item in item_results if item.status == "unverifiable")
        critical_unsupported = sum(
            1 for item in item_results if item.critical and item.status in {"unsupported", "unverifiable"}
        )
        covered_evidence = {eid for item in item_results for eid in item.matched_evidence_ids}
        support_rate = round((supported + 0.5 * partial) / len(item_results), 6) if item_results else 0.0
        coverage = round(len(covered_evidence) / len(evidence), 6) if evidence else 0.0
        suspicious = any(item.suspicious_instruction_present for item in item_results)

        if critical_unsupported and policy.reject_unsupported_critical_claims:
            status = "rejected"
        elif unsupported or unverifiable or partial:
            status = "review_required"
        else:
            status = "passed"

        result = ClaimVerificationResult(
            status=status,
            claim_count=len(item_results),
            supported_count=supported,
            partially_supported_count=partial,
            unsupported_count=unsupported,
            unverifiable_count=unverifiable,
            critical_unsupported_count=critical_unsupported,
            support_rate=support_rate,
            evidence_coverage_rate=coverage,
            item_results=item_results,
            suspicious_instruction_present=suspicious,
            instruction_ignored=suspicious,
            metadata={
                "skill_name": self.spec.name,
                "claim_count": len(item_results),
                "evidence_count": len(evidence),
                "summary_only": True,
            },
        )
        return SkillResult(
            skill_name=self.spec.name,
            version=self.spec.version,
            success=True,
            output=result.to_dict(),
            metadata={
                "status": result.status,
                "claim_count": result.claim_count,
                "supported_count": result.supported_count,
                "unsupported_count": result.unsupported_count,
                "summary_only": True,
            },
        )

    def _verify_claim(
        self,
        claim: ClaimItem,
        evidence_items: Sequence[EvidenceItem],
        policy: ClaimVerificationPolicy,
    ) -> ClaimVerificationItemResult:
        suspicious = _has_suspicious_instruction(claim.claim_text)
        critical = _is_critical_claim(claim)
        allowed_fields = FIELD_MAP.get(claim.claim_type, FIELD_MAP["other"])
        candidates = [
            item
            for item in evidence_items
            if _normalize_field(item.field_name or item.evidence_type) in allowed_fields
            or _normalize_field(item.evidence_type) in allowed_fields
        ]
        if not candidates:
            return ClaimVerificationItemResult(
                claim_id=claim.claim_id,
                status="unverifiable",
                support_score=0.0,
                reason_code="no_field_compatible_evidence",
                critical=critical,
                suspicious_instruction_present=suspicious,
                instruction_ignored=suspicious,
            )

        scored = [(item, self._score_claim_evidence(claim, item, policy)) for item in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_item, best_score = scored[0]
        matched = [item.evidence_id for item, score in scored if score >= policy.partial_support_threshold and item.evidence_id]

        if _contradicts_claim(claim, best_item):
            status = "unsupported"
            reason = "contradicted_by_evidence"
            best_score = min(best_score, 0.25)
        elif best_score >= policy.minimum_support_score:
            status = "supported"
            reason = "field_aware_match"
        elif best_score >= policy.partial_support_threshold:
            status = "partially_supported"
            reason = "partial_field_aware_match"
        else:
            status = "unsupported"
            reason = "insufficient_evidence_overlap"

        return ClaimVerificationItemResult(
            claim_id=claim.claim_id,
            status=status,
            support_score=round(best_score, 6),
            matched_evidence_ids=matched[:5],
            reason_code=reason,
            critical=critical,
            suspicious_instruction_present=suspicious,
            instruction_ignored=suspicious,
        )

    def _score_claim_evidence(
        self,
        claim: ClaimItem,
        evidence: EvidenceItem,
        policy: ClaimVerificationPolicy,
    ) -> float:
        claim_tokens = _content_tokens(claim.claim_text)
        evidence_tokens = _content_tokens(evidence.summary)
        if not claim_tokens or not evidence_tokens:
            return _field_presence_score(claim, evidence)
        overlap = len(claim_tokens & evidence_tokens)
        score = overlap / max(1, len(claim_tokens))
        score = max(score, _field_presence_score(claim, evidence))
        if claim.claim_type == "education" and _degree_tokens(claim.claim_text) & _degree_tokens(evidence.summary):
            score = max(score, 0.85)
        if claim.claim_type == "project" and _shared_keywords(claim.claim_text, evidence.summary):
            score = max(score, 0.82)
        if claim.claim_type == "skill" and any(token in evidence_tokens for token in claim_tokens):
            score = max(score, min(1.0, score + 0.25))
        if policy.allow_semantic_verifier and self.semantic_verifier is not None:
            score = max(score, float(self.semantic_verifier(claim, evidence)))
        return min(1.0, score)


def build_claim_evidence_from_candidate_preview(candidate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    source_document_id = str(candidate.get("source_document_id") or "")
    output: List[Dict[str, Any]] = []

    def add(field: str, summary: Any, evidence_type: Optional[str] = None):
        if not summary:
            return
        values = summary if isinstance(summary, list) else [summary]
        for index, value in enumerate(values):
            text = _truncate(str(value or ""), 500)
            if not text.strip():
                continue
            output.append(
                {
                    "evidence_id": f"{candidate_id}:{field}:{len(output)+1}",
                    "evidence_type": evidence_type or field,
                    "field_name": field,
                    "summary": text,
                    "source_document_id": source_document_id,
                    "candidate_id": candidate_id,
                    "provenance": {
                        "source_field": field,
                        "source_document_id": source_document_id,
                        "evidence_present": True,
                        "summary_only": True,
                    },
                    "summary_only": True,
                }
            )

    add("identity", " ".join(item for item in [candidate_id, str(candidate.get("candidate_name") or candidate.get("name") or "")] if item))
    add("education", candidate.get("education"))
    add("education", candidate.get("education_evidence"))
    add("experience", candidate.get("experience"))
    add("projects", candidate.get("projects"))
    add("projects", candidate.get("project_evidence"))
    for skill, snippets in dict(candidate.get("skill_evidence") or {}).items():
        add("skill_evidence", [f"{skill}: {snippet}" for snippet in snippets], evidence_type="skill_evidence")
    achievements = candidate.get("achievements") if isinstance(candidate.get("achievements"), Mapping) else {}
    for key, values in achievements.items():
        field = "publication" if key in {"research", "publications", "publication"} else str(key)
        add(field, values, evidence_type=field)
    return output


def build_matcher_claims_from_report(report: Mapping[str, Any], candidate: Mapping[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    candidate_id = str(candidate.get("candidate_id") or report.get("candidate_id") or "")
    candidate_name = str(candidate.get("candidate_name") or candidate.get("name") or "")
    if candidate_id:
        claims.append(_claim("identity", f"候选人身份为 {candidate_id}", "identity", "critical"))
    if candidate_name:
        claims.append(_claim("identity_name", f"候选人姓名为 {candidate_name}", "identity", "normal"))
    if candidate.get("education"):
        claims.append(_claim("education", "候选人的学历信息满足岗位评估需要", "education", "critical"))
    if candidate.get("skills"):
        claims.append(_claim("skills", "候选人具备岗位相关技能 " + " ".join(map(str, candidate.get("skills") or [])), "skill", "critical"))
    if candidate.get("projects"):
        claims.append(_claim("projects", "候选人具有岗位相关项目经历", "project", "critical"))
    if candidate.get("experience"):
        claims.append(_claim("experience", "候选人具有相关工作或实践经历", "experience", "normal"))
    reasoning = str(report.get("reasoning") or report.get("recommendation") or "")
    if any(token in reasoning for token in ["论文", "顶会", "发表"]):
        claims.append(_claim("publication", "候选人具有发表论文或顶会成果", "publication", "critical"))
    if any(token in reasoning for token in ["开源", "奖项", "获奖"]):
        claims.append(_claim("achievement", "候选人具有开源或奖项成果", "award", "normal"))
    return claims


def _claim(suffix: str, text: str, claim_type: str, importance: str) -> Dict[str, Any]:
    return {
        "claim_id": f"matcher_{suffix}",
        "claim_text": text,
        "claim_type": claim_type,
        "source_component": "candidate_match",
        "importance": importance,
        "summary_only": True,
    }


def summarize_claim_verification_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "claim_verification_status": str(result.get("status") or ""),
        "claim_count": int(result.get("claim_count") or 0),
        "claim_support_rate": float(result.get("support_rate") or 0.0),
        "unsupported_claim_count": int(result.get("unsupported_count") or 0),
        "critical_unsupported_claim_count": int(result.get("critical_unsupported_count") or 0),
        "evidence_coverage_rate": float(result.get("evidence_coverage_rate") or 0.0),
        "suspicious_instruction_present": bool(result.get("suspicious_instruction_present", False)),
        "summary_only": True,
    }


def _has_suspicious_instruction(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token.lower() in lowered for token in SUSPICIOUS_INSTRUCTIONS)


def _is_critical_claim(claim: ClaimItem) -> bool:
    return claim.importance == "critical" or claim.claim_type in {"identity", "education", "project", "publication", "experience"}


def _normalize_field(value: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "project": "projects",
        "project_evidence": "projects",
        "work_experience": "experience",
        "skill": "skills",
        "skills": "skills",
        "education_evidence": "education",
        "publications": "publication",
    }
    return aliases.get(text, text)


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9+#.]*|[\u4e00-\u9fff]{2,}", str(text or ""))
    stop = {"候选人", "具有", "具备", "相关", "岗位", "项目", "经历", "信息", "满足", "评估", "需要"}
    return {token.lower() for token in raw if token and token not in stop}


def _shared_keywords(left: str, right: str) -> set[str]:
    keywords = {
        "python",
        "rag",
        "langgraph",
        "agent",
        "招聘",
        "检索",
        "匹配",
        "系统",
        "项目",
        "平台",
        "部署",
        "上线",
    }
    left_l = str(left or "").lower()
    right_l = str(right or "").lower()
    return {keyword for keyword in keywords if keyword in left_l and keyword in right_l}


def _field_presence_score(claim: ClaimItem, evidence: EvidenceItem) -> float:
    if not str(evidence.summary or "").strip():
        return 0.0
    field = _normalize_field(evidence.field_name or evidence.evidence_type)
    if claim.claim_type == "project" and field in {"projects", "experience"}:
        return 0.74 if any(token in claim.claim_text for token in ["项目", "贡献", "负责", "构建"]) else 0.2
    if claim.claim_type == "experience" and field == "experience":
        return 0.72 if "多年" not in claim.claim_text and not re.search(r"[三四五六七八九十\d]+\s*年", claim.claim_text) else 0.35
    if claim.claim_type == "skill" and field in {"skills", "skill_evidence", "projects", "experience"}:
        return 0.45
    if claim.claim_type == "education" and field == "education":
        return 0.78 if not _degree_tokens(claim.claim_text) else 0.2
    if claim.claim_type == "identity" and field == "identity":
        return 0.86
    return 0.0


def _degree_tokens(text: str) -> set[str]:
    output = set()
    for token in ("博士", "硕士", "研究生", "本科", "学士", "大专"):
        if token in str(text or ""):
            output.add(token)
    return output


def _contradicts_claim(claim: ClaimItem, evidence: EvidenceItem) -> bool:
    claim_text = claim.claim_text
    evidence_text = evidence.summary
    if claim.claim_type == "publication" and any(token in claim_text for token in ["发表", "顶会", "顶刊"]):
        return any(token in evidence_text for token in ["在投", "投稿", "准备", "未发表"])
    if claim.claim_type == "experience" and re.search(r"[三四五六七八九十\d]+\s*年", claim_text):
        return "实习" in evidence_text and not re.search(r"[三四五六七八九十\d]+\s*年", evidence_text)
    return False


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit]
