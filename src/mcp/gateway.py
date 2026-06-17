import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.mcp.candidate_client import (
    CandidateMCPClient,
    CandidateMCPClientConfig,
    CandidateMCPError,
    CandidateMCPToolCall,
    MCPTimeoutError,
    MCPTransportError,
    MCPSchemaError,
)
from src.mcp.candidate_provider import DEFAULT_ACCESS_SCOPE, SERVER_NAME


ALLOWED_CANDIDATE_TOOLS = {
    "search_candidates",
    "get_candidate_profile",
    "get_resume_evidence",
}


class MCPPermissionDenied(RuntimeError):
    pass


class MCPToolNotAllowed(RuntimeError):
    pass


class MCPArgumentValidationError(RuntimeError):
    pass


class MCPCandidateNotFound(RuntimeError):
    pass


class MCPPayloadLimitExceeded(RuntimeError):
    pass


@dataclass
class CandidateMCPGatewayConfig:
    dataset_dir: str = "evaluation_data/v1"
    tenant_id: str = "public_eval"
    access_scope: str = DEFAULT_ACCESS_SCOPE
    allowed_tools: Sequence[str] = field(default_factory=lambda: sorted(ALLOWED_CANDIDATE_TOOLS))
    max_tool_calls: int = 32
    max_top_k: int = 10
    max_query_chars: int = 1200
    max_payload_chars: int = 12000
    timeout_seconds: float = 8.0
    read_only: bool = True
    fallback_to_direct: bool = True
    summary_only: bool = True


@dataclass
class CandidateMCPGatewayResult:
    output: Dict[str, Any]
    tool_events: List[Dict[str, Any]]
    fallback_used: bool = False
    error_type: str = ""
    summary_only: bool = True


class CandidateMCPGateway:
    def __init__(
        self,
        config: Optional[CandidateMCPGatewayConfig] = None,
        *,
        client: Optional[CandidateMCPClient] = None,
        direct_fallback_callable: Optional[Callable[[Dict[str, Any], Any], Any]] = None,
    ):
        self.config = config or CandidateMCPGatewayConfig()
        self.client = client or CandidateMCPClient(
            CandidateMCPClientConfig(
                dataset_dir=self.config.dataset_dir,
                timeout_seconds=self.config.timeout_seconds,
            )
        )
        self.direct_fallback_callable = direct_fallback_callable
        self.tool_events: List[Dict[str, Any]] = []
        self._tool_calls = 0

    def retrieve_for_skill(self, input_data: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        query = str(input_data.get("query") or "")
        job_requirement = input_data.get("job_requirement") if isinstance(input_data.get("job_requirement"), Mapping) else {}
        top_k = _bounded_top_k(input_data.get("top_k"), self.config.max_top_k)
        required_skills = _skills_from_job_requirement(job_requirement)
        try:
            search = self.call_tool(
                "search_candidates",
                {
                    "query": query,
                    "top_k": top_k,
                    "required_skills": required_skills,
                    "tenant_id": self.config.tenant_id,
                    "access_scope": self.config.access_scope,
                    "request_id": request_id,
                },
                request_id=request_id,
            )
            candidates = []
            evidence = []
            for item in list(search.get("results") or [])[:top_k]:
                candidate_id = str(item.get("candidate_id") or "")
                if not candidate_id:
                    continue
                profile = self.call_tool(
                    "get_candidate_profile",
                    {
                        "candidate_id": candidate_id,
                        "requested_fields": [
                            "identity",
                            "education",
                            "experience",
                            "projects",
                            "skills",
                            "skill_evidence",
                            "achievements",
                            "safety",
                            "provenance",
                        ],
                        "tenant_id": self.config.tenant_id,
                        "access_scope": self.config.access_scope,
                        "request_id": request_id,
                    },
                    request_id=request_id,
                )
                candidate_profile = _profile_to_matcher_candidate(profile)
                candidates.append(candidate_profile)
                evidence_result = self.call_tool(
                    "get_resume_evidence",
                    {
                        "candidate_id": candidate_id,
                        "max_items": 8,
                        "tenant_id": self.config.tenant_id,
                        "access_scope": self.config.access_scope,
                        "request_id": request_id,
                    },
                    request_id=request_id,
                )
                evidence.extend(list(evidence_result.get("evidence") or []))
            return {
                "candidates": candidates,
                "evidence": evidence,
                "metadata": {
                    "source": "candidate_mcp",
                    "candidate_source": "mcp",
                    "mcp_server": SERVER_NAME,
                    "mcp_transport": "stdio",
                    "tool_events": list(self.tool_events),
                    "tool_success_count": sum(1 for event in self.tool_events if event["status"] == "completed"),
                    "mcp_fallback_used": False,
                    "summary_only": True,
                },
            }
        except Exception as exc:
            if self.config.fallback_to_direct and self.direct_fallback_callable is not None and _is_hard_mcp_failure(exc):
                fallback_output = self.direct_fallback_callable(input_data, context)
                if not isinstance(fallback_output, dict):
                    fallback_output = {"evidence": list(fallback_output or [])}
                metadata = dict(fallback_output.get("metadata") or {})
                metadata.update(
                    {
                        "candidate_source": "direct",
                        "mcp_server": SERVER_NAME,
                        "mcp_transport": "stdio",
                        "mcp_fallback_used": True,
                        "mcp_error_type": _error_type(exc),
                        "tool_events": list(self.tool_events),
                        "summary_only": True,
                    }
                )
                fallback_output["metadata"] = metadata
                return fallback_output
            raise

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], *, request_id: str = "") -> Dict[str, Any]:
        self._validate_call(tool_name, arguments)
        event = self._event("tool_started", tool_name, request_id, status="started")
        self.tool_events.append(event)
        started = time.perf_counter()
        try:
            output = self.client.call_tool(tool_name, arguments)
            self._validate_payload(output)
            completed = self._event(
                "tool_completed",
                tool_name,
                request_id,
                status="completed",
                duration_ms=(time.perf_counter() - started) * 1000,
                result_count=_result_count(output),
            )
            self.tool_events.append(completed)
            return output
        except Exception as exc:
            failed = self._event(
                "tool_failed",
                tool_name,
                request_id,
                status="failed",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=_error_type(exc),
                timeout=isinstance(exc, MCPTimeoutError),
            )
            self.tool_events.append(failed)
            raise _sanitize_gateway_error(exc) from exc

    def _validate_call(self, tool_name: str, arguments: Mapping[str, Any]) -> None:
        if tool_name not in set(self.config.allowed_tools):
            raise MCPToolNotAllowed("mcp_tool_not_allowed")
        if tool_name not in ALLOWED_CANDIDATE_TOOLS:
            raise MCPToolNotAllowed("mcp_tool_not_allowed")
        if not self.config.read_only:
            raise MCPPermissionDenied("mcp_read_only_required")
        self._tool_calls += 1
        if self._tool_calls > int(self.config.max_tool_calls):
            raise MCPPermissionDenied("mcp_tool_budget_exceeded")
        if not str(arguments.get("tenant_id") or arguments.get("access_scope") or "").strip():
            raise MCPPermissionDenied("access_scope_required")
        if tool_name == "search_candidates":
            query = str(arguments.get("query") or "")
            if not query.strip() or len(query) > int(self.config.max_query_chars):
                raise MCPArgumentValidationError("query_invalid")
            _bounded_top_k(arguments.get("top_k"), self.config.max_top_k)

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        if len(str(payload)) > int(self.config.max_payload_chars):
            raise MCPPayloadLimitExceeded("mcp_payload_limit_exceeded")

    def _event(
        self,
        event_type: str,
        tool_name: str,
        request_id: str,
        *,
        status: str,
        duration_ms: Optional[float] = None,
        error_type: str = "",
        timeout: bool = False,
        result_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "tool_name": tool_name,
            "mcp_server_name": SERVER_NAME,
            "transport": "stdio",
            "request_id": str(request_id or ""),
            "skill_name": "resume_retrieve",
            "status": status,
            "duration_ms": round(float(duration_ms), 3) if duration_ms is not None else None,
            "error_type": error_type,
            "timeout": bool(timeout),
            "permission_decision": "allowed" if not error_type else "denied" if "Permission" in error_type else "allowed",
            "result_count": int(result_count),
            "fallback_used": False,
            "summary_only": True,
        }


def build_candidate_mcp_retrieve_callable(
    *,
    dataset_dir: str = "evaluation_data/v1",
    direct_fallback_callable: Optional[Callable[[Dict[str, Any], Any], Any]] = None,
    config: Optional[CandidateMCPGatewayConfig] = None,
) -> Callable[[Dict[str, Any], Any], Dict[str, Any]]:
    gateway_config = config or CandidateMCPGatewayConfig(dataset_dir=dataset_dir)

    def retrieve(input_data: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        gateway = CandidateMCPGateway(
            gateway_config,
            direct_fallback_callable=direct_fallback_callable,
        )
        return gateway.retrieve_for_skill(input_data, context)

    return retrieve


def _profile_to_matcher_candidate(profile: Mapping[str, Any]) -> Dict[str, Any]:
    identity = profile.get("identity") if isinstance(profile.get("identity"), Mapping) else {}
    education = profile.get("education") if isinstance(profile.get("education"), Mapping) else {}
    experience = profile.get("experience") if isinstance(profile.get("experience"), Mapping) else {}
    safety = profile.get("safety") if isinstance(profile.get("safety"), Mapping) else {}
    projects = []
    for project in list(profile.get("projects") or []):
        if isinstance(project, Mapping):
            projects.append(
                "；".join(
                    str(value)
                    for value in [
                        project.get("project_name"),
                        ", ".join(project.get("technologies") or []),
                        project.get("task"),
                        project.get("candidate_contribution"),
                        project.get("result"),
                        project.get("evidence_summary"),
                    ]
                    if value
                )
            )
    return {
        "candidate_id": str(profile.get("candidate_id") or identity.get("candidate_id") or ""),
        "name": str(identity.get("display_name") or ""),
        "candidate_name": str(identity.get("display_name") or ""),
        "candidate_name_resolved": bool(identity.get("candidate_name_resolved")),
        "skills": list(profile.get("skills") or []),
        "skill_evidence": dict(profile.get("skill_evidence") or {}),
        "education": "；".join(
            item
            for item in [
                str(education.get("highest_degree") or ""),
                "、".join(education.get("majors") or []),
                "；".join(education.get("evidence_summaries") or []),
            ]
            if item
        ),
        "education_evidence": list(education.get("evidence_summaries") or []),
        "experience": list(experience.get("evidence_summaries") or []),
        "projects": projects,
        "project_evidence": projects,
        "achievements": dict(profile.get("achievements") or {}),
        "safety_signals": {
            "suspicious_instruction_present": bool(safety.get("suspicious_instruction_present")),
            "job_description_like_content": bool(safety.get("job_description_like_content")),
            "keyword_stuffing_signal": bool(safety.get("keyword_stuffing_signal")),
            "filename_injection_signal": bool(safety.get("filename_injection_signal")),
            "invalid_resume_structure_signal": bool(safety.get("invalid_resume_structure_signal")),
            "instruction_treated_as_data": bool(safety.get("instruction_treated_as_data")),
            "summary_only": True,
        },
        "source_document_id": str(profile.get("source_document_id") or ""),
        "source_file_name": str(identity.get("source_file_name") or ""),
        "field_provenance": dict(profile.get("field_provenance") or {}),
        "preview_version": "v2",
        "candidate_profile_preview": True,
        "summary_only": True,
        "metadata": {
            "candidate_profile_preview": True,
            "source": "candidate_mcp",
            "preview_version": "v2",
            "summary_only": True,
        },
    }


def _skills_from_job_requirement(job_requirement: Mapping[str, Any]) -> List[str]:
    for key in ("required_skills", "tech_stack", "skills"):
        value = job_requirement.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item or "").strip()][:12]
    return []


def _bounded_top_k(value: Any, max_top_k: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MCPArgumentValidationError("top_k_invalid") from exc
    if number < 1 or number > int(max_top_k):
        raise MCPArgumentValidationError("top_k_out_of_range")
    return number


def _result_count(output: Mapping[str, Any]) -> int:
    if isinstance(output.get("results"), list):
        return len(output["results"])
    if isinstance(output.get("evidence"), list):
        return len(output["evidence"])
    if output.get("candidate_id"):
        return 1
    return 0


def _sanitize_gateway_error(exc: Exception) -> Exception:
    if isinstance(exc, (MCPPermissionDenied, MCPToolNotAllowed, MCPArgumentValidationError, MCPPayloadLimitExceeded)):
        return exc
    if isinstance(exc, KeyError):
        return MCPCandidateNotFound("candidate_not_found")
    if isinstance(exc, CandidateMCPError):
        return exc
    return MCPTransportError(type(exc).__name__)


def _is_hard_mcp_failure(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            MCPTransportError,
            MCPTimeoutError,
            MCPSchemaError,
            MCPPermissionDenied,
            MCPToolNotAllowed,
            MCPArgumentValidationError,
            MCPPayloadLimitExceeded,
        ),
    )


def _error_type(exc: Exception) -> str:
    if hasattr(exc, "error_type"):
        return str(getattr(exc, "error_type"))
    return type(exc).__name__

