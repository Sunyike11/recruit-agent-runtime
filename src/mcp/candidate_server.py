from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from src.mcp.candidate_provider import (
    DEFAULT_ACCESS_SCOPE,
    SERVER_NAME,
    SERVER_VERSION,
    EvaluationDatasetCandidateProvider,
    ManagedCandidateDataProvider,
)


def create_candidate_mcp_server(
    dataset_dir: str | Path = "evaluation_data/v1",
    *,
    provider_mode: str = "evaluation",
    db_path: str | Path = "storage/sqlite/recruit_api_runtime.sqlite",
) -> FastMCP:
    if provider_mode == "managed":
        provider = ManagedCandidateDataProvider(db_path=db_path)
    else:
        provider = EvaluationDatasetCandidateProvider(dataset_dir=dataset_dir)
    server = FastMCP(SERVER_NAME)

    @server.tool(
        name="search_candidates",
        description="Search summary-only synthetic/anonymized candidate profiles by query.",
    )
    def search_candidates(
        query: str,
        top_k: int = 5,
        required_skills: Optional[List[str]] = None,
        excluded_candidate_ids: Optional[List[str]] = None,
        tenant_id: str = "",
        access_scope: str = DEFAULT_ACCESS_SCOPE,
        request_id: str = "",
    ) -> Dict[str, Any]:
        return provider.search_candidates(
            query=query,
            top_k=top_k,
            required_skills=required_skills,
            excluded_candidate_ids=excluded_candidate_ids,
            tenant_id=tenant_id,
            access_scope=access_scope,
            request_id=request_id,
        )

    @server.tool(
        name="get_candidate_profile",
        description="Get CandidateProfilePreview v2 summary fields for one candidate.",
    )
    def get_candidate_profile(
        candidate_id: str,
        requested_fields: Optional[List[str]] = None,
        tenant_id: str = "",
        access_scope: str = DEFAULT_ACCESS_SCOPE,
        request_id: str = "",
    ) -> Dict[str, Any]:
        return provider.get_candidate_profile(
            candidate_id=candidate_id,
            requested_fields=requested_fields,
            tenant_id=tenant_id,
            access_scope=access_scope,
            request_id=request_id,
        )

    @server.tool(
        name="get_resume_evidence",
        description="Get summary-only evidence snippets and provenance for one candidate.",
    )
    def get_resume_evidence(
        candidate_id: str,
        evidence_ids: Optional[List[str]] = None,
        field_names: Optional[List[str]] = None,
        max_items: int = 10,
        tenant_id: str = "",
        access_scope: str = DEFAULT_ACCESS_SCOPE,
        request_id: str = "",
    ) -> Dict[str, Any]:
        return provider.get_resume_evidence(
            candidate_id=candidate_id,
            evidence_ids=evidence_ids,
            field_names=field_names,
            max_items=max_items,
            tenant_id=tenant_id,
            access_scope=access_scope,
            request_id=request_id,
        )

    return server


def server_metadata() -> Dict[str, Any]:
    return {
        "server_name": SERVER_NAME,
        "server_version": SERVER_VERSION,
        "read_only": True,
        "tool_count": 3,
        "summary_only": True,
    }
