from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.runtime.candidate_preview import (
    build_candidate_preview_quality_audit,
    build_candidate_profile_previews_from_retrieval_results,
    build_candidate_profile_preview_v2,
    candidate_profile_preview_to_matcher_input,
    candidate_profile_preview_v2_to_matcher_input,
)
from src.runtime.event_envelope import build_runtime_event_payload
from src.runtime.variant_runner import (
    build_real_retriever_callable,
    check_llm_env_readiness,
    load_project_dotenv_for_real_wrappers,
)
from src.skills.agent_adapters import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RetrieverSkill,
    invoke_planner_agent_for_skill,
)
from src.skills.claim_verify import (
    ClaimVerifySkill,
    build_claim_evidence_from_candidate_preview,
    build_matcher_claims_from_report,
    summarize_claim_verification_result,
)
from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutionRecorder, SkillExecutor
from src.skills.registry import SkillRegistry


@dataclass
class ProductionSkillGraphConfig:
    enabled: bool = False
    use_real_planner: bool = True
    allow_planner_fallback: bool = False
    use_real_retriever: bool = True
    use_candidate_profile_preview: bool = True
    use_real_matcher: bool = True
    use_real_refiner: bool = True
    max_refine_loops: int = 1
    low_score_threshold: float = 60.0
    enable_claim_verification: bool = True
    candidate_source: str = "direct"
    rollback_on_failure: bool = True
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionSkillGraphState:
    raw_jd: str
    job_requirement: Dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    candidate_previews: List[Dict[str, Any]] = field(default_factory=list)
    match_reports: List[Dict[str, Any]] = field(default_factory=list)
    refined_query: str = ""
    loop_count: int = 0
    next_action: str = ""
    status: str = "created"
    error_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionSkillGraphResult:
    status: str
    success: bool
    job_requirement: Dict[str, Any] = field(default_factory=dict)
    retrieved_count: int = 0
    candidate_profile_preview_count: int = 0
    match_report_count: int = 0
    refined_query_present: bool = False
    loop_count: int = 0
    skill_names: List[str] = field(default_factory=list)
    skill_event_count: int = 0
    fallback_used: bool = False
    rollback_recommended: bool = False
    rollback_target: str = "legacy_default_graph"
    error_type: str = ""
    error_hint: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    candidate_ids: List[str] = field(default_factory=list)
    top_scores: List[float] = field(default_factory=list)
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionSkillGraphRunner:
    def __init__(
        self,
        config: Optional[ProductionSkillGraphConfig] = None,
        *,
        planner_extract_callable: Optional[Callable[..., Any]] = None,
        retrieve_callable: Optional[Callable[..., Any]] = None,
        match_callable: Optional[Callable[..., Any]] = None,
        refine_callable: Optional[Callable[..., Any]] = None,
        retriever_callable_factory: Optional[Callable[[], Callable[..., Any]]] = None,
    ):
        self.config = config or ProductionSkillGraphConfig()
        self.planner_extract_callable = planner_extract_callable
        self.retrieve_callable = retrieve_callable
        self.match_callable = match_callable
        self.refine_callable = refine_callable
        self.retriever_callable_factory = retriever_callable_factory

    def build_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        registry.register(PlannerExtractSkill(extract_callable=self._planner_callable()))
        registry.register(RetrieverSkill(retrieve_callable=self._retriever_callable()))
        registry.register(CandidateMatchSkill(match_callable=self.match_callable))
        registry.register(ClaimVerifySkill())
        registry.register(QueryRefineSkill(refine_callable=self.refine_callable))
        return registry

    def build_executor(self, runtime_store: Any = None) -> SkillExecutor:
        recorder = SkillExecutionRecorder(runtime_store) if runtime_store is not None else None
        return SkillExecutor(self.build_registry(), recorder=recorder)

    def build_initial_context(self, metadata: Optional[Mapping[str, Any]] = None) -> SkillExecutionContext:
        safe_metadata = dict(metadata or {})
        return SkillExecutionContext(
            task_id=str(safe_metadata.get("task_id") or ""),
            session_id=str(safe_metadata.get("session_id") or ""),
            thread_id=str(safe_metadata.get("thread_id") or ""),
            metadata={
                "runner_type": "production_skill_graph",
                "runner_used": "production_skill_graph",
                "graph_mode": "skill",
                "summary_only": True,
                "production_skill_graph_enabled": bool(self.config.enabled),
                "metadata_keys": sorted(str(key) for key in safe_metadata.keys()),
            },
        )

    def run(
        self,
        raw_jd: str,
        memory_context: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            return {
                "status": "skipped",
                "success": False,
                "error_type": "",
                "error_hint": "production_skill_graph_disabled",
                "production_skill_graph_enabled": False,
                "legacy_graph_invoked": False,
                "production_graph_replaced": False,
                "rollback_baseline": "legacy_default_graph",
                "summary_only": True,
                "metadata": {"summary_only": True},
            }
        runtime_store = (metadata or {}).get("runtime_store") if isinstance(metadata, Mapping) else None
        context = self.build_initial_context(metadata)
        if memory_context is not None:
            context.memory_context = memory_context
        if self.config.use_real_planner or self.config.use_real_retriever or self.config.use_real_matcher:
            load_project_dotenv_for_real_wrappers()
        _append_graph_event(runtime_store, "graph_started", context, {"status": "started", "summary_only": True})
        state = ProductionSkillGraphState(raw_jd=raw_jd or "", status="running")
        executor = self.build_executor(runtime_store=runtime_store)
        skill_results = []
        try:
            _append_node_event(runtime_store, "node_started", context, "planner", {"status": "started"})
            planner = executor.execute(
                "planner_extract",
                {"raw_text": raw_jd, "metadata": {"runner_type": "production_skill_graph"}},
                context=context,
            )
            skill_results.append(planner)
            if not planner.success:
                _append_node_event(runtime_store, "node_failed", context, "planner", {"status": "failed", "error_hint": "planner_failed"})
                return self._failed(state, skill_results, "planner_failed", planner.error, runtime_store, context)
            state.job_requirement = dict(planner.output.get("job_requirement") or {})
            _append_node_event(runtime_store, "node_completed", context, "planner", {"status": "completed"})

            while True:
                state.retrieval_query = _derive_query(raw_jd, state.job_requirement, state.refined_query)
                _append_node_event(runtime_store, "node_started", context, "retriever", {"status": "started"})
                retriever = executor.execute(
                    "resume_retrieve",
                    {
                        "job_requirement": state.job_requirement,
                        "query": state.retrieval_query,
                        "top_k": 3,
                        "metadata": {"runner_type": "production_skill_graph"},
                    },
                    context=context,
                )
                skill_results.append(retriever)
                if not retriever.success:
                    _append_node_event(runtime_store, "node_failed", context, "retriever", {"status": "failed", "error_hint": "retriever_failed"})
                    return self._failed(state, skill_results, "retriever_failed", retriever.error, runtime_store, context)

                retrieval_output = dict(retriever.output or {})
                _append_tool_events_from_metadata(runtime_store, context, retrieval_output.get("metadata"))
                retrieval_metadata = dict(retrieval_output.get("metadata") or {})
                for key in (
                    "candidate_source",
                    "mcp_server",
                    "mcp_transport",
                    "tool_success_count",
                    "mcp_fallback_used",
                    "mcp_error_type",
                ):
                    if key in retrieval_metadata:
                        state.metadata[key] = retrieval_metadata[key]
                state.metadata["tool_events"] = list(retrieval_metadata.get("tool_events") or [])
                state.retrieved_documents = list(retrieval_output.get("resume_documents") or [])
                candidates = list(retrieval_output.get("candidates") or [])
                if not candidates and self.config.use_candidate_profile_preview:
                    candidates = _build_candidate_previews(retrieval_output, raw_jd, state.retrieval_query)
                state.candidate_previews = candidates
                _append_node_event(runtime_store, "node_completed", context, "retriever", {"status": "completed"})

                state.match_reports = []
                state.metadata["matcher_input_summaries"] = []
                for candidate in candidates:
                    _append_node_event(
                        runtime_store,
                        "node_started",
                        context,
                        "matcher",
                        {"status": "started", "candidate_id": str(candidate.get("candidate_id") or "")},
                    )
                    matcher_input_summary = _matcher_input_summary(candidate)
                    state.metadata["last_matcher_input_summary"] = matcher_input_summary
                    state.metadata["matcher_input_summaries"].append(matcher_input_summary)
                    match = executor.execute(
                        "candidate_match",
                        {
                            "job_requirement": state.job_requirement,
                            "candidate_profile": candidate,
                            "evidence": list(retrieval_output.get("evidence") or []),
                            "metadata": {"runner_type": "production_skill_graph"},
                        },
                        context=context,
                    )
                    skill_results.append(match)
                    if not match.success:
                        _append_node_event(
                            runtime_store,
                            "node_failed",
                            context,
                            "matcher",
                            {
                                "status": "failed",
                                "error_hint": "matcher_failed",
                                "candidate_id": str(candidate.get("candidate_id") or ""),
                            },
                        )
                        return self._failed(state, skill_results, "matcher_failed", match.error, runtime_store, context)
                    match_report = dict(match.output.get("match_report") or {})
                    if self.config.enable_claim_verification:
                        _append_node_event(
                            runtime_store,
                            "node_started",
                            context,
                            "claim_verify",
                            {"status": "started", "candidate_id": str(candidate.get("candidate_id") or "")},
                        )
                        claims = build_matcher_claims_from_report(match_report, candidate)
                        evidence = build_claim_evidence_from_candidate_preview(candidate)
                        verification = executor.execute(
                            "claim_verify",
                            {
                                "claims": claims,
                                "evidence": evidence,
                                "policy": {"summary_only": True},
                                "metadata": {
                                    "source_component": "production_skill_graph.matcher",
                                    "candidate_id": str(candidate.get("candidate_id") or ""),
                                    "summary_only": True,
                                },
                            },
                            context=context,
                        )
                        skill_results.append(verification)
                        if not verification.success:
                            _append_node_event(
                                runtime_store,
                                "node_failed",
                                context,
                                "claim_verify",
                                {
                                    "status": "failed",
                                    "error_hint": "claim_verify_failed",
                                    "candidate_id": str(candidate.get("candidate_id") or ""),
                                },
                            )
                            return self._failed(state, skill_results, "claim_verify_failed", verification.error, runtime_store, context)
                        verification_summary = summarize_claim_verification_result(dict(verification.output or {}))
                        match_report.update(verification_summary)
                        match_report.setdefault("metadata", {})["claim_verification"] = verification_summary
                        state.metadata.setdefault("claim_verification_summaries", []).append(verification_summary)
                        _append_node_event(
                            runtime_store,
                            "node_completed",
                            context,
                            "claim_verify",
                            {
                                "status": "completed",
                                "candidate_id": str(candidate.get("candidate_id") or ""),
                                **verification_summary,
                            },
                        )
                    state.match_reports.append(match_report)
                    _append_node_event(
                        runtime_store,
                        "node_completed",
                        context,
                        "matcher",
                        {"status": "completed", "candidate_id": str(candidate.get("candidate_id") or "")},
                    )

                if not self._should_refine(state):
                    break
                if state.loop_count >= int(self.config.max_refine_loops):
                    if state.match_reports and state.candidate_previews:
                        state.status = "completed_with_limit"
                        state.metadata["termination_reason"] = "max_refine_loops_reached"
                        state.metadata["max_refine_loops_reached"] = True
                        summary = self.summarize_result(state, skill_results, success=True)
                        _append_graph_event(runtime_store, "graph_completed", context, summary)
                        return summary
                    return self._failed(state, skill_results, "max_loop_exceeded", "max_refine_loops exceeded", runtime_store, context)
                _append_node_event(runtime_store, "node_started", context, "refiner", {"status": "started"})
                refine = executor.execute(
                    "query_refine",
                    {
                        "query": state.retrieval_query or raw_jd,
                        "context": "no candidates or best score below threshold",
                        "metadata": {"runner_type": "production_skill_graph"},
                    },
                    context=context,
                )
                skill_results.append(refine)
                if not refine.success:
                    _append_node_event(runtime_store, "node_failed", context, "refiner", {"status": "failed", "error_hint": "refiner_failed"})
                    return self._failed(state, skill_results, "refiner_failed", refine.error, runtime_store, context)
                state.loop_count += 1
                state.refined_query = str(refine.output.get("refined_query") or "")
                _append_node_event(runtime_store, "node_completed", context, "refiner", {"status": "completed"})

            state.status = "completed"
            summary = self.summarize_result(state, skill_results, success=True)
            _append_graph_event(runtime_store, "graph_completed", context, summary)
            return summary
        except Exception as exc:
            return self._failed(state, skill_results, "graph_exception", type(exc).__name__, runtime_store, context)

    def summarize_result(
        self,
        state: ProductionSkillGraphState,
        skill_results: List[Any],
        *,
        success: bool,
        error_hint: str = "",
        error_type: str = "",
    ) -> Dict[str, Any]:
        planner_summary = _planner_execution_summary(state, skill_results)
        matcher_summary = _matcher_execution_summary(state, skill_results, self.match_callable is None and self.config.use_real_matcher)
        preview_audit = build_candidate_preview_quality_audit(state.candidate_previews)
        result = ProductionSkillGraphResult(
            status=state.status if success and state.status == "completed_with_limit" else ("ok" if success else "failed"),
            success=success,
            job_requirement=_safe_job_requirement_summary(state.job_requirement),
            retrieved_count=len(state.candidate_previews) + len(state.retrieved_documents),
            candidate_profile_preview_count=sum(1 for candidate in state.candidate_previews if _is_preview(candidate)),
            match_report_count=len(state.match_reports),
            refined_query_present=bool(state.refined_query),
            loop_count=int(state.loop_count),
            skill_names=[str(getattr(item, "skill_name", "")) for item in skill_results],
            skill_event_count=len(skill_results),
            fallback_used=bool(planner_summary.get("planner_fallback_used", False)),
            rollback_recommended=bool(not success and self.config.rollback_on_failure),
            rollback_target="legacy_default_graph",
            error_type=error_type,
            error_hint=error_hint,
            provenance=_provenance(state, skill_results, planner_summary),
            candidate_ids=[str(candidate.get("candidate_id") or "") for candidate in state.candidate_previews],
            top_scores=_top_scores(state.match_reports),
            summary_only=True,
            metadata={
                "runner_type": "production_skill_graph",
                "production_skill_graph_enabled": bool(self.config.enabled),
                "legacy_graph_invoked": False,
                "production_graph_replaced": False,
                "rollback_baseline": "legacy_default_graph",
                "summary_only": True,
            },
        )
        data = result.to_dict()
        data.update(
            {
                "candidate_count": len(state.candidate_previews),
                "report_count": len(state.match_reports),
                "match_count": len(state.match_reports),
                "top_score_present": bool(result.top_scores),
                "production_skill_graph_enabled": bool(self.config.enabled),
                "legacy_graph_invoked": False,
                "production_graph_replaced": False,
                "rollback_baseline": "legacy_default_graph",
                "fallback_used": result.fallback_used,
                "termination_reason": str(state.metadata.get("termination_reason") or ""),
                "max_refine_loops_reached": bool(state.metadata.get("max_refine_loops_reached", False)),
                **planner_summary,
                **matcher_summary,
                "candidate_preview_audit": preview_audit,
                "skill_names": result.skill_names,
                "skill_event_count": result.skill_event_count,
                "skill_execution_count": result.skill_event_count,
                "output_keys": [
                    "job_requirement",
                    "candidate_previews",
                    "match_reports",
                    "status",
                    "metadata",
                ],
                "enhanced_candidate_preview_used": any(
                    isinstance(candidate.get("metadata"), Mapping)
                    and bool(candidate["metadata"].get("enhanced_candidate_preview", False))
                    for candidate in state.candidate_previews
                ),
                "candidate_preview_source": "document_chunk_projection_v2" if state.candidate_previews else "",
                "candidate_preview_version": "v2" if state.candidate_previews else "",
                "matcher_input_source": "candidate_profile_preview" if state.candidate_previews else "",
                "candidate_source": _candidate_source(state),
                "mcp_server": _mcp_metadata_value(state, "mcp_server"),
                "mcp_transport": _mcp_metadata_value(state, "mcp_transport"),
                "tool_success_count": _safe_int(_mcp_metadata_value(state, "tool_success_count")),
                "mcp_fallback_used": bool(_mcp_metadata_value(state, "mcp_fallback_used")),
                "mcp_tool_event_count": len(state.metadata.get("tool_events", [])),
                "candidate_previews": [_safe_candidate_preview_summary(candidate) for candidate in state.candidate_previews],
                "match_reports": [_safe_match_report_summary(report) for report in state.match_reports],
                **_claim_verification_summary(state),
            }
        )
        return data

    def _failed(self, state, skill_results, error_hint, error_text, runtime_store, context):
        state.status = "failed"
        error_type = _safe_error_type(error_text, error_hint=error_hint)
        summary = self.summarize_result(
            state,
            skill_results,
            success=False,
            error_hint=error_hint,
            error_type=error_type,
        )
        _append_graph_event(runtime_store, "graph_failed", context, summary)
        return summary

    def _should_refine(self, state: ProductionSkillGraphState) -> bool:
        if not state.candidate_previews:
            return True
        scores = _top_scores(state.match_reports)
        return bool(scores and max(scores) < float(self.config.low_score_threshold))

    def _planner_callable(self):
        if self.planner_extract_callable is not None:
            return self.planner_extract_callable
        if self.config.allow_planner_fallback:
            return _planner_with_optional_fallback
        return None if self.config.use_real_planner else _planner_with_optional_fallback

    def _retriever_callable(self):
        if self.retrieve_callable is not None:
            return self.retrieve_callable
        if self.retriever_callable_factory is not None:
            return self.retriever_callable_factory()
        if self.config.use_real_retriever:
            return build_real_retriever_callable()
        return None


def legacy_state_to_skill_graph_input(state: Mapping[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages") if isinstance(state, Mapping) else []
    raw_jd = ""
    if isinstance(messages, list) and messages:
        raw_jd = str(messages[0].get("content") if isinstance(messages[0], Mapping) else messages[0])
    return {"raw_jd": raw_jd, "summary_only": True}


def skill_graph_result_to_legacy_compatible_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_pool_count": int(result.get("candidate_count") or 0),
        "final_reports_count": int(result.get("report_count") or result.get("match_report_count") or 0),
        "next_action": "end" if result.get("status") == "ok" else "rollback",
        "summary_only": True,
    }


def compare_legacy_and_skill_output_shape(legacy_summary: Mapping[str, Any], skill_summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_count_present": "candidate_count" in skill_summary or "candidate_pool_count" in legacy_summary,
        "report_count_present": "report_count" in skill_summary or "final_reports_count" in legacy_summary,
        "status_present": "status" in skill_summary,
        "summary_only": True,
    }


def build_production_skill_graph_runner(
    config: Optional[ProductionSkillGraphConfig] = None,
    **kwargs: Any,
) -> ProductionSkillGraphRunner:
    return ProductionSkillGraphRunner(config=config, **kwargs)


def _planner_with_optional_fallback(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
    return invoke_planner_agent_for_skill(
        raw_text=input_data["raw_text"],
        metadata={**dict(input_data.get("metadata") or {}), "source": "PlannerAgent"},
        allow_deterministic_fallback=True,
    )


def _build_candidate_previews(retrieval_output: Mapping[str, Any], raw_jd: str, query: str) -> List[Dict[str, Any]]:
    try:
        build_result = build_candidate_profile_previews_from_retrieval_results(
            retrieval_output,
            raw_jd=raw_jd,
            query=query,
        )
        previews = []
        fallback_used = False
        for preview in build_result.previews:
            preview_dict = preview.to_dict()
            try:
                v2_preview = build_candidate_profile_preview_v2(
                    {
                        "candidate_id": preview_dict.get("candidate_id"),
                        "display_name": preview_dict.get("candidate_name"),
                        "source_document_id": preview_dict.get("source_document_id"),
                        "source_file_name": preview_dict.get("source_file_name"),
                        "skills": preview_dict.get("skills") or [],
                        "projects": [preview_dict.get("evidence_summary") or ""],
                        "education": "；".join(preview_dict.get("education_keywords") or []),
                        "work_experience": [preview_dict.get("evidence_summary") or ""],
                        "resume_text": preview_dict.get("evidence_summary") or "",
                        "metadata": {
                            "candidate_id": preview_dict.get("candidate_id"),
                            "candidate_name": preview_dict.get("candidate_name"),
                            "source_document_id": preview_dict.get("source_document_id"),
                            "file_name": preview_dict.get("source_file_name"),
                        },
                    },
                    raw_jd=raw_jd,
                )
                v2_payload = candidate_profile_preview_v2_to_matcher_input(v2_preview)
                v2_payload.setdefault("metadata", {})["enhanced_candidate_preview"] = True
                previews.append(v2_payload)
            except Exception:
                fallback_used = True
                previews.append(candidate_profile_preview_to_matcher_input(preview))
        metadata = dict(retrieval_output.get("metadata") or {})
        metadata.update(
            {
                "candidate_profile_preview_count": len(previews),
                "candidate_profile_preview_source": "document_chunk_projection_v2",
                "enhanced_candidate_preview_used": True,
                "candidate_preview_fallback_used": fallback_used,
                "candidate_preview_grouped_document_count": build_result.grouped_document_count,
                "candidate_preview_skipped_chunk_count": build_result.skipped_chunk_count,
                "candidate_preview_quality_summary": dict(build_result.quality_summary),
                "candidate_preview_version": "v2",
                "summary_only": True,
            }
        )
        retrieval_output["metadata"] = metadata  # type: ignore[index]
        return previews
    except Exception:
        return []


def _derive_query(raw_jd: str, job_requirement: Mapping[str, Any], refined_query: str = "") -> str:
    if refined_query:
        return refined_query
    metadata = job_requirement.get("metadata") if isinstance(job_requirement, Mapping) else {}
    if isinstance(metadata, Mapping) and metadata.get("search_query"):
        return str(metadata["search_query"])
    skills = job_requirement.get("required_skills") or job_requirement.get("tech_stack") if isinstance(job_requirement, Mapping) else []
    if isinstance(skills, list) and skills:
        return " ".join(str(skill) for skill in skills)
    return raw_jd or ""


def _append_graph_event(runtime_store: Any, event_type: str, context: SkillExecutionContext, payload: Mapping[str, Any]) -> None:
    if runtime_store is None or not getattr(context, "task_id", ""):
        return
    runtime_store.append_event(
        event_type,
        session_id=context.session_id,
        task_id=context.task_id,
        payload=build_runtime_event_payload(
            session_id=context.session_id,
            task_id=context.task_id,
            thread_id=context.thread_id,
            graph_mode="skill",
            runner_name="production_skill_graph",
            status=str(payload.get("status") or ("failed" if event_type.endswith("failed") else "completed")),
            error_type=str(payload.get("error_type") or ""),
            error_hint=str(payload.get("error_hint") or ""),
            fallback_used=bool(payload.get("fallback_used", payload.get("planner_fallback_used", False))),
            rollback_recommended=bool(payload.get("rollback_recommended", False)),
            extra=_summary_payload(payload),
        ),
    )


def _append_node_event(
    runtime_store: Any,
    event_type: str,
    context: SkillExecutionContext,
    node_name: str,
    payload: Mapping[str, Any],
) -> None:
    if runtime_store is None or not getattr(context, "task_id", ""):
        return
    runtime_store.append_event(
        event_type,
        session_id=context.session_id,
        task_id=context.task_id,
        payload=build_runtime_event_payload(
            session_id=context.session_id,
            task_id=context.task_id,
            thread_id=context.thread_id,
            graph_mode="skill",
            runner_name="production_skill_graph",
            node_name=node_name,
            status=str(payload.get("status") or ""),
            error_type=str(payload.get("error_type") or ""),
            error_hint=str(payload.get("error_hint") or ""),
            fallback_used=bool(payload.get("fallback_used", False)),
            rollback_recommended=bool(payload.get("rollback_recommended", False)),
            extra=payload,
        ),
    )


def _append_tool_events_from_metadata(runtime_store: Any, context: SkillExecutionContext, metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        return
    events = [event for event in metadata.get("tool_events", []) if isinstance(event, Mapping)]
    if not events:
        return
    for event in events:
        event_type = str(event.get("event_type") or "tool_completed")
        if runtime_store is not None and getattr(context, "task_id", ""):
            runtime_store.append_event(
                event_type,
                session_id=context.session_id,
                task_id=context.task_id,
                payload=build_runtime_event_payload(
                    session_id=context.session_id,
                    task_id=context.task_id,
                    thread_id=context.thread_id,
                    graph_mode="skill",
                    runner_name="production_skill_graph",
                    skill_name=str(event.get("skill_name") or "resume_retrieve"),
                    status=str(event.get("status") or ""),
                    duration_ms=event.get("duration_ms"),
                    error_type=str(event.get("error_type") or ""),
                    fallback_used=bool(event.get("fallback_used", False)),
                    extra={
                        "tool_name": str(event.get("tool_name") or ""),
                        "mcp_server_name": str(event.get("mcp_server_name") or ""),
                        "transport": str(event.get("transport") or ""),
                        "request_id": str(event.get("request_id") or ""),
                        "timeout": bool(event.get("timeout", False)),
                        "permission_decision": str(event.get("permission_decision") or ""),
                        "result_count": _safe_int(event.get("result_count")),
                        "summary_only": True,
                    },
                ),
            )
    context.metadata.setdefault("tool_event_count", 0)
    context.metadata["tool_event_count"] = int(context.metadata["tool_event_count"]) + len(events)


def _summary_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(payload.get("status") or ""),
        "error_type": str(payload.get("error_type") or ""),
        "error_hint": str(payload.get("error_hint") or ""),
        "rollback_recommended": bool(payload.get("rollback_recommended", False)),
        "skill_event_count": int(payload.get("skill_event_count") or 0),
        "skill_execution_count": int(payload.get("skill_execution_count") or payload.get("skill_event_count") or 0),
        "planner_source": str(payload.get("planner_source") or ""),
        "real_planner_invoked": bool(payload.get("real_planner_invoked", False)),
        "real_planner_failed": bool(payload.get("real_planner_failed", False)),
        "planner_fallback_used": bool(payload.get("planner_fallback_used", False)),
        "planner_fallback_type": str(payload.get("planner_fallback_type") or ""),
        "fallback_not_real_planner_success": bool(payload.get("fallback_not_real_planner_success", False)),
        "planner_invocation_stage": str(payload.get("planner_invocation_stage") or ""),
        "provider_error_type": str(payload.get("provider_error_type") or ""),
        "summary_only": True,
    }


def _safe_job_requirement_summary(job_requirement: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(job_requirement, Mapping):
        return {}
    return {
        "keys": sorted(str(key) for key in job_requirement.keys()),
        "required_skill_count": len(job_requirement.get("required_skills") or job_requirement.get("tech_stack") or []),
        "summary_only": True,
    }


def _candidate_source(state: ProductionSkillGraphState) -> str:
    return str(state.metadata.get("candidate_source") or state.metadata.get("retriever_candidate_source") or "")


def _mcp_metadata_value(state: ProductionSkillGraphState, key: str) -> Any:
    return state.metadata.get(key, "")


def _planner_fallback_used(job_requirement: Mapping[str, Any]) -> bool:
    metadata = job_requirement.get("metadata") if isinstance(job_requirement, Mapping) else {}
    return isinstance(metadata, Mapping) and bool(metadata.get("planner_fallback_used", False))


def _provenance(
    state: ProductionSkillGraphState,
    skill_results: List[Any],
    planner_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    job_metadata = state.job_requirement.get("metadata") if isinstance(state.job_requirement, Mapping) else {}
    planner_summary = dict(planner_summary or {})
    return {
        "planner_source": (
            "deterministic_planner_fallback"
            if bool(planner_summary.get("planner_fallback_used", False)) or _planner_fallback_used(state.job_requirement)
            else str(planner_summary.get("planner_source") or (job_metadata or {}).get("source") or "PlannerAgent")
        ),
        "retriever_source": "RetrieverSkill",
        "matcher_source": "CandidateMatchSkill",
        "candidate_preview_source": "document_chunk_projection_v2" if state.candidate_previews else "",
        "candidate_preview_version": "v2" if state.candidate_previews else "",
        "matcher_input_source": "candidate_profile_preview" if state.match_reports else "",
        "skill_names": [str(getattr(item, "skill_name", "")) for item in skill_results],
        "summary_only": True,
    }


def _planner_execution_summary(state: ProductionSkillGraphState, skill_results: List[Any]) -> Dict[str, Any]:
    job_metadata = {}
    if isinstance(state.job_requirement, Mapping):
        job_metadata = dict(state.job_requirement.get("metadata") or {})

    planner_error = ""
    planner_failed = False
    for skill_result in skill_results:
        if getattr(skill_result, "skill_name", "") != "planner_extract":
            continue
        planner_failed = not bool(getattr(skill_result, "success", False))
        planner_error = str(getattr(skill_result, "error", "") or "")
        break

    fallback_used = bool(job_metadata.get("planner_fallback_used", False))
    invocation_stage = (
        str(job_metadata.get("planner_invocation_stage") or "")
        or _extract_marker_value(planner_error, "planner_invocation_stage")
    )
    provider_error_type = (
        str(job_metadata.get("provider_error_type") or "")
        or _extract_marker_value(planner_error, "provider_error_type")
    )
    input_keys_marker = _extract_marker_value(planner_error, "planner_input_keys")
    output_keys_marker = _extract_marker_value(planner_error, "planner_output_keys")
    has_messages = _extract_marker_value(planner_error, "planner_has_messages")
    raw_text_length = (
        str(job_metadata.get("planner_raw_text_length") or "")
        or _extract_marker_value(planner_error, "planner_raw_text_length")
    )

    planner_input_shape: Dict[str, Any] = {}
    if invocation_stage or input_keys_marker or raw_text_length:
        planner_input_shape = {
            "input_keys": sorted(key for key in input_keys_marker.split(",") if key),
            "has_messages": has_messages == "true",
            "raw_text_length": _safe_int(raw_text_length),
            "summary_only": True,
        }
    elif fallback_used:
        planner_input_shape = {
            "input_keys": [],
            "has_messages": False,
            "raw_text_length": len(state.raw_jd or ""),
            "summary_only": True,
        }

    real_invoked = bool(job_metadata.get("real_planner_invoked", False))
    real_failed = bool(job_metadata.get("real_planner_failed", False))
    if planner_failed and invocation_stage:
        real_invoked = True
        real_failed = True

    if fallback_used:
        planner_source = "deterministic_fallback"
    elif real_invoked or planner_failed or job_metadata.get("source"):
        planner_source = str(job_metadata.get("source") or "real_wrapper")
    else:
        planner_source = ""

    return {
        "planner_source": planner_source,
        "real_planner_invoked": real_invoked,
        "real_planner_failed": real_failed,
        "planner_fallback_used": fallback_used,
        "planner_fallback_type": str(job_metadata.get("planner_fallback_type") or ""),
        "fallback_not_real_planner_success": bool(
            fallback_used or job_metadata.get("fallback_not_real_planner_success", False)
        ),
        "planner_invocation_stage": invocation_stage,
        "planner_input_shape": planner_input_shape,
        "planner_output_keys": sorted(key for key in output_keys_marker.split(",") if key),
        "planner_expected_keys": [
            "job_requirement",
            "extracted_jd",
            "tech_stack",
            "education",
            "must_have",
            "search_query",
        ] if planner_failed else [],
        "planner_adapter_error_hint": _planner_adapter_error_hint(planner_error, planner_failed),
        "planner_provider_diagnostics": _planner_provider_diagnostics_from_error(planner_error),
        "provider_error_type": provider_error_type,
        "summary_only": True,
    }


def _planner_adapter_error_hint(error: str, planner_failed: bool) -> str:
    if not planner_failed:
        return ""
    if "planner_schema_adapter_failed" in error:
        return "planner_schema_adapter_failed"
    if "planner_wrapper_failed" in error:
        return "planner_wrapper_failed"
    return "planner_failed"


def _planner_provider_diagnostics_from_error(error: str) -> Dict[str, Any]:
    if not error:
        return {}
    diagnostics = {
        "dotenv_loaded": _parse_marker_bool_or_skip(_extract_marker_value(error, "planner_provider_dotenv_loaded")),
        "dotenv_path_present": _extract_marker_value(error, "planner_provider_dotenv_path_present") == "true",
        "openai_api_key": _safe_set_missing(_extract_marker_value(error, "planner_provider_openai_api_key")),
        "openai_api_base": _safe_set_missing(_extract_marker_value(error, "planner_provider_openai_api_base")),
        "llm_model": _extract_marker_value(error, "planner_provider_llm_model"),
        "planner_agent_class": _extract_marker_value(error, "planner_provider_planner_agent_class") or "PlannerAgent",
        "invocation_method": _extract_marker_value(error, "planner_provider_invocation_method") or "__call__",
        "provider_error_type": _extract_marker_value(error, "provider_error_type"),
        "summary_only": True,
    }
    if not any(value for key, value in diagnostics.items() if key != "summary_only"):
        return {}
    return diagnostics


def _matcher_execution_summary(
    state: ProductionSkillGraphState,
    skill_results: List[Any],
    real_matcher_path: bool,
) -> Dict[str, Any]:
    matcher_results = [
        skill_result
        for skill_result in skill_results
        if getattr(skill_result, "skill_name", "") == "candidate_match"
    ]
    if not matcher_results:
        return {
            "matcher_invocation_stage": "",
            "matcher_input_keys": [],
            "matcher_candidate_id": "",
            "matcher_candidate_name_present": False,
            "matcher_skills_count": 0,
            "matcher_source": "",
            "matcher_input_source": "candidate_profile_preview" if state.candidate_previews else "",
            "real_matcher_invoked": False,
            "real_matcher_failed": False,
            "matcher_adapter_error_hint": "",
            "matcher_provider_error_type": "",
            "matcher_output_keys": [],
            "summary_only": True,
        }

    last = matcher_results[-1]
    failed = not bool(getattr(last, "success", False))
    output = getattr(last, "output", None)
    output_metadata = output.get("metadata") if isinstance(output, Mapping) else {}
    last_input = dict(state.metadata.get("last_matcher_input_summary") or {})
    output_keys = sorted(str(key) for key in output.keys()) if isinstance(output, Mapping) else []
    return {
        "matcher_invocation_stage": "skill_execute" if failed else "completed",
        "matcher_input_keys": sorted(str(key) for key in last_input.get("input_keys", [])),
        "matcher_candidate_id": str(last_input.get("candidate_id") or ""),
        "matcher_candidate_name_present": bool(last_input.get("candidate_name_present", False)),
        "matcher_skills_count": _safe_int(last_input.get("skills_count")),
        "matcher_source": str(
            (output_metadata or {}).get("source")
            or ("MatcherAgent" if real_matcher_path else "CandidateMatchSkill")
        ),
        "matcher_input_source": "candidate_profile_preview" if state.candidate_previews else "",
        "real_matcher_invoked": bool(real_matcher_path),
        "real_matcher_failed": bool(real_matcher_path and failed),
        "matcher_adapter_error_hint": "matcher_wrapper_failed" if failed else "",
        "matcher_provider_error_type": _matcher_provider_error_type(str(getattr(last, "error", "") or ""), failed),
        "matcher_output_keys": output_keys,
        "summary_only": True,
    }


def _matcher_input_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "input_keys": [
            "job_requirement",
            "candidate_profile",
            "evidence",
            "metadata",
        ],
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_name_present": bool(candidate.get("candidate_name_resolved") or candidate.get("name") or candidate.get("candidate_name")),
        "skills_count": len(candidate.get("skills") or []) if isinstance(candidate.get("skills"), list) else 0,
        "summary_only": True,
    }


def _matcher_provider_error_type(error: str, failed: bool) -> str:
    if not failed:
        return ""
    if "TypeError" in error:
        return "TypeError"
    if "ValueError" in error:
        return "ValueError"
    if "RuntimeError" in error:
        return "RuntimeError"
    if "can only concatenate" in error:
        return "TypeError"
    return ""


def _extract_marker_value(text: str, marker: str) -> str:
    token = f"{marker}="
    if token not in text:
        return ""
    return text.split(token, 1)[1].split(" ", 1)[0].strip()


def _parse_marker_bool_or_skip(value: str):
    if value == "true":
        return True
    if value == "false":
        return False
    return "skip"


def _safe_set_missing(value: str) -> str:
    return "set" if value == "set" else "missing"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _top_scores(match_reports: List[Mapping[str, Any]]) -> List[float]:
    scores = []
    for report in match_reports:
        value = report.get("total_score")
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def _safe_candidate_preview_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_name": str(candidate.get("candidate_name") or candidate.get("name") or ""),
        "source_document_id": str(candidate.get("source_document_id") or ""),
        "source_file_name": str(candidate.get("source_file_name") or ""),
        "candidate_name_resolved": bool(candidate.get("candidate_name_resolved", False)),
        "preview_version": str(candidate.get("preview_version") or (candidate.get("metadata") or {}).get("preview_version") or ""),
        "project_keywords_present": bool(candidate.get("project_keywords") or candidate.get("projects")),
        "education_keywords_present": bool(candidate.get("education_keywords") or candidate.get("education")),
        "evidence_summary_present": bool(candidate.get("evidence_text_summary") or candidate.get("evidence_summary")),
        "summary_only": True,
    }


def _safe_match_report_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), Mapping) else {}
    data = {
        "candidate_id": str(report.get("candidate_id") or metadata.get("candidate_id") or ""),
        "candidate_name": str(report.get("candidate_name") or metadata.get("candidate_name") or ""),
        "total_score": report.get("total_score") if isinstance(report.get("total_score"), (int, float)) else 0,
        "summary_only": True,
    }
    if report.get("claim_verification_status"):
        data.update(
            {
                "claim_verification_status": str(report.get("claim_verification_status") or ""),
                "claim_support_rate": float(report.get("claim_support_rate") or 0.0),
                "unsupported_claim_count": int(report.get("unsupported_claim_count") or 0),
                "critical_unsupported_claim_count": int(report.get("critical_unsupported_claim_count") or 0),
            }
        )
    return data


def _claim_verification_summary(state: ProductionSkillGraphState) -> Dict[str, Any]:
    summaries = [
        item
        for item in state.metadata.get("claim_verification_summaries", [])
        if isinstance(item, Mapping)
    ]
    if not summaries:
        return {
            "claim_verification_enabled": False,
            "claim_verification_status": "",
            "claim_verification_case_count": 0,
            "claim_support_pass_rate": 0.0,
            "unsupported_claim_case_rate": 0.0,
            "critical_unsupported_claim_rate": 0.0,
            "evidence_coverage_rate": 0.0,
            "summary_only": True,
        }
    passed = sum(1 for item in summaries if str(item.get("claim_verification_status") or "") == "passed")
    unsupported_cases = sum(1 for item in summaries if int(item.get("unsupported_claim_count") or 0) > 0)
    critical_cases = sum(1 for item in summaries if int(item.get("critical_unsupported_claim_count") or 0) > 0)
    avg_support = sum(float(item.get("claim_support_rate") or 0.0) for item in summaries) / len(summaries)
    avg_coverage = sum(float(item.get("evidence_coverage_rate") or 0.0) for item in summaries) / len(summaries)
    overall_status = "passed" if passed == len(summaries) else ("rejected" if critical_cases else "review_required")
    return {
        "claim_verification_enabled": True,
        "claim_verification_status": overall_status,
        "claim_verification_case_count": len(summaries),
        "claim_support_pass_rate": round(passed / len(summaries), 6),
        "average_claim_support_rate": round(avg_support, 6),
        "unsupported_claim_case_rate": round(unsupported_cases / len(summaries), 6),
        "critical_unsupported_claim_rate": round(critical_cases / len(summaries), 6),
        "evidence_coverage_rate": round(avg_coverage, 6),
        "summary_only": True,
    }


def _is_preview(candidate: Mapping[str, Any]) -> bool:
    metadata = candidate.get("metadata") if isinstance(candidate, Mapping) else {}
    return isinstance(metadata, Mapping) and bool(metadata.get("candidate_profile_preview", False))


def _safe_error_type(error_text: Any, *, error_hint: str = "") -> str:
    text = str(error_text or "")
    if not text:
        return "SkillGraphFailed"
    if error_hint == "matcher_failed":
        return "MatcherSkillFailed"
    if error_hint == "planner_failed":
        return "PlannerSkillFailed"
    if error_hint == "retriever_failed":
        return "RetrieverSkillFailed"
    if error_hint == "refiner_failed":
        return "RefinerSkillFailed"
    if "retriever" in text.lower():
        return "RetrieverSkillFailed"
    if "planner" in text.lower():
        return "PlannerSkillFailed"
    if "matcher" in text.lower():
        return "MatcherSkillFailed"
    if "refiner" in text.lower():
        return "RefinerSkillFailed"
    return text.split()[0][:80]
