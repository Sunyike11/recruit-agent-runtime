import importlib
import importlib.util
import os
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.skills.agent_adapters import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RetrieverSkill,
    invoke_planner_agent_for_skill,
)
from src.skills.context import SkillExecutionContext
from src.skills.execution import SkillExecutor
from src.skills.registry import SkillRegistry
from src.skills.workflow import RecruitmentSkillWorkflow, SkillWorkflowResult
from src.runtime.candidate_preview import (
    build_candidate_preview_quality_audit,
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_skill_backed_variant_runner(top_k: int = 3, low_score_threshold: float = 60.0):
    """Build a deterministic skill-backed variant runner for explicit runtime smoke use.

    This runner uses injected fake callables only. It does not import or call real
    Planner/Matcher/Retriever/Refiner agents.
    """

    def run(raw_jd: str, memory_context=None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        registry = build_deterministic_skill_registry()
        executor = SkillExecutor(registry)
        context = SkillExecutionContext(
            memory_context=memory_context,
            metadata={
                "runner_type": "skill_backed_variant",
                "deterministic_variant": True,
                "summary_only": True,
                "metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
            },
        )
        workflow = RecruitmentSkillWorkflow(executor, low_score_threshold=low_score_threshold)
        result = workflow.run(
            raw_jd=raw_jd,
            top_k=top_k,
            context=context,
            metadata={
                "runner_type": "skill_backed_variant",
                "deterministic_variant": True,
                "memory_context_provided": bool(memory_context is not None),
            },
        )
        return summarize_skill_backed_variant_result(
            result,
            raw_jd_length=len(raw_jd or ""),
            memory_context_provided=bool(memory_context is not None),
            memory_context_summary=(metadata or {}).get("memory_context_summary"),
        )

    return run


def build_real_skill_wrapper_variant_runner(
    *,
    planner_extract_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]] = None,
    retrieve_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]] = None,
    match_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]] = None,
    refine_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]] = None,
    use_real_retriever_callable: bool = False,
    require_embedding_cache_ready: bool = False,
    skip_if_embedding_cache_unavailable: bool = True,
    allow_hf_network_probe: bool = False,
    embedding_readiness_checker: Optional[Callable[..., Dict[str, Any]]] = None,
    allow_planner_deterministic_fallback: bool = False,
    enable_candidate_preview_projection: bool = False,
    llm_readiness_checker: Optional[Callable[..., Dict[str, Any]]] = None,
    top_k: int = 3,
    low_score_threshold: float = 60.0,
):
    """Build an explicit opt-in real wrapper variant runner.

    Missing callables fall back to each skill wrapper's lazy real adapter, except
    retrieval, which remains injected-only in Phase8E to avoid implicit Chroma /
    LlamaIndex / RetrieverAgent initialization.
    """

    def run(raw_jd: str, memory_context=None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        effective_retrieve_callable = retrieve_callable
        if effective_retrieve_callable is None and use_real_retriever_callable:
            effective_retrieve_callable = build_real_retriever_callable(
                require_embedding_cache_ready=require_embedding_cache_ready,
                skip_if_embedding_cache_unavailable=skip_if_embedding_cache_unavailable,
                allow_hf_network_probe=allow_hf_network_probe,
                embedding_readiness_checker=embedding_readiness_checker,
            )
        readiness = _real_wrapper_readiness(
            planner_extract_callable=planner_extract_callable,
            retrieve_callable=effective_retrieve_callable,
            match_callable=match_callable,
            refine_callable=refine_callable,
            llm_readiness_checker=llm_readiness_checker,
        )
        if readiness["status"] != "ok":
            return _real_wrapper_skipped_summary(
                readiness,
                raw_jd_length=len(raw_jd or ""),
                memory_context_provided=bool(memory_context is not None),
            )

        try:
            effective_planner_callable = planner_extract_callable
            if effective_planner_callable is None and allow_planner_deterministic_fallback:
                effective_planner_callable = _planner_callable_with_deterministic_fallback
            registry = build_real_skill_wrapper_registry(
                planner_extract_callable=effective_planner_callable,
                retrieve_callable=effective_retrieve_callable,
                match_callable=match_callable,
                refine_callable=refine_callable,
            )
            executor = SkillExecutor(registry)
            workflow_cls = (
                CandidatePreviewRecruitmentSkillWorkflow
                if enable_candidate_preview_projection
                else RecruitmentSkillWorkflow
            )
            context = SkillExecutionContext(
                memory_context=memory_context,
                metadata={
                    "runner_type": "real_skill_wrapper_variant",
                    "real_skill_wrapper_mode": True,
                    "summary_only": True,
                    "metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
                },
            )
            workflow = workflow_cls(executor, low_score_threshold=low_score_threshold)
            result = workflow.run(
                raw_jd=raw_jd,
                top_k=top_k,
                context=context,
                metadata={
                    "runner_type": "real_skill_wrapper_variant",
                    "real_skill_wrapper_mode": True,
                    "memory_context_provided": bool(memory_context is not None),
                },
            )
            return summarize_skill_backed_variant_result(
                result,
                raw_jd_length=len(raw_jd or ""),
                memory_context_provided=bool(memory_context is not None),
                runner_type="real_skill_wrapper_variant",
                deterministic_variant=False,
                real_skill_wrapper_mode=True,
                env_readiness=readiness.get("env_readiness"),
                memory_context_summary=(metadata or {}).get("memory_context_summary"),
            )
        except Exception as exc:
            return _real_wrapper_failed_summary(
                type(exc).__name__,
                raw_jd_length=len(raw_jd or ""),
                memory_context_provided=bool(memory_context is not None),
                env_readiness=readiness.get("env_readiness"),
            )

    return run


def _planner_callable_with_deterministic_fallback(
    input_data: Dict[str, Any],
    _context: Optional[SkillExecutionContext] = None,
):
    return invoke_planner_agent_for_skill(
        raw_text=input_data["raw_text"],
        metadata={**dict(input_data.get("metadata") or {}), "source": "PlannerAgent"},
        allow_deterministic_fallback=True,
    )


def build_real_retriever_callable(
    search_runner=None,
    retriever_factory=None,
    *,
    require_embedding_cache_ready: bool = False,
    skip_if_embedding_cache_unavailable: bool = True,
    allow_hf_network_probe: bool = False,
    embedding_readiness_checker: Optional[Callable[..., Dict[str, Any]]] = None,
):
    """Build a lazy ResumeRetriever callable adapted for RetrieverSkill.

    The callable returns summary-only document/chunk evidence. It does not expose
    full resume text or metadata values.
    """

    retriever_holder = {"retriever": None, "init_diagnostics": {}}

    def call_retriever(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
        query = _retriever_query_from_input(input_data)
        top_k = input_data.get("top_k")
        if not isinstance(top_k, int) or top_k < 0:
            top_k = 3

        try:
            if search_runner is not None:
                try:
                    raw = search_runner(query, top_k)
                except Exception as exc:
                    raise RuntimeError("retriever_search_failed") from exc
            else:
                if retriever_holder["retriever"] is None:
                    try:
                        retriever, diagnostics = build_resume_retriever_for_runtime(
                            retriever_factory=retriever_factory,
                            require_embedding_cache_ready=require_embedding_cache_ready,
                            skip_if_embedding_cache_unavailable=skip_if_embedding_cache_unavailable,
                            allow_hf_network_probe=allow_hf_network_probe,
                            embedding_readiness_checker=embedding_readiness_checker,
                        )
                        retriever_holder["retriever"] = retriever
                        retriever_holder["init_diagnostics"] = diagnostics
                    except Exception as exc:
                        if isinstance(exc, RuntimeError) and (
                            "retriever_init_failed" in str(exc)
                            or "retriever_embedding_cache_unavailable" in str(exc)
                        ):
                            raise
                        raise RuntimeError("retriever_init_failed retriever_init_stage=instantiate_resume_retriever") from exc
                try:
                    raw = retriever_holder["retriever"].search(query, k=top_k)
                except Exception as exc:
                    raise RuntimeError("retriever_search_failed") from exc
            results = _raw_search_results(raw)
            output = adapt_resume_retriever_results(results)
            metadata = dict(output.get("metadata") or {})
            if search_runner is not None:
                factory_source = "search_runner"
            elif retriever_factory is not None:
                factory_source = "injected_retriever_factory"
            else:
                factory_source = "resume_retriever_runtime"
            metadata.update(
                {
                    "retriever_factory_source": factory_source,
                    "summary_only": True,
                }
            )
            output["metadata"] = metadata
            return output
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("retriever_result_adapter_failed") from exc

    return call_retriever


def build_resume_retriever_for_runtime(
    retriever_factory=None,
    *,
    require_embedding_cache_ready: bool = False,
    skip_if_embedding_cache_unavailable: bool = True,
    allow_hf_network_probe: bool = False,
    embedding_readiness_checker: Optional[Callable[..., Dict[str, Any]]] = None,
):
    diagnostics = {
        "retriever_init_stage": "load_dotenv",
        "retriever_config_summary": {},
        "retriever_init_diagnostics": {},
    }
    try:
        config = resolve_runtime_retriever_config()
        diagnostics["retriever_config_summary"] = dict(config["summary"])
        diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(config=config)
    except Exception as exc:
        raise RuntimeError("retriever_init_failed retriever_init_stage=config_load") from exc
    readiness_checker = embedding_readiness_checker or check_retriever_embedding_readiness
    embedding_readiness = readiness_checker(
        config=config,
        allow_hf_network_probe=allow_hf_network_probe,
    )
    if _should_stop_for_embedding_readiness(
        embedding_readiness,
        require_embedding_cache_ready=require_embedding_cache_ready,
        skip_if_embedding_cache_unavailable=skip_if_embedding_cache_unavailable,
    ):
        diagnostics["retriever_init_stage"] = "embedding_readiness"
        diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
            config=config,
            error_stage="embedding_readiness",
            error_type="RetrieverEmbeddingCacheUnavailable",
            embedding_readiness=embedding_readiness,
        )
        raise RuntimeError(
            "retriever_embedding_cache_unavailable retriever_init_stage=embedding_readiness "
            + _retriever_diagnostic_markers(diagnostics["retriever_init_diagnostics"])
        )

    if retriever_factory is not None:
        diagnostics["retriever_init_stage"] = "instantiate_resume_retriever"
        diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
            config=config,
            resume_retriever_class=_factory_class_name(retriever_factory),
            error_stage="instantiate_resume_retriever",
            embedding_readiness=embedding_readiness,
        )
        try:
            retriever = retriever_factory()
        except Exception as exc:
            raise RuntimeError(
                "retriever_init_failed retriever_init_stage=instantiate_resume_retriever "
                + _retriever_diagnostic_markers(diagnostics["retriever_init_diagnostics"])
            ) from exc
    else:
        diagnostics["retriever_init_stage"] = "import_resume_retriever"
        diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
            config=config,
            error_stage="import_resume_retriever",
            embedding_readiness=embedding_readiness,
        )
        try:
            from src.services.retriever import ResumeRetriever
        except Exception as exc:
            diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
                config=config,
                error_stage="import_resume_retriever",
                error_type=type(exc).__name__,
                embedding_readiness=embedding_readiness,
            )
            raise RuntimeError(
                "retriever_init_failed retriever_init_stage=import_resume_retriever "
                + _retriever_diagnostic_markers(diagnostics["retriever_init_diagnostics"])
            ) from exc

        diagnostics["retriever_init_stage"] = "instantiate_resume_retriever"
        diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
            config=config,
            resume_retriever_class="ResumeRetriever",
            error_stage="instantiate_resume_retriever",
            embedding_readiness=embedding_readiness,
        )
        try:
            retriever = ResumeRetriever(persist_dir=str(config["chroma_dir"]))
        except Exception as exc:
            diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
                config=config,
                resume_retriever_class="ResumeRetriever",
                error_stage="instantiate_resume_retriever",
                error_type=type(exc).__name__,
                embedding_readiness=embedding_readiness,
            )
            raise RuntimeError(
                "retriever_init_failed retriever_init_stage=instantiate_resume_retriever "
                + _retriever_diagnostic_markers(diagnostics["retriever_init_diagnostics"])
            ) from exc

    diagnostics["retriever_init_stage"] = "index_load"
    diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
        config=config,
        resume_retriever_class=type(retriever).__name__,
        error_stage="index_load",
        embedding_readiness=embedding_readiness,
    )
    if getattr(retriever, "index", True) is None:
        raise RuntimeError(
            "retriever_init_failed retriever_init_stage=index_load "
            + _retriever_diagnostic_markers(diagnostics["retriever_init_diagnostics"])
        )
    diagnostics["retriever_init_stage"] = "ready"
    diagnostics["retriever_init_diagnostics"] = build_retriever_init_diagnostics(
        config=config,
        resume_retriever_class=type(retriever).__name__,
        error_stage="",
        embedding_readiness=embedding_readiness,
    )
    return retriever, diagnostics


def resolve_runtime_retriever_config() -> Dict[str, Any]:
    dotenv_status = load_project_dotenv_for_real_wrappers()
    try:
        from src.config import get_settings
    except Exception as exc:
        raise RuntimeError("retriever_init_failed retriever_init_stage=config_load") from exc

    settings = get_settings()
    project_root = Path(settings.project_root)
    data_dir = Path(settings.data_dir)
    chroma_dir = Path(settings.chroma_dir)
    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "chroma_dir": chroma_dir,
        "embedding_model": settings.embedding_model,
        "retriever_top_k": settings.retriever_top_k,
        "summary": {
            "project_root_present": project_root.exists(),
            "data_dir_present": data_dir.exists(),
            "chroma_dir_present": chroma_dir.exists(),
            "chroma_dir_non_empty": chroma_dir.exists() and any(chroma_dir.iterdir()),
            "dotenv_loaded": dotenv_status.get("dotenv_loaded", "skip"),
            "summary_only": True,
        },
    }


def build_retriever_init_diagnostics(
    *,
    config: Optional[Mapping[str, Any]] = None,
    persist_dir: Optional[Any] = None,
    resume_retriever_class: str = "",
    init_method: str = "ResumeRetriever(persist_dir)",
    error_stage: str = "",
    error_type: str = "",
    embedding_readiness: Optional[Mapping[str, Any]] = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Build summary-only retriever init diagnostics without importing real retriever classes."""

    if config is None:
        try:
            config = resolve_runtime_retriever_config()
        except Exception:
            config = {}
    project_root = Path(config.get("project_root") or PROJECT_ROOT)
    data_dir = Path(config.get("data_dir") or project_root / "data")
    chroma_dir = Path(persist_dir or config.get("chroma_dir") or project_root / "chroma_db")
    dependency_imports = _retriever_dependency_import_summary(import_module=import_module)
    embedding = _safe_embedding_readiness(embedding_readiness)
    chroma_file_count = _safe_dir_file_count(chroma_dir)
    return {
        "project_root_present": project_root.exists(),
        "project_root": "set" if project_root else "",
        "data_dir_present": data_dir.exists(),
        "chroma_dir_present": chroma_dir.exists(),
        "chroma_dir_non_empty": chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False,
        "chroma_dir_file_count": chroma_file_count,
        "persist_dir_used": "set" if chroma_dir else "",
        "persist_dir_exists": chroma_dir.exists(),
        "persist_dir_non_empty": chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False,
        "resume_retriever_class": str(resume_retriever_class or ""),
        "init_method": str(init_method or "ResumeRetriever(persist_dir)"),
        "import_stage_ok": bool(dependency_imports.get("resume_retriever_importable", False)),
        "settings_loaded": bool(config),
        "embedding_dependency_importable": bool(dependency_imports.get("embedding_dependency_importable", False)),
        "chroma_dependency_importable": bool(dependency_imports.get("chroma_dependency_importable", False)),
        "llama_index_dependency_importable": bool(dependency_imports.get("llama_index_dependency_importable", False)),
        "embedding_readiness": embedding,
        "error_stage": str(error_stage or ""),
        "error_type": str(error_type or ""),
        "summary_only": True,
    }


def check_retriever_embedding_readiness(
    *,
    config: Optional[Mapping[str, Any]] = None,
    allow_hf_network_probe: bool = False,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Return summary-only embedding/cache readiness without network access by default."""

    if config is None:
        try:
            config = resolve_runtime_retriever_config()
        except Exception:
            config = {}
    model_name = str(config.get("embedding_model") or "")
    dependency_imports = _retriever_dependency_import_summary(import_module=import_module)
    cache_env_set = any(
        os.environ.get(name)
        for name in (
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "SENTENCE_TRANSFORMERS_HOME",
        )
    )
    cache_likely_available: Any = True if cache_env_set else "unknown"
    return {
        "embedding_model_name": model_name,
        "hf_token_status": "set" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") else "missing",
        "hf_home_set": bool(os.environ.get("HF_HOME")),
        "transformers_cache_set": bool(os.environ.get("TRANSFORMERS_CACHE")),
        "sentence_transformers_cache_set": bool(os.environ.get("SENTENCE_TRANSFORMERS_HOME")),
        "cache_probe_supported": True,
        "cache_likely_available": cache_likely_available,
        "network_required_unknown": not bool(allow_hf_network_probe),
        "allow_hf_network_probe": bool(allow_hf_network_probe),
        "embedding_dependency_importable": bool(dependency_imports.get("embedding_dependency_importable", False)),
        "chroma_dependency_importable": bool(dependency_imports.get("chroma_dependency_importable", False)),
        "llama_index_dependency_importable": bool(dependency_imports.get("llama_index_dependency_importable", False)),
        "summary_only": True,
    }


def adapt_resume_retriever_results(results) -> Dict[str, Any]:
    try:
        result_list = [dict(result) for result in (results or [])]
        resume_documents = []
        evidence = []
        source_keys = set()
        for index, result in enumerate(result_list):
            text = result.get("text", "")
            metadata = dict(result.get("metadata") or {})
            score = result.get("score")
            metadata_keys = sorted(str(key) for key in metadata.keys())
            source_keys.update(metadata_keys)
            safe_text = text if isinstance(text, str) else ""
            item = {
                "rank": index + 1,
                "text_length": len(text) if isinstance(text, str) else 0,
                "metadata_keys": metadata_keys,
                "score_present": isinstance(score, (int, float)),
                "skills": _extract_safe_skill_keywords(safe_text),
            }
            file_name = _safe_metadata_identifier(metadata.get("file_name"))
            source = _safe_metadata_identifier(metadata.get("source"))
            if file_name:
                item["file_name"] = file_name
            if source:
                item["source"] = source
            resume_documents.append(dict(item))
            evidence.append(dict(item))
        return {
            "candidates": [],
            "resume_documents": resume_documents,
            "evidence": evidence,
            "metadata": {
                "retriever_invoked": True,
                "summary_only": True,
                "source": "document_chunk_retrieval",
                "candidate_profile_level": False,
                "source_keys": sorted(source_keys),
                "result_count": len(result_list),
            },
        }
    except Exception as exc:
        raise RuntimeError("retriever_result_adapter_failed") from exc


def build_deterministic_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=_fake_planner_extract))
    registry.register(RetrieverSkill(retrieve_callable=_fake_resume_retrieve))
    registry.register(CandidateMatchSkill(match_callable=_fake_candidate_match))
    registry.register(QueryRefineSkill(refine_callable=_fake_query_refine))
    return registry


def build_real_skill_wrapper_registry(
    *,
    planner_extract_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]],
    retrieve_callable: Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any],
    match_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]],
    refine_callable: Optional[Callable[[Dict[str, Any], Optional[SkillExecutionContext]], Any]],
) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=planner_extract_callable))
    registry.register(RetrieverSkill(retrieve_callable=retrieve_callable))
    registry.register(CandidateMatchSkill(match_callable=match_callable))
    registry.register(QueryRefineSkill(refine_callable=refine_callable))
    return registry


class CandidatePreviewRecruitmentSkillWorkflow(RecruitmentSkillWorkflow):
    def run(
        self,
        raw_jd: str,
        top_k: int = 5,
        context: Optional[SkillExecutionContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillWorkflowResult:
        result = super().run(raw_jd=raw_jd, top_k=top_k, context=context, metadata=metadata)
        return result

    def _execute_step(
        self,
        workflow_result: SkillWorkflowResult,
        skill_name: str,
        input_data: Dict[str, Any],
        context: SkillExecutionContext,
    ):
        result = super()._execute_step(workflow_result, skill_name, input_data, context)
        if skill_name == "resume_retrieve" and result.success:
            output = dict(result.output or {})
            candidates = list(output.get("candidates", []))
            if not candidates:
                previews, preview_metadata = _build_candidate_previews_for_workflow(output, input_data)
                if previews:
                    output["candidates"] = previews
                    metadata = dict(output.get("metadata") or {})
                    metadata.update(
                        {
                            "candidate_profile_preview_count": len(previews),
                            "candidate_profile_preview_source": preview_metadata.get(
                                "candidate_profile_preview_source",
                                "document_chunk_projection",
                            ),
                            "enhanced_candidate_preview_used": bool(
                                preview_metadata.get("enhanced_candidate_preview_used", False)
                            ),
                            "candidate_preview_fallback_used": bool(
                                preview_metadata.get("candidate_preview_fallback_used", False)
                            ),
                            "candidate_preview_grouped_document_count": _safe_int(
                                preview_metadata.get("candidate_preview_grouped_document_count")
                            ),
                            "candidate_preview_skipped_chunk_count": _safe_int(
                                preview_metadata.get("candidate_preview_skipped_chunk_count")
                            ),
                            "candidate_preview_quality_summary": dict(
                                preview_metadata.get("candidate_preview_quality_summary") or {}
                            ),
                            "summary_only": True,
                        }
                    )
                    output["metadata"] = metadata
                    result.output = output
        return result


SAFE_SKILL_KEYWORDS = [
    "Python",
    "RAG",
    "LangGraph",
    "PyTorch",
    "3DGS",
    "Diffusion",
    "LLM",
    "Agent",
    "Chroma",
    "LlamaIndex",
]


def project_retrieval_documents_to_candidate_previews(retriever_output: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Project summary-only document/chunk retrieval output into matcher-safe previews.

    Phase10A uses the enhanced CandidateProfilePreview builder first. The
    legacy projection remains as a fallback for compatibility and diagnostics.
    """

    if not isinstance(retriever_output, Mapping):
        return []
    try:
        build_result = build_candidate_profile_previews_from_retrieval_results(retriever_output)
        previews = [
            candidate_profile_preview_to_matcher_input(preview)
            for preview in build_result.previews
        ]
        if previews:
            return previews
    except Exception:
        return _legacy_project_retrieval_documents_to_candidate_previews(retriever_output)
    return _legacy_project_retrieval_documents_to_candidate_previews(retriever_output)


def _build_candidate_previews_for_workflow(
    retriever_output: Mapping[str, Any],
    input_data: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        job_requirement = input_data.get("job_requirement")
        query = str(input_data.get("query") or "")
        raw_jd = ""
        if isinstance(job_requirement, Mapping):
            raw_jd = " ".join(_as_string_list(job_requirement.get("required_skills")))
            metadata = job_requirement.get("metadata")
            if isinstance(metadata, Mapping):
                raw_jd = " ".join([raw_jd, str(metadata.get("search_query") or "")]).strip()
        build_result = build_candidate_profile_previews_from_retrieval_results(
            retriever_output,
            raw_jd=raw_jd,
            query=query,
        )
        previews = [candidate_profile_preview_to_matcher_input(preview) for preview in build_result.previews]
        if previews:
            return previews, {
                "candidate_profile_preview_source": "document_chunk_projection",
                "enhanced_candidate_preview_used": True,
                "candidate_preview_fallback_used": False,
                "candidate_preview_grouped_document_count": build_result.grouped_document_count,
                "candidate_preview_skipped_chunk_count": build_result.skipped_chunk_count,
                "candidate_preview_quality_summary": dict(build_result.quality_summary),
                "summary_only": True,
            }
    except Exception:
        pass
    previews = _legacy_project_retrieval_documents_to_candidate_previews(retriever_output)
    return previews, {
        "candidate_profile_preview_source": "document_chunk_projection",
        "enhanced_candidate_preview_used": False,
        "candidate_preview_fallback_used": bool(previews),
        "candidate_preview_grouped_document_count": len(previews),
        "candidate_preview_skipped_chunk_count": 0,
        "candidate_preview_quality_summary": audit_candidate_profile_previews(previews),
        "summary_only": True,
    }


def _legacy_project_retrieval_documents_to_candidate_previews(retriever_output: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Legacy Phase8L one-document-one-preview projection."""

    documents = list(retriever_output.get("resume_documents") or [])
    evidence_items = list(retriever_output.get("evidence") or [])
    previews = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            continue
        evidence = evidence_items[index] if index < len(evidence_items) and isinstance(evidence_items[index], Mapping) else {}
        source_identifier = _preview_source_identifier(document, evidence, index)
        skills = _safe_preview_skills(document, evidence)
        previews.append(
            {
                "candidate_id": _stable_candidate_preview_id(source_identifier, index),
                "name": _safe_metadata_identifier(source_identifier),
                "skills": skills,
                "source_document_id": _safe_metadata_identifier(source_identifier),
                "source_keys": sorted(
                    set(_as_string_list(document.get("metadata_keys")) + _as_string_list(evidence.get("metadata_keys")))
                ),
                "evidence_summary": {
                    "text_length": _safe_int(document.get("text_length") or evidence.get("text_length")),
                    "metadata_keys": sorted(
                        set(_as_string_list(document.get("metadata_keys")) + _as_string_list(evidence.get("metadata_keys")))
                    ),
                    "score_present": bool(document.get("score_present") or evidence.get("score_present")),
                    "summary_only": True,
                },
                "metadata": {
                    "candidate_profile_preview": True,
                    "source": "document_chunk_projection",
                    "summary_only": True,
                },
            }
        )
    return previews


def summarize_skill_backed_variant_result(
    result: SkillWorkflowResult,
    *,
    raw_jd_length: int = 0,
    memory_context_provided: bool = False,
    runner_type: str = "skill_backed_variant",
    deterministic_variant: bool = True,
    real_skill_wrapper_mode: bool = False,
    env_readiness: Any = None,
    memory_context_summary: Any = None,
) -> Dict[str, Any]:
    match_reports = list(result.match_reports or [])
    retrieved_candidates = list(result.retrieved_candidates or [])
    preview_count = sum(1 for candidate in retrieved_candidates if _is_candidate_profile_preview(candidate))
    skill_names = [step.skill_name for step in result.steps]
    output_keys = [
        key
        for key, value in result.to_dict().items()
        if value not in (None, "", [], {})
    ]
    error_type = "" if result.success else "SkillWorkflowFailed"
    error_hint = "" if result.success else _classify_workflow_error_hint(result.error, result=result)
    planner_diagnostics = _planner_failure_diagnostics(error_hint, result)
    retriever_diagnostics = _retriever_failure_diagnostics(error_hint, result)
    planner_fallback = _planner_fallback_summary(result)
    provenance = build_variant_provenance_summary(
        result,
        runner_type=runner_type,
        deterministic_variant=deterministic_variant,
        real_skill_wrapper_mode=real_skill_wrapper_mode,
        planner_fallback=planner_fallback,
    )
    preview_audit = audit_candidate_profile_previews(retrieved_candidates)
    retriever_metadata = _step_output_metadata(result, "resume_retrieve")
    preview_quality_summary = _safe_candidate_preview_quality_summary(
        retriever_metadata.get("candidate_preview_quality_summary")
    )
    memory_summary = _safe_memory_context_summary(
        memory_context_summary,
        memory_context_provided=memory_context_provided,
    )
    return {
        "status": "ok" if result.success else "failed",
        "workflow_status": result.status,
        "candidate_count": len(retrieved_candidates),
        "candidate_profile_preview_count": preview_count,
        "candidate_preview_audit": preview_audit,
        "enhanced_candidate_preview_used": bool(retriever_metadata.get("enhanced_candidate_preview_used", False)),
        "candidate_preview_fallback_used": bool(retriever_metadata.get("candidate_preview_fallback_used", False)),
        "candidate_preview_grouped_document_count": _safe_int(
            retriever_metadata.get("candidate_preview_grouped_document_count")
        ),
        "candidate_preview_skipped_chunk_count": _safe_int(
            retriever_metadata.get("candidate_preview_skipped_chunk_count")
        ),
        "candidate_preview_quality_summary": preview_quality_summary,
        "candidate_name_present_count": _safe_int(preview_audit.get("candidate_name_present_count")),
        "skills_present_count": _safe_int(preview_audit.get("skills_present_count")),
        "project_keywords_present_count": _safe_int(preview_audit.get("project_keywords_present_count")),
        "evidence_summary_present_count": _safe_int(preview_audit.get("evidence_summary_present_count")),
        "source_document_id_present_count": _safe_int(preview_audit.get("source_document_id_present_count")),
        "retrieved_count": len(retrieved_candidates) + len(result.resume_documents or []) + len(result.evidence or []),
        "match_count": len(match_reports),
        "report_count": len(match_reports),
        "top_score_present": any(isinstance(report, Mapping) and "total_score" in report for report in match_reports),
        "refined_query_present": bool(result.refined_query),
        "skill_names": skill_names,
        "skill_event_count": len(result.skill_results),
        "output_keys": sorted(output_keys),
        "error_type": error_type,
        "error_hint": error_hint,
        **planner_diagnostics,
        **planner_fallback,
        **retriever_diagnostics,
        **provenance,
        **memory_summary,
        "env_readiness": _safe_env_readiness(env_readiness),
        "real_skill_wrapper_mode": bool(real_skill_wrapper_mode),
        "metadata": {
            "runner_type": runner_type,
            "deterministic_variant": bool(deterministic_variant),
            "real_skill_wrapper_mode": bool(real_skill_wrapper_mode),
            "production_graph_invoked": False,
            "production_graph_replaced": False,
            "summary_only": True,
            "raw_jd_length": int(raw_jd_length),
            "memory_context_provided": bool(memory_context_provided),
            "env_readiness": _safe_env_readiness(env_readiness),
            "provenance": provenance,
            "candidate_preview_audit": preview_audit,
            "candidate_preview_quality_summary": preview_quality_summary,
            "memory_context_summary": memory_summary,
        },
    }


def _safe_memory_context_summary(value: Any, *, memory_context_provided: bool) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    provided = bool(value.get("provided", memory_context_provided))
    return {
        "memory_source": str(value.get("memory_source") or (value.get("metadata") or {}).get("memory_source") or "none"),
        "memory_db_path_present": bool(value.get("memory_db_path_present", False)),
        "memory_store_loaded": bool(value.get("memory_store_loaded", False)),
        "memory_records_seen": _safe_int(value.get("memory_records_seen")),
        "memory_context_requested": bool(value.get("enabled", provided)),
        "memory_context_provided": provided,
        "memory_context_eligible_count": _safe_int(value.get("eligible_count")),
        "memory_context_denied_count": _safe_int(value.get("denied_count")),
        "memory_context_requires_review_count": _safe_int(value.get("requires_review_count")),
        "memory_context_rendered_char_count": _safe_int(value.get("rendered_char_count")),
        "memory_context_governance_applied": bool(value.get("governance_applied", False)),
        "memory_context_revoked_filtered_count": _safe_int(value.get("revoked_filtered_count")),
        "memory_context_expired_filtered_count": _safe_int(value.get("expired_filtered_count")),
        "memory_context_superseded_filtered_count": _safe_int(value.get("superseded_filtered_count")),
        "memory_context_demo_mode": bool(
            (value.get("metadata") or {}).get("demo_memory_context", False)
            if isinstance(value.get("metadata"), Mapping)
            else False
        ),
        "memory_context_source": "runtime_preview" if provided else "",
        "memory_context_summary_only": True,
    }


def _safe_candidate_preview_quality_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "candidate_profile_preview_count": 0,
            "candidate_name_present_count": 0,
            "skills_present_count": 0,
            "project_keywords_present_count": 0,
            "evidence_summary_present_count": 0,
            "source_document_id_present_count": 0,
            "summary_only": True,
        }
    return {
        "candidate_profile_preview_count": _safe_int(value.get("candidate_profile_preview_count")),
        "candidate_id_present": _safe_int(value.get("candidate_id_present")),
        "candidate_name_present_count": _safe_int(
            value.get("candidate_name_present_count", value.get("candidate_name_present"))
        ),
        "skills_count": _safe_int(value.get("skills_count")),
        "skills_present_count": _safe_int(value.get("skills_present_count")),
        "project_keywords_present_count": _safe_int(value.get("project_keywords_present_count")),
        "education_keywords_present_count": _safe_int(value.get("education_keywords_present_count")),
        "experience_keywords_present_count": _safe_int(value.get("experience_keywords_present_count")),
        "evidence_summary_present_count": _safe_int(
            value.get("evidence_summary_present_count", value.get("evidence_summary_present"))
        ),
        "source_document_id_present_count": _safe_int(
            value.get("source_document_id_present_count", value.get("source_document_id_present"))
        ),
        "summary_only": True,
    }


def build_variant_provenance_summary(
    result: SkillWorkflowResult,
    *,
    runner_type: str,
    deterministic_variant: bool,
    real_skill_wrapper_mode: bool,
    planner_fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build summary-only provenance for the optional variant path."""

    fallback = dict(planner_fallback or _planner_fallback_summary(result))
    job_metadata = {}
    if isinstance(result.job_requirement, Mapping):
        job_metadata = dict(result.job_requirement.get("metadata") or {})
    retriever_metadata = _step_output_metadata(result, "resume_retrieve")
    matcher_metadata = _first_matcher_metadata(result)
    preview_count = sum(1 for candidate in result.retrieved_candidates if _is_candidate_profile_preview(candidate))
    planner_source = str(job_metadata.get("source") or "")
    if fallback.get("planner_fallback_used"):
        planner_source = "deterministic_planner_fallback"
    elif deterministic_variant:
        planner_source = planner_source or "deterministic_variant_planner"
    elif real_skill_wrapper_mode:
        planner_source = planner_source or "PlannerAgent"
    retriever_source = str(
        retriever_metadata.get("source")
        or ("deterministic_variant_retriever" if deterministic_variant else "")
    )
    matcher_source = str(
        matcher_metadata.get("source")
        or ("deterministic_variant_matcher" if deterministic_variant and result.match_reports else "")
    )
    candidate_preview_source = ""
    if preview_count:
        candidate_preview_source = str(
            retriever_metadata.get("candidate_profile_preview_source")
            or "document_chunk_projection"
        )
    matcher_input_source = ""
    if result.match_reports:
        matcher_input_source = "candidate_profile_preview" if preview_count else "candidate_profile"
    return {
        "planner_source": planner_source,
        "retriever_source": retriever_source,
        "matcher_source": matcher_source,
        "real_planner_invoked": bool(fallback.get("real_planner_invoked", False)),
        "planner_fallback_used": bool(fallback.get("planner_fallback_used", False)),
        "fallback_not_real_planner_success": bool(
            fallback.get("fallback_not_real_planner_success", False)
        ),
        "retriever_factory_source": str(retriever_metadata.get("retriever_factory_source") or ""),
        "candidate_preview_source": candidate_preview_source,
        "matcher_input_source": matcher_input_source,
        "provenance_summary_only": True,
        "variant_runner_type": str(runner_type or ""),
    }


def audit_candidate_profile_previews(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return summary-only quality counters for CandidateProfile-like previews."""

    previews = [candidate for candidate in candidates if _is_candidate_profile_preview(candidate)]
    audit = build_candidate_preview_quality_audit(previews)
    return {
        **audit,
        "candidate_name_present": _safe_int(audit.get("candidate_name_present")),
        "evidence_summary_present": _safe_int(audit.get("evidence_summary_present")),
        "source_document_id_present": _safe_int(audit.get("source_document_id_present")),
        "summary_only": True,
    }


def _step_output_metadata(result: SkillWorkflowResult, skill_name: str) -> Dict[str, Any]:
    for step in result.steps:
        if step.skill_name != skill_name:
            continue
        output = getattr(step.result, "output", None)
        if isinstance(output, Mapping):
            metadata = output.get("metadata")
            if isinstance(metadata, Mapping):
                return dict(metadata)
    return {}


def _first_matcher_metadata(result: SkillWorkflowResult) -> Dict[str, Any]:
    for step in result.steps:
        if step.skill_name != "candidate_match":
            continue
        output = getattr(step.result, "output", None)
        if not isinstance(output, Mapping):
            continue
        metadata = output.get("metadata")
        if isinstance(metadata, Mapping) and metadata:
            return dict(metadata)
        report = output.get("match_report")
        if isinstance(report, Mapping) and isinstance(report.get("metadata"), Mapping):
            return dict(report["metadata"])
    for report in result.match_reports:
        if isinstance(report, Mapping) and isinstance(report.get("metadata"), Mapping):
            return dict(report["metadata"])
    return {}


def _real_wrapper_readiness(
    *,
    planner_extract_callable,
    retrieve_callable,
    match_callable,
    refine_callable,
    llm_readiness_checker: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    readiness_checker = llm_readiness_checker or check_llm_env_readiness
    env_readiness = readiness_checker(load_dotenv=False)
    if retrieve_callable is None:
        return {
            "status": "skipped",
            "error_type": "",
            "error_hint": "retriever_callable_required",
            "reason": "Phase8E requires injected retrieve_callable; real Retriever wrapper is deferred",
            "env_readiness": env_readiness,
        }
    all_callables_injected = all(
        callable(item)
        for item in (
            planner_extract_callable,
            retrieve_callable,
            match_callable,
            refine_callable,
        )
    )
    if not all_callables_injected:
        env_readiness = readiness_checker(load_dotenv=True)
    if not all_callables_injected and env_readiness.get("openai_api_key") != "set":
        return {
            "status": "skipped",
            "error_type": "",
            "error_hint": "llm_env_not_detected_for_lazy_wrappers",
            "reason": "LLM environment not detected for lazy Planner/Matcher/Refiner wrappers",
            "env_readiness": env_readiness,
        }
    return {
        "status": "ok",
        "error_type": "",
        "error_hint": "",
        "reason": "",
        "env_readiness": env_readiness,
    }


def load_project_dotenv_for_real_wrappers(import_module: Callable[[str], Any] = importlib.import_module) -> Dict[str, Any]:
    """Load the project .env for explicit real wrapper runs without exposing values."""

    dotenv_path = Path(os.environ.get("RECRUIT_AGENT_DOTENV_PATH") or PROJECT_ROOT / ".env")
    try:
        dotenv_module = import_module("dotenv")
    except ImportError:
        return {
            "dotenv_loaded": "skip",
            "dotenv_path_present": dotenv_path.exists(),
            "dotenv_error_type": "ImportError",
            "summary_only": True,
        }
    try:
        loaded = bool(dotenv_module.load_dotenv(dotenv_path=dotenv_path, override=False))
    except Exception as exc:
        return {
            "dotenv_loaded": False,
            "dotenv_path_present": dotenv_path.exists(),
            "dotenv_error_type": type(exc).__name__,
            "summary_only": True,
        }
    return {
        "dotenv_loaded": loaded,
        "dotenv_path_present": dotenv_path.exists(),
        "dotenv_error_type": "",
        "summary_only": True,
    }


def check_llm_env_readiness(
    *,
    load_dotenv: bool = True,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    dotenv_status = {
        "dotenv_loaded": "skip",
        "dotenv_path_present": (PROJECT_ROOT / ".env").exists(),
        "dotenv_error_type": "",
        "summary_only": True,
    }
    if load_dotenv:
        dotenv_status = load_project_dotenv_for_real_wrappers(import_module=import_module)
    return {
        "dotenv_loaded": dotenv_status.get("dotenv_loaded", "skip"),
        "dotenv_path_present": bool(dotenv_status.get("dotenv_path_present", False)),
        "dotenv_error_type": str(dotenv_status.get("dotenv_error_type") or ""),
        "openai_api_key": "set" if os.environ.get("OPENAI_API_KEY") else "missing",
        "openai_api_base": "set" if os.environ.get("OPENAI_API_BASE") else "missing",
        "summary_only": True,
    }


def _real_wrapper_skipped_summary(
    readiness: Mapping[str, Any],
    *,
    raw_jd_length: int,
    memory_context_provided: bool,
) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "workflow_status": "skipped",
        "real_skill_wrapper_mode": True,
        "candidate_count": 0,
        "retrieved_count": 0,
        "match_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "refined_query_present": False,
        "skill_names": [],
        "skill_event_count": 0,
        "output_keys": ["status", "workflow_status", "error_hint"],
        "error_type": str(readiness.get("error_type") or ""),
        "error_hint": str(readiness.get("error_hint") or "readiness_skipped"),
        "env_readiness": _safe_env_readiness(readiness.get("env_readiness")),
        "metadata": {
            "runner_type": "real_skill_wrapper_variant",
            "deterministic_variant": False,
            "real_skill_wrapper_mode": True,
            "production_graph_invoked": False,
            "production_graph_replaced": False,
            "summary_only": True,
            "raw_jd_length": int(raw_jd_length),
            "memory_context_provided": bool(memory_context_provided),
            "readiness_status": str(readiness.get("status") or "skipped"),
            "env_readiness": _safe_env_readiness(readiness.get("env_readiness")),
        },
    }


def _real_wrapper_failed_summary(
    error_type: str,
    *,
    raw_jd_length: int,
    memory_context_provided: bool,
    env_readiness: Any = None,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "workflow_status": "failed",
        "real_skill_wrapper_mode": True,
        "candidate_count": 0,
        "retrieved_count": 0,
        "match_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "refined_query_present": False,
        "skill_names": [],
        "skill_event_count": 0,
        "output_keys": ["error_type", "status", "workflow_status"],
        "error_type": str(error_type or "Error"),
        "error_hint": "real_wrapper_execution_failed",
        "env_readiness": _safe_env_readiness(env_readiness),
        "metadata": {
            "runner_type": "real_skill_wrapper_variant",
            "deterministic_variant": False,
            "real_skill_wrapper_mode": True,
            "production_graph_invoked": False,
            "production_graph_replaced": False,
            "summary_only": True,
            "raw_jd_length": int(raw_jd_length),
            "memory_context_provided": bool(memory_context_provided),
            "env_readiness": _safe_env_readiness(env_readiness),
        },
    }


def _retriever_query_from_input(input_data: Mapping[str, Any]) -> str:
    query = input_data.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    job_requirement = input_data.get("job_requirement")
    if isinstance(job_requirement, Mapping):
        metadata = job_requirement.get("metadata")
        if isinstance(metadata, Mapping):
            search_query = metadata.get("search_query")
            if isinstance(search_query, str) and search_query.strip():
                return search_query.strip()
        required = job_requirement.get("required_skills")
        if isinstance(required, list) and required:
            return " ".join(str(item) for item in required)
    return ""


def _load_resume_retriever_factory():
    try:
        from src.config import get_settings
        from src.services.retriever import ResumeRetriever
    except Exception as exc:
        raise RuntimeError("retriever_dependency_missing") from exc

    def factory():
        settings = get_settings()
        return ResumeRetriever(persist_dir=str(settings.chroma_dir))

    return factory


def _raw_search_results(raw: Any):
    if isinstance(raw, Mapping):
        return list(raw.get("results") or raw.get("matches") or [])
    return list(raw or [])


def _retriever_dependency_import_summary(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, bool]:
    return {
        "resume_retriever_importable": _is_importable("src.services.retriever", import_module=import_module),
        "llama_index_dependency_importable": _is_importable("llama_index.core", import_module=import_module),
        "embedding_dependency_importable": _is_importable(
            "llama_index.embeddings.huggingface",
            import_module=import_module,
        ),
        "chroma_dependency_importable": _is_importable("chromadb", import_module=import_module),
    }


def _is_importable(module_name: str, *, import_module: Callable[[str], Any] = importlib.import_module) -> bool:
    if import_module is importlib.import_module:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False
    try:
        import_module(module_name)
        return True
    except Exception:
        return False


def _safe_dir_file_count(path: Path) -> int:
    try:
        if not path.exists() or not path.is_dir():
            return 0
        return sum(1 for _item in path.iterdir())
    except Exception:
        return 0


def _factory_class_name(factory: Callable[..., Any]) -> str:
    name = getattr(factory, "__name__", "")
    if name and name != "<lambda>":
        return str(name)
    return type(factory).__name__


def _safe_metadata_identifier(value: Any, max_length: int = 80) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    safe = value.replace("\\", "/").split("/")[-1].strip()
    if len(safe) <= max_length:
        return safe
    return safe[: max_length - 3] + "..."


def _safe_path_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    return text[:240]


def _marker_safe_text(value: Any) -> str:
    text = _safe_path_string(value)
    if not text:
        return ""
    return text.replace(" ", "%20")


def _unmarker_safe_text(value: str) -> str:
    return str(value or "").replace("%20", " ")


def _preview_source_identifier(document: Mapping[str, Any], evidence: Mapping[str, Any], index: int) -> str:
    for key in ("file_name", "source", "source_document_id", "document_id"):
        value = document.get(key) or evidence.get(key)
        safe = _safe_metadata_identifier(value)
        if safe:
            return safe
    return f"document_preview_{index + 1}"


def _stable_candidate_preview_id(source_identifier: str, index: int) -> str:
    digest = hashlib.sha1(f"{source_identifier}:{index}".encode("utf-8")).hexdigest()[:12]
    return f"candidate_preview_{digest}"


def _safe_preview_skills(document: Mapping[str, Any], evidence: Mapping[str, Any]) -> List[str]:
    skills = []
    for source in (document, evidence):
        skills.extend(_as_string_list(source.get("skills")))
    if skills:
        lowered = {str(skill).lower(): str(skill) for skill in skills}
        return [keyword for keyword in SAFE_SKILL_KEYWORDS if keyword.lower() in lowered]
    return []


def _extract_safe_skill_keywords(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [keyword for keyword in SAFE_SKILL_KEYWORDS if keyword.lower() in lowered]


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _is_candidate_profile_preview(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    metadata = candidate.get("metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("candidate_profile_preview", False))


def _classify_workflow_error_hint(error: str, result: Optional[SkillWorkflowResult] = None) -> str:
    text = str(error or "")
    for marker in (
        "retriever_embedding_cache_unavailable",
        "retriever_dependency_missing",
        "retriever_init_failed",
        "retriever_search_failed",
        "retriever_result_adapter_failed",
    ):
        if marker in text:
            return marker
    if result is not None:
        for skill_result in result.skill_results:
            if getattr(skill_result, "success", True):
                continue
            skill_name = str(getattr(skill_result, "skill_name", "") or "")
            if skill_name == "planner_extract":
                return "planner_wrapper_failed"
            if skill_name == "candidate_match":
                return "matcher_wrapper_failed"
            if skill_name == "query_refine":
                return "refiner_wrapper_failed"
            if skill_name == "resume_retrieve":
                return "retriever_search_failed"
    if "planner_extract" in text:
        return "planner_wrapper_failed"
    if "candidate_match" in text:
        return "matcher_wrapper_failed"
    if "query_refine" in text:
        return "refiner_wrapper_failed"
    return "real_wrapper_execution_failed"


def _retriever_failure_diagnostics(error_hint: str, result: SkillWorkflowResult) -> Dict[str, Any]:
    if error_hint not in {
        "retriever_embedding_cache_unavailable",
        "retriever_dependency_missing",
        "retriever_init_failed",
        "retriever_search_failed",
        "retriever_result_adapter_failed",
    }:
        return {}
    error = str(result.error or "")
    return {
        "retriever_init_stage": _extract_marker_value(error, "retriever_init_stage") or "",
        "retriever_config_summary": _safe_retriever_config_summary(_parse_retriever_config_markers(error)),
        "retriever_init_diagnostics": _safe_retriever_init_diagnostics(
            _parse_retriever_init_diagnostic_markers(error)
        ),
        "retriever_embedding_readiness": _safe_embedding_readiness(
            _parse_embedding_readiness_markers(error)
        ),
    }


def _planner_failure_diagnostics(error_hint: str, result: SkillWorkflowResult) -> Dict[str, Any]:
    if error_hint != "planner_wrapper_failed":
        return {}
    observed_keys = []
    adapter_hint = "planner_wrapper_failed"
    invocation_stage = ""
    input_shape = {}
    provider_error_type = ""
    for skill_result in result.skill_results:
        if getattr(skill_result, "skill_name", "") != "planner_extract" or getattr(skill_result, "success", True):
            continue
        error = str(getattr(skill_result, "error", "") or "")
        adapter_hint = "planner_schema_adapter_failed" if "planner_schema_adapter_failed" in error else adapter_hint
        invocation_stage = _extract_marker_value(error, "planner_invocation_stage")
        provider_error_type = _extract_marker_value(error, "provider_error_type")
        input_keys_marker = _extract_marker_value(error, "planner_input_keys")
        raw_text_length = _extract_marker_value(error, "planner_raw_text_length")
        has_messages = _extract_marker_value(error, "planner_has_messages")
        if invocation_stage:
            input_shape = {
                "input_keys": sorted(key for key in input_keys_marker.split(",") if key),
                "has_messages": has_messages == "true",
                "raw_text_length": _safe_int(raw_text_length),
                "summary_only": True,
            }
        output_keys_marker = _extract_marker_value(error, "planner_output_keys")
        if output_keys_marker:
            observed_keys = [key for key in output_keys_marker.split(",") if key]
        if "observed_keys=" in error:
            observed_part = error.split("observed_keys=", 1)[1].split(" ", 1)[0]
            observed_keys = [key for key in observed_part.split(",") if key]
        break
    return {
        "planner_output_keys": sorted(observed_keys),
        "planner_expected_keys": [
            "job_requirement",
            "extracted_jd",
            "tech_stack",
            "education",
            "must_have",
            "search_query",
        ],
        "planner_adapter_error_hint": adapter_hint,
        "planner_invocation_stage": invocation_stage,
        "planner_input_shape": input_shape,
        "provider_error_type": provider_error_type,
        "planner_provider_diagnostics": _planner_provider_diagnostics_from_error(error),
    }


def _planner_fallback_summary(result: SkillWorkflowResult) -> Dict[str, Any]:
    metadata = {}
    if isinstance(result.job_requirement, Mapping):
        metadata = dict(result.job_requirement.get("metadata") or {})
    fallback_used = bool(metadata.get("planner_fallback_used", False))
    planner_error = ""
    for skill_result in result.skill_results:
        if getattr(skill_result, "skill_name", "") == "planner_extract" and not getattr(skill_result, "success", True):
            planner_error = str(getattr(skill_result, "error", "") or "")
            break
    invoked_and_failed = (
        _extract_marker_value(planner_error, "planner_invocation_stage") == "invoke_planner_agent"
    )
    return {
        "planner_fallback_used": fallback_used,
        "planner_fallback_type": str(metadata.get("planner_fallback_type") or ""),
        "real_planner_invoked": bool(metadata.get("real_planner_invoked", False)),
        "real_planner_failed": bool(metadata.get("real_planner_failed", False)),
        "fallback_not_real_planner_success": True,
        "provider_error_type": str(metadata.get("provider_error_type") or ""),
    } if fallback_used else {
        "planner_fallback_used": False,
        "planner_fallback_type": "",
        "real_planner_invoked": bool(invoked_and_failed),
        "real_planner_failed": bool(invoked_and_failed),
        "fallback_not_real_planner_success": False,
    }


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _retriever_config_markers(summary: Mapping[str, Any]) -> str:
    safe = _safe_retriever_config_summary(summary)
    return " ".join(
        f"{key}={'true' if value is True else 'false'}"
        for key, value in safe.items()
        if key != "summary_only"
    )


def _retriever_diagnostic_markers(summary: Mapping[str, Any]) -> str:
    safe = _safe_retriever_init_diagnostics(summary)
    marker_values = {
        "project_root_present": safe["project_root_present"],
        "project_root": _marker_safe_text(safe["project_root"]),
        "data_dir_present": safe["data_dir_present"],
        "chroma_dir_present": safe["chroma_dir_present"],
        "chroma_dir_non_empty": safe["chroma_dir_non_empty"],
        "chroma_dir_file_count": safe["chroma_dir_file_count"],
        "persist_dir_exists": safe["persist_dir_exists"],
        "persist_dir_non_empty": safe["persist_dir_non_empty"],
        "persist_dir_used": _marker_safe_text(safe["persist_dir_used"]),
        "resume_retriever_class": _marker_safe_text(safe["resume_retriever_class"]),
        "init_method": _marker_safe_text(safe["init_method"]),
        "import_stage_ok": safe["import_stage_ok"],
        "settings_loaded": safe["settings_loaded"],
        "embedding_dependency_importable": safe["embedding_dependency_importable"],
        "chroma_dependency_importable": safe["chroma_dependency_importable"],
        "llama_index_dependency_importable": safe["llama_index_dependency_importable"],
        "embedding_model_name": _marker_safe_text(safe["embedding_readiness"]["embedding_model_name"]),
        "hf_token_status": _marker_safe_text(safe["embedding_readiness"]["hf_token_status"]),
        "hf_home_set": safe["embedding_readiness"]["hf_home_set"],
        "transformers_cache_set": safe["embedding_readiness"]["transformers_cache_set"],
        "sentence_transformers_cache_set": safe["embedding_readiness"]["sentence_transformers_cache_set"],
        "cache_probe_supported": safe["embedding_readiness"]["cache_probe_supported"],
        "cache_likely_available": _marker_safe_text(str(safe["embedding_readiness"]["cache_likely_available"])),
        "network_required_unknown": safe["embedding_readiness"]["network_required_unknown"],
        "allow_hf_network_probe": safe["embedding_readiness"]["allow_hf_network_probe"],
        "error_stage": _marker_safe_text(safe["error_stage"]),
        "error_type": _marker_safe_text(safe["error_type"]),
    }
    return " ".join(
        f"{key}={'true' if value is True else 'false' if value is False else value}"
        for key, value in marker_values.items()
    )


def _parse_retriever_config_markers(text: str) -> Dict[str, bool]:
    keys = (
        "project_root_present",
        "data_dir_present",
        "chroma_dir_present",
        "chroma_dir_non_empty",
    )
    return {key: _extract_marker_value(text, key) == "true" for key in keys if f"{key}=" in text}


def _parse_retriever_init_diagnostic_markers(text: str) -> Dict[str, Any]:
    bool_keys = (
        "project_root_present",
        "data_dir_present",
        "chroma_dir_present",
        "chroma_dir_non_empty",
        "persist_dir_exists",
        "persist_dir_non_empty",
        "import_stage_ok",
        "settings_loaded",
        "embedding_dependency_importable",
        "chroma_dependency_importable",
        "llama_index_dependency_importable",
        "hf_home_set",
        "transformers_cache_set",
        "sentence_transformers_cache_set",
        "cache_probe_supported",
        "network_required_unknown",
        "allow_hf_network_probe",
    )
    parsed: Dict[str, Any] = {
        key: _extract_marker_value(text, key).lower() == "true"
        for key in bool_keys
        if f"{key}=" in text
    }
    for key in ("chroma_dir_file_count",):
        if f"{key}=" in text:
            parsed[key] = _safe_int(_extract_marker_value(text, key))
    for key in (
        "persist_dir_used",
        "project_root",
        "resume_retriever_class",
        "init_method",
        "error_stage",
        "error_type",
        "embedding_model_name",
        "hf_token_status",
        "cache_likely_available",
    ):
        if f"{key}=" in text:
            parsed[key] = _unmarker_safe_text(_extract_marker_value(text, key))
    return parsed


def _safe_retriever_config_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    return {
        "project_root_present": bool(value.get("project_root_present", False)),
        "data_dir_present": bool(value.get("data_dir_present", False)),
        "chroma_dir_present": bool(value.get("chroma_dir_present", False)),
        "chroma_dir_non_empty": bool(value.get("chroma_dir_non_empty", False)),
        "summary_only": True,
    }


def _safe_retriever_init_diagnostics(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    return {
        "project_root_present": bool(value.get("project_root_present", False)),
        "project_root": _safe_path_string(value.get("project_root")),
        "data_dir_present": bool(value.get("data_dir_present", False)),
        "chroma_dir_present": bool(value.get("chroma_dir_present", False)),
        "chroma_dir_non_empty": bool(value.get("chroma_dir_non_empty", False)),
        "chroma_dir_file_count": _safe_int(value.get("chroma_dir_file_count")),
        "persist_dir_used": _safe_path_string(value.get("persist_dir_used")),
        "persist_dir_exists": bool(value.get("persist_dir_exists", False)),
        "persist_dir_non_empty": bool(value.get("persist_dir_non_empty", False)),
        "resume_retriever_class": str(value.get("resume_retriever_class") or ""),
        "init_method": str(value.get("init_method") or ""),
        "import_stage_ok": bool(value.get("import_stage_ok", False)),
        "settings_loaded": bool(value.get("settings_loaded", False)),
        "embedding_dependency_importable": bool(value.get("embedding_dependency_importable", False)),
        "chroma_dependency_importable": bool(value.get("chroma_dependency_importable", False)),
        "llama_index_dependency_importable": bool(value.get("llama_index_dependency_importable", False)),
        "embedding_readiness": _safe_embedding_readiness(value.get("embedding_readiness") or value),
        "error_stage": str(value.get("error_stage") or ""),
        "error_type": str(value.get("error_type") or ""),
        "summary_only": True,
    }


def _parse_embedding_readiness_markers(text: str) -> Dict[str, Any]:
    parsed = {}
    bool_keys = (
        "hf_home_set",
        "transformers_cache_set",
        "sentence_transformers_cache_set",
        "cache_probe_supported",
        "network_required_unknown",
        "allow_hf_network_probe",
        "embedding_dependency_importable",
        "chroma_dependency_importable",
        "llama_index_dependency_importable",
    )
    for key in bool_keys:
        if f"{key}=" in text:
            parsed[key] = _extract_marker_value(text, key) == "true"
    for key in ("embedding_model_name", "hf_token_status", "cache_likely_available"):
        if f"{key}=" in text:
            parsed[key] = _unmarker_safe_text(_extract_marker_value(text, key))
    return parsed


def _safe_embedding_readiness(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    cache_value = value.get("cache_likely_available", "unknown")
    if cache_value is True or str(cache_value).lower() == "true":
        cache_likely_available: Any = True
    elif cache_value is False or str(cache_value).lower() == "false":
        cache_likely_available = False
    else:
        cache_likely_available = "unknown"
    return {
        "embedding_model_name": str(value.get("embedding_model_name") or ""),
        "hf_token_status": "set" if value.get("hf_token_status") == "set" else "missing",
        "hf_home_set": _safe_bool(value.get("hf_home_set", False)),
        "transformers_cache_set": _safe_bool(value.get("transformers_cache_set", False)),
        "sentence_transformers_cache_set": _safe_bool(value.get("sentence_transformers_cache_set", False)),
        "cache_probe_supported": _safe_bool(value.get("cache_probe_supported", False)),
        "cache_likely_available": cache_likely_available,
        "network_required_unknown": _safe_bool(value.get("network_required_unknown", True)),
        "allow_hf_network_probe": _safe_bool(value.get("allow_hf_network_probe", False)),
        "embedding_dependency_importable": _safe_bool(value.get("embedding_dependency_importable", False)),
        "chroma_dependency_importable": _safe_bool(value.get("chroma_dependency_importable", False)),
        "llama_index_dependency_importable": _safe_bool(value.get("llama_index_dependency_importable", False)),
        "summary_only": True,
    }


def _should_stop_for_embedding_readiness(
    readiness: Mapping[str, Any],
    *,
    require_embedding_cache_ready: bool,
    skip_if_embedding_cache_unavailable: bool,
) -> bool:
    if not require_embedding_cache_ready or not skip_if_embedding_cache_unavailable:
        return False
    safe = _safe_embedding_readiness(readiness)
    return safe["cache_likely_available"] is not True


def _safe_env_readiness(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    dotenv_loaded = value.get("dotenv_loaded", "skip")
    if dotenv_loaded not in {True, False, "skip"}:
        dotenv_loaded = "skip"
    return {
        "dotenv_loaded": dotenv_loaded,
        "dotenv_path_present": bool(value.get("dotenv_path_present", False)),
        "dotenv_error_type": str(value.get("dotenv_error_type") or ""),
        "openai_api_key": "set" if value.get("openai_api_key") == "set" else "missing",
        "openai_api_base": "set" if value.get("openai_api_base") == "set" else "missing",
        "summary_only": True,
    }


def _fake_planner_extract(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
    raw_text = str(input_data.get("raw_text") or "")
    required_skills = _extract_known_skills(raw_text)
    return {
        "job_requirement": {
            "job_id": "variant_job_1",
            "title": "AI Agent Engineer",
            "raw_text_length": len(raw_text),
            "required_skills": required_skills,
            "education": "",
            "metadata": {
                "search_query": " ".join(required_skills) or "AI Agent",
                "source": "deterministic_variant_planner",
            },
        },
        "extracted_keywords": required_skills,
        "metadata": {"source": "deterministic_variant_planner"},
    }


def _fake_resume_retrieve(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
    top_k = input_data.get("top_k")
    candidates = [
        {
            "candidate_id": "variant_candidate_1",
            "name": "Candidate One",
            "skills": ["Python", "RAG", "LangGraph", "AI Agent"],
            "metadata": {"source": "deterministic_variant_retriever"},
        },
        {
            "candidate_id": "variant_candidate_2",
            "name": "Candidate Two",
            "skills": ["Python", "LLM"],
            "metadata": {"source": "deterministic_variant_retriever"},
        },
    ]
    if isinstance(top_k, int) and top_k >= 0:
        candidates = candidates[:top_k]
    return {
        "candidates": candidates,
        "evidence": [
            {
                "candidate_id": candidate["candidate_id"],
                "matched_skill_count": len(candidate["skills"]),
            }
            for candidate in candidates
        ],
        "metadata": {"source": "deterministic_variant_retriever"},
    }


def _fake_candidate_match(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
    job_requirement = dict(input_data.get("job_requirement") or {})
    candidate = dict(input_data.get("candidate_profile") or {})
    required = set(job_requirement.get("required_skills") or [])
    candidate_skills = set(candidate.get("skills") or [])
    matched = sorted(required.intersection(candidate_skills))
    score = 75.0 if not required else round((len(matched) / len(required)) * 100, 2)
    candidate_id = str(candidate.get("candidate_id") or "")
    recommendation = "strong_match" if score >= 80 else "possible_match"
    return {
        "total_score": score,
        "recommendation": recommendation,
        "match_report": {
            "candidate_id": candidate_id,
            "total_score": score,
            "matched_skills": matched,
            "recommendation": recommendation,
            "metadata": {"source": "deterministic_variant_matcher"},
        },
        "metadata": {"source": "deterministic_variant_matcher"},
    }


def _fake_query_refine(input_data: Dict[str, Any], _context: Optional[SkillExecutionContext] = None):
    query = str(input_data.get("query") or "")
    return {
        "refined_query": f"{query} Python RAG LangGraph".strip(),
        "reason": "deterministic low-result refinement",
        "metadata": {"source": "deterministic_variant_refiner"},
    }


def _extract_known_skills(text: str):
    known = ["Python", "RAG", "LangGraph", "AI Agent", "LLM", "Chroma", "LlamaIndex"]
    lowered = text.lower()
    return [skill for skill in known if skill.lower() in lowered]
