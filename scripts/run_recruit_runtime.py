import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.entry import (  # noqa: E402
    RuntimeEntryConfig,
    RuntimeEntryHarness,
    build_default_graph_runner,
    build_demo_mode_runner,
    load_jd_text,
)
from src.core.graph_factory import resolve_recruit_graph_factory_config  # noqa: E402
from src.runtime.variant_runner import (  # noqa: E402
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    build_skill_backed_variant_runner,
)
from src.runtime.memory_context import (  # noqa: E402
    RuntimeMemorySourceConfig,
    build_readonly_runtime_memory_context,
)
from src.integration.production_skill_graph import (  # noqa: E402
    ProductionSkillGraphConfig,
    build_production_skill_graph_runner,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the recruiting workflow through the runtime-managed entry harness."
    )
    parser.add_argument("--jd", default=None, help="Job description text.")
    parser.add_argument("--jd-file", default=None, help="Path to a UTF-8 job description file.")
    parser.add_argument("--db-path", default=None, help="Optional SQLite runtime store path.")
    parser.add_argument(
        "--graph-mode",
        choices=["legacy", "skill"],
        default=None,
        help="Unified graph mode. Defaults to RECRUIT_GRAPH_MODE or skill.",
    )
    parser.add_argument("--demo-mode", action="store_true", help="Request the limited demo mode runner.")
    parser.add_argument(
        "--enable-demo-mode",
        action="store_true",
        help="Explicitly allow the limited demo mode runner instead of falling back to default.",
    )
    parser.add_argument(
        "--use-skill-backed-variant",
        action="store_true",
        help="Request the skill-backed graph variant path.",
    )
    parser.add_argument(
        "--use-production-skill-graph",
        action="store_true",
        help="Compatibility alias for --graph-mode skill.",
    )
    parser.add_argument(
        "--use-real-skill-wrappers",
        action="store_true",
        help="Use the explicit real skill wrapper variant runner; implies --use-skill-backed-variant.",
    )
    parser.add_argument(
        "--use-real-retriever",
        action="store_true",
        help="Allow the real skill wrapper variant to use a lazy ResumeRetriever callable.",
    )
    parser.add_argument(
        "--allow-planner-fallback",
        action="store_true",
        help="Allow explicit deterministic Planner fallback in the real skill wrapper variant.",
    )
    parser.add_argument(
        "--allow-memory-context",
        action="store_true",
        help="Allow an already-built read-only memory context preview to reach demo/variant runners.",
    )
    parser.add_argument(
        "--use-demo-memory-context",
        action="store_true",
        help="Build an in-memory governed demo memory preview for explicit variant smoke.",
    )
    parser.add_argument(
        "--memory-source",
        choices=["none", "demo", "sqlite"],
        default="none",
        help="Explicit read-only memory source for variant memory context.",
    )
    parser.add_argument("--memory-db-path", default=None, help="Optional MemorySQLiteStore path.")
    parser.add_argument("--governance-db-path", default=None, help="Reserved governance store path marker.")
    parser.add_argument("--memory-max-items", type=int, default=5, help="Max memory preview items.")
    parser.add_argument("--memory-max-chars", type=int, default=1200, help="Max memory preview characters.")
    parser.add_argument("--memory-tag", action="append", default=None, help="Memory tag filter; repeatable.")
    parser.add_argument(
        "--no-ab-required",
        action="store_true",
        help="Do not require A/B smoke pass before demo mode variant use.",
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Do not automatically roll back to default when a variant runner fails.",
    )
    parser.add_argument(
        "--disable-legacy-fallback",
        action="store_true",
        help="Disable the one-shot legacy fallback when default skill graph has a hard failure.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on failed runtime execution.")
    return parser.parse_args(argv)


def run_cli(
    argv=None,
    default_runner=None,
    demo_runner=None,
    variant_runner=None,
    real_skill_wrapper_runner=None,
    real_retriever_callable=None,
    planner_extract_callable=None,
    match_callable=None,
    refine_callable=None,
    production_skill_graph_runner=None,
    memory_context=None,
):
    args = parse_args(argv)
    jd_text = load_jd_text(args.jd, args.jd_file)
    default = default_runner or build_default_graph_runner()
    graph_factory_config = resolve_recruit_graph_factory_config(
        requested_graph_mode=args.graph_mode,
        use_production_skill_graph_alias=bool(args.use_production_skill_graph),
        allow_planner_fallback=bool(args.allow_planner_fallback),
        summary_only=True,
    )
    effective_memory_source = args.memory_source
    if args.use_demo_memory_context:
        effective_memory_source = "demo"
    memory_context_summary = {
        "enabled": bool(args.allow_memory_context),
        "provided": False,
        "memory_source": effective_memory_source,
        "memory_db_path_present": bool(args.memory_db_path),
        "memory_store_loaded": False,
        "memory_records_seen": 0,
        "eligible_count": 0,
        "denied_count": 0,
        "requires_review_count": 0,
        "rendered_char_count": 0,
        "governance_applied": False,
        "metadata": {
            "summary_only": True,
            "read_only": True,
            "demo_memory_context": bool(args.use_demo_memory_context),
        },
        "summary_only": True,
    }
    if args.allow_memory_context and (args.use_skill_backed_variant or args.use_real_skill_wrappers):
        memory_result = build_readonly_runtime_memory_context(
            RuntimeMemorySourceConfig(
                source=effective_memory_source,
                memory_db_path=args.memory_db_path,
                governance_db_path=args.governance_db_path,
                max_items=args.memory_max_items,
                max_chars=args.memory_max_chars,
                tags=args.memory_tag or (["runtime_demo"] if effective_memory_source == "demo" else None),
                require_governance=True,
                metadata={"demo_memory_context": bool(args.use_demo_memory_context)},
            ),
            target_context={"tags": ["runtime_demo"]} if effective_memory_source == "demo" else None,
        )
        memory_context = memory_context or memory_result.memory_context_preview
        memory_context_summary = memory_result.to_summary()
        if effective_memory_source == "none":
            memory_context_summary["enabled"] = True
            memory_context_summary["reason"] = "memory context requested but unavailable"
    elif args.allow_memory_context:
        memory_context_summary["reason"] = "memory context requested but default graph ignored"
    variant = variant_runner
    use_variant = bool(args.use_skill_backed_variant or args.use_real_skill_wrappers)
    production_runner = production_skill_graph_runner
    if production_runner is None and graph_factory_config.mode.value == "skill":
        production_runner = build_production_skill_graph_runner(
            ProductionSkillGraphConfig(
                enabled=True,
                allow_planner_fallback=bool(args.allow_planner_fallback),
                use_real_retriever=True,
                use_candidate_profile_preview=True,
                rollback_on_failure=not args.no_rollback,
                summary_only=True,
            ),
            planner_extract_callable=planner_extract_callable,
            retrieve_callable=real_retriever_callable,
            match_callable=match_callable,
            refine_callable=refine_callable,
        ).run
    if variant is None and args.use_real_skill_wrappers:
        retriever_callable = None
        if args.use_real_retriever:
            retriever_callable = real_retriever_callable or build_real_retriever_callable()
        variant = real_skill_wrapper_runner or build_real_skill_wrapper_variant_runner(
            planner_extract_callable=planner_extract_callable,
            retrieve_callable=retriever_callable,
            match_callable=match_callable,
            refine_callable=refine_callable,
            use_real_retriever_callable=bool(args.use_real_retriever and retriever_callable is None),
            allow_planner_deterministic_fallback=bool(args.allow_planner_fallback),
            enable_candidate_preview_projection=bool(args.allow_planner_fallback),
        )
    elif variant is None and use_variant:
        variant = build_skill_backed_variant_runner()
    demo = demo_runner or build_demo_mode_runner(
        default,
        variant_runner=variant,
        require_ab_smoke_pass=not args.no_ab_required,
        rollback_on_variant_failure=not args.no_rollback,
        allow_memory_context=bool(args.allow_memory_context),
    )
    config = RuntimeEntryConfig(
        use_demo_mode=bool(args.demo_mode),
        demo_mode_enabled=bool(args.enable_demo_mode),
        use_skill_backed_variant=use_variant,
        use_production_skill_graph=bool(args.use_production_skill_graph),
        graph_mode=str(args.graph_mode or ""),
        legacy_fallback_enabled=not args.disable_legacy_fallback,
        allow_memory_context=bool(args.allow_memory_context),
        allow_planner_fallback=bool(args.allow_planner_fallback),
        require_ab_smoke_pass=not args.no_ab_required,
        rollback_on_variant_failure=not args.no_rollback,
        db_path=args.db_path,
        summary_only=True,
        metadata={
            "cli": True,
            "use_real_skill_wrappers": bool(args.use_real_skill_wrappers),
            "use_real_retriever": bool(args.use_real_retriever),
            "allow_planner_fallback": bool(args.allow_planner_fallback),
            "use_production_skill_graph": bool(args.use_production_skill_graph),
            "requested_graph_mode": str(args.graph_mode or ""),
            "selected_graph_mode": graph_factory_config.mode.value,
            "selection_reason": graph_factory_config.selection_reason,
            "legacy_alias_used": bool(graph_factory_config.legacy_alias_used),
            "legacy_fallback_enabled": not args.disable_legacy_fallback,
            "use_demo_memory_context": bool(args.use_demo_memory_context),
            "memory_context_summary": memory_context_summary,
        },
    )
    try:
        result = RuntimeEntryHarness().run(
            jd_text,
            default_runner=default,
            demo_runner=demo,
            variant_runner=variant,
            production_skill_graph_runner=production_runner,
            memory_context=memory_context,
            config=config,
        )
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "summary_only": True,
            "production_graph_replaced": False,
        }
        _emit(payload, as_json=args.json)
        return 1 if args.strict else 0

    payload = result.to_dict()
    _emit(payload, as_json=args.json)
    output_status = result.output_summary.get("status") if isinstance(result.output_summary, dict) else ""
    if args.strict and (result.status != "ok" or output_status in {"failed", "skipped"}):
        return 1
    return 0


def _emit(payload, *, as_json: bool):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"status={payload.get('status')}")
    print(f"runner_used={payload.get('runner_used', '')}")
    print(f"task_status={payload.get('task_status', '')}")
    print(f"event_count={payload.get('event_count', 0)}")
    print(f"error_type={payload.get('error_type', '')}")
    print("summary_only=true")


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
