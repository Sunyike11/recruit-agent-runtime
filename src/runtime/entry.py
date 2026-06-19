from dataclasses import asdict, dataclass, field
from pathlib import Path
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from src.core.graph_factory import RecruitGraphFactory, resolve_recruit_graph_factory_config
from src.runtime.event_envelope import build_runtime_event_payload
from src.runtime.models import TaskStatus
from src.runtime.sqlite_store import SQLiteRuntimeStore
from src.runtime.store import InMemoryRuntimeStore


@dataclass
class RuntimeEntryConfig:
    use_demo_mode: bool = False
    demo_mode_enabled: bool = False
    use_skill_backed_variant: bool = False
    use_production_skill_graph: bool = False
    graph_mode: str = ""
    legacy_fallback_enabled: bool = True
    allow_memory_context: bool = False
    allow_planner_fallback: bool = False
    require_ab_smoke_pass: bool = True
    rollback_on_variant_failure: bool = True
    db_path: Optional[str] = None
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeEntryResult:
    status: str
    session_id: str
    task_id: str
    thread_id: str
    runner_used: str
    task_status: str
    event_count: int
    output_summary: Dict[str, Any]
    error_type: str = ""
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeEntryHarness:
    """Runtime-managed entry point for default or explicitly enabled demo runners."""

    def run(
        self,
        raw_jd: str,
        default_runner: Callable[[str], Mapping[str, Any]],
        demo_runner: Optional[Callable[[str], Mapping[str, Any]]] = None,
        config: Optional[RuntimeEntryConfig] = None,
        store: Optional[Any] = None,
        variant_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
        production_skill_graph_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
        memory_context: Any = None,
    ) -> RuntimeEntryResult:
        entry_config = config or RuntimeEntryConfig()
        runtime_store = store or _create_store(entry_config)
        session = runtime_store.create_session(
            metadata={
                "mode": "runtime_entry",
                "summary_only": bool(entry_config.summary_only),
                "metadata_keys": sorted(str(key) for key in entry_config.metadata.keys()),
            }
        )
        task = runtime_store.create_task(
            session.session_id,
            input_payload={
                "jd_text": raw_jd,
                "metadata": {
                    "summary_only": True,
                    "jd_length": len(raw_jd or ""),
                    "use_demo_mode": bool(entry_config.use_demo_mode),
                    "demo_mode_enabled": bool(entry_config.demo_mode_enabled),
                    "use_skill_backed_variant": bool(entry_config.use_skill_backed_variant),
                    "use_production_skill_graph": bool(entry_config.use_production_skill_graph),
                    "graph_mode": str(entry_config.graph_mode or ""),
                    "legacy_fallback_enabled": bool(entry_config.legacy_fallback_enabled),
                    "candidate_source": str(entry_config.metadata.get("candidate_source") or ""),
                    "allow_memory_context": bool(entry_config.allow_memory_context),
                    "allow_planner_fallback": bool(entry_config.allow_planner_fallback),
                },
            },
        )
        runner, runner_used, selection_metadata = self._select_runner(
            default_runner,
            demo_runner,
            variant_runner,
            production_skill_graph_runner,
            entry_config,
        )
        runtime_store.update_task_status(task.task_id, TaskStatus.RUNNING)
        runtime_store.append_event(
            "task_started",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                runner_name=runner_used,
                status="started",
                extra={"runner_used": runner_used, "selection": selection_metadata},
            ),
        )

        primary_attempt_id = str(uuid.uuid4())
        _append_graph_attempt_event(
            runtime_store,
            "graph_primary_started",
            session,
            task,
            graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
            runner_name=runner_used,
            attempt_id=primary_attempt_id,
            status="started",
        )
        try:
            output = self._call_runner(
                runner,
                raw_jd,
                runner_used,
                entry_config,
                memory_context,
                runtime_store,
                session,
                task,
            )
            output_summary = _apply_graph_selection_summary(
                summarize_runner_output(output),
                runner_used=runner_used,
                selection_metadata=selection_metadata,
            )
            _apply_memory_context_summary(output_summary, entry_config)
            primary_health = _classify_graph_health(output_summary)
            _append_graph_attempt_event(
                runtime_store,
                "graph_primary_failed" if primary_health["graph_health_status"] == "critical" else "graph_primary_completed",
                session,
                task,
                graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                runner_name=runner_used,
                attempt_id=primary_attempt_id,
                status="failed" if primary_health["graph_health_status"] == "critical" else "completed",
                error_type=str(output_summary.get("error_type") or output_summary.get("runner_error_type") or ""),
                error_hint=str(output_summary.get("error_hint") or ""),
                fallback_attempted=primary_health["graph_health_status"] == "critical",
            )
        except Exception as exc:
            error_type = type(exc).__name__
            output_summary = _apply_graph_selection_summary(
                _failed_output_summary(error_type, stage="runner_call"),
                runner_used=runner_used,
                selection_metadata=selection_metadata,
            )
            _append_graph_attempt_event(
                runtime_store,
                "graph_primary_failed",
                session,
                task,
                graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                runner_name=runner_used,
                attempt_id=primary_attempt_id,
                status="failed",
                error_type=error_type,
                error_hint="runner_call",
                fallback_attempted=False,
            )
            runtime_store.update_task_status(task.task_id, TaskStatus.FAILED, result=output_summary, error=error_type)
            runtime_store.append_event(
                "task_failed",
                session_id=session.session_id,
                task_id=task.task_id,
                payload=build_runtime_event_payload(
                    session_id=session.session_id,
                    task_id=task.task_id,
                    thread_id=task.thread_id,
                    graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                    runner_name=runner_used,
                    status="failed",
                    error_type=error_type,
                    extra={"runner_used": runner_used},
                ),
            )
            failed_task = runtime_store.get_task(task.task_id)
            return RuntimeEntryResult(
                status="failed",
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                runner_used=runner_used,
                task_status=_task_status_value(failed_task.status),
                event_count=len(runtime_store.list_events(task_id=task.task_id)),
                output_summary=output_summary,
                error_type=error_type,
                summary_only=True,
                metadata=_result_metadata(raw_jd, entry_config, selection_metadata),
            )

        if _should_attempt_legacy_fallback(output_summary, runner_used, entry_config):
            fallback_summary = self._run_legacy_fallback(
                default_runner,
                raw_jd,
                primary_summary=output_summary,
                entry_config=entry_config,
                runtime_store=runtime_store,
                session=session,
                task=task,
                selection_metadata=selection_metadata,
                primary_attempt_id=primary_attempt_id,
            )
            if fallback_summary.get("fallback_succeeded"):
                runtime_store.update_task_status(
                    task.task_id,
                    TaskStatus.COMPLETED_WITH_FALLBACK,
                    result=fallback_summary,
                )
                runtime_store.append_event(
                    "task_completed",
                    session_id=session.session_id,
                    task_id=task.task_id,
                    payload=build_runtime_event_payload(
                        session_id=session.session_id,
                        task_id=task.task_id,
                        thread_id=task.thread_id,
                        graph_mode="skill",
                        runner_name=runner_used,
                        status="completed_with_fallback",
                        fallback_used=True,
                        extra={"runner_used": runner_used, "final_runner_used": "default_graph"},
                    ),
                )
                completed_task = runtime_store.get_task(task.task_id)
                return RuntimeEntryResult(
                    status="completed_with_fallback",
                    session_id=session.session_id,
                    task_id=task.task_id,
                    thread_id=task.thread_id,
                    runner_used=runner_used,
                    task_status=_task_status_value(completed_task.status),
                    event_count=len(runtime_store.list_events(task_id=task.task_id)),
                    output_summary=fallback_summary,
                    error_type="",
                    summary_only=True,
                    metadata=_result_metadata(raw_jd, entry_config, selection_metadata),
                )
            output_summary = fallback_summary

        if output_summary.get("status") == "failed" or output_summary.get("error_type"):
            error_type = str(output_summary.get("error_type") or "RunnerFailed")
            runtime_store.update_task_status(task.task_id, TaskStatus.FAILED, result=output_summary, error=error_type)
            runtime_store.append_event(
                "task_failed",
                session_id=session.session_id,
                task_id=task.task_id,
                payload=build_runtime_event_payload(
                    session_id=session.session_id,
                    task_id=task.task_id,
                    thread_id=task.thread_id,
                    graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                    runner_name=runner_used,
                    status="failed",
                    error_type=error_type,
                    error_hint=str(output_summary.get("error_hint") or ""),
                    rollback_recommended=bool(output_summary.get("rollback_recommended", False)),
                    extra={"runner_used": runner_used},
                ),
            )
            failed_task = runtime_store.get_task(task.task_id)
            return RuntimeEntryResult(
                status="failed",
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                runner_used=runner_used,
                task_status=_task_status_value(failed_task.status),
                event_count=len(runtime_store.list_events(task_id=task.task_id)),
                output_summary=output_summary,
                error_type=error_type,
                summary_only=True,
                metadata=_result_metadata(raw_jd, entry_config, selection_metadata),
            )

        runtime_store.update_task_status(task.task_id, TaskStatus.COMPLETED, result=output_summary)
        runtime_store.append_event(
            "task_completed",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode=str(selection_metadata.get("selected_graph_mode") or ""),
                runner_name=runner_used,
                status="completed",
                extra={"runner_used": runner_used},
            ),
        )
        completed_task = runtime_store.get_task(task.task_id)
        return RuntimeEntryResult(
            status="ok",
            session_id=session.session_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            runner_used=runner_used,
            task_status=_task_status_value(completed_task.status),
            event_count=len(runtime_store.list_events(task_id=task.task_id)),
            output_summary=output_summary,
            error_type="",
            summary_only=True,
            metadata=_result_metadata(raw_jd, entry_config, selection_metadata),
        )

    @staticmethod
    def _run_legacy_fallback(
        default_runner,
        raw_jd: str,
        *,
        primary_summary: Mapping[str, Any],
        entry_config: RuntimeEntryConfig,
        runtime_store,
        session,
        task,
        selection_metadata: Mapping[str, Any],
        primary_attempt_id: str,
    ) -> Dict[str, Any]:
        fallback_attempt_id = str(uuid.uuid4())
        primary_error_type = str(primary_summary.get("error_type") or primary_summary.get("runner_error_type") or "")
        primary_error_hint = str(primary_summary.get("error_hint") or "")
        runtime_store.append_event(
            "graph_fallback_requested",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode="skill",
                runner_name="production_skill_graph",
                execution_id=primary_attempt_id,
                status="fallback_requested",
                error_type=primary_error_type,
                error_hint=primary_error_hint,
                fallback_used=True,
                rollback_recommended=True,
                extra={
                    "attempt_type": "primary",
                    "attempt_id": primary_attempt_id,
                    "primary_graph_mode": "skill",
                    "fallback_graph_mode": "legacy",
                    "fallback_attempted": True,
                },
            ),
        )
        runtime_store.append_event(
            "graph_fallback_started",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode="legacy",
                runner_name="default_graph",
                execution_id=fallback_attempt_id,
                status="started",
                fallback_used=True,
                extra={
                    "attempt_type": "fallback",
                    "attempt_id": fallback_attempt_id,
                    "primary_graph_mode": "skill",
                    "fallback_graph_mode": "legacy",
                    "fallback_attempted": True,
                },
            ),
        )
        try:
            fallback_output = RuntimeEntryHarness._call_runner(
                default_runner,
                raw_jd,
                "default_graph",
                RuntimeEntryConfig(
                    graph_mode="legacy",
                    legacy_fallback_enabled=False,
                    allow_planner_fallback=bool(entry_config.allow_planner_fallback),
                    summary_only=True,
                    metadata={"fallback_attempt": True},
                ),
                None,
                runtime_store,
                session,
                task,
            )
            fallback_summary = _apply_graph_selection_summary(
                summarize_runner_output(fallback_output),
                runner_used="default_graph",
                selection_metadata={
                    "selected_graph_mode": "legacy",
                    "graph_mode": "legacy",
                    "requested_graph_mode": "legacy",
                    "runner_name": "default_graph",
                    "rollback_target": "legacy",
                    "selection_reason": "legacy_fallback",
                    "legacy_explicitly_requested": False,
                    "skill_default_used": False,
                    "default_graph_mode": "skill",
                    "selection_source": "fallback",
                },
            )
        except Exception as exc:
            fallback_summary = _apply_graph_selection_summary(
                _failed_output_summary(type(exc).__name__, stage="legacy_fallback", error_hint="legacy_fallback_failed"),
                runner_used="default_graph",
                selection_metadata={
                    "selected_graph_mode": "legacy",
                    "graph_mode": "legacy",
                    "requested_graph_mode": "legacy",
                    "runner_name": "default_graph",
                    "rollback_target": "legacy",
                    "selection_reason": "legacy_fallback",
                },
            )

        fallback_succeeded = not fallback_summary.get("error_type") and fallback_summary.get("status") not in {"failed", "skipped"}
        runtime_store.append_event(
            "graph_fallback_completed" if fallback_succeeded else "graph_fallback_failed",
            session_id=session.session_id,
            task_id=task.task_id,
            payload=build_runtime_event_payload(
                session_id=session.session_id,
                task_id=task.task_id,
                thread_id=task.thread_id,
                graph_mode="legacy",
                runner_name="default_graph",
                execution_id=fallback_attempt_id,
                status="completed" if fallback_succeeded else "failed",
                error_type="" if fallback_succeeded else str(fallback_summary.get("error_type") or "FallbackFailed"),
                error_hint="" if fallback_succeeded else str(fallback_summary.get("error_hint") or "legacy_fallback_failed"),
                fallback_used=True,
                extra={
                    "attempt_type": "fallback",
                    "attempt_id": fallback_attempt_id,
                    "primary_attempt_id": primary_attempt_id,
                    "fallback_succeeded": fallback_succeeded,
                    "fallback_attempted": True,
                },
            ),
        )
        return _merge_fallback_summary(
            primary_summary=primary_summary,
            fallback_summary=fallback_summary,
            primary_attempt_id=primary_attempt_id,
            fallback_attempt_id=fallback_attempt_id,
            fallback_succeeded=fallback_succeeded,
        )

    @staticmethod
    def _select_runner(default_runner, demo_runner, variant_runner, production_skill_graph_runner, config: RuntimeEntryConfig):
        graph_factory_config = resolve_recruit_graph_factory_config(
            requested_graph_mode=config.graph_mode,
            use_production_skill_graph_alias=bool(config.use_production_skill_graph),
            allow_planner_fallback=bool(config.allow_planner_fallback),
            summary_only=bool(config.summary_only),
        )
        graph_factory = RecruitGraphFactory(
            legacy_runner=default_runner,
            skill_runner=production_skill_graph_runner,
            config=graph_factory_config,
        )
        metadata = {
            "demo_mode_requested": bool(config.use_demo_mode),
            "demo_mode_enabled": bool(config.demo_mode_enabled),
            "skill_backed_variant_requested": bool(config.use_skill_backed_variant),
            "production_skill_graph_requested": bool(config.use_production_skill_graph),
            "requested_graph_mode": str(config.graph_mode or ""),
            "selected_graph_mode": graph_factory_config.mode.value,
            "default_graph_mode": graph_factory_config.default_graph_mode.value,
            "graph_mode": graph_factory_config.mode.value,
            "runner_name": "production_skill_graph" if graph_factory_config.mode.value == "skill" else "default_graph",
            "rollback_target": graph_factory_config.rollback_target.value,
            "legacy_alias_used": bool(graph_factory_config.legacy_alias_used),
            "legacy_explicitly_requested": bool(graph_factory_config.legacy_explicitly_requested),
            "skill_default_used": bool(graph_factory_config.skill_default_used),
            "selection_source": str(graph_factory_config.selection_source),
            "allow_memory_context": bool(config.allow_memory_context),
            "demo_mode_requested_but_disabled": False,
            "variant_requested_but_unavailable": False,
            "production_skill_graph_unavailable": False,
            "config_error": str(graph_factory_config.config_error or ""),
            "runner_selection_reason": str(graph_factory_config.selection_reason or "default_legacy"),
            "selection_reason": str(graph_factory_config.selection_reason or "default_legacy"),
        }
        if graph_factory_config.config_error:
            selection = graph_factory.create_runner()
            metadata.update(selection.to_dict())
            return selection.runner, selection.runner_name, metadata
        if config.use_production_skill_graph and (config.use_skill_backed_variant or config.use_demo_mode):
            metadata["config_error"] = "conflicting_runtime_runner_flags"
            metadata["runner_selection_reason"] = "config_error"
            return _config_error_runner("conflicting_runtime_runner_flags"), "runtime_config_error", metadata
        if config.use_demo_mode and config.demo_mode_enabled and demo_runner is not None:
            metadata["runner_selection_reason"] = "demo_mode_enabled"
            return demo_runner, "demo_mode", metadata
        if config.use_demo_mode and not config.demo_mode_enabled:
            metadata["demo_mode_requested_but_disabled"] = True
            metadata["runner_selection_reason"] = "demo_mode_disabled_default_fallback"
            return default_runner, "default_graph", metadata
        if config.use_demo_mode and demo_runner is None:
            metadata["demo_mode_requested_but_disabled"] = True
            metadata["runner_selection_reason"] = "demo_runner_missing_default_fallback"
            return default_runner, "default_graph", metadata
        if config.use_skill_backed_variant:
            if variant_runner is not None:
                metadata["runner_selection_reason"] = "skill_backed_variant_requested"
                return variant_runner, "skill_backed_variant", metadata
            metadata["variant_requested_but_unavailable"] = True
            metadata["runner_selection_reason"] = "variant_runner_missing_default_fallback"
            return default_runner, "default_graph", metadata
        if graph_factory_config.mode.value == "skill":
            selection = graph_factory.create_runner()
            metadata.update(selection.to_dict())
            if selection.selected_mode.value == "legacy" and production_skill_graph_runner is None:
                metadata["production_skill_graph_unavailable"] = True
            return selection.runner, selection.runner_name, metadata
        selection = graph_factory.create_runner()
        metadata.update(selection.to_dict())
        return selection.runner, selection.runner_name, metadata

    @staticmethod
    def _call_runner(runner, raw_jd: str, runner_used: str, config: RuntimeEntryConfig, memory_context: Any, runtime_store=None, session=None, task=None):
        effective_memory_context = memory_context if config.allow_memory_context else None
        runner_metadata = {
            "summary_only": True,
            "runner_used": runner_used,
            "graph_mode": "skill" if runner_used == "production_skill_graph" else ("legacy" if runner_used == "default_graph" else runner_used),
            "allow_memory_context": bool(config.allow_memory_context),
            "allow_planner_fallback": bool(config.allow_planner_fallback),
            "memory_context_provided": bool(effective_memory_context is not None),
            "require_ab_smoke_pass": bool(config.require_ab_smoke_pass),
            "rollback_on_variant_failure": bool(config.rollback_on_variant_failure),
            "runtime_store": runtime_store,
            "session_id": getattr(session, "session_id", ""),
            "task_id": getattr(task, "task_id", ""),
            "thread_id": getattr(task, "thread_id", ""),
        }
        if runner_used == "production_skill_graph":
            runner_metadata.update(
                {
                    "allow_memory_context": bool(config.allow_memory_context),
                }
            )
        if isinstance(config.metadata, Mapping) and "memory_context_summary" in config.metadata:
            runner_metadata["memory_context_summary"] = dict(config.metadata["memory_context_summary"] or {})
            runtime_store.append_event(
                "memory_context_injected" if effective_memory_context is not None else "memory_context_built",
                session_id=getattr(session, "session_id", ""),
                task_id=getattr(task, "task_id", ""),
                payload=build_runtime_event_payload(
                    session_id=getattr(session, "session_id", ""),
                    task_id=getattr(task, "task_id", ""),
                    thread_id=getattr(task, "thread_id", ""),
                    graph_mode=runner_metadata["graph_mode"],
                    runner_name=runner_used,
                    status="completed",
                    extra={
                        "memory_mode": str(config.metadata.get("memory_mode") or "off"),
                        "memory_context_provided": bool(effective_memory_context is not None),
                        "eligible_count": int((config.metadata.get("memory_context_summary") or {}).get("eligible_count") or 0),
                        "denied_count": int((config.metadata.get("memory_context_summary") or {}).get("denied_count") or 0),
                        "summary_only": True,
                    },
                ),
            )
        try:
            if runner_used == "default_graph":
                return runner(raw_jd, metadata=runner_metadata)
            return runner(raw_jd, memory_context=effective_memory_context, metadata=runner_metadata)
        except TypeError:
            return runner(raw_jd)


def build_default_graph_runner():
    """Build a lazy real graph runner for manual CLI use only."""

    def run(raw_jd: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        runtime_store = (metadata or {}).get("runtime_store") if isinstance(metadata, Mapping) else None
        session_id = str((metadata or {}).get("session_id") or "")
        task_id = str((metadata or {}).get("task_id") or "")
        thread_id = str((metadata or {}).get("thread_id") or "")
        graph = None
        state = {}
        config = {}
        try:
            load_project_dotenv()
            from src.core.graph import create_recruit_graph

            graph = create_recruit_graph()
        except Exception as exc:
            _append_legacy_graph_event(runtime_store, "graph_failed", session_id, task_id, thread_id, type(exc).__name__, "create_graph")
            return _graph_failed_summary(
                type(exc).__name__,
                stage="create_graph",
                error_hint=_classify_error_hint(exc),
                state=state,
                config=config,
            )

        try:
            state = build_default_graph_initial_state(raw_jd)
            config = build_default_graph_config()
        except Exception as exc:
            _append_legacy_graph_event(runtime_store, "graph_failed", session_id, task_id, thread_id, type(exc).__name__, "build_initial_state")
            return _graph_failed_summary(
                type(exc).__name__,
                stage="build_initial_state",
                error_hint="missing_or_invalid_initial_state",
                state=state,
                config=config,
            )

        try:
            _append_legacy_graph_event(runtime_store, "graph_started", session_id, task_id, thread_id, "", "")
            events = list(graph.stream(state, config))
            final_state = _safe_get_graph_state(graph, config)
            summary = summarize_graph_events(events, final_state=final_state, input_state=state, config=config)
            _append_legacy_graph_event(runtime_store, "graph_completed", session_id, task_id, thread_id, "", "")
            return summary
        except Exception as exc:
            _append_legacy_graph_event(runtime_store, "graph_failed", session_id, task_id, thread_id, type(exc).__name__, "graph_stream")
            return _graph_failed_summary(
                type(exc).__name__,
                stage="graph_stream",
                error_hint=_classify_error_hint(exc),
                state=state,
                config=config,
            )

    return run


def load_project_dotenv() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv())


def build_default_graph_initial_state(raw_jd: str) -> Dict[str, Any]:
    from src.core.state import create_initial_state

    return dict(create_initial_state(raw_jd))


def build_default_graph_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _append_legacy_graph_event(runtime_store, event_type: str, session_id: str, task_id: str, thread_id: str, error_type: str, error_hint: str):
    if runtime_store is None or not task_id:
        return
    runtime_store.append_event(
        event_type,
        session_id=session_id,
        task_id=task_id,
        payload=build_runtime_event_payload(
            session_id=session_id,
            task_id=task_id,
            thread_id=thread_id,
            graph_mode="legacy",
            runner_name="default_graph",
            status="failed" if event_type.endswith("failed") else ("started" if event_type.endswith("started") else "completed"),
            error_type=error_type,
            error_hint=error_hint,
            extra={"runner_used": "default_graph"},
        ),
    )


def _append_graph_attempt_event(
    runtime_store,
    event_type: str,
    session,
    task,
    *,
    graph_mode: str,
    runner_name: str,
    attempt_id: str,
    status: str,
    error_type: str = "",
    error_hint: str = "",
    fallback_attempted: bool = False,
):
    if runtime_store is None or task is None:
        return
    runtime_store.append_event(
        event_type,
        session_id=session.session_id,
        task_id=task.task_id,
        payload=build_runtime_event_payload(
            session_id=session.session_id,
            task_id=task.task_id,
            thread_id=task.thread_id,
            graph_mode=graph_mode,
            runner_name=runner_name,
            execution_id=attempt_id,
            status=status,
            error_type=error_type,
            error_hint=error_hint,
            fallback_used=bool(fallback_attempted),
            rollback_recommended=bool(fallback_attempted),
            extra={
                "attempt_type": "primary",
                "attempt_id": attempt_id,
                "primary_graph_mode": graph_mode,
                "fallback_graph_mode": "legacy" if fallback_attempted else "",
                "fallback_attempted": bool(fallback_attempted),
            },
        ),
    )


def build_demo_mode_runner(
    default_runner=None,
    variant_runner=None,
    *,
    require_ab_smoke_pass: bool = True,
    rollback_on_variant_failure: bool = True,
    allow_memory_context: bool = False,
):
    """Build a lazy limited demo runner; it remains default-only unless configured elsewhere."""

    def run(raw_jd: str, memory_context=None, metadata=None) -> Dict[str, Any]:
        from src.integration.demo_mode import DemoModeConfig, LimitedProductionDemoHarness

        safe_default = default_runner or build_default_graph_runner()
        result = LimitedProductionDemoHarness().run(
            raw_jd,
            default_runner=safe_default,
            variant_runner=variant_runner,
            config=DemoModeConfig(
                enabled=True,
                use_skill_backed_variant=variant_runner is not None,
                allow_memory_context=bool(allow_memory_context),
                require_ab_smoke_pass=bool(require_ab_smoke_pass),
                rollback_on_variant_failure=bool(rollback_on_variant_failure),
                summary_only=True,
            ),
            memory_context=memory_context,
        )
        data = result.to_dict()
        summary = dict(data.get("output_summary") or {})
        output_status = summary.get("status")
        summary["status"] = (
            output_status
            if output_status in {"failed", "skipped"}
            else data.get("status") or output_status or "unknown"
        )
        summary["runner_mode"] = "limited_demo_default_only"
        if variant_runner is not None:
            summary["runner_mode"] = "limited_demo_with_variant"
        summary["rollback_to_default"] = bool(data.get("rollback_to_default", False))
        summary["rollback_reason"] = str(data.get("rollback_reason") or "")
        summary["ab_smoke_summary"] = _safe_ab_summary(data.get("ab_smoke_summary"))
        summary["metadata"] = {
            "keys": sorted(str(key) for key in (data.get("metadata") or {}).keys()),
            "summary_only": True,
            "memory_context_allowed": bool(allow_memory_context),
            "memory_context_provided": bool(memory_context is not None and allow_memory_context),
            "input_metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
        }
        return summary

    return run


def build_fake_variant_runner_for_tests(
    *,
    candidate_count: int = 1,
    report_count: int = 1,
    top_score_present: bool = True,
    status: str = "ok",
):
    """Deterministic variant runner for structure tests; never imports real Agents."""

    def run(raw_jd: str, memory_context=None, metadata=None) -> Dict[str, Any]:
        return {
            "status": status,
            "candidate_count": candidate_count,
            "report_count": report_count,
            "top_score_present": top_score_present,
            "output_keys": ["candidate_count", "report_count", "status", "variant_marker"],
            "variant_marker": "skill_backed_variant_fake",
            "memory_context_seen": bool(memory_context is not None),
            "metadata": {
                "summary_only": True,
                "raw_jd_length": len(raw_jd or ""),
                "memory_context_key": "present" if memory_context is not None else "absent",
                "input_metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
            },
        }

    return run


def summarize_graph_events(events, final_state=None, input_state=None, config=None) -> Dict[str, Any]:
    event_list = list(events or [])
    output_keys = set()
    candidate_count = 0
    report_count = 0
    top_score_present = False
    for event in event_list:
        if not isinstance(event, Mapping):
            continue
        for update in event.values():
            if not isinstance(update, Mapping):
                continue
            output_keys.update(str(key) for key in update.keys())
            candidates = update.get("candidate_pool") or update.get("retrieved_candidates") or []
            reports = update.get("final_reports") or update.get("match_reports") or []
            if isinstance(candidates, list):
                candidate_count = max(candidate_count, len(candidates))
            if isinstance(reports, list):
                report_count = max(report_count, len(reports))
                top_score_present = top_score_present or any(
                    isinstance(report, Mapping) and "total_score" in report for report in reports
                )
    final_values = _final_state_values(final_state)
    if final_values:
        output_keys.update(str(key) for key in final_values.keys())
        final_candidates = final_values.get("candidate_pool") or []
        final_reports = final_values.get("final_reports") or []
        if isinstance(final_candidates, list):
            candidate_count = max(candidate_count, len(final_candidates))
        if isinstance(final_reports, list):
            report_count = max(report_count, len(final_reports))
            top_score_present = top_score_present or any(
                isinstance(report, Mapping) and "total_score" in report for report in final_reports
            )
    return {
        "status": "ok",
        "event_count": len(event_list),
        "candidate_count": candidate_count,
        "report_count": report_count,
        "top_score_present": top_score_present,
        "output_keys": sorted(output_keys),
        "error_type": "",
        "graph_input_keys": sorted(str(key) for key in (input_state or {}).keys()),
        "graph_input_shape": _shape_summary(input_state or {}),
        "graph_config_has_thread_id": _config_has_thread_id(config or {}),
        "graph_result_keys": sorted(str(key) for key in final_values.keys()),
        "runner_error_type": "",
        "runner_error_stage": "",
        "error_hint": "",
        "candidate_pool": _safe_candidate_pool_summary(final_values.get("candidate_pool") or []),
        "final_reports": _safe_final_report_summary(final_values.get("final_reports") or []),
        "metadata": {
            "summary_only": True,
            "production_graph_invoked": True,
            "diagnostics_available": True,
        },
    }


def summarize_runner_output(output: Any) -> Dict[str, Any]:
    data = dict(output or {}) if isinstance(output, Mapping) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    output_keys = data.get("output_keys")
    if output_keys is None:
        output_keys = [
            key
            for key in data.keys()
            if key not in {"metadata", "raw_jd", "jd_text", "reasoning", "llm_response", "resume_text"}
        ]
    return {
        "status": str(data.get("status") or "unknown"),
        "candidate_count": _safe_int(data.get("candidate_count")),
        "candidate_profile_preview_count": _safe_int(data.get("candidate_profile_preview_count")),
        "candidate_preview_audit": _safe_candidate_preview_audit(data.get("candidate_preview_audit")),
        "report_count": _safe_int(data.get("report_count")),
        "top_score_present": bool(data.get("top_score_present", data.get("score_present", False))),
        "output_keys": sorted(str(key) for key in output_keys),
        "error_type": str(data.get("error_type") or ""),
        "graph_input_keys": sorted(str(key) for key in data.get("graph_input_keys", [])),
        "graph_input_shape": dict(data.get("graph_input_shape") or {}),
        "graph_config_has_thread_id": bool(data.get("graph_config_has_thread_id", False)),
        "graph_result_keys": sorted(str(key) for key in data.get("graph_result_keys", [])),
        "runner_error_type": str(data.get("runner_error_type") or data.get("error_type") or ""),
        "runner_error_stage": str(data.get("runner_error_stage") or ""),
        "error_hint": str(data.get("error_hint") or ""),
        "planner_output_keys": sorted(str(key) for key in data.get("planner_output_keys", [])),
        "planner_expected_keys": sorted(str(key) for key in data.get("planner_expected_keys", [])),
        "planner_adapter_error_hint": str(data.get("planner_adapter_error_hint") or ""),
        "planner_invocation_stage": str(data.get("planner_invocation_stage") or ""),
        "planner_input_shape": _safe_planner_input_shape(data.get("planner_input_shape")),
        "provider_error_type": str(data.get("provider_error_type") or ""),
        "planner_provider_diagnostics": _safe_planner_provider_diagnostics(
            data.get("planner_provider_diagnostics")
        ),
        "planner_fallback_used": bool(data.get("planner_fallback_used", False)),
        "planner_fallback_type": str(data.get("planner_fallback_type") or ""),
        "fallback_not_real_planner_success": bool(data.get("fallback_not_real_planner_success", False)),
        "fallback_used": bool(data.get("fallback_used", data.get("planner_fallback_used", False))),
        "real_planner_invoked": bool(data.get("real_planner_invoked", False)),
        "real_planner_failed": bool(data.get("real_planner_failed", False)),
        "planner_source": str(data.get("planner_source") or ""),
        "retriever_source": str(data.get("retriever_source") or ""),
        "matcher_source": str(data.get("matcher_source") or ""),
        "matcher_invocation_stage": str(data.get("matcher_invocation_stage") or ""),
        "matcher_input_keys": sorted(str(key) for key in data.get("matcher_input_keys", [])),
        "matcher_candidate_id": str(data.get("matcher_candidate_id") or ""),
        "matcher_candidate_name_present": bool(data.get("matcher_candidate_name_present", False)),
        "matcher_skills_count": _safe_int(data.get("matcher_skills_count")),
        "real_matcher_invoked": bool(data.get("real_matcher_invoked", False)),
        "real_matcher_failed": bool(data.get("real_matcher_failed", False)),
        "matcher_adapter_error_hint": str(data.get("matcher_adapter_error_hint") or ""),
        "matcher_provider_error_type": str(data.get("matcher_provider_error_type") or ""),
        "matcher_output_keys": sorted(str(key) for key in data.get("matcher_output_keys", [])),
        "claim_verification_enabled": bool(data.get("claim_verification_enabled", False)),
        "claim_verification_status": str(data.get("claim_verification_status") or ""),
        "claim_verification_case_count": _safe_int(data.get("claim_verification_case_count")),
        "claim_support_pass_rate": float(data.get("claim_support_pass_rate") or 0.0),
        "average_claim_support_rate": float(data.get("average_claim_support_rate") or 0.0),
        "unsupported_claim_case_rate": float(data.get("unsupported_claim_case_rate") or 0.0),
        "critical_unsupported_claim_rate": float(data.get("critical_unsupported_claim_rate") or 0.0),
        "evidence_coverage_rate": float(data.get("evidence_coverage_rate") or 0.0),
        "retriever_factory_source": str(data.get("retriever_factory_source") or ""),
        "candidate_preview_source": str(data.get("candidate_preview_source") or ""),
        "candidate_preview_version": str(data.get("candidate_preview_version") or ""),
        "matcher_input_source": str(data.get("matcher_input_source") or ""),
        **_safe_mcp_summary(data),
        "candidate_ids": [str(item) for item in data.get("candidate_ids", []) if str(item)],
        "top_scores": [float(item) for item in data.get("top_scores", []) if isinstance(item, (int, float))],
        "skill_names": [str(item) for item in data.get("skill_names", []) if str(item)],
        "skill_event_count": _safe_int(data.get("skill_event_count")),
        "skill_execution_count": _safe_int(data.get("skill_execution_count", data.get("skill_event_count"))),
        "production_skill_graph_enabled": bool(data.get("production_skill_graph_enabled", False)),
        "legacy_graph_invoked": bool(data.get("legacy_graph_invoked", False)),
        "rollback_recommended": bool(data.get("rollback_recommended", False)),
        "rollback_target": str(data.get("rollback_target") or ""),
        "rollback_baseline": str(data.get("rollback_baseline") or ""),
        "provenance_summary_only": bool(data.get("provenance_summary_only", False)),
        "memory_context_requested": bool(data.get("memory_context_requested", False)),
        "memory_context_provided": bool(data.get("memory_context_provided", False)),
        "memory_source": str(data.get("memory_source") or "none"),
        "memory_db_path_present": bool(data.get("memory_db_path_present", False)),
        "memory_store_loaded": bool(data.get("memory_store_loaded", False)),
        "memory_records_seen": _safe_int(data.get("memory_records_seen")),
        "memory_context_eligible_count": _safe_int(data.get("memory_context_eligible_count")),
        "memory_context_denied_count": _safe_int(data.get("memory_context_denied_count")),
        "memory_context_requires_review_count": _safe_int(data.get("memory_context_requires_review_count")),
        "memory_context_rendered_char_count": _safe_int(data.get("memory_context_rendered_char_count")),
        "memory_context_governance_applied": bool(data.get("memory_context_governance_applied", False)),
        "memory_context_revoked_filtered_count": _safe_int(data.get("memory_context_revoked_filtered_count")),
        "memory_context_expired_filtered_count": _safe_int(data.get("memory_context_expired_filtered_count")),
        "memory_context_superseded_filtered_count": _safe_int(data.get("memory_context_superseded_filtered_count")),
        "memory_context_demo_mode": bool(data.get("memory_context_demo_mode", False)),
        "memory_context_source": str(data.get("memory_context_source") or ""),
        "memory_context_summary_only": bool(data.get("memory_context_summary_only", False)),
        "retriever_init_stage": str(data.get("retriever_init_stage") or ""),
        "retriever_config_summary": (
            _safe_retriever_config_summary(data.get("retriever_config_summary"))
            if "retriever_config_summary" in data
            else {}
        ),
        "retriever_init_diagnostics": (
            _safe_retriever_init_diagnostics(data.get("retriever_init_diagnostics"))
            if "retriever_init_diagnostics" in data
            else {}
        ),
        "retriever_embedding_readiness": (
            _safe_embedding_readiness(data.get("retriever_embedding_readiness"))
            if "retriever_embedding_readiness" in data
            else {}
        ),
        "rollback_to_default": bool(data.get("rollback_to_default", False)),
        "rollback_reason": str(data.get("rollback_reason") or ""),
        "ab_smoke_summary": _safe_ab_summary(data.get("ab_smoke_summary")),
        "env_readiness": _safe_env_readiness(data.get("env_readiness")),
        "metadata": {
            "keys": sorted(str(key) for key in metadata.keys()),
            "summary_only": True,
        },
    }


def _create_store(config: RuntimeEntryConfig):
    if config.db_path:
        return SQLiteRuntimeStore(config.db_path)
    return InMemoryRuntimeStore()


def _failed_output_summary(error_type: str, stage: str = "runner_call", error_hint: str = "graph_invoke_error") -> Dict[str, Any]:
    return {
        "status": "failed",
        "candidate_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "output_keys": ["error_type", "status"],
        "error_type": str(error_type or "Error"),
        "graph_input_keys": [],
        "graph_input_shape": {},
        "graph_config_has_thread_id": False,
        "graph_result_keys": [],
        "runner_error_type": str(error_type or "Error"),
        "runner_error_stage": stage,
        "error_hint": error_hint,
        "rollback_to_default": False,
        "rollback_reason": "",
        "ab_smoke_summary": {},
        "env_readiness": {},
        "metadata": {"keys": [], "summary_only": True},
    }


def _config_error_runner(error_hint: str):
    def run(_raw_jd: str, **_kwargs) -> Dict[str, Any]:
        return {
            "status": "failed",
            "error_type": "RuntimeConfigError",
            "error_hint": str(error_hint or "runtime_config_error"),
            "candidate_count": 0,
            "report_count": 0,
            "top_score_present": False,
            "production_graph_replaced": False,
            "summary_only": True,
            "metadata": {"summary_only": True},
        }

    return run


def _safe_mcp_summary(data: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_source = str(data.get("candidate_source") or "")
    mcp_server = str(data.get("mcp_server") or "")
    mcp_transport = str(data.get("mcp_transport") or "")
    tool_success_count = _safe_int(data.get("tool_success_count"))
    mcp_tool_event_count = _safe_int(data.get("mcp_tool_event_count"))
    mcp_fallback_used = bool(data.get("mcp_fallback_used", False))
    if not any([candidate_source, mcp_server, mcp_transport, tool_success_count, mcp_tool_event_count, mcp_fallback_used]):
        return {}
    return {
        "candidate_source": candidate_source,
        "mcp_server": mcp_server,
        "mcp_transport": mcp_transport,
        "tool_success_count": tool_success_count,
        "mcp_fallback_used": mcp_fallback_used,
        "mcp_tool_event_count": mcp_tool_event_count,
    }


def _result_metadata(raw_jd: str, config: RuntimeEntryConfig, selection_metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    metadata = {
        "mode": "runtime_entry_harness",
        "jd_length": len(raw_jd or ""),
        "use_demo_mode": bool(config.use_demo_mode),
        "demo_mode_enabled": bool(config.demo_mode_enabled),
        "use_skill_backed_variant": bool(config.use_skill_backed_variant),
        "use_production_skill_graph": bool(config.use_production_skill_graph),
        "graph_mode": str(config.graph_mode or ""),
        "legacy_fallback_enabled": bool(config.legacy_fallback_enabled),
        "allow_memory_context": bool(config.allow_memory_context),
        "allow_planner_fallback": bool(config.allow_planner_fallback),
        "require_ab_smoke_pass": bool(config.require_ab_smoke_pass),
        "rollback_on_variant_failure": bool(config.rollback_on_variant_failure),
        "summary_only": bool(config.summary_only),
        "metadata_keys": sorted(str(key) for key in config.metadata.keys()),
        "production_graph_replaced": False,
    }
    if selection_metadata:
        metadata.update(
            {
                "demo_mode_requested": bool(selection_metadata.get("demo_mode_requested", False)),
                "skill_backed_variant_requested": bool(
                    selection_metadata.get("skill_backed_variant_requested", False)
                ),
                "production_skill_graph_requested": bool(
                    selection_metadata.get("production_skill_graph_requested", False)
                ),
                "requested_graph_mode": str(selection_metadata.get("requested_graph_mode") or ""),
                "selected_graph_mode": str(selection_metadata.get("selected_graph_mode") or ""),
                "default_graph_mode": str(selection_metadata.get("default_graph_mode") or "skill"),
                "graph_mode": str(selection_metadata.get("graph_mode") or ""),
                "runner_name": str(selection_metadata.get("runner_name") or ""),
                "rollback_target": str(selection_metadata.get("rollback_target") or ""),
                "selection_reason": str(selection_metadata.get("selection_reason") or ""),
                "selection_source": str(selection_metadata.get("selection_source") or ""),
                "legacy_alias_used": bool(selection_metadata.get("legacy_alias_used", False)),
                "legacy_explicitly_requested": bool(selection_metadata.get("legacy_explicitly_requested", False)),
                "skill_default_used": bool(selection_metadata.get("skill_default_used", False)),
                "demo_mode_requested_but_disabled": bool(
                    selection_metadata.get("demo_mode_requested_but_disabled", False)
                ),
                "variant_requested_but_unavailable": bool(
                    selection_metadata.get("variant_requested_but_unavailable", False)
                ),
                "production_skill_graph_unavailable": bool(
                    selection_metadata.get("production_skill_graph_unavailable", False)
                ),
                "config_error": str(selection_metadata.get("config_error") or ""),
                "runner_selection_reason": str(selection_metadata.get("runner_selection_reason") or ""),
            }
        )
    return metadata


def _apply_graph_selection_summary(
    output_summary: Dict[str, Any],
    *,
    runner_used: str,
    selection_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_mode = str(selection_metadata.get("selected_graph_mode") or selection_metadata.get("graph_mode") or "")
    if not selected_mode:
        selected_mode = "skill" if runner_used == "production_skill_graph" else "legacy"
    output_summary["graph_mode"] = selected_mode
    output_summary["selected_graph_mode"] = selected_mode
    output_summary["requested_graph_mode"] = str(selection_metadata.get("requested_graph_mode") or "")
    output_summary["runner_name"] = str(selection_metadata.get("runner_name") or runner_used)
    output_summary["selection_reason"] = str(selection_metadata.get("selection_reason") or selection_metadata.get("runner_selection_reason") or "")
    output_summary["selection_source"] = str(selection_metadata.get("selection_source") or "")
    output_summary["default_graph_mode"] = str(selection_metadata.get("default_graph_mode") or "skill")
    output_summary["legacy_alias_used"] = bool(selection_metadata.get("legacy_alias_used", False))
    output_summary["legacy_explicitly_requested"] = bool(selection_metadata.get("legacy_explicitly_requested", False))
    output_summary["skill_default_used"] = bool(selection_metadata.get("skill_default_used", False))
    output_summary["rollback_target"] = str(
        output_summary.get("rollback_target") or selection_metadata.get("rollback_target") or "legacy"
    )
    output_summary["rollback_baseline"] = str(output_summary.get("rollback_baseline") or "legacy_default_graph")
    output_summary["legacy_graph_invoked"] = bool(selected_mode == "legacy" and runner_used == "default_graph")
    output_summary["skill_graph_invoked"] = bool(selected_mode == "skill" and runner_used == "production_skill_graph")
    output_summary["production_skill_graph_invoked"] = output_summary["skill_graph_invoked"]
    output_summary["production_graph_replaced"] = False
    output_summary["fallback_attempted"] = bool(output_summary.get("fallback_attempted", False))
    output_summary["fallback_succeeded"] = bool(output_summary.get("fallback_succeeded", False))
    output_summary["fallback_graph_mode"] = str(output_summary.get("fallback_graph_mode") or "")
    output_summary["primary_graph_mode"] = str(output_summary.get("primary_graph_mode") or selected_mode)
    output_summary["final_runner_used"] = str(output_summary.get("final_runner_used") or runner_used)
    output_summary["final_status"] = str(output_summary.get("final_status") or output_summary.get("status") or "")
    health = _classify_graph_health(output_summary)
    output_summary["graph_health_status"] = health["graph_health_status"]
    output_summary["health_warning_codes"] = health["health_warning_codes"]
    output_summary["critical_failure_codes"] = health["critical_failure_codes"]
    output_summary["summary_only"] = True
    return output_summary


def _apply_memory_context_summary(output_summary: Dict[str, Any], config: RuntimeEntryConfig) -> None:
    if not isinstance(config.metadata, Mapping):
        return
    summary = config.metadata.get("memory_context_summary")
    if not isinstance(summary, Mapping):
        output_summary.setdefault("memory_mode", str(config.metadata.get("memory_mode") or "off"))
        output_summary["memory_context_requested"] = bool(
            output_summary.get("memory_context_requested", False) or config.allow_memory_context
        )
        output_summary.setdefault("memory_context_provided", False)
        return
    output_summary["memory_mode"] = str(config.metadata.get("memory_mode") or "off")
    output_summary["memory_context_summary"] = dict(summary)
    output_summary["memory_context_requested"] = bool(config.allow_memory_context or output_summary["memory_mode"] == "governed")
    output_summary["memory_context_provided"] = bool(summary.get("provided", False))
    output_summary["memory_records_seen"] = int(summary.get("memory_records_seen") or 0)
    output_summary["memory_context_eligible_count"] = int(summary.get("eligible_count") or 0)
    output_summary["memory_context_denied_count"] = int(summary.get("denied_count") or 0)
    output_summary["memory_context_rendered_char_count"] = int(summary.get("rendered_char_count") or 0)
    output_summary["memory_context_governance_applied"] = bool(summary.get("governance_applied", False))
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), Mapping) else {}
    output_summary["memory_ids_used"] = list(metadata.get("memory_ids_used") or [])
    output_summary["memory_versions_used"] = list(metadata.get("memory_versions_used") or [])


def _classify_graph_health(summary: Mapping[str, Any]) -> Dict[str, Any]:
    critical = []
    warnings = []
    status = str(summary.get("status") or "")
    if status == "failed" or summary.get("error_type"):
        critical.append(str(summary.get("error_hint") or summary.get("error_type") or "graph_failed"))
    if _safe_int(summary.get("candidate_count")) == 0 and summary.get("selected_graph_mode") == "skill":
        critical.append("empty_candidates")
    if _safe_int(summary.get("report_count")) == 0 and summary.get("selected_graph_mode") == "skill":
        critical.append("empty_reports")
    if status == "completed_with_limit":
        warnings.append("completed_with_limit")
    if bool(summary.get("rollback_recommended", False)) and not critical:
        warnings.append("rollback_recommended")
    if critical:
        health = "critical"
    elif warnings:
        health = "degraded"
    else:
        health = "healthy"
    return {
        "graph_health_status": health,
        "health_warning_codes": sorted(set(warnings)),
        "critical_failure_codes": sorted(set(code for code in critical if code)),
    }


def _should_attempt_legacy_fallback(summary: Mapping[str, Any], runner_used: str, config: RuntimeEntryConfig) -> bool:
    if runner_used != "production_skill_graph":
        return False
    if not bool(config.legacy_fallback_enabled):
        return False
    if str(summary.get("graph_health_status") or "") != "critical":
        return False
    if bool(summary.get("fallback_attempted", False)):
        return False
    return True


def _merge_fallback_summary(
    *,
    primary_summary: Mapping[str, Any],
    fallback_summary: Mapping[str, Any],
    primary_attempt_id: str,
    fallback_attempt_id: str,
    fallback_succeeded: bool,
) -> Dict[str, Any]:
    final = dict(fallback_summary)
    final["status"] = "completed_with_fallback" if fallback_succeeded else "failed"
    final["final_status"] = final["status"]
    final["primary_graph_mode"] = "skill"
    final["fallback_graph_mode"] = "legacy"
    final["fallback_attempted"] = True
    final["fallback_succeeded"] = bool(fallback_succeeded)
    final["fallback_attempt_id"] = fallback_attempt_id
    final["primary_attempt_id"] = primary_attempt_id
    final["primary_error_type"] = str(primary_summary.get("error_type") or primary_summary.get("runner_error_type") or "")
    final["primary_error_hint"] = str(primary_summary.get("error_hint") or "")
    final["primary_summary"] = _safe_primary_summary(primary_summary)
    final["fallback_summary"] = _safe_fallback_summary(fallback_summary)
    final["final_runner_used"] = "default_graph" if fallback_succeeded else "none"
    final["runner_name"] = "production_skill_graph"
    final["selected_graph_mode"] = "skill"
    final["graph_mode"] = "skill"
    final["legacy_graph_invoked"] = bool(fallback_succeeded)
    final["skill_graph_invoked"] = True
    final["production_skill_graph_invoked"] = True
    final["rollback_target"] = "legacy"
    final["rollback_baseline"] = "legacy_default_graph"
    final["rollback_recommended"] = not fallback_succeeded
    final["graph_health_status"] = "degraded" if fallback_succeeded else "critical"
    final["health_warning_codes"] = ["completed_with_fallback"] if fallback_succeeded else []
    final["critical_failure_codes"] = [] if fallback_succeeded else ["fallback_failed"]
    final["summary_only"] = True
    final["output_keys"] = sorted(set([*final.get("output_keys", []), "fallback_attempted", "fallback_succeeded"]))
    return final


def _safe_primary_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(summary.get("status") or ""),
        "error_type": str(summary.get("error_type") or ""),
        "error_hint": str(summary.get("error_hint") or ""),
        "candidate_count": _safe_int(summary.get("candidate_count")),
        "report_count": _safe_int(summary.get("report_count")),
        "graph_health_status": str(summary.get("graph_health_status") or ""),
        "summary_only": True,
    }


def _safe_fallback_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(summary.get("status") or ""),
        "error_type": str(summary.get("error_type") or ""),
        "error_hint": str(summary.get("error_hint") or ""),
        "candidate_count": _safe_int(summary.get("candidate_count")),
        "report_count": _safe_int(summary.get("report_count")),
        "top_score_present": bool(summary.get("top_score_present", False)),
        "summary_only": True,
    }


def _safe_ab_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "case_id": str(value.get("case_id") or ""),
        "default_status": str(value.get("default_status") or ""),
        "variant_status": str(value.get("variant_status") or ""),
        "rollback_to_default": bool(value.get("rollback_to_default", False)),
        "rollback_reason": str(value.get("rollback_reason") or ""),
        "risk_level": str(value.get("risk_level") or ""),
        "summary_only": True,
    }


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


def _safe_candidate_preview_audit(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    return {
        "candidate_profile_preview_count": _safe_int(value.get("candidate_profile_preview_count")),
        "candidate_id_present": _safe_int(value.get("candidate_id_present")),
        "candidate_name_present": _safe_int(value.get("candidate_name_present")),
        "skills_count": _safe_int(value.get("skills_count")),
        "evidence_summary_present": _safe_int(value.get("evidence_summary_present")),
        "source_document_id_present": _safe_int(value.get("source_document_id_present")),
        "summary_only": True,
    }


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
        "project_root": str(value.get("project_root") or "")[:240],
        "data_dir_present": bool(value.get("data_dir_present", False)),
        "chroma_dir_present": bool(value.get("chroma_dir_present", False)),
        "chroma_dir_non_empty": bool(value.get("chroma_dir_non_empty", False)),
        "chroma_dir_file_count": _safe_int(value.get("chroma_dir_file_count")),
        "persist_dir_used": str(value.get("persist_dir_used") or "")[:240],
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


def _safe_planner_input_shape(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    raw_keys = value.get("input_keys") or []
    if not isinstance(raw_keys, list):
        raw_keys = []
    return {
        "input_keys": sorted(str(key) for key in raw_keys),
        "has_messages": bool(value.get("has_messages", False)),
        "raw_text_length": _safe_int(value.get("raw_text_length")),
        "summary_only": True,
    }


def _safe_planner_provider_diagnostics(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if not value:
        return {}
    dotenv_loaded = value.get("dotenv_loaded", "skip")
    if dotenv_loaded not in {True, False, "skip"}:
        dotenv_loaded = "skip"
    return {
        "dotenv_loaded": dotenv_loaded,
        "dotenv_path_present": bool(value.get("dotenv_path_present", False)),
        "openai_api_key": "set" if value.get("openai_api_key") == "set" else "missing",
        "openai_api_base": "set" if value.get("openai_api_base") == "set" else "missing",
        "llm_model": str(value.get("llm_model") or ""),
        "planner_agent_class": str(value.get("planner_agent_class") or ""),
        "invocation_method": str(value.get("invocation_method") or ""),
        "provider_error_type": str(value.get("provider_error_type") or ""),
        "summary_only": True,
    }


def _task_status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _graph_failed_summary(error_type: str, *, stage: str, error_hint: str, state: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    if stage == "create_graph" and error_hint == "graph_invoke_error":
        error_hint = "llm_or_provider_error"
    summary = _failed_output_summary(error_type, stage=stage, error_hint=error_hint)
    summary["graph_input_keys"] = sorted(str(key) for key in state.keys())
    summary["graph_input_shape"] = _shape_summary(state)
    summary["graph_config_has_thread_id"] = _config_has_thread_id(config)
    summary["metadata"] = {
        "summary_only": True,
        "production_graph_invoked": stage not in {"create_graph", "build_initial_state"},
        "diagnostics_available": True,
    }
    return summary


def _safe_get_graph_state(graph, config):
    try:
        state_snapshot = graph.get_state(config)
    except Exception:
        return {}
    values = getattr(state_snapshot, "values", None)
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(state_snapshot, Mapping):
        return dict(state_snapshot)
    return {}


def _final_state_values(final_state) -> Dict[str, Any]:
    if isinstance(final_state, Mapping):
        return dict(final_state)
    return {}


def _safe_candidate_pool_summary(candidates: Any) -> list:
    if not isinstance(candidates, list):
        return []
    safe = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        safe.append(
            {
                "candidate_id": str(item.get("candidate_id") or metadata.get("candidate_id") or ""),
                "candidate_name": str(item.get("candidate_name") or item.get("name") or metadata.get("candidate_name") or metadata.get("name") or ""),
                "source_document_id": str(item.get("source_document_id") or item.get("document_id") or metadata.get("source_document_id") or metadata.get("document_id") or ""),
                "source_file_name": _basename(item.get("file_name") or item.get("source") or metadata.get("file_name") or metadata.get("source")),
                "metadata": {
                    "candidate_id": str(metadata.get("candidate_id") or ""),
                    "candidate_name": str(metadata.get("candidate_name") or metadata.get("name") or ""),
                    "document_id": str(metadata.get("document_id") or metadata.get("source_document_id") or ""),
                    "file_name": _basename(metadata.get("file_name") or metadata.get("source")),
                    "summary_only": True,
                },
                "summary_only": True,
            }
        )
    return safe


def _safe_final_report_summary(reports: Any) -> list:
    if not isinstance(reports, list):
        return []
    safe = []
    for item in reports:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        score = item.get("total_score")
        safe.append(
            {
                "candidate_id": str(item.get("candidate_id") or metadata.get("candidate_id") or ""),
                "candidate_name": str(item.get("candidate_name") or metadata.get("candidate_name") or ""),
                "total_score": score if isinstance(score, (int, float)) else 0,
                "summary_only": True,
            }
        )
    return safe


def _basename(value: Any) -> str:
    if value is None:
        return ""
    return Path(str(value).replace("\\", "/")).name


def _shape_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    shape = {}
    for key, item in value.items():
        if isinstance(item, list):
            shape[str(key)] = {"type": "list", "count": len(item)}
        elif isinstance(item, dict):
            shape[str(key)] = {"type": "dict", "keys": sorted(str(k) for k in item.keys())}
        elif isinstance(item, str):
            shape[str(key)] = {"type": "str", "length": len(item)}
        else:
            shape[str(key)] = {"type": type(item).__name__}
    return shape


def _config_has_thread_id(config: Mapping[str, Any]) -> bool:
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    return isinstance(configurable, Mapping) and bool(configurable.get("thread_id"))


def _classify_error_hint(exc: Exception) -> str:
    error_type = type(exc).__name__
    text = str(exc).lower()
    if error_type == "ValueError":
        if "parse" in text or "json" in text or "structured" in text:
            return "structured_output_parse_error"
        if "state" in text or "messages" in text or "input" in text:
            return "missing_or_invalid_initial_state"
        return "unknown_value_error"
    if any(
        marker in text
        for marker in (
            "openai",
            "api",
            "provider",
            "llm",
            "huggingface",
            "nodename",
            "servname",
            "network",
        )
    ):
        return "llm_or_provider_error"
    return "graph_invoke_error"


def load_jd_text(jd_text: Optional[str] = None, jd_file: Optional[str] = None) -> str:
    if jd_file:
        return Path(jd_file).read_text(encoding="utf-8")
    return jd_text or "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师"
