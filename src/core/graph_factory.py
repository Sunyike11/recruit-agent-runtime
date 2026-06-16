import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class RecruitGraphMode(str, Enum):
    LEGACY = "legacy"
    SKILL = "skill"


@dataclass
class GraphFallbackConfig:
    enabled: bool = True
    fallback_mode: RecruitGraphMode = RecruitGraphMode.LEGACY
    fallback_on_provider_error: bool = True
    fallback_on_retriever_error: bool = True
    fallback_on_schema_error: bool = True
    fallback_on_empty_candidates: bool = True
    fallback_on_empty_reports: bool = True
    fallback_on_quality_warning: bool = False
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fallback_mode"] = self.fallback_mode.value
        return data


@dataclass
class RecruitGraphFactoryConfig:
    mode: RecruitGraphMode = RecruitGraphMode.SKILL
    allow_planner_fallback: bool = False
    rollback_target: RecruitGraphMode = RecruitGraphMode.LEGACY
    summary_only: bool = True
    selection_reason: str = "default_skill"
    selection_source: str = "default"
    default_graph_mode: RecruitGraphMode = RecruitGraphMode.SKILL
    requested_graph_mode: str = ""
    legacy_alias_used: bool = False
    legacy_explicitly_requested: bool = False
    skill_default_used: bool = False
    config_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["rollback_target"] = self.rollback_target.value
        data["default_graph_mode"] = self.default_graph_mode.value
        return data


@dataclass
class RecruitGraphSelection:
    runner: Callable[..., Any]
    selected_mode: RecruitGraphMode
    runner_name: str
    rollback_target: RecruitGraphMode
    selection_reason: str
    selection_source: str = ""
    default_graph_mode: RecruitGraphMode = RecruitGraphMode.SKILL
    legacy_alias_used: bool = False
    legacy_explicitly_requested: bool = False
    skill_default_used: bool = False
    config_error: str = ""
    summary_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_mode": self.selected_mode.value,
            "selected_graph_mode": self.selected_mode.value,
            "runner_name": self.runner_name,
            "rollback_target": self.rollback_target.value,
            "default_graph_mode": self.default_graph_mode.value,
            "selection_reason": self.selection_reason,
            "selection_source": self.selection_source,
            "runner_selection_reason": self.selection_reason,
            "legacy_alias_used": bool(self.legacy_alias_used),
            "legacy_explicitly_requested": bool(self.legacy_explicitly_requested),
            "skill_default_used": bool(self.skill_default_used),
            "config_error": self.config_error,
            "summary_only": True,
        }


class RecruitGraphFactory:
    def __init__(
        self,
        *,
        legacy_runner: Callable[..., Any],
        skill_runner: Optional[Callable[..., Any]] = None,
        config: Optional[RecruitGraphFactoryConfig] = None,
    ):
        self.legacy_runner = legacy_runner
        self.skill_runner = skill_runner
        self.config = config or RecruitGraphFactoryConfig()

    def create_runner(self) -> RecruitGraphSelection:
        if self.config.config_error:
            return RecruitGraphSelection(
                runner=_config_error_runner(self.config.config_error),
                selected_mode=self.config.mode,
                runner_name=self._runner_name(self.config.mode),
                rollback_target=self.config.rollback_target,
                selection_reason="config_error",
                selection_source=self.config.selection_source,
                default_graph_mode=self.config.default_graph_mode,
                legacy_alias_used=self.config.legacy_alias_used,
                legacy_explicitly_requested=self.config.legacy_explicitly_requested,
                skill_default_used=self.config.skill_default_used,
                config_error=self.config.config_error,
            )
        if self.config.mode == RecruitGraphMode.SKILL:
            if self.skill_runner is None:
                return RecruitGraphSelection(
                    runner=self.legacy_runner,
                    selected_mode=RecruitGraphMode.LEGACY,
                    runner_name="default_graph",
                    rollback_target=self.config.rollback_target,
                    selection_reason="skill_runner_missing_legacy_fallback",
                    selection_source=self.config.selection_source,
                    default_graph_mode=self.config.default_graph_mode,
                    legacy_alias_used=self.config.legacy_alias_used,
                    legacy_explicitly_requested=self.config.legacy_explicitly_requested,
                    skill_default_used=self.config.skill_default_used,
                    config_error="",
                )
            return RecruitGraphSelection(
                runner=self.skill_runner,
                selected_mode=RecruitGraphMode.SKILL,
                runner_name="production_skill_graph",
                rollback_target=self.config.rollback_target,
                selection_reason=self.config.selection_reason or "graph_mode_skill",
                selection_source=self.config.selection_source,
                default_graph_mode=self.config.default_graph_mode,
                legacy_alias_used=self.config.legacy_alias_used,
                legacy_explicitly_requested=self.config.legacy_explicitly_requested,
                skill_default_used=self.config.skill_default_used,
            )
        return RecruitGraphSelection(
            runner=self.legacy_runner,
            selected_mode=RecruitGraphMode.LEGACY,
            runner_name="default_graph",
            rollback_target=self.config.rollback_target,
            selection_reason=self.config.selection_reason or "graph_mode_legacy",
            selection_source=self.config.selection_source,
            default_graph_mode=self.config.default_graph_mode,
            legacy_alias_used=self.config.legacy_alias_used,
            legacy_explicitly_requested=self.config.legacy_explicitly_requested,
            skill_default_used=self.config.skill_default_used,
        )

    def describe_selection(self) -> Dict[str, Any]:
        return self.create_runner().to_dict()

    @staticmethod
    def _runner_name(mode: RecruitGraphMode) -> str:
        return "production_skill_graph" if mode == RecruitGraphMode.SKILL else "default_graph"


def resolve_recruit_graph_factory_config(
    *,
    requested_graph_mode: Optional[str] = None,
    use_production_skill_graph_alias: bool = False,
    env: Optional[Dict[str, str]] = None,
    allow_planner_fallback: bool = False,
    summary_only: bool = True,
) -> RecruitGraphFactoryConfig:
    env_values = env if env is not None else os.environ
    explicit = (requested_graph_mode or "").strip().lower()
    env_mode = str(env_values.get("RECRUIT_GRAPH_MODE") or "").strip().lower()
    config_error = ""
    legacy_alias_used = bool(use_production_skill_graph_alias)

    if explicit and explicit not in {"legacy", "skill"}:
        config_error = "invalid_graph_mode"
    if env_mode and env_mode not in {"legacy", "skill"} and not explicit:
        config_error = "invalid_env_graph_mode"

    if explicit and use_production_skill_graph_alias and explicit != "skill":
        config_error = "conflicting_graph_mode_flags"

    selected = explicit or ("skill" if use_production_skill_graph_alias else "") or env_mode or "skill"
    if selected not in {"legacy", "skill"}:
        selected = "legacy"

    if explicit:
        reason = "explicit_cli_graph_mode"
        source = "cli"
    elif use_production_skill_graph_alias:
        reason = "deprecated_alias_use_production_skill_graph"
        source = "cli_alias"
    elif env_mode:
        reason = "environment_graph_mode"
        source = "environment"
    else:
        reason = "default_skill"
        source = "default"

    return RecruitGraphFactoryConfig(
        mode=RecruitGraphMode(selected),
        allow_planner_fallback=bool(allow_planner_fallback),
        rollback_target=RecruitGraphMode.LEGACY,
        summary_only=bool(summary_only),
        selection_reason=reason,
        selection_source=source,
        default_graph_mode=RecruitGraphMode.SKILL,
        requested_graph_mode=explicit,
        legacy_alias_used=legacy_alias_used,
        legacy_explicitly_requested=explicit == "legacy" or (not explicit and env_mode == "legacy"),
        skill_default_used=not explicit and not use_production_skill_graph_alias and not env_mode,
        config_error=config_error,
    )


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
