from typing import Callable, Dict, Optional

from src.skills.context import SkillExecutionContext
from src.skills.models import SkillResult


class SkillNodeAdapter:
    """Adapt a registered skill into a graph-node-like callable."""

    def __init__(
        self,
        registry,
        skill_name: str,
        input_mapper: Callable[[dict], dict],
        output_mapper: Callable[[SkillResult, dict], dict],
        skill_version: Optional[str] = None,
        context_builder: Optional[Callable[[dict], SkillExecutionContext]] = None,
        skill_executor=None,
    ):
        self.registry = registry
        self.skill_name = skill_name
        self.skill_version = skill_version
        self.input_mapper = input_mapper
        self.output_mapper = output_mapper
        self.context_builder = context_builder
        self.skill_executor = skill_executor

    def __call__(self, state: dict) -> dict:
        input_data = self.input_mapper(state)
        context = self._build_context(state)
        if self.skill_executor is not None:
            result = self.skill_executor.execute(
                self.skill_name,
                input_data,
                context=context,
                version=self.skill_version,
            )
        else:
            skill = self.registry.get(self.skill_name, version=self.skill_version)
            result = skill.execute(input_data, context=context)
        state_update = self.output_mapper(result, state) or {}

        if not result.success and "skill_error" not in state_update:
            state_update["skill_error"] = result.error

        state_update["skill_execution_metadata"] = self._execution_metadata(
            result=result,
            input_data=input_data,
            state_update=state_update,
        )
        return state_update

    def _build_context(self, state: dict) -> SkillExecutionContext:
        if self.context_builder is not None:
            return self.context_builder(state)
        return SkillExecutionContext(
            task_id=state.get("task_id"),
            session_id=state.get("session_id"),
            thread_id=state.get("thread_id"),
            memory_context=state.get("memory_context"),
            runtime_context=state.get("runtime_context"),
            metadata=state.get("metadata", {}),
        )

    def _execution_metadata(self, result: SkillResult, input_data: dict, state_update: dict) -> Dict:
        output_keys = [
            key
            for key in state_update.keys()
            if key not in {"skill_execution_metadata"}
        ]
        return {
            "skill_name": result.skill_name,
            "version": result.version,
            "success": result.success,
            "error": result.error,
            "input_keys": sorted(input_data.keys()),
            "output_keys": sorted(output_keys),
        }
