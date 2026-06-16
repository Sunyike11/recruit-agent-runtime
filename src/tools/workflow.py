from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.skills.execution import summarize_value
from src.tools.models import ToolExecutionContext, ToolResult


@dataclass
class ToolWorkflowStep:
    tool_name: str
    input_mapper: Callable[[Dict[str, Any]], Dict[str, Any]]
    output_key: str
    tool_version: Optional[str] = None
    continue_on_failure: bool = False


@dataclass
class ToolWorkflowResult:
    status: str
    success: bool
    outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "outputs": dict(self.outputs),
            "tool_results": list(self.tool_results),
            "steps": list(self.steps),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class LocalToolWorkflow:
    """Deterministic fake-tool composition harness.

    Internal workflow state carries prior tool output for later input mapping.
    Public results retain only output summaries and execution outcome metadata.
    """

    def __init__(
        self,
        registry: Any,
        executor: Any,
        context: Optional[ToolExecutionContext] = None,
        steps: Optional[Iterable[ToolWorkflowStep]] = None,
    ):
        self.registry = registry
        self.executor = executor
        self.context = context or ToolExecutionContext()
        self.steps = list(steps or [])

    def run(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        steps: Optional[Iterable[ToolWorkflowStep]] = None,
    ) -> ToolWorkflowResult:
        workflow_state = dict(initial_state or {})
        workflow_steps = list(steps) if steps is not None else list(self.steps)
        result = ToolWorkflowResult(
            status="running",
            success=False,
            metadata={"workflow": "local_fake_tool_workflow"},
        )
        had_failure = False

        for index, step in enumerate(workflow_steps):
            try:
                input_data = step.input_mapper(dict(workflow_state))
                if not isinstance(input_data, dict):
                    raise TypeError("input_mapper must return a dict")
            except Exception as exc:
                error = f"input mapping failed for {step.tool_name}: {exc}"
                self._record_mapping_failure(result, index, step, error)
                had_failure = True
                if not step.continue_on_failure:
                    return self._finish_failed(result, error, workflow_state)
                continue

            tool_result = self.executor.execute(
                step.tool_name,
                input_data,
                context=self.context,
                version=step.tool_version,
            )
            self._record_execution(result, index, step, input_data, tool_result)

            if tool_result.success:
                workflow_state[step.output_key] = tool_result.output
                result.outputs[step.output_key] = summarize_value(tool_result.output)
                continue

            had_failure = True
            if not result.error:
                result.error = tool_result.error
            if not step.continue_on_failure:
                return self._finish_failed(result, tool_result.error, workflow_state)

        result.status = "partial" if had_failure else "completed"
        result.success = not had_failure
        result.metadata.update(
            {
                "step_count": len(workflow_steps),
                "executed_step_count": len(result.steps),
                "state_keys": sorted(workflow_state.keys()),
            }
        )
        return result

    def _record_execution(
        self,
        workflow_result: ToolWorkflowResult,
        index: int,
        step: ToolWorkflowStep,
        input_data: Dict[str, Any],
        tool_result: ToolResult,
    ) -> None:
        tool_summary = {
            "tool_name": tool_result.tool_name,
            "version": tool_result.version,
            "success": tool_result.success,
            "output_summary": summarize_value(tool_result.output),
            "error": tool_result.error,
        }
        workflow_result.tool_results.append(tool_summary)
        workflow_result.steps.append(
            {
                "index": index,
                "tool_name": step.tool_name,
                "tool_version": step.tool_version,
                "output_key": step.output_key,
                "continue_on_failure": step.continue_on_failure,
                "input_summary": summarize_value(input_data),
                "success": tool_result.success,
                "error": tool_result.error,
            }
        )

    def _record_mapping_failure(
        self,
        workflow_result: ToolWorkflowResult,
        index: int,
        step: ToolWorkflowStep,
        error: str,
    ) -> None:
        workflow_result.tool_results.append(
            {
                "tool_name": step.tool_name,
                "version": step.tool_version or "",
                "success": False,
                "output_summary": summarize_value(None),
                "error": error,
            }
        )
        workflow_result.steps.append(
            {
                "index": index,
                "tool_name": step.tool_name,
                "tool_version": step.tool_version,
                "output_key": step.output_key,
                "continue_on_failure": step.continue_on_failure,
                "input_summary": summarize_value(None),
                "success": False,
                "error": error,
            }
        )

    def _finish_failed(
        self,
        result: ToolWorkflowResult,
        error: str,
        workflow_state: Dict[str, Any],
    ) -> ToolWorkflowResult:
        result.status = "failed"
        result.success = False
        result.error = error
        result.metadata.update(
            {
                "executed_step_count": len(result.steps),
                "state_keys": sorted(workflow_state.keys()),
            }
        )
        return result
