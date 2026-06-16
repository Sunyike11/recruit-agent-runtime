import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.runtime.models import utc_now
from src.skills.execution import summarize_value
from src.tools.approval import InMemoryToolApprovalStore, ToolApprovalRequest
from src.tools.models import ToolExecutionContext, ToolResult
from src.tools.policy import ToolPermissionDecision, ToolPermissionPolicy
from src.tools.sandbox import SandboxDecision, SandboxPolicy, ToolSandboxContext


def new_tool_execution_id() -> str:
    return f"tool_execution_{uuid.uuid4()}"


@dataclass
class ToolExecutionRecord:
    execution_id: str = field(default_factory=new_tool_execution_id)
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    tool_name: str = ""
    tool_version: str = ""
    status: str = "started"
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Optional[Dict[str, Any]] = None
    error: str = ""
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    caller_type: str = ""
    caller_name: str = ""
    permissions_required: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    permission_decision: Optional[Dict[str, Any]] = None
    sandbox_decision: Optional[Dict[str, Any]] = None
    approval_id: Optional[str] = None
    approval_status: str = ""
    approval_decision: Optional[Dict[str, Any]] = None

    def to_event_payload(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "version": self.tool_version,
            "status": self.status,
            "success": self.status == "completed",
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "caller_type": self.caller_type,
            "caller_name": self.caller_name,
            "permissions_required": self.permissions_required,
            "permission_decision": self.permission_decision,
            "sandbox_decision": self.sandbox_decision,
            "approval_id": self.approval_id,
            "approval_status": self.approval_status,
            "approval_decision": self.approval_decision,
            "metadata": self.metadata,
        }


class ToolExecutionRecorder:
    def __init__(self, runtime_store):
        self.runtime_store = runtime_store

    def tool_started(self, record: ToolExecutionRecord):
        self._append_tool_event("tool_started", record)

    def tool_completed(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "completed"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_completed", record)

    def tool_failed(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "failed"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_failed", record)

    def tool_denied(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "denied"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_denied", record)

    def tool_approval_required(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "requires_approval"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_approval_required", record)

    def tool_approval_granted(self, record: ToolExecutionRecord):
        record.status = "approval_granted"
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_approval_granted", record)

    def tool_approval_rejected(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "approval_rejected"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_approval_rejected", record)

    def tool_sandbox_denied(self, record: ToolExecutionRecord, result: ToolResult):
        record.status = "sandbox_denied"
        record.output_summary = summarize_value(result.output)
        record.error = result.error
        record.completed_at = utc_now()
        record.duration_ms = _duration_ms(record.started_at, record.completed_at)
        self._append_tool_event("tool_sandbox_denied", record)

    def _append_tool_event(self, event_type: str, record: ToolExecutionRecord):
        if not record.task_id:
            return None
        return self.runtime_store.append_event(
            event_type,
            session_id=record.session_id,
            task_id=record.task_id,
            payload=record.to_event_payload(),
        )


class ToolExecutor:
    def __init__(
        self,
        registry,
        recorder: Optional[ToolExecutionRecorder] = None,
        permission_policy: Optional[ToolPermissionPolicy] = None,
        sandbox_policy: Optional[SandboxPolicy] = None,
        sandbox_context: Optional[ToolSandboxContext] = None,
        approval_store: Optional[InMemoryToolApprovalStore] = None,
    ):
        self.registry = registry
        self.recorder = recorder
        self.permission_policy = permission_policy
        self.sandbox_policy = sandbox_policy
        self.sandbox_context = sandbox_context
        self.approval_store = approval_store

    def execute(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
        version: Optional[str] = None,
        approval_id: Optional[str] = None,
    ) -> ToolResult:
        tool = self.registry.get(tool_name, version=version)
        execution_context = context or ToolExecutionContext()
        record = ToolExecutionRecord(
            task_id=execution_context.task_id,
            session_id=execution_context.session_id,
            thread_id=execution_context.thread_id,
            tool_name=tool.spec.name,
            tool_version=tool.spec.version,
            status="started",
            input_summary=summarize_value(input_data),
            caller_type=execution_context.caller_type,
            caller_name=execution_context.caller_name,
            permissions_required=list(tool.spec.permissions_required),
            metadata=dict(execution_context.metadata),
        )

        permission_decision = self._evaluate_permission(tool.spec, execution_context)
        if permission_decision is not None:
            record.permission_decision = permission_decision.to_dict()
            if permission_decision.status == "denied":
                result = self._permission_result(tool, permission_decision)
                if self.recorder is not None:
                    self.recorder.tool_denied(record, result)
                return result
            if permission_decision.status == "requires_approval":
                approval_outcome = self._resolve_approval(
                    tool,
                    execution_context,
                    input_data,
                    permission_decision,
                    record,
                    approval_id,
                )
                if approval_outcome is not None:
                    return approval_outcome

        sandbox_decision = self._evaluate_sandbox(tool.spec, execution_context)
        if sandbox_decision is not None:
            record.sandbox_decision = sandbox_decision.to_dict()
            if sandbox_decision.status == "denied":
                result = self._sandbox_result(tool, sandbox_decision)
                if self.recorder is not None:
                    self.recorder.tool_sandbox_denied(record, result)
                return result

        if self.recorder is not None:
            record.status = "started"
            self.recorder.tool_started(record)

        started = time.perf_counter()
        result = tool.execute(input_data, context=execution_context)
        if record.approval_id:
            result.metadata.update(
                {
                    "approval_id": record.approval_id,
                    "approval_status": record.approval_status,
                    "approval_decision": record.approval_decision,
                }
            )
        record.duration_ms = round((time.perf_counter() - started) * 1000, 3)

        if self.recorder is not None:
            if result.success:
                self.recorder.tool_completed(record, result)
            else:
                self.recorder.tool_failed(record, result)

        return result

    def _evaluate_permission(self, tool_spec, execution_context) -> Optional[ToolPermissionDecision]:
        if self.permission_policy is None:
            return None
        return self.permission_policy.evaluate(tool_spec, execution_context)

    def _evaluate_sandbox(self, tool_spec, execution_context) -> Optional[SandboxDecision]:
        if self.sandbox_policy is None:
            return None
        return self.sandbox_policy.evaluate(tool_spec, execution_context, self.sandbox_context)

    def _permission_result(self, tool, decision: ToolPermissionDecision) -> ToolResult:
        error = "approval required" if decision.status == "requires_approval" else decision.reason
        return ToolResult(
            tool_name=tool.spec.name,
            version=tool.spec.version,
            success=False,
            output=None,
            error=error,
            metadata={"permission_decision": decision.to_dict()},
        )

    def _sandbox_result(self, tool, decision: SandboxDecision) -> ToolResult:
        return ToolResult(
            tool_name=tool.spec.name,
            version=tool.spec.version,
            success=False,
            output=None,
            error=decision.reason,
            metadata={"sandbox_decision": decision.to_dict()},
        )

    def _resolve_approval(
        self,
        tool,
        execution_context: ToolExecutionContext,
        input_data: Dict[str, Any],
        permission_decision: ToolPermissionDecision,
        record: ToolExecutionRecord,
        approval_id: Optional[str],
    ) -> Optional[ToolResult]:
        if approval_id and self.approval_store is not None:
            request = self.approval_store.get_request(approval_id)
            self._validate_approval_request(request, tool, execution_context)
            decision = self.approval_store.get_decision(approval_id)
            record.approval_id = approval_id
            record.approval_status = request.status
            record.approval_decision = decision.to_dict() if decision is not None else None
            if decision is not None and decision.approved:
                if self.recorder is not None:
                    self.recorder.tool_approval_granted(record)
                return None
            if decision is not None and not decision.approved:
                result = ToolResult(
                    tool_name=tool.spec.name,
                    version=tool.spec.version,
                    success=False,
                    output=None,
                    error="approval rejected",
                    metadata={
                        "permission_decision": permission_decision.to_dict(),
                        "approval_id": approval_id,
                        "approval_decision": decision.to_dict(),
                    },
                )
                if self.recorder is not None:
                    self.recorder.tool_approval_rejected(record, result)
                return result
            result = self._permission_result(tool, permission_decision)
            result.metadata.update({"approval_id": approval_id, "approval_request": request.to_dict()})
            if self.recorder is not None:
                self.recorder.tool_approval_required(record, result)
            return result

        request = None
        if self.approval_store is not None:
            request = ToolApprovalRequest(
                task_id=execution_context.task_id,
                session_id=execution_context.session_id,
                thread_id=execution_context.thread_id,
                tool_name=tool.spec.name,
                tool_version=tool.spec.version,
                reason=permission_decision.reason,
                requested_permissions=list(permission_decision.required_permissions),
                side_effects=tool.spec.side_effects,
                input_summary=summarize_value(input_data),
                metadata={"approval_permission": permission_decision.metadata.get("approval_permission", "")},
            )
            self.approval_store.create_request(request)
            record.approval_id = request.approval_id
            record.approval_status = request.status

        result = self._permission_result(tool, permission_decision)
        if request is not None:
            result.metadata.update(
                {
                    "approval_id": request.approval_id,
                    "approval_request": request.to_dict(),
                }
            )
        if self.recorder is not None:
            self.recorder.tool_approval_required(record, result)
        return result

    def _validate_approval_request(self, request: ToolApprovalRequest, tool, context: ToolExecutionContext) -> None:
        if request.tool_name != tool.spec.name or request.tool_version != tool.spec.version:
            raise ValueError("Approval request does not match the requested tool")
        if request.task_id is not None and context.task_id != request.task_id:
            raise ValueError("Approval request does not match the execution task")


def _duration_ms(started_at: datetime, completed_at: datetime) -> float:
    return round((completed_at - started_at).total_seconds() * 1000, 3)
