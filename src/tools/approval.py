import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.runtime.models import utc_now


def new_approval_id() -> str:
    return f"tool_approval_{uuid.uuid4()}"


@dataclass
class ToolApprovalRequest:
    approval_id: str = field(default_factory=new_approval_id)
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    tool_name: str = ""
    tool_version: str = ""
    reason: str = ""
    requested_permissions: List[str] = field(default_factory=list)
    side_effects: str = "none"
    input_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "reason": self.reason,
            "requested_permissions": list(self.requested_permissions),
            "side_effects": self.side_effects,
            "input_summary": dict(self.input_summary),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


@dataclass
class ToolApprovalDecision:
    approval_id: str
    approved: bool
    decided_by: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approved": self.approved,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "decided_at": self.decided_at.isoformat(),
        }


class InMemoryToolApprovalStore:
    def __init__(self):
        self.requests: Dict[str, ToolApprovalRequest] = {}
        self.decisions: Dict[str, ToolApprovalDecision] = {}

    def create_request(self, request: ToolApprovalRequest) -> ToolApprovalRequest:
        if request.approval_id in self.requests:
            raise ValueError(f"Approval request already exists: {request.approval_id}")
        self.requests[request.approval_id] = request
        return request

    def get_request(self, approval_id: str) -> ToolApprovalRequest:
        if approval_id not in self.requests:
            raise KeyError(approval_id)
        return self.requests[approval_id]

    def record_decision(self, decision: ToolApprovalDecision) -> ToolApprovalDecision:
        request = self.get_request(decision.approval_id)
        if request.status != "pending":
            raise ValueError(f"Approval request is already resolved: {decision.approval_id}")
        request.status = "approved" if decision.approved else "rejected"
        self.decisions[decision.approval_id] = decision
        return decision

    def get_decision(self, approval_id: str) -> Optional[ToolApprovalDecision]:
        return self.decisions.get(approval_id)

    def list_requests_by_task(self, task_id: str) -> List[ToolApprovalRequest]:
        return [request for request in self.requests.values() if request.task_id == task_id]

    def list_pending_requests(self, task_id: Optional[str] = None) -> List[ToolApprovalRequest]:
        return [
            request
            for request in self.requests.values()
            if request.status == "pending" and (task_id is None or request.task_id == task_id)
        ]
