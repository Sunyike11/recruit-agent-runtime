import asyncio
import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.api.errors import IdempotencyConflict, QueueFull, TaskNotFound, TenantAccessDenied
from src.api.schemas import CreateMatchingTaskRequest
from src.runtime.entry import RuntimeEntryResult
from src.runtime.inspect import RuntimeInspector


TERMINAL_STATUSES = {"completed", "completed_with_fallback", "failed", "cancelled"}


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class APITaskRecord:
    task_id: str
    session_id: str
    tenant_id: str
    idempotency_key: str
    request_fingerprint: str
    status: str = "queued"
    graph_mode: str = "skill"
    candidate_source: str = "direct"
    task_type: str = "matching"
    created_at: str = field(default_factory=utc_text)
    started_at: str = ""
    completed_at: str = ""
    runtime_task_id: str = ""
    runtime_session_id: str = ""
    runtime_thread_id: str = ""
    result_summary: Dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    cancel_requested: bool = False
    queue_entered_at: float = field(default_factory=time.perf_counter)
    queue_started_at: float = 0.0
    completed_perf: float = 0.0


@dataclass
class IdempotencyEntry:
    task_id: str
    fingerprint: str


class AsyncTaskManager:
    def __init__(
        self,
        *,
        store: Any,
        runtime_submitter: Callable[[CreateMatchingTaskRequest, str], RuntimeEntryResult],
        ingestion_submitter: Optional[Callable[[Any, str], RuntimeEntryResult]] = None,
        worker_count: int = 2,
        queue_max_size: int = 20,
        task_timeout_seconds: float = 120.0,
    ):
        self.store = store
        self.runtime_submitter = runtime_submitter
        self.ingestion_submitter = ingestion_submitter
        self.worker_count = int(worker_count)
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=int(queue_max_size))
        self.task_timeout_seconds = float(task_timeout_seconds)
        self.tasks: Dict[str, APITaskRecord] = {}
        self.requests: Dict[str, CreateMatchingTaskRequest] = {}
        self.idempotency: Dict[tuple[str, str], IdempotencyEntry] = {}
        self.workers: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=max(1, self.worker_count))
        self.shutdown_requested = False
        self.metrics: Dict[str, Any] = {
            "request_count": 0,
            "request_error_count": 0,
            "task_created_count": 0,
            "task_success_count": 0,
            "task_failure_count": 0,
            "task_cancel_count": 0,
            "fallback_count": 0,
            "queue_wait_ms": [],
            "end_to_end_task_ms": [],
            "sse_first_event_ms": [],
            "mcp_tool_success_count": 0,
            "mcp_tool_failure_count": 0,
            "candidate_created_count": 0,
            "resume_upload_count": 0,
            "duplicate_upload_count": 0,
            "ingestion_success_count": 0,
            "ingestion_failure_count": 0,
            "ingestion_cancel_count": 0,
            "ingestion_end_to_end_ms": [],
            "parse_ms": [],
            "evidence_extract_ms": [],
            "index_ms": [],
            "active_version_switch_count": 0,
        }

    async def start(self) -> None:
        self.shutdown_requested = False
        self.workers = [asyncio.create_task(self._worker(index)) for index in range(self.worker_count)]

    async def stop(self) -> None:
        self.shutdown_requested = True
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def submit(self, tenant_id: str, idempotency_key: str, request: CreateMatchingTaskRequest) -> tuple[APITaskRecord, bool]:
        self.metrics["request_count"] += 1
        fingerprint = _fingerprint_request(request)
        idem_key = (tenant_id, idempotency_key)
        existing = self.idempotency.get(idem_key)
        if existing:
            if existing.fingerprint != fingerprint:
                self.metrics["request_error_count"] += 1
                raise IdempotencyConflict("Idempotency key already used for a different request")
            return self.tasks[existing.task_id], True
        if self.queue.full():
            self.metrics["request_error_count"] += 1
            raise QueueFull("Task queue is full")
        record = APITaskRecord(
            task_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            candidate_source=request.candidate_source,
        )
        self.tasks[record.task_id] = record
        self.requests[record.task_id] = request
        self.idempotency[idem_key] = IdempotencyEntry(record.task_id, fingerprint)
        self.metrics["task_created_count"] += 1
        await self.queue.put(record.task_id)
        return record, False

    async def submit_ingestion(self, tenant_id: str, idempotency_key: str, request: Any) -> tuple[APITaskRecord, bool]:
        self.metrics["request_count"] += 1
        fingerprint = _fingerprint_ingestion_request(request)
        idem_key = (tenant_id, idempotency_key)
        existing = self.idempotency.get(idem_key)
        if existing:
            if existing.fingerprint != fingerprint:
                self.metrics["request_error_count"] += 1
                raise IdempotencyConflict("Idempotency key already used for a different request")
            return self.tasks[existing.task_id], True
        if self.queue.full():
            self.metrics["request_error_count"] += 1
            raise QueueFull("Task queue is full")
        record = APITaskRecord(
            task_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            candidate_source="managed",
            task_type="candidate_ingestion",
        )
        self.tasks[record.task_id] = record
        self.requests[record.task_id] = request
        self.idempotency[idem_key] = IdempotencyEntry(record.task_id, fingerprint)
        self.metrics["task_created_count"] += 1
        await self.queue.put(record.task_id)
        return record, False

    def get_task(self, tenant_id: str, task_id: str) -> APITaskRecord:
        record = self.tasks.get(task_id)
        if record is None:
            raise TaskNotFound("Task not found")
        if record.tenant_id != tenant_id:
            raise TenantAccessDenied("Tenant cannot access this task")
        return record

    def cancel(self, tenant_id: str, task_id: str) -> APITaskRecord:
        record = self.get_task(tenant_id, task_id)
        if record.status in TERMINAL_STATUSES:
            return record
        record.cancel_requested = True
        if record.status == "queued":
            record.status = "cancelled"
            record.completed_at = utc_text()
            self.metrics["task_cancel_count"] += 1
        elif record.status == "running":
            record.status = "cancel_requested"
        return record

    def events_for_task(self, tenant_id: str, task_id: str, *, after_event_id: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        record = self.get_task(tenant_id, task_id)
        events: List[Dict[str, Any]] = [
            {
                "event_id": f"api:{record.task_id}:queued",
                "event_type": "task_queued",
                "task_id": record.task_id,
                "session_id": record.session_id,
                "status": "queued",
                "summary_only": True,
            }
        ]
        if record.runtime_task_id:
            events.extend(RuntimeInspector().inspect_events(record.runtime_task_id, self.store))
        if record.status in TERMINAL_STATUSES:
            events.append(
                {
                    "event_id": f"api:{record.task_id}:terminal",
                    "event_type": f"task_{record.status}",
                    "task_id": record.task_id,
                    "session_id": record.session_id,
                    "status": record.status,
                    "summary_only": True,
                }
            )
        if after_event_id:
            ids = [event["event_id"] for event in events]
            if after_event_id in ids:
                events = events[ids.index(after_event_id) + 1 :]
        return events[: max(1, min(int(limit), 200))]

    def add_feedback(self, tenant_id: str, task_id: str, feedback_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        record = self.get_task(tenant_id, task_id)
        target_task_id = record.runtime_task_id or record.task_id
        if hasattr(self.store, "add_human_feedback") and record.runtime_task_id:
            feedback = self.store.add_human_feedback(target_task_id, feedback_type, dict(payload))
        else:
            feedback = {
                "feedback_id": str(uuid.uuid4()),
                "task_id": target_task_id,
                "feedback_type": feedback_type,
                "payload": dict(payload),
                "created_at": utc_text(),
            }
        return feedback

    def metrics_summary(self) -> Dict[str, Any]:
        return {
            "request_count": int(self.metrics["request_count"]),
            "request_error_count": int(self.metrics["request_error_count"]),
            "task_created_count": int(self.metrics["task_created_count"]),
            "task_success_count": int(self.metrics["task_success_count"]),
            "task_failure_count": int(self.metrics["task_failure_count"]),
            "task_cancel_count": int(self.metrics["task_cancel_count"]),
            "queue_depth": self.queue.qsize(),
            "queue_wait_ms": _latency_summary(self.metrics["queue_wait_ms"]),
            "end_to_end_task_ms": _latency_summary(self.metrics["end_to_end_task_ms"]),
            "sse_first_event_ms": _latency_summary(self.metrics["sse_first_event_ms"]),
            "fallback_count": int(self.metrics["fallback_count"]),
            "mcp_tool_success_count": int(self.metrics["mcp_tool_success_count"]),
            "mcp_tool_failure_count": int(self.metrics["mcp_tool_failure_count"]),
            "candidate_created_count": int(self.metrics["candidate_created_count"]),
            "resume_upload_count": int(self.metrics["resume_upload_count"]),
            "duplicate_upload_count": int(self.metrics["duplicate_upload_count"]),
            "ingestion_success_count": int(self.metrics["ingestion_success_count"]),
            "ingestion_failure_count": int(self.metrics["ingestion_failure_count"]),
            "ingestion_cancel_count": int(self.metrics["ingestion_cancel_count"]),
            "ingestion_end_to_end_ms": _latency_summary(self.metrics["ingestion_end_to_end_ms"]),
            "parse_ms": _latency_summary(self.metrics["parse_ms"]),
            "evidence_extract_ms": _latency_summary(self.metrics["evidence_extract_ms"]),
            "index_ms": _latency_summary(self.metrics["index_ms"]),
            "active_version_switch_count": int(self.metrics["active_version_switch_count"]),
            "summary_only": True,
        }

    async def _worker(self, worker_index: int) -> None:
        while not self.shutdown_requested:
            task_id = await self.queue.get()
            try:
                await self._run_task(task_id, worker_index)
            finally:
                self.queue.task_done()

    async def _run_task(self, task_id: str, worker_index: int) -> None:
        record = self.tasks.get(task_id)
        request = self.requests.get(task_id)
        if record is None or request is None:
            return
        if record.cancel_requested or record.status == "cancelled":
            record.status = "cancelled"
            record.completed_at = utc_text()
            return
        record.status = "running"
        record.started_at = utc_text()
        record.queue_started_at = time.perf_counter()
        self.metrics["queue_wait_ms"].append((record.queue_started_at - record.queue_entered_at) * 1000)
        loop = asyncio.get_running_loop()
        try:
            submitter = self.ingestion_submitter if record.task_type == "candidate_ingestion" else self.runtime_submitter
            if submitter is None:
                raise RuntimeError("submitter_missing")
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, submitter, request, record.tenant_id),
                timeout=self.task_timeout_seconds,
            )
            record.runtime_task_id = result.task_id
            record.runtime_session_id = result.session_id
            record.runtime_thread_id = result.thread_id
            record.result_summary = dict(result.output_summary or {})
            final_status = result.status if result.status in {"completed_with_fallback", "failed"} else result.task_status
            if str(final_status) in {"completed", "ok"}:
                record.status = "completed"
                self.metrics["task_success_count"] += 1
                if record.task_type == "candidate_ingestion":
                    self.metrics["ingestion_success_count"] += 1
            elif str(final_status) == "completed_with_fallback":
                record.status = "completed_with_fallback"
                self.metrics["task_success_count"] += 1
            else:
                record.status = "failed"
                self.metrics["task_failure_count"] += 1
                if record.task_type == "candidate_ingestion":
                    self.metrics["ingestion_failure_count"] += 1
            if record.result_summary.get("fallback_succeeded"):
                self.metrics["fallback_count"] += 1
            self.metrics["mcp_tool_success_count"] += int(record.result_summary.get("tool_success_count") or 0)
            if record.task_type == "candidate_ingestion":
                self.metrics["resume_upload_count"] += 1
                self.metrics["parse_ms"].append(float(record.result_summary.get("parse_duration_ms") or 0))
                self.metrics["evidence_extract_ms"].append(float(record.result_summary.get("evidence_extract_duration_ms") or 0))
                self.metrics["index_ms"].append(float(record.result_summary.get("index_duration_ms") or 0))
                if record.result_summary.get("active_version_switched"):
                    self.metrics["active_version_switch_count"] += 1
        except asyncio.TimeoutError:
            record.status = "failed"
            record.error_type = "TaskTimeout"
            self.metrics["task_failure_count"] += 1
            if record.task_type == "candidate_ingestion":
                self.metrics["ingestion_failure_count"] += 1
        except Exception as exc:
            record.status = "failed"
            record.error_type = type(exc).__name__
            self.metrics["task_failure_count"] += 1
            if record.task_type == "candidate_ingestion":
                self.metrics["ingestion_failure_count"] += 1
        finally:
            record.completed_at = utc_text()
            record.completed_perf = time.perf_counter()
            self.metrics["end_to_end_task_ms"].append((record.completed_perf - record.queue_entered_at) * 1000)
            if record.task_type == "candidate_ingestion":
                self.metrics["ingestion_end_to_end_ms"].append((record.completed_perf - record.queue_entered_at) * 1000)


def _fingerprint_request(request: CreateMatchingTaskRequest) -> str:
    canonical = "|".join(
        [
            str(len(request.jd_text or "")),
            hashlib.sha256((request.jd_text or "").strip().encode("utf-8")).hexdigest(),
            request.candidate_source,
            "1" if request.allow_legacy_fallback else "0",
            ",".join(sorted(str(key) for key in request.metadata.keys())),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_ingestion_request(request: Any) -> str:
    canonical = "|".join(
        [
            str(getattr(request, "candidate_id", "")),
            str(getattr(request, "resume_version_id", "")),
            str(getattr(request, "content_hash", "")),
            str(getattr(request, "file_size", "")),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _latency_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "summary_only": True}
    ordered = sorted(float(value) for value in values)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "count": len(ordered),
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "summary_only": True,
    }
