import asyncio
import json
import time
from typing import AsyncIterator

from src.api.task_manager import TERMINAL_STATUSES, AsyncTaskManager


def format_sse(event: dict) -> str:
    event_id = str(event.get("event_id") or "")
    event_type = str(event.get("event_type") or "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"


async def task_event_stream(
    manager: AsyncTaskManager,
    *,
    tenant_id: str,
    task_id: str,
    last_event_id: str = "",
    poll_interval_seconds: float = 0.2,
    keepalive_seconds: float = 2.0,
) -> AsyncIterator[str]:
    cursor = last_event_id
    first = True
    last_emit = time.perf_counter()
    while True:
        events = manager.events_for_task(tenant_id, task_id, after_event_id=cursor, limit=100)
        if events:
            for event in events:
                cursor = str(event.get("event_id") or cursor)
                if first:
                    manager.metrics["sse_first_event_ms"].append(0.0)
                    first = False
                last_emit = time.perf_counter()
                yield format_sse(event)
        record = manager.get_task(tenant_id, task_id)
        if record.status in TERMINAL_STATUSES and cursor.endswith(":terminal"):
            break
        if time.perf_counter() - last_emit >= keepalive_seconds:
            last_emit = time.perf_counter()
            yield ": keepalive\n\n"
        await asyncio.sleep(poll_interval_seconds)
