from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    build_candidate_store,
    build_ingestion_submitter,
    build_resume_blob_store,
    build_review_memory_store,
    build_runtime_store,
    build_runtime_submitter,
)
from src.api.errors import APIError, ServiceNotReady, api_error_handler, http_error_handler
from src.api.routes.tasks import router as tasks_router
from src.api.routes.candidates import router as candidates_router
from src.api.routes.reviews import router as reviews_router
from src.api.schemas import HealthResponse
from src.api.task_manager import AsyncTaskManager
from src.core.graph_factory import resolve_recruit_graph_factory_config


def create_app(
    *,
    store: Any = None,
    runtime_submitter: Optional[Callable[..., Any]] = None,
    db_path: str | None = None,
    worker_count: int = 2,
    queue_max_size: int = 20,
    task_timeout_seconds: float = 120.0,
    max_body_bytes: int = 128_000,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime_store = store or build_runtime_store(db_path)
        candidate_store = build_candidate_store(db_path)
        review_memory_store = build_review_memory_store(db_path)
        blob_store = build_resume_blob_store()
        submitter = runtime_submitter or build_runtime_submitter(runtime_store, review_memory_store)
        ingestion_submitter = build_ingestion_submitter(
            runtime_store=runtime_store,
            candidate_store=candidate_store,
            blob_store=blob_store,
        )
        manager = AsyncTaskManager(
            store=runtime_store,
            runtime_submitter=submitter,
            ingestion_submitter=ingestion_submitter,
            worker_count=worker_count,
            queue_max_size=queue_max_size,
            task_timeout_seconds=task_timeout_seconds,
        )
        app.state.runtime_store = runtime_store
        app.state.candidate_store = candidate_store
        app.state.review_memory_store = review_memory_store
        app.state.resume_blob_store = blob_store
        app.state.task_manager = manager
        app.state.ready = True
        await manager.start()
        try:
            yield
        finally:
            app.state.ready = False
            await manager.stop()

    app = FastAPI(
        title="Recruit-Graph Runtime Service",
        version="16.0.0-mvp",
        lifespan=lifespan,
    )
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-Tenant-ID", "Idempotency-Key", "Last-Event-ID", "Content-Type"],
    )

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and int(length) > max_body_bytes:
            raise HTTPException(status_code=413, detail="Request body too large")
        return await call_next(request)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return HealthResponse(status="ok")

    @app.get("/readyz")
    async def readyz(request: Request):
        if not getattr(request.app.state, "ready", False):
            raise ServiceNotReady("Service is not ready")
        graph_config = resolve_recruit_graph_factory_config()
        return {
            "status": "ready",
            "runtime_store_available": bool(getattr(request.app.state, "runtime_store", None)),
            "graph_factory_default_mode": graph_config.mode.value,
            "candidate_source_values": ["direct", "mcp"],
            "candidate_ingestion_available": True,
            "summary_only": True,
        }

    @app.get("/metrics/summary")
    async def metrics_summary(request: Request):
        data = request.app.state.task_manager.metrics_summary()
        review_store = getattr(request.app.state, "review_memory_store", None)
        if review_store is not None:
            data.update(review_store.metrics_summary())
        return data

    app.include_router(tasks_router)
    app.include_router(candidates_router)
    app.include_router(reviews_router)
    return app


app = create_app()
