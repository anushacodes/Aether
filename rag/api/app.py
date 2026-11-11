"""Run local HTTP endpoints with explicit client lifetimes and dependency health."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from opensearchpy import OpenSearch, TransportError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from rag import __version__
from rag.config import PipelineConfig, load_config
from rag.search_store import create_search_client


def get_search_client(request: Request) -> OpenSearch:
    """Share the application's thread-safe search connection pool across requests."""
    return cast(OpenSearch, request.app.state.search)


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    """Create a configurable application without connecting to models during import."""
    config_path = Path("config.toml")
    settings = config or load_config(config_path if config_path.exists() else None)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        with create_search_client(settings) as client:
            application.state.search = client
            yield

    application = FastAPI(title="S3 Knowledge Engine", version=__version__, lifespan=lifespan)
    application.state.config = settings

    @application.middleware("http")
    async def request_identity(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.get("/health/live")
    def liveness() -> dict[str, str]:
        """Report process liveness without making a dependency or model request."""
        return {"status": "ok"}

    @application.get("/health/ready")
    def readiness(client: Annotated[OpenSearch, Depends(get_search_client)]) -> dict[str, object]:
        """Check local search and indexed content without invoking any paid model."""
        try:
            health = client.cluster.health(request_timeout=5)
            if health["status"] not in {"green", "yellow"}:
                raise HTTPException(503, detail={"code": "search_unavailable"})
            count = client.count(index=settings.index.name)["count"]
            if count == 0:
                raise HTTPException(503, detail={"code": "corpus_not_ready"})
            return {
                "status": "ready",
                "cluster_status": health["status"],
                "index": settings.index.name,
                "documents": count,
            }
        except TransportError as error:
            raise HTTPException(503, detail={"code": "search_unavailable"}) from error

    return application


app = create_app()
