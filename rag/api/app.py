"""Run local HTTP endpoints with explicit client lifetimes and dependency health."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager, closing, contextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from opensearchpy import OpenSearch, TransportError
from starlette.responses import Response, StreamingResponse

from rag import __version__
from rag.api.comparison import comparison_response
from rag.api.errors import dependency_failure
from rag.api.ingestion import create_ingestion_router
from rag.api.limits import RequestLimitError, SearchLimiter
from rag.api.observability import RequestLogMiddleware, configure_event_logging, observation
from rag.api.stream import streaming_response
from rag.aws_clients import create_bedrock_client
from rag.config import PipelineConfig, load_config
from rag.embeddings import Embedder
from rag.indexing import resolve_read_config, verify_index
from rag.models import (
    AnswerResult,
    ComparisonRequest,
    ComparisonResult,
    RetrievalResult,
    SearchRequest,
)
from rag.pipeline import answer_question, retrieve
from rag.search_store import create_search_client


def get_search_client(request: Request) -> OpenSearch:
    """Share the application's thread-safe search connection pool across requests."""
    return cast(OpenSearch, request.app.state.search)


@contextmanager
def _dependency_errors() -> Iterator[None]:
    """Classify expected integration failures without sending provider details to the browser."""
    try:
        yield
    except Exception as error:
        failure = dependency_failure(error)
        if failure is None:
            raise
        status, code = failure
        headers = {"Retry-After": "60"} if isinstance(error, RequestLimitError) else None
        raise HTTPException(status, detail={"code": code}, headers=headers) from error


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    """Create a configurable application without connecting to models during import."""
    config_path = Path("config.toml")
    settings = config or load_config(config_path if config_path.exists() else None)
    configure_event_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        with create_search_client(settings) as client:
            application.state.search = client
            yield

    application = FastAPI(title="S3 Knowledge Engine", version=__version__, lifespan=lifespan)
    application.state.config = settings
    application.include_router(create_ingestion_router(settings, get_search_client))
    limiter = SearchLimiter(settings.api)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
        expose_headers=["X-Request-ID"],
    )

    application.add_middleware(RequestLogMiddleware, config=settings)

    @application.exception_handler(HTTPException)
    async def classified_error(request: Request, error: HTTPException) -> Response:
        """Retain the already-public application error code without logging any detail payload."""
        current = observation(request)
        code = error.detail.get("code") if isinstance(error.detail, dict) else None
        if current is not None and isinstance(code, str) and code.isidentifier():
            current.error_code = code
        return await http_exception_handler(request, error)

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
            pinned = resolve_read_config(client, settings)
            verify_index(client, pinned)
            count = client.count(index=pinned.index.name)["count"]
            if count == 0 and not client.indices.exists_alias(name=settings.index.name):
                raise HTTPException(503, detail={"code": "corpus_not_ready"})
            return {
                "status": "ready",
                "cluster_status": health["status"],
                "index": settings.index.name,
                "documents": count,
            }
        except TransportError as error:
            raise HTTPException(503, detail={"code": "search_unavailable"}) from error
        except ValueError as error:
            raise HTTPException(503, detail={"code": "invalid_index_configuration"}) from error

    @application.post("/retrieval", response_model=RetrievalResult)
    def retrieve_passages(
        body: SearchRequest,
        request: Request,
        client: Annotated[OpenSearch, Depends(get_search_client)],
    ) -> RetrievalResult:
        """Run blocking search/model SDK calls in FastAPI's worker thread pool."""
        current = observation(request)
        if current is not None:
            current.describe(body)
        if body.workflow:
            raise HTTPException(400, detail={"code": "workflow_requires_answer_endpoint"})
        with _dependency_errors(), limiter.admit(), ExitStack() as stack:
            embedder = None
            if body.mode != "bm25":
                runtime = stack.enter_context(closing(create_bedrock_client(settings)))
                embedder = Embedder(settings, runtime)
            result = retrieve(client, settings, body.query, body.scope, body.mode, embedder)
            if current is not None:
                current.record(result)
            return result

    @application.post("/search", response_model=AnswerResult)
    def search_answer(
        body: SearchRequest,
        request: Request,
        client: Annotated[OpenSearch, Depends(get_search_client)],
    ) -> AnswerResult:
        """Use the same original-context, generation, and citation path as the CLI."""
        current = observation(request)
        if current is not None:
            current.describe(body)
        with _dependency_errors(), limiter.admit():
            result = answer_question(client, settings, body)
            if current is not None:
                current.record(result)
            return result

    @application.post("/retrieval/compare", response_model=ComparisonResult)
    async def compare_passages(
        body: ComparisonRequest,
        request: Request,
        client: Annotated[OpenSearch, Depends(get_search_client)],
    ) -> ComparisonResult:
        """Compare two live source searches with one admission and a shared concrete snapshot."""
        current = observation(request)
        if current is not None:
            current.describe(body)
        with _dependency_errors():
            return await comparison_response(body, request, client, settings, limiter)

    @application.post("/search/stream")
    async def streamed_answer(
        body: SearchRequest,
        request: Request,
        client: Annotated[OpenSearch, Depends(get_search_client)],
    ) -> StreamingResponse:
        """Reserve bounded capacity before returning a POST-capable SSE response."""
        current = observation(request)
        if current is not None:
            current.describe(body)
        with _dependency_errors():
            lease = limiter.acquire()
        try:
            return streaming_response(body, request, client, settings, lease)
        except Exception:
            lease.close()
            raise

    return application


app = create_app()
