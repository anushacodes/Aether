"""Run local HTTP endpoints with explicit client lifetimes and dependency health."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager, closing, contextmanager
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import Depends, FastAPI, HTTPException, Request
from opensearchpy import OpenSearch, TransportError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from rag import __version__
from rag.aws_clients import create_bedrock_client
from rag.config import ConfigurationError, PipelineConfig, load_config
from rag.embeddings import Embedder
from rag.models import AnswerResult, RetrievalResult, SearchRequest
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
    except ConfigurationError as error:
        raise HTTPException(503, detail={"code": "model_not_configured"}) from error
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            raise HTTPException(429, detail={"code": "model_throttled"}) from error
        raise HTTPException(502, detail={"code": "model_request_failed"}) from error
    except BotoCoreError as error:
        raise HTTPException(503, detail={"code": "model_unavailable"}) from error
    except TransportError as error:
        raise HTTPException(503, detail={"code": "search_unavailable"}) from error
    except ValueError as error:
        raise HTTPException(502, detail={"code": "invalid_dependency_response"}) from error


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

    @application.post("/retrieval", response_model=RetrievalResult)
    def retrieve_passages(
        body: SearchRequest, client: Annotated[OpenSearch, Depends(get_search_client)]
    ) -> RetrievalResult:
        """Run blocking search/model SDK calls in FastAPI's worker thread pool."""
        with _dependency_errors(), ExitStack() as stack:
            embedder = None
            if body.mode != "bm25":
                runtime = stack.enter_context(closing(create_bedrock_client(settings)))
                embedder = Embedder(settings, runtime)
            return retrieve(client, settings, body.query, body.scope, body.mode, embedder)

    @application.post("/search", response_model=AnswerResult)
    def search_answer(
        body: SearchRequest, client: Annotated[OpenSearch, Depends(get_search_client)]
    ) -> AnswerResult:
        """Use the same original-context, generation, and citation path as the CLI."""
        with _dependency_errors():
            return answer_question(client, settings, body)

    return application


app = create_app()
