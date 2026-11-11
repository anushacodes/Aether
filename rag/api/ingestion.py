"""Authenticate local corpus administration before reading state or accepting durable work."""

import hmac
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opensearchpy import OpenSearch, TransportError

from rag.config import ConfigurationError, PipelineConfig
from rag.ingestion import ingestion_catalog, submit_ingestion
from rag.jobs import IngestionJob, IngestionRequest, JobConflict, JobStore

security = HTTPBearer(auto_error=False, scheme_name="LocalAdministrator")
JobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except JobConflict as error:
        raise HTTPException(409, detail={"code": str(error)}) from error
    except KeyError as error:
        raise HTTPException(404, detail={"code": "job_not_found"}) from error
    except ConfigurationError as error:
        raise HTTPException(503, detail={"code": "model_not_configured"}) from error
    except TransportError as error:
        raise HTTPException(503, detail={"code": "search_unavailable"}) from error
    except OSError as error:
        raise HTTPException(503, detail={"code": "source_files_unavailable"}) from error
    except ValueError as error:
        raise HTTPException(400, detail={"code": "invalid_ingestion_inputs"}) from error


def create_ingestion_router(
    config: PipelineConfig, search_dependency: Callable[[Request], OpenSearch]
) -> APIRouter:
    """Keep shared administrator credentials server-side and disable mutation when unset."""

    def require_admin(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> None:
        configured = config.ingestion.admin_token
        if configured is None or not configured.get_secret_value():
            raise HTTPException(503, detail={"code": "administration_disabled"})
        provided = credentials.credentials if credentials else ""
        if not hmac.compare_digest(
            provided.encode("utf-8"), configured.get_secret_value().encode("utf-8")
        ):
            raise HTTPException(
                401,
                detail={"code": "administrator_required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    def get_store(_: Annotated[None, Depends(require_admin)]) -> JobStore:
        return JobStore(config.publication.state_path)

    router = APIRouter(
        prefix="/ingestion",
        tags=["ingestion"],
        dependencies=[Depends(require_admin)],
    )

    @router.get("/catalog")
    def catalog(client: Annotated[OpenSearch, Depends(search_dependency)]) -> dict[str, object]:
        """Expose approved manifest identifiers and the base revision needed for a mutation."""
        with _errors():
            return ingestion_catalog(client, config)

    @router.post("", response_model=IngestionJob, status_code=202)
    def submit(
        body: IngestionRequest,
        key: Annotated[
            str, Header(alias="Idempotency-Key", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
        ],
        store: Annotated[JobStore, Depends(get_store)],
        client: Annotated[OpenSearch, Depends(search_dependency)],
    ) -> IngestionJob:
        with _errors():
            return submit_ingestion(client, config, store, key, body)

    @router.get("/{job_id}", response_model=IngestionJob)
    def status(job_id: JobId, store: Annotated[JobStore, Depends(get_store)]) -> IngestionJob:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, detail={"code": "job_not_found"})
        return job

    @router.post("/{job_id}/cancel", response_model=IngestionJob)
    def cancel(job_id: JobId, store: Annotated[JobStore, Depends(get_store)]) -> IngestionJob:
        with _errors():
            return store.cancel(job_id)

    @router.post("/{job_id}/replay", response_model=IngestionJob, status_code=202)
    def replay(job_id: JobId, store: Annotated[JobStore, Depends(get_store)]) -> IngestionJob:
        with _errors():
            return store.replay(job_id, config.ingestion.max_attempts)

    return router
