"""Connect to OpenSearch and verify the local storage boundary.

The setup check uses a uniquely named, temporary text index. Corpus indexing
and retrieval will use their own mappings in the next implementation stages.
"""

from __future__ import annotations

from uuid import uuid4

from opensearchpy import OpenSearch

from rag.config import PipelineConfig

REQUEST_TIMEOUT_SECONDS = 10
HEALTH_TIMEOUT = "5s"
CHECK_DOCUMENT_ID = "setup-check"
CHECK_TEXT = "S3 versioning preserves multiple versions of an object."


def create_search_client(config: PipelineConfig) -> OpenSearch:
    """Create a bounded client from resolved settings without making a request."""
    return OpenSearch(
        hosts=[str(config.opensearch_url)],
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        retry_on_timeout=False,
        verify_certs=True,
    )


def check_search_store(client: OpenSearch) -> dict[str, str]:
    """Verify health and keyword search, removing the temporary index on failure too."""
    # This client treats `timeout` as an HTTP timeout unless request_timeout is explicit.
    health = client.cluster.health(
        wait_for_status="yellow", timeout=HEALTH_TIMEOUT, request_timeout=REQUEST_TIMEOUT_SECONDS
    )
    if health.get("timed_out") or health["status"] not in {"green", "yellow"}:
        raise ValueError(f"OpenSearch is not ready: cluster status {health['status']}")

    index_name = f"s3-rag-check-{uuid4().hex}"
    created = False
    try:
        client.indices.create(
            index=index_name,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {"properties": {"text": {"type": "text"}}},
            },
        )
        created = True
        client.index(
            index=index_name,
            id=CHECK_DOCUMENT_ID,
            body={"text": CHECK_TEXT},
            refresh="wait_for",
        )
        response = client.search(
            index=index_name, body={"query": {"match": {"text": "versioning"}}, "size": 1}
        )
        hits = response["hits"]["hits"]
        if (
            not hits
            or hits[0]["_id"] != CHECK_DOCUMENT_ID
            or hits[0]["_source"]["text"] != CHECK_TEXT
        ):
            raise ValueError("OpenSearch setup query did not return the expected document")
    finally:
        if created:
            client.indices.delete(index=index_name)

    return {"cluster_status": health["status"], "keyword_search": "passed", "cleanup": "passed"}
