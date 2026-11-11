FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY rag ./rag
# Source-only edits must rebuild the project wheel even when uv's metadata cache is unchanged.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --reinstall-package aws-s3-knowledge-engine

FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AWS_EC2_METADATA_DISABLED=true \
    LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false
RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --no-create-home rag \
    && mkdir -p /app/data \
    && ln -s /app/manifests /app/data/manifests \
    && chown rag:rag /app/data
COPY --from=builder /app/.venv /app/.venv
COPY config.toml ./
# Keep the approved catalog in the image; the data volume must not shadow later catalog updates.
COPY data/manifests ./manifests
USER rag
EXPOSE 8000
CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
