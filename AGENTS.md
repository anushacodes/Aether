# Project guidance

- AWS S3 documentation RAG: Python/uv backend in `rag/`, Next.js frontend in `frontend/`.
- Keep implementation minimal. Detailed plans and decisions belong in `docs/`, not this file.
- Record material architecture choices, reasons, alternatives, and tradeoffs in [architecture decisions](docs/architecture_decisions.md); omit routine implementation details.
- Follow the [engineering rulebook](<docs/Python Engineering Rulebook — Agent Coding Standards.md>) for all coding work.
- Follow `docs/aws_s3_rag_atomic_build_checklist.md`. Validate backend changes with `make check`; frontend changes with `npm --prefix frontend run build`.
- Never commit or push without an explicit request for that specific action. A commit request does not authorize a push.
- “Revert commits” means reset local history, preserving files unless asked to discard them; do not create a revert commit or force-push.
