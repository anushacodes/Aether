"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { WorkspaceHeader } from "../workspace-header";
import {
  AdminError,
  changeJob,
  getCatalog,
  getJobs,
  submitJob,
  type Catalog,
  type Job,
  type Mutation,
} from "./admin-api";

const STATUS: Record<Job["status"], string> = {
  queued: "Queued",
  running: "Running",
  ready: "Published",
  failed: "Failed",
  cancelled: "Cancelled",
};
const ERRORS: Record<string, string> = {
  administrator_required:
    "The administrator token was rejected. Enter the token configured for this API.",
  administration_disabled:
    "Administration is disabled. Configure RAG_ADMIN_TOKEN in the API environment and restart the API.",
  administration_connection_failed:
    "Couldn’t reach the administration API. Check the local API, then try again.",
  stale_base_revision:
    "The active corpus changed. Refresh the corpus and review the mutation against its new revision.",
  ingestion_queue_full:
    "The ingestion queue is full. Let the worker finish pending jobs before submitting more.",
  idempotency_key_conflict:
    "This submission key belongs to different accepted work. Refresh jobs to inspect the original request.",
  attempt_limit_reached:
    "This job reached its replay limit. Review its failures before creating a new mutation.",
  publication_started:
    "Publication has started and can no longer be cancelled. Refresh to see its outcome.",
  job_already_terminal:
    "This job has already finished. Refresh its durable status.",
  only_failed_jobs_can_be_replayed:
    "Only failed jobs can be replayed. Refresh its durable status.",
  model_not_configured:
    "Embedding configuration is missing. Use a text-only rebuild or configure the API before requesting embeddings.",
  search_unavailable:
    "OpenSearch is unavailable. Start the local search service and try again.",
  source_files_unavailable:
    "An approved manifest or source artifact is unavailable on the server. Check the local files before retrying.",
  job_state_unavailable:
    "The durable job store is unavailable. Check the API state storage before retrying.",
  invalid_ingestion_inputs:
    "The requested manifest or section selection is no longer valid. Refresh and choose approved inputs.",
  legacy_snapshot_requires_rebuild:
    "Rebuild from an approved manifest before deleting sections from this older snapshot.",
  deletion_must_preserve_vector_coverage:
    "Vector coverage changed. Refresh the corpus before reviewing this deletion again.",
  invalid_response:
    "The API returned an unexpected administration response. Refresh jobs before submitting more work.",
};
const FAILURES: Record<string, string> = {
  stale_base_revision:
    "The corpus changed before execution. Refresh and submit a new mutation against the current revision; replay cannot rebase this job.",
  search_request_failed:
    "The search service rejected the build or publication. Check OpenSearch before replaying.",
  model_request_failed:
    "The embedding model request failed. Check model access and configuration before replaying.",
  model_unavailable:
    "The embedding service could not be reached. Check connectivity and configuration before replaying.",
  local_files_unavailable:
    "The worker could not read local source artifacts. Restore the pinned files before replaying.",
  invalid_build_inputs:
    "Accepted build inputs failed validation. Review the approved manifest and pinned sources; replay keeps the same accepted inputs.",
  worker_interrupted:
    "The worker stopped during this attempt. A later worker can recover the accepted job.",
};
type Pending = { body: Mutation; key: string };

function JobDetails({
  job,
  busy,
  onAction,
}: {
  job: Job;
  busy: boolean;
  onAction: (id: string, action: "cancel" | "replay") => void;
}) {
  return (
    <details className="admin-job">
      <summary>
        <span className="admin-job-summary">
          <span>{STATUS[job.status]}</span>
          <span>
            {job.request.operation === "delete"
              ? "Delete sections"
              : `Rebuild ${job.request.catalog_id}`}
          </span>
          <time dateTime={job.created_at}>
            {new Date(job.created_at).toLocaleString()}
          </time>
        </span>
      </summary>
      <div className="admin-job-body">
        <p className="source-note">
          Job <code>{job.job_id}</code> · Stage: {job.stage}
          {job.cancel_requested ? " · Cancellation requested" : ""}
        </p>
        <p className="source-note">
          Base revision:{" "}
          <code>{job.request.base_revision ?? "First publication"}</code>
        </p>
        {job.request.remove_section_ids.length > 0 && (
          <p>Sections: {job.request.remove_section_ids.join(", ")}</p>
        )}
        {job.snapshot_id && (
          <p className="source-note">
            Published snapshot: <code>s3-rag-snapshot-{job.snapshot_id}</code>.
            A newer job may now be active.
          </p>
        )}
        {!job.attempts.length ? (
          <p>
            No attempt yet. The local worker must be running to process queued
            work.
          </p>
        ) : (
          <ol className="admin-attempts">
            {job.attempts.map((attempt, index) => (
              <li key={`${attempt.started_at}-${index}`}>
                <strong>
                  Attempt {index + 1}: {attempt.outcome}
                </strong>
                {attempt.error_code && (
                  <>
                    <p>
                      {FAILURES[attempt.error_code] ??
                        "The worker could not finish this attempt. Inspect its error code and server configuration before retrying."}
                    </p>
                    <p className="source-note">
                      Code: <code>{attempt.error_code}</code> ·{" "}
                      {attempt.retryable
                        ? "Potentially transient"
                        : "Review required"}
                    </p>
                  </>
                )}
              </li>
            ))}
          </ol>
        )}
        <div className="actions">
          {job.status === "failed" && (
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => onAction(job.job_id, "replay")}
            >
              Replay failed job
            </button>
          )}
          {(job.status === "queued" ||
            (job.status === "running" &&
              job.stage !== "publishing" &&
              !job.cancel_requested)) && (
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => onAction(job.job_id, "cancel")}
            >
              Cancel job
            </button>
          )}
        </div>
      </div>
    </details>
  );
}

export default function AdminPage() {
  const [draftToken, setDraftToken] = useState("");
  const [token, setToken] = useState("");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [before, setBefore] = useState<number | null>(null);
  const [catalogId, setCatalogId] = useState("");
  const [embeddings, setEmbeddings] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [review, setReview] = useState<Pending | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), []);

  function lock() {
    active.current?.abort();
    active.current = null;
    setToken("");
    setDraftToken("");
    setCatalog(null);
    setJobs([]);
    setBefore(null);
    setSelected([]);
    setReview(null);
    setPending(null);
    setError("");
    setNotice("");
    setBusy(false);
  }
  async function perform(work: (signal: AbortSignal) => Promise<void>) {
    if (active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work(controller.signal);
    } catch (cause) {
      if (controller.signal.aborted) return;
      const code =
        cause instanceof AdminError
          ? cause.code
          : "administration_connection_failed";
      if (cause instanceof AdminError && cause.status === 401) lock();
      setError(
        ERRORS[code] ??
          "The administration request failed. Refresh the corpus and jobs before trying again.",
      );
    } finally {
      if (active.current === controller) {
        active.current = null;
        setBusy(false);
      }
    }
  }
  async function load(credential: string, signal: AbortSignal) {
    const [nextCatalog, page] = await Promise.all([
      getCatalog(credential, signal),
      getJobs(credential, signal),
    ]);
    if (signal.aborted) return;
    setCatalog(nextCatalog);
    setJobs(page.jobs);
    setBefore(page.next_before);
    setCatalogId((current) =>
      Object.hasOwn(nextCatalog.catalog, current)
        ? current
        : (Object.keys(nextCatalog.catalog)[0] ?? ""),
    );
    setSelected([]);
    setReview(null);
  }
  async function unlock(event: FormEvent) {
    event.preventDefault();
    const credential = draftToken.trim();
    await perform(async (signal) => {
      await load(credential, signal);
      if (!signal.aborted) {
        setToken(credential);
        setDraftToken("");
      }
    });
  }
  async function submit(captured: Pending) {
    setReview(null);
    setPending(captured);
    await perform(async (signal) => {
      try {
        const accepted = await submitJob(
          token,
          signal,
          captured.body,
          captured.key,
        );
        if (signal.aborted) return;
        setJobs((current) => [
          accepted,
          ...current.filter((job) => job.job_id !== accepted.job_id),
        ]);
        setPending(null);
        setSelected([]);
        setNotice(
          `Job ${accepted.job_id} accepted. Its durable status is ${STATUS[accepted.status].toLowerCase()}. Refresh to follow worker progress.`,
        );
      } catch (cause) {
        if (
          cause instanceof AdminError &&
          cause.status >= 400 &&
          cause.status < 500
        )
          setPending(null);
        throw cause;
      }
    });
  }
  function capture(body: Mutation): Pending {
    return { body, key: crypto.randomUUID() };
  }
  function rebuild(event: FormEvent) {
    event.preventDefault();
    if (!catalog || !catalogId || busy || pending) return;
    void submit(
      capture({
        operation: "rebuild",
        catalog_id: catalogId,
        base_revision: catalog.active_revision,
        with_embeddings: embeddings,
        remove_section_ids: [],
      }),
    );
  }
  function inspectDeletion() {
    if (
      !catalog?.active_revision ||
      !catalog.active_snapshot ||
      !selected.length
    )
      return;
    setReview(
      capture({
        operation: "delete",
        catalog_id: "pilot",
        base_revision: catalog.active_revision,
        with_embeddings: catalog.active_snapshot.has_vectors,
        remove_section_ids: [...selected],
      }),
    );
  }
  function act(id: string, action: "cancel" | "replay") {
    void perform(async (signal) => {
      const changed = await changeJob(token, signal, id, action);
      if (signal.aborted) return;
      setJobs((current) =>
        current.map((job) => (job.job_id === id ? changed : job)),
      );
      setNotice(
        action === "replay"
          ? "The same accepted job is queued again. Its inputs and attempt history are preserved."
          : "Cancellation recorded. Running work stops at its next safe checkpoint.",
      );
    });
  }
  const activeSnapshot = catalog?.active_snapshot;
  const sections = activeSnapshot?.sections ?? [];
  const mutationDisabled = busy || pending !== null;
  return (
    <main>
      <WorkspaceHeader current="admin" />
      <section className="question-area" aria-labelledby="admin-heading">
        <h1 id="admin-heading">Manage the source corpus.</h1>
        <p className="intro">
          Publish approved documentation changes and follow their durable
          ingestion jobs. Searches keep using the current snapshot until
          publication finishes.
        </p>
        {!token ? (
          <form className="admin-unlock" onSubmit={unlock}>
            <label htmlFor="admin-token">Administrator token</label>
            <input
              id="admin-token"
              type="password"
              autoComplete="off"
              value={draftToken}
              onChange={(event) => setDraftToken(event.target.value)}
              required
              disabled={busy}
              aria-describedby="admin-token-help"
            />
            <p id="admin-token-help" className="source-note">
              Use the token configured for this API. It stays in this page’s
              memory and is cleared when you lock, reload, or leave
              administration.
            </p>
            <button className="primary" disabled={busy || !draftToken.trim()}>
              {busy ? "Connecting…" : "Unlock administration"}
            </button>
          </form>
        ) : (
          <>
            <div className="admin-toolbar">
              <button
                type="button"
                className="secondary"
                disabled={busy}
                onClick={() => void perform((signal) => load(token, signal))}
              >
                Refresh corpus and jobs
              </button>
              <button type="button" className="secondary" onClick={lock}>
                Lock administration
              </button>
              <p>Refresh after the worker runs to see published changes.</p>
            </div>
            {catalog && (
              <>
                <section
                  className="admin-section"
                  aria-labelledby="active-heading"
                >
                  <h2 id="active-heading">Active corpus</h2>
                  <p className="source-note">
                    Revision:{" "}
                    <code>
                      {catalog.active_revision ?? "No publication yet"}
                    </code>
                  </p>
                  {activeSnapshot ? (
                    <p>
                      {activeSnapshot.corpus_id} ·{" "}
                      {activeSnapshot.document_count} passages ·{" "}
                      {activeSnapshot.has_vectors
                        ? "Includes embeddings"
                        : "Text only"}
                    </p>
                  ) : (
                    <p>
                      {catalog.active_revision
                        ? "This revision has no registered input artifacts. Rebuild from an approved manifest to manage its sections."
                        : "Choose an approved manifest below to publish the first snapshot."}
                    </p>
                  )}
                  <form onSubmit={rebuild} className="admin-rebuild">
                    <label htmlFor="admin-manifest">
                      Approved replacement manifest
                    </label>
                    <select
                      id="admin-manifest"
                      value={catalogId}
                      onChange={(event) => setCatalogId(event.target.value)}
                      disabled={mutationDisabled}
                      required
                    >
                      {Object.entries(catalog.catalog).map(([id, item]) => (
                        <option key={id} value={id}>
                          {id} · {item.sections.length} sections
                        </option>
                      ))}
                    </select>
                    <p className="source-note">
                      Rebuild replaces the active corpus with this approved
                      manifest. It does not add arbitrary files or URLs.
                    </p>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={embeddings}
                        disabled={mutationDisabled}
                        onChange={(event) =>
                          setEmbeddings(event.target.checked)
                        }
                      />
                      Build embeddings
                    </label>
                    <p className="source-note">
                      Off by default. Embeddings require configured model access
                      and may incur provider charges when the worker runs.
                    </p>
                    <button
                      className="primary"
                      disabled={mutationDisabled || !catalogId}
                    >
                      Queue rebuild
                    </button>
                  </form>
                </section>
                <section
                  className="admin-section"
                  aria-labelledby="sections-heading"
                >
                  <h2 id="sections-heading">Included sections</h2>
                  {activeSnapshot?.artifact_state ===
                  "legacy-rebuild-required" ? (
                    <p>
                      Rebuild this older snapshot before selecting sections for
                      deletion.
                    </p>
                  ) : !sections.length ? (
                    <p>
                      No verified sections are available for deletion. An empty
                      published corpus returns no evidence.
                    </p>
                  ) : (
                    <>
                      <fieldset
                        className="admin-section-list"
                        disabled={mutationDisabled || review !== null}
                      >
                        <legend>
                          Select exact sections to remove from the active corpus
                        </legend>
                        {sections.map((section) => (
                          <label
                            className="checkbox-label"
                            key={section.section_id}
                          >
                            <input
                              type="checkbox"
                              checked={selected.includes(section.section_id)}
                              onChange={(event) =>
                                setSelected((current) =>
                                  event.target.checked
                                    ? [...current, section.section_id]
                                    : current.filter(
                                        (id) => id !== section.section_id,
                                      ),
                                )
                              }
                            />
                            <span>
                              {section.heading_path.join(" › ")}
                              <small>{section.section_id}</small>
                            </span>
                          </label>
                        ))}
                      </fieldset>
                      {!review && (
                        <button
                          type="button"
                          className="secondary"
                          disabled={
                            mutationDisabled ||
                            !selected.length ||
                            selected.length > 100
                          }
                          onClick={inspectDeletion}
                        >
                          Review deletion
                          {selected.length ? ` (${selected.length})` : ""}
                        </button>
                      )}
                      {selected.length > 100 && (
                        <p className="source-note">
                          Choose at most 100 sections per deletion.
                        </p>
                      )}
                      {review && (
                        <div
                          className="admin-deletion-review"
                          role="region"
                          aria-label="Confirm section deletion"
                        >
                          <h3>
                            Remove {review.body.remove_section_ids.length}{" "}
                            sections?
                          </h3>
                          <p>
                            The following sections will leave revision{" "}
                            <code>{review.body.base_revision}</code> when this
                            job publishes:
                          </p>
                          <ul>
                            {review.body.remove_section_ids.map((id) => (
                              <li key={id}>
                                {sections
                                  .find((section) => section.section_id === id)
                                  ?.heading_path.join(" › ")}{" "}
                                <code>{id}</code>
                              </li>
                            ))}
                          </ul>
                          {review.body.remove_section_ids.length ===
                            sections.length && (
                            <p>
                              This removes all current sections. Searches will
                              return no evidence after publication.
                            </p>
                          )}
                          {review.body.with_embeddings && (
                            <p>
                              This snapshot includes embeddings. Deletion
                              preserves vector coverage; missing cached vectors
                              may incur provider calls.
                            </p>
                          )}
                          <div className="actions">
                            <button
                              type="button"
                              className="secondary"
                              disabled={busy}
                              onClick={() => setReview(null)}
                            >
                              Keep sections
                            </button>
                            <button
                              type="button"
                              className="primary"
                              disabled={mutationDisabled}
                              onClick={() => void submit(review)}
                            >
                              Confirm and queue deletion
                            </button>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </section>
              </>
            )}
            <section className="admin-section" aria-labelledby="jobs-heading">
              <div className="section-heading">
                <h2 id="jobs-heading">Recent jobs</h2>
                <span>Newest submissions first</span>
              </div>
              {!jobs.length ? (
                <p>No jobs have been submitted to this API’s durable store.</p>
              ) : (
                jobs.map((job) => (
                  <JobDetails
                    key={job.job_id}
                    job={job}
                    busy={mutationDisabled}
                    onAction={act}
                  />
                ))
              )}
              {before !== null && (
                <button
                  className="secondary admin-older"
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    void perform(async (signal) => {
                      const page = await getJobs(token, signal, before);
                      if (signal.aborted) return;
                      setJobs((current) => [
                        ...current,
                        ...page.jobs.filter(
                          (job) =>
                            !current.some(
                              (existing) => existing.job_id === job.job_id,
                            ),
                        ),
                      ]);
                      setBefore(page.next_before);
                    })
                  }
                >
                  Load older jobs
                </button>
              )}
            </section>
          </>
        )}
        {busy && (
          <p className="source-note" role="status">
            Loading administration state…
          </p>
        )}
        {error && (
          <p className="error-message" role="alert">
            {error}
          </p>
        )}
        {pending && token && !busy && (
          <div className="notice" role="status">
            <p>
              Submission outcome is uncertain. The server may already have
              accepted it. Retry the exact same request and key to recover its
              durable record; new mutations are paused.
            </p>
            <button
              className="secondary"
              type="button"
              onClick={() => void submit(pending)}
            >
              Retry same submission
            </button>
          </div>
        )}
        {notice && (
          <p className="notice" role="status">
            {notice}
          </p>
        )}
      </section>
    </main>
  );
}
