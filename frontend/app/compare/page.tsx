"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  compareSources,
  type ComparisonArm,
  type ComparisonResult,
  type Guide,
  type RetrievalMode,
} from "../search-api";
import { SourceInspector } from "../source-inspector";
import { WorkspaceHeader } from "../workspace-header";

const MODES: { value: RetrievalMode; label: string }[] = [
  { value: "bm25", label: "Keyword" },
  { value: "vector", label: "Vector" },
  { value: "hybrid", label: "Hybrid" },
];
const ERRORS: Record<string, string> = {
  vector_index_not_ready:
    "This snapshot has no complete vector index. Try keyword search, or build embeddings before using this mode.",
  invalid_index_configuration:
    "This snapshot does not match the configured embedding space. Check the index configuration before trying again.",
  model_not_configured:
    "This mode needs model configuration. Keyword search can run without embeddings.",
  search_unavailable:
    "The search service is unavailable. Check that the local API and OpenSearch are running, then try again.",
  model_throttled: "The model is busy. Wait a minute, then try again.",
  model_unavailable:
    "The model is unavailable. Try again when it is reachable.",
  model_request_failed:
    "The model request failed. Check the model configuration and try again.",
  model_timeout:
    "The model took too long. Try again, or compare without this mode.",
  reranking_failed:
    "The configured reranker failed. Check its configuration or disable reranking before trying again.",
  reranking_timeout:
    "The configured reranker took too long. Try again or disable reranking.",
  rate_limited: "Too many requests. Wait a minute, then try again.",
  search_busy:
    "Search is busy. Wait for the current work to finish, then try again.",
  request_timeout:
    "The comparison took too long. Try again with keyword search as the first mode.",
};

function ResultColumn({ arm, side }: { arm: ComparisonArm; side: string }) {
  const [selected, setSelected] = useState(0);
  const hits = arm.result?.hits ?? [];
  const source = hits[selected]?.chunk;
  return (
    <section className="comparison-column" aria-labelledby={`${side}-heading`}>
      <div className="section-heading">
        <h2 id={`${side}-heading`}>
          {MODES.find((mode) => mode.value === arm.mode)?.label}
        </h2>
        <span>
          {arm.status === "error" ? "Unavailable" : `${hits.length} passages`}
        </span>
      </div>
      <dl className="comparison-stats">
        <div>
          <dt>Search time</dt>
          <dd>{arm.elapsed_ms.toFixed(1)} ms</dd>
        </div>
        <div>
          <dt>Model calls</dt>
          <dd>{arm.model_calls}</dd>
        </div>
        <div>
          <dt>Model cost</dt>
          <dd>{arm.model_cost_usd === 0 ? "$0" : "Unknown"}</dd>
        </div>
      </dl>
      {arm.embedding_cache_hits > 0 && (
        <p className="source-note">Query embedding reused from cache.</p>
      )}
      {arm.reranking?.status === "fallback" && (
        <p className="notice">
          Reranking fell back to the original retrieval order.
        </p>
      )}
      {arm.status === "error" ? (
        <div className="error-message">
          <h3>Couldn’t search this mode</h3>
          <p>
            {ERRORS[arm.error_code ?? ""] ??
              "The service could not complete this mode. Try again."}
          </p>
        </div>
      ) : !source ? (
        <div className="empty-state">
          <h3>No matching sources</h3>
          <p>
            No passages matched this question and guide selection. Try different
            terms or guides.
          </p>
        </div>
      ) : (
        <>
          <div className="comparison-passage-picker">
            <label htmlFor={`${side}-passage`}>Passage</label>
            <select
              id={`${side}-passage`}
              value={selected}
              onChange={(event) => setSelected(Number(event.target.value))}
            >
              {hits.map((hit, index) => (
                <option key={hit.chunk.chunk_id} value={index}>
                  {index + 1}. {hit.chunk.heading_path.at(-1)}
                </option>
              ))}
            </select>
          </div>
          <div className="source-preview">
            <SourceInspector chunk={source} />
          </div>
        </>
      )}
    </section>
  );
}

export default function ComparePage() {
  const [question, setQuestion] = useState("");
  const [modes, setModes] = useState<[RetrievalMode, RetrievalMode]>([
    "bm25",
    "hybrid",
  ]);
  const [guides, setGuides] = useState<Guide[]>(["s3-userguide", "s3-api"]);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopped, setStopped] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  function changeMode(index: number, mode: RetrievalMode) {
    setModes((current) =>
      index === 0
        ? [mode, current[1] === mode ? current[0] : current[1]]
        : [current[0] === mode ? current[1] : current[0], mode],
    );
  }

  async function compare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (controller.current || !question.trim()) return;
    const active = new AbortController();
    controller.current = active;
    const timeout = setTimeout(
      () => active.abort(new Error("request_timeout")),
      185_000,
    );
    setBusy(true);
    setStopped(false);
    setError("");
    setResult(null);
    try {
      const response = await compareSources(
        {
          query: question.trim(),
          modes,
          scope: { permitted_guides: guides, source_scope: "general-purpose" },
        },
        active.signal,
      );
      if (!active.signal.aborted) setResult(response);
    } catch (cause) {
      if (
        active.signal.aborted &&
        !active.signal.reason?.message?.includes("request_timeout")
      ) {
        setStopped(true);
      } else {
        const code = active.signal.aborted
          ? "request_timeout"
          : cause instanceof Error
            ? cause.message
            : "";
        setError(
          ERRORS[code] ??
            "Couldn’t complete the comparison. Check the local API, then try again.",
        );
      }
    } finally {
      clearTimeout(timeout);
      controller.current = null;
      setBusy(false);
    }
  }

  return (
    <main>
      <WorkspaceHeader current="compare" />
      <section className="question-area" aria-labelledby="compare-heading">
        <h1 id="compare-heading">One question. Two ways to find sources.</h1>
        <p className="intro">
          Compare live retrieval modes, then inspect the original passages from
          the same snapshot.
        </p>
        <form onSubmit={compare}>
          <label htmlFor="comparison-question">Your question</label>
          <textarea
            id="comparison-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={2000}
            required
            rows={3}
            disabled={busy}
            placeholder="What happens when I delete an object in a versioned bucket?"
            aria-describedby="comparison-help"
          />
          <div className="comparison-controls">
            {modes.map((mode, index) => (
              <div className="mode-label" key={index}>
                <label htmlFor={`comparison-mode-${index}`}>
                  {index === 0 ? "First mode" : "Second mode"}
                </label>
                <select
                  id={`comparison-mode-${index}`}
                  value={mode}
                  disabled={busy}
                  onChange={(event) =>
                    changeMode(index, event.target.value as RetrievalMode)
                  }
                >
                  {MODES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                      {option.value !== "bm25" ? " · requires embeddings" : ""}
                    </option>
                  ))}
                </select>
              </div>
            ))}
            <fieldset disabled={busy}>
              <legend>Search in</legend>
              {(["s3-userguide", "s3-api"] as Guide[]).map((guide) => (
                <label className="checkbox-label" key={guide}>
                  <input
                    type="checkbox"
                    checked={guides.includes(guide)}
                    disabled={guides.length === 1 && guides.includes(guide)}
                    onChange={(event) =>
                      setGuides((current) =>
                        event.target.checked
                          ? [...current, guide]
                          : current.filter((item) => item !== guide),
                      )
                    }
                  />
                  {guide === "s3-api" ? "API reference" : "User guide"}
                </label>
              ))}
            </fieldset>
          </div>
          <div className="form-footer">
            <p id="comparison-help">
              Vector and hybrid require indexed embeddings. They—and any
              configured reranker—can make model calls. Answers are not
              generated.
            </p>
            <div className="actions">
              {busy ? (
                <button
                  className="secondary"
                  type="button"
                  onClick={(event) => {
                    event.preventDefault();
                    controller.current?.abort();
                  }}
                >
                  Stop
                </button>
              ) : (
                <button
                  className="primary"
                  type="submit"
                  disabled={!question.trim()}
                >
                  Compare sources
                </button>
              )}
            </div>
          </div>
        </form>
      </section>
      <div className="result-status" role="status" aria-live="polite">
        <span className={`status-indicator${busy ? " active" : ""}`} />
        {busy
          ? "Comparing sources…"
          : stopped
            ? "Comparison stopped"
            : error
              ? "Comparison unavailable"
              : result
                ? "Comparison ready"
                : "Ready to compare"}
      </div>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {stopped && (
        <p className="notice">
          Comparison stopped. In-flight work may take a moment to finish; no
          further mode will be started after cancellation is received.
        </p>
      )}
      {result ? (
        <>
          <p className="asked-question">{result.query}</p>
          <p className="source-note comparison-snapshot">
            Shared snapshot: {result.snapshot_index}
          </p>
          <p className="source-note">
            Modes run in the selected order and share an embedding cache. These
            timings describe this request, not a benchmark. Model cost excludes
            hosting; paid-call prices are not configured.
          </p>
          <div className="comparison-results">
            {result.arms.map((arm, index) => (
              <ResultColumn
                key={`${result.total_ms}-${index}`}
                arm={arm}
                side={`comparison-${index}`}
              />
            ))}
          </div>
          <details className="provenance comparison-settings">
            <summary>Retrieval settings</summary>
            <pre className="passage">
              {JSON.stringify(
                {
                  retrieval: result.retrieval_settings,
                  reranking: result.reranking_settings,
                },
                null,
                2,
              )}
            </pre>
          </details>
        </>
      ) : (
        !busy &&
        !error &&
        !stopped && (
          <div className="empty-state comparison-empty">
            <h2>Inspect differences in the evidence</h2>
            <p>
              Choose two modes for the same question and guides. A mode that is
              unavailable will show its own explanation while the other side
              keeps its sources.
            </p>
          </div>
        )
      )}
      <footer className="page-footer">
        Live source inspection · No quality score or winning configuration is
        inferred from one question.
      </footer>
    </main>
  );
}
