"use client";

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  retrieveSources,
  streamAnswer,
  type Chunk,
  type Guide,
  type RetrievalMode,
  type SearchRequest,
  type AnswerResult,
} from "./search-api";

import { SourceInspector } from "./source-inspector";
import { WorkspaceHeader } from "./workspace-header";

type Phase =
  | "idle"
  | "retrieval"
  | "assessment"
  | "refinement"
  | "generation"
  | "complete"
  | "sources"
  | "insufficient"
  | "failed"
  | "stopped";
interface Result {
  phase: Phase;
  question: string;
  text: string;
  evidence: Chunk[];
  citations: string[];
  timings: Record<string, number>;
  error?: string;
  requestId?: string;
  workflow?: AnswerResult["workflow"];
}

const INITIAL: Result = {
  phase: "idle",
  question: "",
  text: "",
  evidence: [],
  citations: [],
  timings: {},
};
const GUIDES: { value: Guide; label: string }[] = [
  { value: "s3-userguide", label: "User guide" },
  { value: "s3-api", label: "API reference" },
];
const MESSAGES: Record<string, string> = {
  model_not_configured:
    "Answers need a configured model. Use Find sources to read the matching passages for now.",
  search_unavailable:
    "The search service is unavailable. Check that the local API and OpenSearch are running, then try again.",
  model_unavailable:
    "The model is unavailable. Try again, or use Find sources.",
  model_request_failed:
    "The model request failed. Try again, or use Find sources.",
  model_throttled: "The model is busy. Wait a minute, then try again.",
  rate_limited: "Too many requests. Wait a minute, then try again.",
  search_busy:
    "Search is busy. Wait for the current requests to finish, then try again.",
  request_timeout:
    "The answer took too long. Try a shorter question, or use Find sources.",
  invalid_citations:
    "The answer did not pass citation checks. Read the original sources or try again.",
  incomplete_stream:
    "The connection ended before the answer finished. Any partial text is unverified. Try again.",
  incomplete_generation:
    "The model stopped before completing an answer. Any partial text is unverified. Try again.",
  invalid_response: "The service returned an unreadable response. Try again.",
  invalid_dependency_response:
    "The model response could not be validated. Review the sources or try again.",
  workflow_budget_exhausted:
    "The evidence check reached its model-call limit. Try answering without the extra check, or use Find sources.",
};

function answerWithCitations(
  text: string,
  evidence: Chunk[],
  select: (index: number) => void,
): ReactNode[] {
  return text
    .split(/(\[[a-zA-Z0-9][a-zA-Z0-9._-]*\])/g)
    .map((part, position) => {
      const index = part.startsWith("[")
        ? evidence.findIndex((chunk) => chunk.chunk_id === part.slice(1, -1))
        : -1;
      if (index === -1) return part;
      return (
        <button
          className="citation"
          key={position}
          onClick={() => select(index)}
          aria-label={`Show source ${index + 1}: ${evidence[index].heading_path.at(-1)}`}
        >
          Source {index + 1}
        </button>
      );
    });
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [guides, setGuides] = useState<Guide[]>(["s3-userguide", "s3-api"]);
  const [mode, setMode] = useState<RetrievalMode>("bm25");
  const [workflow, setWorkflow] = useState(false);
  const [result, setResult] = useState<Result>(INITIAL);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const sourcePanel = useRef<HTMLElement>(null);
  const queryInput = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => controller.current?.abort(), []);

  function selectSource(index: number) {
    setSelected(index);
    sourcePanel.current?.focus({ preventScroll: true });
    sourcePanel.current?.scrollIntoView({ block: "nearest" });
  }

  async function run(sourcesOnly: boolean) {
    if (controller.current || !question.trim()) return;
    const request: SearchRequest = {
      query: question.trim(),
      mode,
      scope: { permitted_guides: guides, source_scope: "general-purpose" },
      workflow: !sourcesOnly && workflow,
    };
    const active = new AbortController();
    controller.current = active;
    const deadline = setTimeout(
      () => active.abort(new Error("request_timeout")),
      95_000,
    );
    setBusy(true);
    setSelected(0);
    setResult({ ...INITIAL, phase: "retrieval", question: request.query });
    try {
      if (sourcesOnly) {
        const response = await retrieveSources(request, active.signal);
        setResult((current) => ({
          ...current,
          phase: "sources",
          evidence: response.hits.map((hit) => hit.chunk),
          timings: response.timings_ms,
        }));
      } else {
        await streamAnswer(request, active.signal, (event) => {
          if (event.event === "sources" || event.event === "done")
            setSelected(0);
          setResult((current) => {
            if (event.event === "status")
              return { ...current, phase: event.data.stage };
            if (event.event === "sources")
              return { ...current, evidence: event.data.evidence };
            if (event.event === "token")
              return { ...current, text: current.text + event.data.text };
            if (event.event === "error")
              return {
                ...current,
                phase: "failed",
                error: event.data.code,
                requestId: event.data.request_id,
              };
            const answer = event.data;
            return {
              ...current,
              phase:
                answer.status === "answered"
                  ? "complete"
                  : answer.status === "insufficient_evidence"
                    ? "insufficient"
                    : "failed",
              text: answer.answer,
              evidence: answer.evidence,
              citations: answer.citation_ids,
              timings: answer.timings_ms,
              workflow: answer.workflow,
              error:
                answer.status === "failed" ? "invalid_citations" : undefined,
            };
          });
        });
      }
    } catch (error) {
      const timedOut =
        active.signal.reason instanceof Error &&
        active.signal.reason.message === "request_timeout";
      setResult((current) => ({
        ...current,
        phase: active.signal.aborted && !timedOut ? "stopped" : "failed",
        error: timedOut
          ? "request_timeout"
          : error instanceof TypeError
            ? "search_unavailable"
            : error instanceof Error
              ? error.message
              : "generation_failed",
      }));
    } finally {
      clearTimeout(deadline);
      if (controller.current === active) controller.current = null;
      setBusy(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void run(false);
  }

  const labels: Record<Phase, string> = {
    idle: "Ready for your question",
    retrieval: "Finding supporting passages…",
    assessment: "Checking the available evidence…",
    refinement: "Trying a more focused search…",
    generation: "Writing answer…",
    complete: "Answer complete",
    sources: result.evidence.length
      ? `${result.evidence.length} sources found`
      : "No matching sources",
    insufficient: "Not enough evidence",
    failed: "Answer unavailable",
    stopped: "Stopped",
  };
  const partial =
    ["generation", "failed", "stopped"].includes(result.phase) && !!result.text;
  const displayedSource = result.evidence[selected];

  return (
    <main>
      <WorkspaceHeader current="ask" />
      <section className="question-area" aria-labelledby="question-heading">
        <h1 id="question-heading">Ask about S3. Inspect the sources.</h1>
        <p className="intro">
          Explore a curated set of S3 documentation. Follow each answer back to
          the original passage.
        </p>
        <form onSubmit={submit}>
          <label htmlFor="question">Your question</label>
          <textarea
            id="question"
            ref={queryInput}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="What happens when I delete an object in a versioned bucket?"
            rows={3}
            maxLength={2000}
            required
            disabled={busy}
            aria-describedby="question-help"
          />
          <div className="form-settings">
            <fieldset disabled={busy}>
              <legend>Search in</legend>
              {GUIDES.map((guide) => (
                <label className="checkbox-label" key={guide.value}>
                  <input
                    type="checkbox"
                    checked={guides.includes(guide.value)}
                    disabled={
                      guides.length === 1 && guides.includes(guide.value)
                    }
                    onChange={(event) =>
                      setGuides((current) =>
                        event.target.checked
                          ? [...current, guide.value]
                          : current.filter((item) => item !== guide.value),
                      )
                    }
                  />
                  {guide.label}
                </label>
              ))}
            </fieldset>
            <label className="mode-label" htmlFor="retrieval-mode">
              Retrieval
              <select
                id="retrieval-mode"
                value={mode}
                disabled={busy}
                onChange={(event) =>
                  setMode(event.target.value as RetrievalMode)
                }
              >
                <option value="bm25">Keyword · local</option>
                <option value="hybrid">Hybrid · requires embeddings</option>
                <option value="vector">Vector · requires embeddings</option>
              </select>
            </label>
            <div className="workflow-option">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={workflow}
                  disabled={busy}
                  onChange={(event) => setWorkflow(event.target.checked)}
                />
                Check evidence before answering
              </label>
              <p>May search twice and use extra model calls.</p>
            </div>
          </div>
          <div className="form-footer">
            <p id="question-help">
              General-purpose buckets. Find sources works without an answer
              model.
            </p>
            <div className="actions">
              {busy ? (
                <button
                  type="button"
                  className="secondary"
                  onClick={() => controller.current?.abort()}
                >
                  Stop
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="secondary"
                    disabled={!question.trim()}
                    onClick={() => void run(true)}
                  >
                    Find sources
                  </button>
                  <button
                    type="submit"
                    className="primary"
                    disabled={!question.trim()}
                  >
                    Ask question
                  </button>
                </>
              )}
            </div>
          </div>
        </form>
      </section>
      <div
        className="result-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <span
          className={busy ? "status-indicator active" : "status-indicator"}
          aria-hidden="true"
        />
        {labels[result.phase]}
        {typeof result.timings.retrieval_total === "number" && (
          <span className="timing">
            Retrieval {(result.timings.retrieval_total / 1000).toFixed(2)} s
          </span>
        )}
      </div>
      <div className="workspace">
        <section
          className="answer-area"
          aria-labelledby="answer-heading"
          aria-busy={busy}
        >
          <h2 id="answer-heading">Answer</h2>
          {result.question && (
            <p className="asked-question">{result.question}</p>
          )}
          {result.phase === "idle" && (
            <div className="empty-state">
              <h3>A useful place to start</h3>
              <p>
                The pilot includes versioning and object deletion. Try a
                question from these topics.
              </p>
              <button
                className="example-question"
                onClick={() => {
                  setQuestion(
                    "What happens when I delete an object in a versioned bucket?",
                  );
                  queryInput.current?.focus();
                }}
              >
                What happens when I delete an object in a versioned bucket?
              </button>
            </div>
          )}
          {["retrieval", "assessment", "refinement"].includes(result.phase) && (
            <p className="muted">
              {result.phase === "retrieval"
                ? "Searching the selected guides for original passages."
                : result.phase === "assessment"
                  ? "Checking whether these passages cover your question."
                  : "Keeping your original question and guides while adding focused search terms."}
            </p>
          )}
          {result.phase === "generation" && (
            <p className="notice">
              Draft answer · citations have not been checked yet.
            </p>
          )}
          {result.phase === "failed" && (
            <div role="alert" className="error-message">
              <h3>Couldn’t complete the answer</h3>
              <p>
                {MESSAGES[result.error || ""] ||
                  "The request failed. Try again, or use Find sources."}
              </p>
              {result.requestId && (
                <p className="request-id">Request: {result.requestId}</p>
              )}
            </div>
          )}
          {result.phase === "stopped" && (
            <p className="notice">
              You stopped the request.{" "}
              {partial
                ? "The partial answer is unverified."
                : "No answer was completed."}
            </p>
          )}
          {result.phase === "sources" && (
            <p>
              {result.evidence.length
                ? "Read the matching passages in Sources. Ask question to generate an answer from this documentation."
                : "No passages matched this question in the selected guides. Try different wording or include both guides."}
            </p>
          )}
          {result.phase === "insufficient" && (
            <p className="notice">
              The selected sources don’t provide enough evidence to answer this
              question. Try a question about versioning or object deletion.
            </p>
          )}
          {result.text && result.phase !== "insufficient" && (
            <>
              {partial && result.phase !== "generation" && (
                <p className="partial-label">Unverified partial answer</p>
              )}
              <div className="answer-text">
                {answerWithCitations(
                  result.text,
                  result.evidence,
                  selectSource,
                )}
              </div>
            </>
          )}
          {result.phase === "complete" && (
            <p className="answer-note">
              Citation IDs checked against the supplied passages. Review the
              sources to assess the answer.
            </p>
          )}
          {result.workflow && (
            <p className="source-note">
              {result.workflow.retrieval_passes}{" "}
              {result.workflow.retrieval_passes === 1 ? "search" : "searches"} ·{" "}
              {result.workflow.model_calls} model calls
            </p>
          )}
        </section>
        <section className="sources-area" aria-labelledby="sources-heading">
          <div className="section-heading">
            <h2 id="sources-heading">Sources</h2>
            <span>{result.evidence.length} passages</span>
          </div>
          {!result.evidence.length ? (
            <div className="empty-state">
              <h3>Original passages, right here</h3>
              <p>
                Search to inspect the text, guide, and snapshot behind each
                source.
              </p>
            </div>
          ) : (
            <>
              <div className="source-list" aria-label="Retrieved passages">
                {result.evidence.map((chunk, index) => (
                  <button
                    className="source-choice"
                    key={chunk.chunk_id}
                    aria-pressed={selected === index}
                    onClick={() => selectSource(index)}
                  >
                    <span className="source-number">{index + 1}</span>
                    <span>
                      <strong>{chunk.heading_path.at(-1)}</strong>
                      <span className="source-kind">
                        {chunk.provenance.guide === "s3-api"
                          ? "API reference"
                          : "User guide"}
                        {result.citations.includes(chunk.chunk_id)
                          ? " · Cited"
                          : ""}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
              <article
                className="source-preview"
                ref={sourcePanel}
                tabIndex={-1}
                aria-label={`Source ${selected + 1} original passage`}
              >
                {displayedSource && <SourceInspector chunk={displayedSource} />}
              </article>
            </>
          )}
        </section>
      </div>
      <footer className="page-footer">
        S3 documentation subset · Answers describe the source material, not your
        live AWS account.
      </footer>
    </main>
  );
}
