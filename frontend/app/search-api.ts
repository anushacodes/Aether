/** Browser contracts and bounded transport for the local search API. */
export type Guide = "s3-userguide" | "s3-api";
export type RetrievalMode = "bm25" | "vector" | "hybrid";

export interface Chunk {
  chunk_id: string;
  section_id: string;
  corpus_id: string;
  original_text: string;
  source_scope: "general-purpose";
  heading_path: string[];
  section_raw_start: number;
  section_raw_end: number;
  provenance: {
    source_id: string;
    guide: Guide;
    revision: string;
    original_filename: string;
    repository_url: string;
    publisher_url: string;
    sha256: string;
  };
}

export interface SearchRequest {
  query: string;
  mode: RetrievalMode;
  scope: { permitted_guides: Guide[]; source_scope: "general-purpose" };
  workflow?: boolean;
}

export interface RetrievalResult {
  hits: { chunk: Chunk; method: RetrievalMode; rank: number; score: number }[];
  mode: RetrievalMode;
  timings_ms: Record<string, number>;
}

export interface AnswerResult {
  answer: string;
  citation_ids: string[];
  status: "answered" | "insufficient_evidence" | "failed";
  evidence: Chunk[];
  timings_ms: Record<string, number>;
  usage_tokens: Record<string, number>;
  workflow?: {
    snapshot_index: string;
    retrieval_passes: number;
    model_calls: number;
    assessment_prompt_version: string;
    steps: {
      retrieval_pass: number;
      query: string;
      decision: "answer" | "retry" | "insufficient";
      missing_evidence: string;
      context_chunk_ids?: string[];
    }[];
  } | null;
}

export type SearchEvent =
  | {
      event: "status";
      data: { stage: "retrieval" | "assessment" | "refinement" | "generation" };
    }
  | { event: "sources"; data: { evidence: Chunk[] } }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: AnswerResult }
  | { event: "error"; data: { code: string; request_id?: string } };

const API_URL = (
  process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8000"
).replace(/\/$/, "");
const MAX_FRAME_CHARACTERS = 1_000_000;
const MAX_RESPONSE_CHARACTERS = 32_000;

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("invalid_response");
  return value as Record<string, unknown>;
}

function strings(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function numbers(value: unknown): value is Record<string, number> {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.values(value).every(
      (item) => typeof item === "number" && Number.isFinite(item) && item >= 0,
    )
  );
}

function evidence(value: unknown): Chunk[] {
  if (!Array.isArray(value) || value.length > 500)
    throw new Error("invalid_response");
  const ids = new Set<string>();
  for (const item of value) {
    const chunk = object(item);
    const source = object(chunk.provenance);
    const textFields = [
      chunk.chunk_id,
      chunk.section_id,
      chunk.corpus_id,
      chunk.original_text,
      source.revision,
      source.original_filename,
      source.repository_url,
      source.publisher_url,
      source.sha256,
    ];
    if (
      textFields.some((field) => typeof field !== "string" || !field) ||
      !strings(chunk.heading_path) ||
      !chunk.heading_path.length ||
      !["s3-userguide", "s3-api"].includes(String(source.guide)) ||
      chunk.source_scope !== "general-purpose" ||
      !Number.isSafeInteger(chunk.section_raw_start) ||
      !Number.isSafeInteger(chunk.section_raw_end) ||
      Number(chunk.section_raw_start) < 0 ||
      Number(chunk.section_raw_end) <= Number(chunk.section_raw_start) ||
      ids.has(String(chunk.chunk_id))
    )
      throw new Error("invalid_response");
    ids.add(String(chunk.chunk_id));
  }
  return value as Chunk[];
}

function answer(value: unknown): AnswerResult {
  const result = object(value);
  const chunks = evidence(result.evidence);
  const known = new Set(chunks.map((chunk) => chunk.chunk_id));
  if (
    typeof result.answer !== "string" ||
    result.answer.length > MAX_RESPONSE_CHARACTERS ||
    !strings(result.citation_ids) ||
    result.citation_ids.some((id) => !known.has(id)) ||
    !["answered", "insufficient_evidence", "failed"].includes(
      String(result.status),
    ) ||
    (result.status === "answered" && !result.citation_ids.length) ||
    !numbers(result.timings_ms) ||
    !numbers(result.usage_tokens)
  )
    throw new Error("invalid_response");
  if (result.workflow !== undefined && result.workflow !== null) {
    const report = object(result.workflow);
    if (
      typeof report.snapshot_index !== "string" ||
      typeof report.assessment_prompt_version !== "string" ||
      !Number.isInteger(report.retrieval_passes) ||
      Number(report.retrieval_passes) < 0 ||
      Number(report.retrieval_passes) > 2 ||
      !Number.isInteger(report.model_calls) ||
      Number(report.model_calls) < 0 ||
      Number(report.model_calls) > 5 ||
      !Array.isArray(report.steps) ||
      report.steps.length > 2
    )
      throw new Error("invalid_response");
    for (const item of report.steps) {
      const step = object(item);
      if (
        !Number.isInteger(step.retrieval_pass) ||
        Number(step.retrieval_pass) < 1 ||
        Number(step.retrieval_pass) > 2 ||
        typeof step.query !== "string" ||
        typeof step.missing_evidence !== "string" ||
        !["answer", "retry", "insufficient"].includes(String(step.decision))
      )
        throw new Error("invalid_response");
    }
  }
  return result as unknown as AnswerResult;
}

function parseEvent(frame: string): SearchEvent | null {
  let name = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!data.length) return null; // Keepalive comments are not search events.
  const payload = object(JSON.parse(data.join("\n")));
  if (name === "done") return { event: name, data: answer(payload) };
  if (name === "sources")
    return { event: name, data: { evidence: evidence(payload.evidence) } };
  if (name === "token" && typeof payload.text === "string")
    return { event: name, data: { text: payload.text } };
  if (
    name === "status" &&
    (payload.stage === "retrieval" ||
      payload.stage === "assessment" ||
      payload.stage === "refinement" ||
      payload.stage === "generation")
  ) {
    return { event: name, data: { stage: payload.stage } };
  }
  if (name === "error" && typeof payload.code === "string") {
    return {
      event: name,
      data: {
        code: payload.code,
        request_id:
          typeof payload.request_id === "string"
            ? payload.request_id
            : undefined,
      },
    };
  }
  throw new Error("invalid_response");
}

/** Decode bytes incrementally: network chunks can split both UTF-8 and SSE boundaries. */
export async function* readSearchEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SearchEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let pendingCR = false;
  let responseCharacters = 0;
  let terminal = false;
  try {
    while (!terminal) {
      const { value, done } = await reader.read();
      let text = decoder.decode(value, { stream: !done });
      if (pendingCR) {
        text = "\r" + text;
        pendingCR = false;
      }
      if (!done && text.endsWith("\r")) {
        pendingCR = true;
        text = text.slice(0, -1);
      }
      buffer += text.replace(/\r\n?/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        if (boundary > MAX_FRAME_CHARACTERS)
          throw new Error("invalid_response");
        const event = parseEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (event) {
          if (event.event === "token")
            responseCharacters += event.data.text.length;
          if (responseCharacters > MAX_RESPONSE_CHARACTERS)
            throw new Error("invalid_response");
          terminal = event.event === "done" || event.event === "error";
          yield event;
          if (terminal) return;
        }
      }
      if (buffer.length > MAX_FRAME_CHARACTERS)
        throw new Error("invalid_response");
      if (done) throw new Error("incomplete_stream");
    }
  } finally {
    try {
      await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
}

async function post(
  path: string,
  request: SearchRequest,
  signal: AbortSignal,
): Promise<Response> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const code = payload?.detail?.code;
    throw new Error(
      typeof code === "string" ? code : `http_${response.status}`,
    );
  }
  return response;
}

export async function retrieveSources(
  request: SearchRequest,
  signal: AbortSignal,
): Promise<RetrievalResult> {
  const payload = object(
    await (await post("/retrieval", request, signal)).json(),
  );
  if (
    !Array.isArray(payload.hits) ||
    !numbers(payload.timings_ms) ||
    payload.mode !== request.mode
  ) {
    throw new Error("invalid_response");
  }
  evidence(payload.hits.map((hit) => object(hit).chunk));
  for (const hit of payload.hits) {
    if (
      !Number.isInteger(hit.rank) ||
      hit.rank < 1 ||
      typeof hit.score !== "number" ||
      !Number.isFinite(hit.score) ||
      hit.method !== request.mode
    )
      throw new Error("invalid_response");
  }
  return payload as unknown as RetrievalResult;
}

export async function streamAnswer(
  request: SearchRequest,
  signal: AbortSignal,
  onEvent: (event: SearchEvent) => void,
): Promise<void> {
  const response = await post("/search/stream", request, signal);
  if (
    !response.body ||
    !response.headers.get("content-type")?.includes("text/event-stream")
  ) {
    throw new Error("invalid_response");
  }
  for await (const event of readSearchEvents(response.body)) {
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    onEvent(event);
  }
}

/** Source URLs come from response data; never turn non-web schemes into links. */
export function sourceLink(value: string): string | undefined {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:"
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}
