/** Protected local administration contracts. Credentials are supplied per request, never persisted. */
export interface Section {
  section_id: string;
  heading_path: string[];
}
export interface Catalog {
  active_revision: string | null;
  active_snapshot: {
    snapshot_id: string;
    corpus_id: string;
    document_count: number;
    has_vectors: boolean;
    artifact_state: "verified" | "legacy-rebuild-required";
    sections: Section[] | null;
  } | null;
  catalog: Record<string, { corpus_id: string; sections: Section[] }>;
}
export interface Mutation {
  operation: "rebuild" | "delete";
  catalog_id: string;
  base_revision: string | null;
  with_embeddings: boolean;
  remove_section_ids: string[];
}
export interface Job {
  job_id: string;
  request: Mutation;
  status: "queued" | "running" | "ready" | "failed" | "cancelled";
  stage: string;
  created_at: string;
  updated_at: string;
  cancel_requested: boolean;
  snapshot_id: string | null;
  attempts: {
    started_at: string;
    finished_at: string | null;
    outcome: "running" | "ready" | "failed" | "cancelled" | "interrupted";
    error_code: string | null;
    retryable: boolean;
  }[];
}
export interface JobPage {
  jobs: Job[];
  next_before: number | null;
}
export class AdminError extends Error {
  code: string;
  status: number;

  constructor(code: string, status = 0) {
    super(code);
    this.code = code;
    this.status = status;
  }
}

const API_URL = (
  process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8000"
).replace(/\/$/, "");
const ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const JOB_ID = /^[0-9a-f]{32}$/;
const REVISION = /^s3-rag-snapshot-[0-9a-f]{32}$/;
const STATUSES = ["queued", "running", "ready", "failed", "cancelled"];
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new AdminError("invalid_response");
  return value as Record<string, unknown>;
}
function validId(value: unknown, pattern = ID): value is string {
  return typeof value === "string" && pattern.test(value);
}
function integer(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}
function timestamp(value: unknown): boolean {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}
function sections(value: unknown): Section[] {
  if (
    !Array.isArray(value) ||
    value.some((item) => {
      const section = record(item);
      return (
        !validId(section.section_id) ||
        !Array.isArray(section.heading_path) ||
        !section.heading_path.length ||
        section.heading_path.some((text) => typeof text !== "string")
      );
    }) ||
    new Set(value.map((item) => item.section_id)).size !== value.length
  )
    throw new AdminError("invalid_response");
  return value as Section[];
}
function mutation(value: unknown): Mutation {
  const body = record(value);
  if (
    !["rebuild", "delete"].includes(String(body.operation)) ||
    !validId(body.catalog_id) ||
    (body.base_revision !== null && !validId(body.base_revision, REVISION)) ||
    typeof body.with_embeddings !== "boolean" ||
    !Array.isArray(body.remove_section_ids) ||
    body.remove_section_ids.length > 100 ||
    body.remove_section_ids.some((id) => !validId(id)) ||
    new Set(body.remove_section_ids).size !== body.remove_section_ids.length ||
    (body.operation === "rebuild" && body.remove_section_ids.length !== 0) ||
    (body.operation === "delete" &&
      (!body.base_revision || !body.remove_section_ids.length))
  )
    throw new AdminError("invalid_response");
  return body as unknown as Mutation;
}
function job(value: unknown): Job {
  const row = record(value);
  mutation(row.request);
  if (
    !validId(row.job_id, JOB_ID) ||
    !STATUSES.includes(String(row.status)) ||
    ![
      "queued",
      "building",
      "publishing",
      "ready",
      "failed",
      "cancelled",
    ].includes(String(row.stage)) ||
    !timestamp(row.created_at) ||
    !timestamp(row.updated_at) ||
    typeof row.cancel_requested !== "boolean" ||
    (row.snapshot_id !== null && !validId(row.snapshot_id, JOB_ID)) ||
    !Array.isArray(row.attempts)
  )
    throw new AdminError("invalid_response");
  for (const value of row.attempts) {
    const attempt = record(value);
    if (
      !timestamp(attempt.started_at) ||
      (attempt.finished_at !== null && !timestamp(attempt.finished_at)) ||
      !["running", "ready", "failed", "cancelled", "interrupted"].includes(
        String(attempt.outcome),
      ) ||
      (attempt.error_code !== null && !validId(attempt.error_code)) ||
      typeof attempt.retryable !== "boolean"
    )
      throw new AdminError("invalid_response");
  }
  return row as unknown as Job;
}
async function request(
  path: string,
  token: string,
  signal: AbortSignal,
  body?: Mutation,
  key?: string,
): Promise<unknown> {
  if (!token || /[\r\n]/.test(token))
    throw new AdminError("administrator_required", 401);
  try {
    const response = await fetch(`${API_URL}${path}`, {
      method:
        body || path.endsWith("/replay") || path.endsWith("/cancel")
          ? "POST"
          : "GET",
      signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(key ? { "Idempotency-Key": key } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload && record(payload).detail;
      const code =
        detail && !Array.isArray(detail) ? record(detail).code : null;
      throw new AdminError(
        validId(code)
          ? code
          : response.status === 422
            ? "invalid_ingestion_inputs"
            : "administration_request_failed",
        response.status,
      );
    }
    return payload;
  } catch (error) {
    if (error instanceof AdminError) throw error;
    throw new AdminError("administration_connection_failed");
  }
}
export async function getCatalog(
  token: string,
  signal: AbortSignal,
): Promise<Catalog> {
  const result = record(await request("/ingestion/catalog", token, signal));
  if (
    result.active_revision !== null &&
    !validId(result.active_revision, REVISION)
  )
    throw new AdminError("invalid_response");
  for (const [id, value] of Object.entries(record(result.catalog))) {
    const entry = record(value);
    if (!validId(id) || !validId(entry.corpus_id))
      throw new AdminError("invalid_response");
    sections(entry.sections);
  }
  if (result.active_snapshot !== null) {
    const active = record(result.active_snapshot);
    if (
      !result.active_revision ||
      !validId(active.snapshot_id, JOB_ID) ||
      !validId(active.corpus_id) ||
      !integer(active.document_count) ||
      typeof active.has_vectors !== "boolean" ||
      !["verified", "legacy-rebuild-required"].includes(
        String(active.artifact_state),
      )
    )
      throw new AdminError("invalid_response");
    if (active.artifact_state === "verified") sections(active.sections);
    else if (active.sections !== null) throw new AdminError("invalid_response");
  }
  return result as unknown as Catalog;
}
export async function getJobs(
  token: string,
  signal: AbortSignal,
  before?: number,
): Promise<JobPage> {
  if (before !== undefined && (!integer(before) || before < 1))
    throw new AdminError("invalid_request");
  const result = record(
    await request(
      `/ingestion?limit=20${before === undefined ? "" : `&before=${before}`}`,
      token,
      signal,
    ),
  );
  if (
    !Array.isArray(result.jobs) ||
    result.jobs.length > 20 ||
    (result.next_before !== null &&
      (!integer(result.next_before) ||
        result.next_before < 1 ||
        result.jobs.length !== 20 ||
        (before !== undefined && result.next_before >= before)))
  )
    throw new AdminError("invalid_response");
  const jobs = result.jobs.map(job);
  if (new Set(jobs.map((item) => item.job_id)).size !== jobs.length)
    throw new AdminError("invalid_response");
  return { jobs, next_before: result.next_before as number | null };
}
export async function submitJob(
  token: string,
  signal: AbortSignal,
  body: Mutation,
  key: string,
): Promise<Job> {
  mutation(body);
  if (!validId(key)) throw new AdminError("invalid_request");
  const accepted = job(await request("/ingestion", token, signal, body, key));
  if (JSON.stringify(accepted.request) !== JSON.stringify(body)) {
    // Compare fields, independent of server JSON property ordering.
    const expected = mutation(body);
    if (
      Object.keys(expected).some(
        (field) =>
          JSON.stringify(accepted.request[field as keyof Mutation]) !==
          JSON.stringify(expected[field as keyof Mutation]),
      )
    )
      throw new AdminError("invalid_response");
  }
  return accepted;
}
export async function changeJob(
  token: string,
  signal: AbortSignal,
  id: string,
  action: "cancel" | "replay",
): Promise<Job> {
  if (!validId(id, JOB_ID) || !["cancel", "replay"].includes(action))
    throw new AdminError("invalid_request");
  const result = job(
    await request(`/ingestion/${id}/${action}`, token, signal),
  );
  if (result.job_id !== id) throw new AdminError("invalid_response");
  return result;
}
