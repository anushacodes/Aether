import assert from "node:assert/strict";
import test from "node:test";
import {
  AdminError,
  changeJob,
  getCatalog,
  getJobs,
  submitJob,
} from "../app/admin/admin-api.ts";

const id = "1".repeat(32);
const mutation = {
  operation: "rebuild",
  catalog_id: "pilot",
  base_revision: null,
  with_embeddings: false,
  remove_section_ids: [],
};
const job = {
  job_id: id,
  request: mutation,
  status: "failed",
  stage: "failed",
  created_at: "2026-11-11T12:00:00Z",
  updated_at: "2026-11-11T12:01:00Z",
  cancel_requested: false,
  snapshot_id: null,
  attempts: [
    {
      started_at: "2026-11-11T12:00:00Z",
      finished_at: "2026-11-11T12:01:00Z",
      outcome: "failed",
      error_code: "search_request_failed",
      retryable: true,
    },
  ],
};
const catalog = {
  active_revision: null,
  active_snapshot: null,
  catalog: {
    pilot: {
      corpus_id: "s3-pilot-v1",
      sections: [{ section_id: "versioning", heading_path: ["Versioning"] }],
    },
  },
};

test("admin sends credentials only in protected headers and preserves exact retry intent", async (context) => {
  const calls = [];
  context.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, options });
    assert.equal(
      options.headers.Authorization,
      "Bearer manually-entered-test-token",
    );
    assert.equal(options.cache, "no-store");
    assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error");
    assert.equal(new URL(url).searchParams.has("token"), false);
    assert.ok(options.signal instanceof AbortSignal);
    if (url.endsWith("/catalog")) return Response.json(catalog);
    if (options.method === "GET")
      return Response.json({ jobs: [job], next_before: null });
    if (calls.length === 3) throw new Error("lost after possible acceptance");
    return Response.json(job);
  });
  const signal = new AbortController().signal;
  assert.equal(
    (await getCatalog("manually-entered-test-token", signal)).catalog.pilot
      .sections[0].section_id,
    "versioning",
  );
  assert.equal(
    (await getJobs("manually-entered-test-token", signal)).jobs[0].attempts[0]
      .retryable,
    true,
  );
  await assert.rejects(
    submitJob("manually-entered-test-token", signal, mutation, "stable-key"),
    (error) => error instanceof AdminError && error.status === 0,
  );
  assert.equal(
    (
      await submitJob(
        "manually-entered-test-token",
        signal,
        mutation,
        "stable-key",
      )
    ).job_id,
    id,
  );
  assert.deepEqual(calls[2].options.body, calls[3].options.body);
  assert.equal(
    calls[2].options.headers["Idempotency-Key"],
    calls[3].options.headers["Idempotency-Key"],
  );
  assert.equal(
    (await changeJob("manually-entered-test-token", signal, id, "replay"))
      .job_id,
    id,
  );
  assert.equal(calls.at(-1).options.method, "POST");
  await assert.rejects(
    changeJob("manually-entered-test-token", signal, "../catalog", "cancel"),
    /invalid_request/,
  );
});

test("admin rejects malformed identities, pagination and accepted intent, and classifies denied access", async (context) => {
  let payload;
  let status = 200;
  context.mock.method(globalThis, "fetch", async () =>
    Response.json(payload, { status }),
  );
  const signal = new AbortController().signal;
  for (const changed of [
    { ...catalog, active_revision: "unsafe-alias" },
    { ...catalog, active_snapshot: { snapshot_id: id } },
    {
      ...catalog,
      catalog: {
        pilot: {
          corpus_id: "x",
          sections: [
            { section_id: "duplicate", heading_path: ["One"] },
            { section_id: "duplicate", heading_path: ["Two"] },
          ],
        },
      },
    },
  ]) {
    payload = changed;
    await assert.rejects(getCatalog("token", signal), /invalid_response/);
  }
  for (const changed of [
    { jobs: [job, job], next_before: null },
    {
      jobs: [{ ...job, attempts: [{ ...job.attempts[0], retryable: "yes" }] }],
      next_before: null,
    },
    { jobs: [job], next_before: 4 },
    { jobs: [{ ...job, job_id: "../catalog" }], next_before: null },
  ]) {
    payload = changed;
    await assert.rejects(getJobs("token", signal), /invalid_response/);
  }
  payload = { ...job, request: { ...mutation, with_embeddings: true } };
  await assert.rejects(
    submitJob("token", signal, mutation, "key"),
    /invalid_response/,
  );
  payload = { ...job, job_id: "2".repeat(32) };
  await assert.rejects(
    changeJob("token", signal, id, "cancel"),
    /invalid_response/,
  );
  payload = { detail: { code: "administrator_required" } };
  status = 401;
  await assert.rejects(
    getJobs("token", signal),
    (error) => error.code === "administrator_required" && error.status === 401,
  );
  payload = { detail: { code: "administration_disabled" } };
  status = 503;
  await assert.rejects(
    getCatalog("token", signal),
    (error) => error.code === "administration_disabled" && error.status === 503,
  );
});
