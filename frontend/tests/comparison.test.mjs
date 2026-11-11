import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { compareSources } from "../app/search-api.ts";

const chunk = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/records.json", import.meta.url),
    "utf8",
  ),
).chunk;
const request = {
  query: "object versions",
  modes: ["bm25", "hybrid"],
  scope: {
    permitted_guides: ["s3-userguide"],
    source_scope: "general-purpose",
  },
};

function response() {
  const base = {
    elapsed_ms: 12,
    model_calls: 0,
    embedding_cache_hits: 0,
    known_embedding_input_tokens: 0,
    model_cost_usd: 0,
    reranking: null,
  };
  return {
    query: request.query,
    scope: request.scope,
    snapshot_index: "same-concrete-snapshot",
    execution: "sequential_shared_cache",
    total_ms: 25,
    retrieval_settings: {},
    reranking_settings: {},
    arms: [
      {
        ...base,
        mode: "bm25",
        status: "ok",
        error_code: null,
        result: {
          mode: "bm25",
          hits: [{ chunk, method: "bm25", rank: 1, score: 1 }],
          timings_ms: { retrieval_total: 10 },
        },
      },
      {
        ...base,
        mode: "hybrid",
        status: "error",
        error_code: "vector_index_not_ready",
        result: null,
      },
    ],
  };
}

test("live comparison retains original sources beside an unavailable mode and sends shared scope once", async (context) => {
  const signal = new AbortController().signal;
  const payload = response();
  const fetch = context.mock.method(
    globalThis,
    "fetch",
    async (url, options) => {
      assert.equal(new URL(url).pathname, "/retrieval/compare");
      assert.deepEqual(JSON.parse(options.body), request);
      assert.equal(options.signal, signal);
      return Response.json(payload);
    },
  );
  const result = await compareSources(request, signal);
  assert.equal(fetch.mock.callCount(), 1);
  assert.equal(
    result.arms[0].result.hits[0].chunk.original_text,
    chunk.original_text,
  );
  assert.equal(result.arms[1].error_code, "vector_index_not_ready");
  payload.arms[1].model_calls = 1;
  payload.arms[1].model_cost_usd = null;
  assert.equal(
    (await compareSources(request, signal)).arms[1].model_cost_usd,
    null,
  );
});

test("comparison rejects mixed questions, scopes, outcomes, and false known paid cost", async (context) => {
  let payload = response();
  context.mock.method(globalThis, "fetch", async () => Response.json(payload));
  const signal = new AbortController().signal;
  for (const change of [
    (value) => {
      value.query = "another question";
    },
    (value) => {
      value.scope = { ...request.scope, permitted_guides: ["s3-api"] };
    },
    (value) => {
      value.arms.reverse();
    },
    (value) => {
      value.arms[0].model_calls = 1;
    },
    (value) => {
      value.arms[1].result = value.arms[0].result;
    },
    (value) => {
      value.arms[0].result.hits[0].chunk.provenance.guide = "s3-api";
    },
    (value) => {
      value.arms[0].elapsed_ms = -1;
    },
  ]) {
    payload = structuredClone(response());
    change(payload);
    await assert.rejects(compareSources(request, signal), /invalid_response/);
  }
  await assert.rejects(
    compareSources({ ...request, modes: ["bm25", "bm25"] }, signal),
    /invalid_request/,
  );
});
