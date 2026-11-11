import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readSearchEvents, sourceLink } from "../app/search-api.ts";

const records = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/records.json", import.meta.url),
    "utf8",
  ),
);
const chunk = records.chunk;
const completed = {
  answer: `Versioning preserves objects. [${chunk.chunk_id}]`,
  citation_ids: [chunk.chunk_id],
  status: "answered",
  evidence: [chunk],
  timings_ms: { retrieval_total: 12 },
  usage_tokens: { generation_input: 10, generation_output: 5 },
  workflow: {
    snapshot_index: "test-snapshot",
    retrieval_passes: 2,
    model_calls: 3,
    assessment_prompt_version: "evidence-assessment-v1",
    steps: [
      {
        retrieval_pass: 1,
        query: "versioning",
        decision: "retry",
        missing_evidence: "missing_context",
      },
      {
        retrieval_pass: 2,
        query: "versioning delete marker",
        decision: "answer",
        missing_evidence: "none",
      },
    ],
  },
};
const frame = (event, data) =>
  `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;

function body(text, byteSize = 1) {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.length) return controller.close();
      controller.enqueue(bytes.slice(offset, offset + byteSize));
      offset += byteSize;
    },
  });
}

test("fragmented UTF-8 and CRLF frames preserve sources and require a validated terminal result", async () => {
  const wire =
    ": keepalive\r\n\r\n" +
    frame("status", { stage: "assessment" }) +
    frame("status", { stage: "refinement" }) +
    frame("sources", { evidence: [chunk] }) +
    frame("token", { text: "A café’s object" }) +
    frame("done", completed);
  const events = await Array.fromAsync(readSearchEvents(body(wire)));
  assert.deepEqual(
    events.map((event) => event.event),
    ["status", "status", "sources", "token", "done"],
  );
  assert.equal(events[3].data.text, "A café’s object");
  assert.equal(events[2].data.evidence[0].original_text, chunk.original_text);
  assert.deepEqual(events[4].data, completed);
});

test("a terminal error cancels the reader and cannot be followed by success", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          frame("error", { code: "invalid_citations" }) +
            frame("done", completed),
        ),
      );
    },
    cancel() {
      cancelled = true;
    },
  });
  const events = await Array.fromAsync(readSearchEvents(stream));
  assert.deepEqual(events, [
    {
      event: "error",
      data: { code: "invalid_citations", request_id: undefined },
    },
  ]);
  assert.equal(cancelled, true);
});

test("truncated streams retain their draft event without inventing completion", async () => {
  const events = [];
  await assert.rejects(async () => {
    for await (const event of readSearchEvents(
      body(frame("token", { text: "Partial answer" })),
    ))
      events.push(event);
  }, /incomplete_stream/);
  assert.deepEqual(
    events.map((event) => event.event),
    ["token"],
  );
});

test("unknown source citations and oversized drafts fail at the transport boundary", async () => {
  await assert.rejects(
    Array.fromAsync(
      readSearchEvents(
        body(
          frame("done", {
            ...completed,
            citation_ids: ["outside-the-supplied-evidence"],
          }),
          4096,
        ),
      ),
    ),
    /invalid_response/,
  );
  await assert.rejects(
    Array.fromAsync(
      readSearchEvents(
        body(frame("token", { text: "x".repeat(32_001) }), 4096),
      ),
    ),
    /invalid_response/,
  );
  assert.equal(sourceLink("javascript:alert(1)"), undefined);
});

test("leaving a stream early cancels its source rather than continuing background reads", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(frame("token", { text: "Draft" })),
      );
    },
    cancel() {
      cancelled = true;
    },
  });
  for await (const event of readSearchEvents(stream)) {
    assert.equal(event.event, "token");
    break;
  }
  assert.equal(cancelled, true);
});
