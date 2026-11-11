import { sourceLink, type Chunk } from "./search-api";

export function SourceInspector({ chunk }: { chunk: Chunk }) {
  const snapshot = sourceLink(chunk.provenance.repository_url);
  const publisher = sourceLink(chunk.provenance.publisher_url);
  return (
    <>
      <div className="passage-heading">
        <h3>{chunk.heading_path.at(-1)}</h3>
        <p>
          {chunk.provenance.guide === "s3-api" ? "API reference" : "User guide"}{" "}
          · General-purpose buckets
        </p>
      </div>
      <pre className="passage">{chunk.original_text}</pre>
      <details className="provenance">
        <summary>Source details</summary>
        <dl>
          <dt>Section</dt>
          <dd>{chunk.heading_path.join(" / ")}</dd>
          <dt>Corpus</dt>
          <dd>{chunk.corpus_id}</dd>
          <dt>Chunk</dt>
          <dd>{chunk.chunk_id}</dd>
          <dt>Snapshot revision</dt>
          <dd>{chunk.provenance.revision}</dd>
          <dt>File SHA-256</dt>
          <dd>{chunk.provenance.sha256}</dd>
          <dt>Section raw characters</dt>
          <dd>
            {chunk.section_raw_start}–{chunk.section_raw_end}
          </dd>
        </dl>
      </details>
      <div className="source-links">
        {snapshot && (
          <a href={snapshot} target="_blank" rel="noreferrer">
            Pinned source snapshot
          </a>
        )}
        {publisher && (
          <a href={publisher} target="_blank" rel="noreferrer">
            Current AWS documentation
          </a>
        )}
      </div>
      <p className="source-note">
        Passage text comes from the pinned snapshot. The current AWS page may
        have changed.
      </p>
    </>
  );
}
