"""Chunk cleaned source text without rewriting evidence or performing I/O.

Whitespace-based sizes are estimates, not Bedrock token counts. Small recognized
examples stay intact; larger examples retain ordered continuation metadata.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

from rag.config import ChunkingConfig
from rag.models import ChunkRecord, ExampleContinuation, SectionRecord

CHUNKER_VERSION = "chunks-v1"
BOUNDARY_PREFERENCE_FRACTION = 0.6
TOKEN = re.compile(r"\S+")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
HEADING = re.compile(r"^#{1,6}\s+\S")
SENTENCE_END = re.compile(r"[.!?][ \t\r\n]+(?=[A-Z0-9\"'\[(])")


@dataclass(frozen=True)
class TextBlock:
    """An exact cleaned-text span and its recognized formatting role."""

    start: int
    end: int
    kind: Literal["example", "heading", "whitespace", "prose"]


@dataclass(frozen=True)
class _ExampleSpan:
    block: TextBlock
    token_start: int
    token_end: int


def _sha256(content: bytes) -> str:
    """Keep deterministic identities independent of the filesystem layer."""
    return hashlib.sha256(content).hexdigest()


def _example_end(text: str, lines: list[str], starts: list[int], index: int) -> int | None:
    """Recognize explicit examples; do not guess a language for ordinary prose."""
    line = lines[index]
    fence = FENCE.match(line)
    if fence:
        delimiter = fence.group(1)
        for end in range(index + 1, len(lines)):
            if re.fullmatch(
                rf"[ \t]*{re.escape(delimiter[0])}{{{len(delimiter)},}}[ \t\r\n]*", lines[end]
            ):
                return end + 1
        return len(lines)  # Preserve an unfinished fenced example to EOF.
    if line.startswith(("    ", "\t")) and line.strip():
        end = index + 1
        while end < len(lines) and (
            lines[end].startswith(("    ", "\t")) or not lines[end].strip()
        ):
            end += 1
        return end
    stripped = line.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            _, end = json.JSONDecoder().raw_decode(text, starts[index] + len(line) - len(stripped))
        except ValueError:
            return None
        end_line = bisect.bisect_left(starts, end)
        if end_line == len(starts) or not text[end : starts[end_line]].strip():
            return end_line
    if re.match(r"<[A-Za-z_][\w:.-]*(?:\s|>|/)", stripped):
        parser = ET.XMLPullParser(events=("start", "end"))
        depth = 0
        seen_root = False
        try:
            for end in range(index, len(lines)):
                parser.feed(lines[end])
                for event, _ in parser.read_events():
                    seen_root = True
                    depth += 1 if event == "start" else -1
                if seen_root and depth == 0:
                    return end + 1
        except ET.ParseError:
            return None
    return None


def _example_spans(blocks: list[TextBlock], token_starts: list[int]) -> list[_ExampleSpan]:
    """Map recognized examples to half-open whitespace-token ranges."""
    return [
        _ExampleSpan(
            block=block,
            token_start=bisect.bisect_left(token_starts, block.start),
            token_end=bisect.bisect_left(token_starts, block.end),
        )
        for block in blocks
        if block.kind == "example"
    ]


def _chunker_version(section: SectionRecord, config: ChunkingConfig) -> str:
    """Fingerprint settings and cleaning lineage for reproducible chunk identities."""
    identity = {
        "algorithm": CHUNKER_VERSION,
        "settings": config.model_dump(),
        "cleaning_version": section.cleaning_version,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return f"{CHUNKER_VERSION}-{_sha256(serialized.encode('utf-8'))[:16]}"


def _chunk_spans(
    text: str,
    token_starts: list[int],
    blocks: list[TextBlock],
    examples: list[_ExampleSpan],
    config: ChunkingConfig,
) -> list[tuple[int, int]]:
    """Choose advancing, size-bounded windows while protecting short examples."""
    protected = [
        example for example in examples if example.token_end - example.token_start <= config.size
    ]
    boundaries = sorted({bisect.bisect_left(token_starts, block.end) for block in blocks})
    heading_starts = [
        bisect.bisect_left(token_starts, block.start) for block in blocks if block.kind == "heading"
    ]
    heading_ends = {
        bisect.bisect_left(token_starts, block.end) for block in blocks if block.kind == "heading"
    }
    # A heading alone is not useful evidence; prefer a boundary after its content.
    boundaries = [boundary for boundary in boundaries if boundary not in heading_ends]
    spans: list[tuple[int, int]] = []
    token_count = len(token_starts)
    start_token, previous_end_token = 0, 0
    while start_token < token_count:
        end_token = min(start_token + config.size, token_count)
        if config.strategy == "heading-aware":
            headings = [value for value in heading_starts if start_token < value < end_token]
            if headings:
                end_token = headings[0]
            elif end_token < token_count:
                minimum_boundary = (
                    start_token + (end_token - start_token) * BOUNDARY_PREFERENCE_FRACTION
                )
                preferred = [
                    value for value in boundaries if minimum_boundary <= value <= end_token
                ]
                if preferred:
                    end_token = preferred[-1]
        for example in protected:
            if example.token_start < end_token < example.token_end:
                if example.token_start > start_token:
                    end_token = example.token_start
                else:
                    end_token = example.token_end
        if end_token <= previous_end_token:
            # Drop overlap if it would produce no new evidence.
            start_token = previous_end_token
            continue
        start_char = 0 if start_token == 0 else token_starts[start_token]
        for example in examples:
            if example.token_start == start_token:
                start_char = min(start_char, example.block.start)
        end_char = len(text) if end_token == token_count else token_starts[end_token]
        spans.append((start_char, end_char))
        if end_token == token_count:
            break
        previous_end_token = end_token
        previous_start_token = start_token
        start_token = max(start_token + 1, end_token - config.overlap)
        for example in protected:
            if example.token_start < start_token < example.token_end:
                if example.token_start > previous_start_token:
                    start_token = example.token_start
                else:
                    start_token = example.token_end
    return spans


def _build_chunk(section: SectionRecord, start: int, end: int, version: str) -> ChunkRecord:
    """Bind an exact cleaned-text substring to its source and build identity."""
    original_text = section.text[start:end]
    content_hash = _sha256(original_text.encode("utf-8"))
    identity = {
        "source": section.provenance.sha256,
        "source_id": section.source_id,
        "revision": section.provenance.revision,
        "section_id": section.section_id,
        "corpus_id": section.corpus_id,
        "source_scope": section.source_scope,
        "heading_path": section.heading_path,
        "version": version,
        "text_start": start,
        "text_end": end,
        "content_sha256": content_hash,
    }
    chunk_id = "chunk-" + _sha256(json.dumps(identity, sort_keys=True).encode("utf-8"))
    return ChunkRecord(
        chunk_id=chunk_id,
        section_id=section.section_id,
        corpus_id=section.corpus_id,
        original_text=original_text,
        source_scope=section.source_scope,
        heading_path=section.heading_path,
        section_raw_start=section.raw_start,
        section_raw_end=section.raw_end,
        chunker_version=version,
        provenance=section.provenance,
        text_start=start,
        text_end=end,
        content_sha256=content_hash,
        estimated_tokens=len(TOKEN.findall(original_text)),
    )


def _attach_continuations(
    section: SectionRecord,
    chunks: list[ChunkRecord],
    spans: list[tuple[int, int]],
    tokens: list[re.Match[str]],
    examples: list[_ExampleSpan],
    config: ChunkingConfig,
    version: str,
) -> None:
    """Attach ordered fragment metadata without changing any source text."""
    for example in examples:
        if example.token_end - example.token_start <= config.size:
            continue
        meaningful_start = tokens[example.token_start].start()
        meaningful_end = tokens[example.token_end - 1].end()
        members = [
            index
            for index, (start, end) in enumerate(spans)
            if start < meaningful_end and end > meaningful_start
        ]
        if len(members) < 2:
            continue
        identity = (
            f"{section.corpus_id}:{section.provenance.sha256}:"
            f"{section.section_id}:{version}:{example.block.start}:{example.block.end}"
        )
        block_id = "example-" + _sha256(identity.encode("utf-8"))[:24]
        for part_index, chunk_index in enumerate(members, 1):
            chunks[chunk_index].continuations.append(
                ExampleContinuation(
                    block_id=block_id, part_index=part_index, part_count=len(members)
                )
            )


def split_blocks(text: str) -> list[TextBlock]:
    """Partition exact text spans into examples, headings, prose, and whitespace."""
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    blocks = []
    index = 0
    while index < len(lines):
        end = _example_end(text, lines, starts, index)
        if end is not None:
            blocks.append(TextBlock(starts[index], starts[end], "example"))
            index = end
            continue
        if HEADING.match(lines[index]):
            blocks.append(TextBlock(starts[index], starts[index + 1], "heading"))
            index += 1
            continue
        if not lines[index].strip():
            blocks.append(TextBlock(starts[index], starts[index + 1], "whitespace"))
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip():
            if HEADING.match(lines[end]) or _example_end(text, lines, starts, end) is not None:
                break
            end += 1
        start_char, end_char = starts[index], starts[end]
        cursor = start_char
        for match in SENTENCE_END.finditer(text, start_char, end_char):
            blocks.append(TextBlock(cursor, match.end(), "prose"))
            cursor = match.end()
        if cursor < end_char:
            blocks.append(TextBlock(cursor, end_char, "prose"))
        index = end
    return blocks


def chunk_section(section: SectionRecord, config: ChunkingConfig) -> list[ChunkRecord]:
    """Return exact source substrings with reproducible IDs and explicit fragment metadata."""
    tokens = list(TOKEN.finditer(section.text))
    if not tokens:
        return []
    token_starts = [token.start() for token in tokens]
    blocks = split_blocks(section.text)
    examples = _example_spans(blocks, token_starts)
    version = _chunker_version(section, config)
    spans = _chunk_spans(section.text, token_starts, blocks, examples, config)
    chunks = [_build_chunk(section, start, end, version) for start, end in spans]
    _attach_continuations(section, chunks, spans, tokens, examples, config, version)
    return chunks


def chunk_inventory(
    sections: list[SectionRecord], chunks: list[ChunkRecord], config: ChunkingConfig
) -> dict[str, str | int | float]:
    """Summarize estimated sizes and continuation usage without performing I/O."""
    sizes = sorted(len(TOKEN.findall(chunk.original_text)) for chunk in chunks)
    return {
        "sections": len(sections),
        "chunks": len(chunks),
        "counting_method": "whitespace-estimate",
        "strategy": config.strategy,
        "size": config.size,
        "overlap": config.overlap,
        "estimated_tokens_min": sizes[0] if sizes else 0,
        "estimated_tokens_median": statistics.median(sizes) if sizes else 0,
        "estimated_tokens_max": sizes[-1] if sizes else 0,
        "continuation_chunks": sum(bool(chunk.continuations) for chunk in chunks),
        "oversized_chunks": sum(size > config.size for size in sizes),
    }
