# Domain glossary

The vocabulary this codebase is built from. Names here are load-bearing — use
them in code, commits, and future architecture reviews.

- **Transcript** — a Claude Code session as saved JSONL under
  `~/.claude/projects/<slug>/<session-id>.jsonl`. A flat list of rows
  (`user`, `assistant`, `attachment`, `ai-title`, …) linked by `parentUuid`.
  The raw input; never mutated.

- **Span tree** — the in-memory list of `SpanRecord`s built from a Transcript.
  This is *meaning*, before any wire format: what happened, in the session's
  own terms. Produced by `build_span_tree(rows)`.

- **SpanRecord** — one span described in its own terms: semantic attributes
  (the `gen_ai.*` keys), nanosecond times, parent link, optional status —
  *before* OTLP encoding. This is the **test surface**: assert on these.

- **OTLP encoder** — `to_otlp(records, trace_id, dataset)`. The only place that
  knows how Honeycomb receives spans (OTLP/HTTP JSON envelope, `intValue` as
  strings, resource attributes). Swappable without touching the Span tree.

- **Sub-agent lane** — a `Task`/`Agent` spawn rendered as its own
  `invoke_agent` span, named for what it did (`{subagent_type}: {description}`)
  so a fan-out reads as distinct agents rather than one blob. Sidechain turns
  nest under their spawning lane via `sourceToolUseID`.

- **Session catalog** — the discoverable set of Transcripts plus the rules for
  picking one (list / resolve a selector / current). Today this is spread
  across several functions in `convert.py`; consolidating it is a known
  deepening opportunity (see the architecture review).
