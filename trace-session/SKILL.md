---
name: trace-session
description: Send a Claude Code session to Honeycomb as an OpenTelemetry trace. Converts a session transcript (JSONL) into GenAI-convention spans (chat / execute_tool / invoke_agent) that land natively in Honeycomb's Agent Timeline. Use when the user wants to "trace this session", "send this session to Honeycomb", "observe my Claude Code run", or debug/analyze an agent session as a trace.
license: MIT
---

# trace-session

Turn a Claude Code session into a Honeycomb trace you can see in **Agent Timeline**. Each session becomes one trace; each assistant turn is a `chat` span, each tool call an `execute_tool` span nested inside it, and each sub-agent spawn an `invoke_agent` span with its own lane named for what it did.

The converter lives at `~/.claude/skills/trace-session/convert.py` (stdlib-only Python 3). It resolves sessions itself — you rarely need to hand it a path.

## When to use

The user says any of: "trace this session", "send this/that session to Honeycomb", "observe this run", "show this session as a trace", "put my Claude Code session in the Agent Timeline."

## What it emits (OTel GenAI semantic conventions)

| Transcript element | Span | `gen_ai.operation.name` |
|---|---|---|
| whole session | `invoke_agent claude-code` (root) | `invoke_agent` |
| each assistant message | `chat {model}` | `chat` |
| each tool call | `execute_tool {name}` | `execute_tool` |
| Task/Agent spawn | `invoke_agent {type}: {description}` | `invoke_agent` |

Key attrs: `gen_ai.conversation.id` (session id), `gen_ai.agent.name`, `gen_ai.agent.type`/`gen_ai.agent.description` (on spawns), `gen_ai.request/response.model`, `gen_ai.usage.{input,output,cache_read,cache_creation}_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.tool.name/call.id/call.arguments/call.result`. Prompts and responses ride as `gen_ai.input.messages` / `gen_ai.output.messages` span events. Tool errors set span status ERROR.

**Sub-agents get their own lane.** Each Task/Agent spawn becomes an `invoke_agent` span whose `gen_ai.agent.name` is `{subagent_type}: {description}` (e.g. `general-purpose: Backfill batch A`), so a fan-out shows as distinct agents rather than one blob. When a session has inline sidechain turns, those turns are linked back to their spawning Task (via `sourceToolUseID`) and nested under that agent — so they read as the sub-agent, not the orchestrator.

**Sending is append-only — NOT idempotent.** Honeycomb stores each span as an immutable event; there is no upsert on `trace_id`+`span_id`. Sending the same session twice **duplicates every span** in the trace. So: send each finished session **once**. If you're iterating on the converter, stay in `--dry-run` and only `--send` when done; to get a clean view after test sends, reset the dataset in the Honeycomb UI (or send to a fresh `--dataset`). `trace_id` is still derived from the session UUID so a session maps to a stable, findable trace id.

## Workflow

### 1. Pick the session (the script resolves it)

- **Current session** (default) — run with **no session argument**; it uses the most recently written transcript in the current project.
- **A specific past session** — pass a selector as the first arg: a session-id prefix (`5ce4`), a title substring (`"roadmap comments"`), a slug, or a full path. Add `--all` to search every project, not just the current one.
- **Browse** — `--list` (add `--all` for every project) prints a numbered table (age, title, id, size), newest first. Show it to the user, let them pick by number/title/id, then send that one.

```bash
python3 ~/.claude/skills/trace-session/convert.py --list            # recognizable menu
python3 ~/.claude/skills/trace-session/convert.py --list --all      # across all projects
```

### 2. Dry run first (always)

```bash
python3 ~/.claude/skills/trace-session/convert.py                     # current session
python3 ~/.claude/skills/trace-session/convert.py <selector>          # a chosen session
```
Prints the span tree stats (turns / chats / tools / tokens / duration) and writes the OTLP payload to `$TMPDIR` — **sends nothing**. Show the user these stats and confirm before sending.

### 3. Send to Honeycomb

Needs an ingest key in `HONEYCOMB_API_KEY`. If it isn't set, tell the user to run (they can paste `!` in the prompt to run it in-session):
```bash
export HONEYCOMB_API_KEY=hcaik_...   # an ingest key from Honeycomb → Environment settings → API keys
```
Then add `--send` (append it to the same invocation you dry-ran):
```bash
python3 ~/.claude/skills/trace-session/convert.py <selector> --send
```
Send **once** per session (see the append-only note above).

Options:
- `--dataset <name>` — target dataset / `service.name` (default `claude-code-sessions`; env `HONEYCOMB_DATASET` also works).
- `--no-messages` — omit prompt/response/tool bodies (leaner + keeps content out of Honeycomb; keeps structure + tokens + timings).

### 4. Point the user at the trace

After a successful send, print the dataset name and `trace.trace_id`. The user finds it in Honeycomb: open the dataset → **Agent Timeline** (or query `trace.trace_id = <id>` for the raw waterfall). Do not fabricate a deep-link URL — the team/environment slug isn't known from the key.

**Tell them to widen the time range.** Spans are backdated to when the work actually happened, so a session from yesterday/last week will NOT appear in Honeycomb's default "last 2 hours" view — it looks like nothing sent. The send output prints the span time range; point the user at it and have them set the query window to cover it.

## Notes

- Stdlib-only Python 3, no dependencies. Endpoint: `https://api.honeycomb.io/v1/traces` (OTLP/HTTP JSON).
- EU or self-serve region: the endpoint is hard-coded to US; for EU change `OTLP_ENDPOINT` in `convert.py` to `https://api.eu1.honeycomb.io/v1/traces`.
- Large sessions produce thousands of spans; that's fine for Honeycomb ingest but use `--no-messages` if you want to keep transcript content out.
- Timing model: `chat` span spans from when input became available (the prior prompt/tool-result timestamp) to when the last of its tool results returned; `llm.generation_ms` isolates just the generation latency. Timestamps come from transcript log times, so treat sub-second latencies as approximate.
