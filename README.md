# trace-session

A [Claude Code](https://claude.com/claude-code) skill that turns a Claude Code session into an OpenTelemetry trace and ships it to [Honeycomb](https://honeycomb.io) to see in **Agent Timeline**.

Each session becomes one trace. Each assistant turn is a `chat` span, each tool call an `execute_tool` span nested inside it, and each sub-agent spawn an `invoke_agent` span with its own lane named for what it did. Because it emits the OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), the trace lands natively in Honeycomb's Agent Timeline rather than as a generic waterfall.

```
invoke_agent claude-code                         root — whole session (tokens, models, turns, cwd, branch)
 └─ chat claude-opus-4-8                          one per assistant turn (input/output tokens, finish reason)
     ├─ execute_tool Bash                         one per tool call (args, result, duration, errors → span status)
     └─ invoke_agent general-purpose: Backfill…   Task/Agent spawns get their own lane, named by their job
```

## Install

Clone anywhere and copy (or symlink) the skill folder into your Claude Code skills directory:

```bash
git clone https://github.com/gsiener/trace-session.git
ln -s "$(pwd)/trace-session/trace-session" ~/.claude/skills/trace-session
```

Then in Claude Code: `/trace-session`.

## Use

No dependencies — stdlib Python 3. You can also run the converter directly:

```bash
# dry run (default): prints span-tree stats, writes the OTLP payload to $TMPDIR, sends nothing
python3 ~/.claude/skills/trace-session/convert.py

# browse recent sessions, then pick one by number / id / title
python3 ~/.claude/skills/trace-session/convert.py --list
python3 ~/.claude/skills/trace-session/convert.py "roadmap comments"

# send it (needs an ingest key)
export HONEYCOMB_API_KEY=hcaik_...        # Honeycomb → Environment settings → API keys
python3 ~/.claude/skills/trace-session/convert.py --send
```

With no argument it traces the **current** session (the most recently written transcript in the current project). Pass a selector — a session-id prefix, a title substring, or a path — to trace a specific one; add `--all` to reach sessions in other projects.

### Options

| Flag | Effect |
|------|--------|
| `--send` | POST to Honeycomb (default is a dry run) |
| `--list` | Print a numbered table of recent sessions and exit |
| `--all` | Scan every project, not just the current one |
| `--dataset <name>` | Target dataset / `service.name` (default `claude-code-sessions`) |
| `--no-messages` | Omit prompt/response/tool bodies — structure, tokens, and timings only |

## What lands in Honeycomb

Spans carry the GenAI conventions: `gen_ai.operation.name` (`chat` / `execute_tool` / `invoke_agent`), `gen_ai.conversation.id` (the session id), `gen_ai.agent.name` / `type` / `description`, `gen_ai.request/response.model`, `gen_ai.usage.{input,output,cache_read,cache_creation}_tokens`, `gen_ai.response.finish_reasons`, and `gen_ai.tool.name/call.id/call.arguments/call.result`. With bodies included, prompts and responses ride along as `gen_ai.input.messages` / `gen_ai.output.messages` span events. Tool errors set the span status to ERROR.

## Notes

- **Sending is append-only, not idempotent.** Honeycomb stores each span as an immutable event, so sending the same session twice duplicates its spans. Send each finished session once; stay in dry-run while iterating.
- **Spans are backdated** to when the work happened, so set your Honeycomb query time range to cover the session — a trace from last week won't show in the default "last 2 hours" view.
- **Region:** the endpoint defaults to US (`api.honeycomb.io`). For EU, change `OTLP_ENDPOINT` in `convert.py` to `https://api.eu1.honeycomb.io/v1/traces`.
- **Privacy:** use `--no-messages` to keep conversation content out of Honeycomb while still capturing the shape, tokens, and timings of a run.

## License

MIT — see [LICENSE](LICENSE).

---

Built with [Claude Code](https://claude.com/claude-code).
