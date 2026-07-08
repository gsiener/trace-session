#!/usr/bin/env python3
"""
Convert a Claude Code session transcript (JSONL) into an OpenTelemetry trace
and (optionally) ship it to Honeycomb over OTLP/HTTP.

Emits OTel GenAI semantic conventions so the trace lands natively in
Honeycomb's Agent Timeline:
  - root         -> invoke_agent claude-code   (gen_ai.operation.name=invoke_agent)
  - assistant    -> chat {model}               (gen_ai.operation.name=chat)
  - tool_use     -> execute_tool {name}        (gen_ai.operation.name=execute_tool)
  - Task/Agent   -> invoke_agent {subagent}    (promoted, so sub-agents get a lane)

Stdlib only. Defaults to --dry-run (writes payload + prints tree; sends nothing).

Usage:
  convert.py <session.jsonl>                 # dry run: stats + payload to scratchpad
  convert.py <session.jsonl> --send          # POST to Honeycomb (needs HONEYCOMB_API_KEY)
  convert.py <session.jsonl> --dataset foo    # override dataset / service.name
  convert.py <session.jsonl> --no-messages    # omit prompt/response bodies (leaner, private)
"""
import argparse, hashlib, json, os, sys, urllib.request, urllib.error
from dataclasses import dataclass
from datetime import datetime

OTLP_ENDPOINT = "https://api.honeycomb.io/v1/traces"
DEFAULT_DATASET = "claude-code-sessions"
GEN_AI_SYSTEM = "anthropic"
MAX_STR = 4000          # truncate long attribute strings
MAX_MSG = 12000         # truncate message-body events

# ---- OTLP value/attribute builders ----------------------------------------

def _val(v):
    if isinstance(v, bool):   return {"boolValue": v}
    if isinstance(v, int):    return {"intValue": str(v)}
    if isinstance(v, float):  return {"doubleValue": v}
    if isinstance(v, list):   return {"arrayValue": {"values": [_val(x) for x in v]}}
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    if len(s) > MAX_STR:
        s = s[:MAX_STR] + f"...[+{len(s)-MAX_STR} chars]"
    return {"stringValue": s}

def attrs(d):
    return [{"key": k, "value": _val(v)} for k, v in d.items() if v is not None]

# ---- id / time helpers ------------------------------------------------------

def trace_id(session_id):
    h = session_id.replace("-", "")
    return h if len(h) == 32 else hashlib.sha1(session_id.encode()).hexdigest()[:32]

def span_id(seed):
    return hashlib.sha1(seed.encode()).hexdigest()[:16]

def nanos(iso):
    if not iso: return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1e9)
    except Exception:
        return None

FINISH = {"tool_use": ["tool_calls"], "end_turn": ["stop"],
          "max_tokens": ["length"], "stop_sequence": ["stop"]}

def truncate(s, n=MAX_MSG):
    s = s if isinstance(s, str) else json.dumps(s, default=str)
    return s if len(s) <= n else s[:n] + f"...[+{len(s)-n} chars]"

# ---- transcript parsing -----------------------------------------------------

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rows

def is_user_prompt(r):
    """A real human prompt, not a tool_result or an injected system/meta message."""
    if r.get("type") != "user" or r.get("isMeta"): return False
    c = r.get("message", {}).get("content")
    if isinstance(c, str):
        t = c.lstrip()
        return bool(t) and not t.startswith("<local-command") and not t.startswith("<system-reminder")
    if isinstance(c, list):
        # a prompt made of text blocks with no tool_result
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return False
        return any(isinstance(b, dict) and b.get("type") == "text" for b in c)
    return False

def prompt_text(r):
    c = r.get("message", {}).get("content")
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""

def result_text(tool_result_block):
    c = tool_result_block.get("content")
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c
                         if isinstance(b, dict) and b.get("type") == "text")
    return json.dumps(c, default=str) if c is not None else ""

# ---- core: transcript -> span tree -> OTLP ---------------------------------
#
# Two seams, so meaning is separable from wire format:
#   build_span_tree(rows) -> ([SpanRecord], stats)   what happened (pure, testable)
#   to_otlp(records, ...)  -> OTLP payload            how Honeycomb receives it
# build_spans() composes them for callers that just want a payload.

@dataclass
class SpanRecord:
    """One span, described in its own terms (semantic attributes, ns times) —
    before any OTLP encoding. This is the test surface: assert on these."""
    span_id: str
    parent_id: str | None          # None == trace root
    name: str
    kind: int
    start_ns: int
    end_ns: int
    attributes: dict               # raw values; encoded to OTLP later
    status: dict | None = None

def build_spans(rows, dataset, include_messages=True):
    """Convenience composition: transcript rows -> (OTLP payload, stats)."""
    records, stats = build_span_tree(rows, include_messages=include_messages)
    stats["dataset"] = dataset
    return to_otlp(records, stats["trace_id"], dataset), stats

def build_span_tree(rows, include_messages=True):
    session_id = next((r.get("sessionId") for r in rows if r.get("sessionId")), "unknown")
    tid = trace_id(session_id)
    by_uuid = {r["uuid"]: r for r in rows if r.get("uuid")}

    # map every tool_use id -> its result row (from a later user tool_result)
    tool_result = {}
    for r in rows:
        if r.get("type") != "user": continue
        c = r.get("message", {}).get("content")
        if not isinstance(c, list): continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tool_result[b.get("tool_use_id")] = (r, b)

    assistants = [r for r in rows if r.get("type") == "assistant"
                  and isinstance(r.get("message", {}).get("content"), list)]
    all_ts = [nanos(r.get("timestamp")) for r in rows if nanos(r.get("timestamp"))]
    t_start, t_end = (min(all_ts), max(all_ts)) if all_ts else (0, 0)

    # Map each Task/Agent spawn to a readable agent label, so sub-agents show as
    # their own lane (named by what they did) instead of all being "claude-code".
    task_agent = {}
    for a in assistants:
        for b in a.get("message", {}).get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in ("Task", "Agent"):
                task_agent[b["id"]] = task_label(b)[2]

    def source_tool_id(a):
        """Walk up from a sidechain row to the Task tool_use that spawned it."""
        r = a
        for _ in range(8):
            if not r: break
            if r.get("sourceToolUseID"): return r["sourceToolUseID"]
            r = by_uuid.get(r.get("parentUuid"))
        return None

    def triggering_prompt(a):
        """The user prompt that triggered this chat, if any. The immediate parent is
        usually an injected 'attachment' row, so skip those (and meta) — but stop at
        a tool-result/assistant, since a mid-turn chat's input is the tool result,
        not the original prompt."""
        r = by_uuid.get(a.get("parentUuid"))
        for _ in range(5):
            if not r: return None
            if r.get("type") == "attachment" or r.get("isMeta"):
                r = by_uuid.get(r.get("parentUuid")); continue
            return r if is_user_prompt(r) else None
        return None

    root_meta = next((r for r in rows if r.get("cwd")), {})
    root_id = span_id("root:" + session_id)
    spans = []
    tot_in = tot_out = tot_cache_r = tot_cache_w = 0
    tool_count = 0
    models = set()

    for a in assistants:
        m = a.get("message", {})
        model = m.get("model", "unknown")
        if model == "<synthetic>":   # Claude Code's injected non-LLM turns; not real chat calls
            continue
        models.add(model)
        usage = m.get("usage", {}) or {}
        # Anthropic reports input_tokens as the UNCACHED delta only (often tiny);
        # real context = uncached + cache-read + cache-creation. Sum them so the
        # Agent Timeline token panels reflect true context size (matches hny).
        uncached_in = usage.get("input_tokens")
        cache_r = usage.get("cache_read_input_tokens") or 0
        cache_w = usage.get("cache_creation_input_tokens") or 0
        context_in = (uncached_in + cache_r + cache_w) if uncached_in is not None else None
        tot_in    += context_in or 0
        tot_out   += usage.get("output_tokens", 0) or 0
        tot_cache_r += cache_r
        tot_cache_w += cache_w

        parent = by_uuid.get(a.get("parentUuid"))
        a_ts = nanos(a.get("timestamp"))
        in_ts = nanos(parent.get("timestamp")) if parent else a_ts   # when input became available
        chat_start = in_ts or a_ts
        sidechain = a.get("isSidechain", False)
        # a sub-agent's own turns are named for the agent that was spawned, and
        # nested under that invoke_agent span; the orchestrator's turns are claude-code.
        src_tool = source_tool_id(a) if sidechain else None
        agent_name = task_agent.get(src_tool, "subagent") if sidechain else "claude-code"
        chat_parent = span_id("tool:" + src_tool) if (sidechain and src_tool in task_agent) else root_id
        chat_id = span_id("chat:" + a["uuid"])

        tool_uses = [b for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        assistant_text = "\n".join(b.get("text", "") for b in m["content"]
                                   if isinstance(b, dict) and b.get("type") == "text")

        # tool child spans; track their end to size the chat span
        child_spans, child_ends = [], []
        for tu in tool_uses:
            tool_count += 1
            name = tu.get("name", "tool")
            res_row, res_block = tool_result.get(tu.get("id"), (None, None))
            r_ts = nanos(res_row.get("timestamp")) if res_row else None
            tstart = a_ts or chat_start
            tend = r_ts or tstart
            child_ends.append(tend)
            is_err = bool(res_block.get("is_error")) if res_block else False

            # promote Task/Agent tool calls to invoke_agent so sub-agents get their
            # own lane, named for what they did (subagent_type + description).
            sub_type = sub_desc = sub_label = None
            if name in ("Task", "Agent"):
                sub_type, sub_desc, sub_label = task_label(tu)
                op, sp_name = "invoke_agent", f"invoke_agent {sub_label}"
            else:
                op, sp_name = "execute_tool", f"execute_tool {name}"

            ta = {
                "gen_ai.operation.name": op,
                "gen_ai.system": GEN_AI_SYSTEM,
                "gen_ai.conversation.id": session_id,
                "gen_ai.agent.name": sub_label if op == "invoke_agent" else agent_name,
                "gen_ai.agent.type": sub_type,
                "gen_ai.agent.description": sub_desc,
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": tu.get("id"),
                "gen_ai.tool.type": "agent" if op == "invoke_agent" else "function",
                "duration_ms": int((tend - tstart) / 1e6) if (tend and tstart) else None,
                "mcp.server": a.get("attributionMcpServer"),
                "mcp.tool": a.get("attributionMcpTool"),
                "claude.skill": a.get("attributionSkill"),
                "claude.plugin": a.get("attributionPlugin"),
            }
            if include_messages:
                ta["gen_ai.tool.call.arguments"] = truncate(tu.get("input", {}))
                if res_block is not None:
                    ta["gen_ai.tool.call.result"] = truncate(result_text(res_block))
            child_spans.append(SpanRecord(
                span_id=span_id("tool:" + tu.get("id", tu_fallback(tu, a))),
                parent_id=chat_id, name=sp_name, kind=1,
                start_ns=tstart, end_ns=tend, attributes=ta,
                status={"code": 2, "message": "tool returned is_error"} if is_err else None,
            ))

        chat_end = max([a_ts or chat_start] + child_ends) if child_ends else (a_ts or chat_start)
        ca = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": GEN_AI_SYSTEM,
            "gen_ai.conversation.id": session_id,
            "gen_ai.agent.name": agent_name,
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": context_in,                 # total context (uncached + cache) — for Timeline fidelity
            "gen_ai.usage.uncached_input_tokens": uncached_in,       # raw semconv input_tokens, preserved
            "gen_ai.usage.output_tokens": usage.get("output_tokens"),
            "gen_ai.usage.cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "gen_ai.usage.cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "gen_ai.response.finish_reasons": FINISH.get(m.get("stop_reason"), [m.get("stop_reason")] if m.get("stop_reason") else None),
            "gen_ai.response.id": m.get("id"),
            "llm.generation_ms": int(((a_ts or chat_start) - chat_start) / 1e6) if a_ts else None,
            "duration_ms": int((chat_end - chat_start) / 1e6) if chat_end else None,
            "tool.count": len(tool_uses),
            "session.is_sidechain": sidechain,
        }
        # Messages ride as span ATTRIBUTES (not events) so Honeycomb's Agent
        # Timeline renders them in the span's Messages panel, not a separate
        # Span Events tab.
        if include_messages:
            prompt_row = triggering_prompt(a)
            if prompt_row:
                ca["gen_ai.input.messages"] = truncate(prompt_text(prompt_row))
            if assistant_text.strip():
                ca["gen_ai.output.messages"] = truncate(assistant_text)
        chat_status = None
        if a.get("isApiErrorMessage") or a.get("error"):
            chat_status = {"code": 2, "message": truncate(str(a.get("error") or "api error"), 300)}
        spans.append(SpanRecord(
            span_id=chat_id, parent_id=chat_parent, name=f"chat {model}", kind=3,
            start_ns=chat_start, end_ns=chat_end, attributes=ca, status=chat_status,
        ))
        spans.extend(child_spans)

    prompts = [r for r in rows if is_user_prompt(r)]
    root_attrs = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.system": GEN_AI_SYSTEM,
        "gen_ai.agent.name": "claude-code",
        "gen_ai.conversation.id": session_id,
        "gen_ai.usage.input_tokens": tot_in,
        "gen_ai.usage.output_tokens": tot_out,
        "gen_ai.usage.cache_read_input_tokens": tot_cache_r,
        "gen_ai.usage.cache_creation_input_tokens": tot_cache_w,
        "session.id": session_id,
        "session.turns": len(prompts),
        "session.assistant_steps": len(assistants),
        "session.tool_calls": tool_count,
        "session.models": sorted(models),
        "session.cwd": root_meta.get("cwd"),
        "session.git_branch": root_meta.get("gitBranch"),
        "session.cli_version": root_meta.get("version"),
        "session.entrypoint": root_meta.get("entrypoint"),
        "duration_ms": int((t_end - t_start) / 1e6) if t_end else None,
    }
    spans.insert(0, SpanRecord(
        span_id=root_id, parent_id=None, name="invoke_agent claude-code", kind=1,
        start_ns=t_start, end_ns=t_end, attributes=root_attrs,
    ))

    stats = {
        "session_id": session_id, "trace_id": tid,
        "spans": len(spans), "turns": len(prompts), "chats": len(assistants),
        "tools": tool_count, "tokens_in": tot_in, "tokens_out": tot_out,
        "models": sorted(models),
        "duration_s": round((t_end - t_start) / 1e9, 1) if t_end else 0,
        "oldest_epoch": int(t_start / 1e9) if t_start else 0,
        "newest_epoch": int(t_end / 1e9) if t_end else 0,
    }
    return spans, stats

def to_otlp(records, trace_id, dataset):
    """Encode SpanRecords into an OTLP/HTTP JSON payload for Honeycomb."""
    spans = []
    for r in records:
        s = {"traceId": trace_id, "spanId": r.span_id}
        if r.parent_id is not None:
            s["parentSpanId"] = r.parent_id
        s["name"] = r.name
        s["kind"] = r.kind
        s["startTimeUnixNano"] = str(r.start_ns)
        s["endTimeUnixNano"] = str(r.end_ns)
        s["attributes"] = attrs(r.attributes)
        if r.status is not None:
            s["status"] = r.status
        spans.append(s)
    return {"resourceSpans": [{
        "resource": {"attributes": attrs({
            "service.name": dataset, "gen_ai.system": GEN_AI_SYSTEM,
            "telemetry.sdk.name": "claude-code-trace-session",
            "telemetry.sdk.language": "python",
        })},
        "scopeSpans": [{
            "scope": {"name": "claude-code.session-trace", "version": "1.0.0"},
            "spans": spans,
        }],
    }]}

def tu_fallback(tu, a):
    return (tu.get("id") or "") + a.get("uuid", "")

def task_label(tu):
    """A readable agent identity for a Task/Agent spawn: '<type>: <what it did>'."""
    inp = tu.get("input", {}) or {}
    subtype = inp.get("subagent_type") or "subagent"
    desc = " ".join((inp.get("description") or "").split())[:48]
    return subtype, desc, (f"{subtype}: {desc}" if desc else subtype)

# ---- session discovery / selection -----------------------------------------

PROJECTS = os.path.expanduser("~/.claude/projects")

def project_dir_for_cwd():
    import re
    # Claude Code names the project folder by replacing every non-alphanumeric
    # char in the cwd with '-' (so /, ., @, and spaces all become '-').
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.getcwd())
    d = os.path.join(PROJECTS, slug)
    return d if os.path.isdir(d) else None

def scan_dirs(all_projects):
    import glob
    if all_projects:
        return [d for d in glob.glob(os.path.join(PROJECTS, "*")) if os.path.isdir(d)]
    d = project_dir_for_cwd()
    return [d] if d else [x for x in glob.glob(os.path.join(PROJECTS, "*")) if os.path.isdir(x)]

def human_age(secs):
    if secs < 90: return "just now"
    for unit, n in (("m", 60), ("h", 3600), ("d", 86400)):
        if secs < n * (60 if unit == "m" else 24 if unit == "h" else 3650):
            return f"{int(secs / n)}{unit} ago"
    return f"{int(secs/86400)}d ago"

def human_size(b):
    for u in ("B", "K", "M", "G"):
        if b < 1024: return f"{b:.0f}{u}" if u == "B" else f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}T"

import re as _re

def clean_label(text):
    """Strip slash-command wrapper tags so a fallback label reads cleanly."""
    text = _re.sub(r"<command-(message|name|args)>.*?</command-\1>", " ", text, flags=_re.S)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text[:70] if text else "(command)"

def peek(path, max_lines=300):
    """Cheap metadata read: title/first-prompt from the head of the file.
    (slug lives too deep in newer transcripts to read cheaply; id comes from the filename.)"""
    slug = title = branch = cwd = prompt = None
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i > max_lines: break
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                slug = slug or r.get("slug")
                branch = branch or r.get("gitBranch")
                cwd = cwd or r.get("cwd")
                if r.get("type") == "ai-title" and not title:
                    title = r.get("aiTitle")
                if not prompt and is_user_prompt(r):
                    prompt = clean_label(prompt_text(r))
                if title and prompt: break
    except OSError:
        pass
    return {"slug": slug, "title": title, "branch": branch, "cwd": cwd, "prompt": prompt}

@dataclass
class Session:
    """One discoverable transcript. Cheap fields come from the filename + stat;
    title/prompt/slug/branch/cwd are filled by peek() only when needed."""
    path: str
    id: str                # filename minus ".jsonl" == the session id
    mtime: float
    size: int
    project: str           # short project name
    title: str | None = None
    prompt: str | None = None
    slug: str | None = None
    branch: str | None = None
    cwd: str | None = None

    def age(self, now):
        return human_age(now - self.mtime) if now else ""

    @classmethod
    def at(cls, path, peeked=False):
        st = os.stat(path)
        s = cls(path=path, id=os.path.basename(path)[:-6], mtime=st.st_mtime,
                size=st.st_size,
                project=os.path.basename(os.path.dirname(path)).rsplit("-", 1)[-1])
        if peeked:
            for k, v in peek(path).items():
                setattr(s, k, v)
        return s

@dataclass
class Resolution:
    """Outcome of resolving a selector. status: 'found' | 'ambiguous' | 'notfound'.
    A typed result so the caller doesn't decode an untyped (path, hits) pair."""
    status: str
    session: "Session | None" = None
    candidates: list | None = None

    @property
    def found(self):
        return self.status == "found"

class SessionCatalog:
    """The discoverable session transcripts + the rules for picking one, in one
    place. Pass explicit `dirs` to point it at a fixture directory (tests);
    otherwise it scans ~/.claude/projects (current project, or all)."""
    def __init__(self, all_projects=False, now=0, dirs=None):
        self._dirs = dirs if dirs is not None else scan_dirs(all_projects)
        self.now = now

    def _paths(self):
        import glob
        fs = []
        for d in self._dirs:
            for p in glob.glob(os.path.join(d, "*.jsonl")):
                try: fs.append((os.stat(p).st_mtime, p))
                except OSError: pass
        fs.sort(reverse=True)
        return [p for _, p in fs]   # newest first

    def current(self):
        ps = self._paths()
        return ps[0] if ps else None

    def list(self, limit=25):
        # stat+sort is cheap; peek only the top N so a huge history stays fast
        return [Session.at(p, peeked=True) for p in self._paths()[:limit]]

    def resolve(self, selector):
        """selector: a path, a session-id (or prefix), a slug (or prefix), or a
        title substring. Returns a typed Resolution — never silently guesses."""
        if os.path.exists(selector):
            return Resolution("found", Session.at(selector))
        # 1) session-id prefix — the filename IS the id, so this needs no reads
        by_id = [p for p in self._paths() if os.path.basename(p)[:-6].startswith(selector)]
        if len(by_id) == 1:
            return Resolution("found", Session.at(by_id[0]))
        if len(by_id) > 1:   # ambiguous id — report the candidates, never guess
            return Resolution("ambiguous", candidates=[Session.at(p, peeked=True) for p in by_id])
        # 2) no id hit → fuzzy over title / slug only (NOT prompt bodies, which
        #    match far too loosely — a stray digit would pick a random session)
        s = selector.lower()
        hits = [x for x in self.list(limit=200)
                if s in (x.title or "").lower() or (x.slug or "").lower().startswith(s)]
        if len(hits) == 1:
            return Resolution("found", hits[0])
        return Resolution("ambiguous" if hits else "notfound", candidates=hits)

def print_table(sessions, now):
    if not sessions:
        print("no sessions found."); return
    multi = len({s.project for s in sessions}) > 1
    hdr = f'{"#":>2}  {"AGE":<9}  {"TITLE":<46}  {"ID":<8}  SIZE'
    if multi: hdr += "   PROJECT"
    print(hdr)
    for i, s in enumerate(sessions, 1):
        label = (s.title or s.prompt or "?")[:46]
        row = f'{i:>2}  {s.age(now):<9}  {label:<46}  {s.id[:8]:<8}  {human_size(s.size):>6}'
        if multi: row += f'   {s.project}'
        print(row)

# ---- send -------------------------------------------------------------------

def send(payload, api_key, dataset):
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "x-honeycomb-team": api_key,
        "x-honeycomb-dataset": dataset,   # honored by Classic keys; ignored by E&S (uses service.name)
    }
    req = urllib.request.Request(OTLP_ENDPOINT, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        return None, str(e)

# Verification (querying the trace back) is done by the skill orchestrator via
# the Honeycomb MCP after a send — see SKILL.md. It isn't reimplemented here so
# the script stays dependency- and key-free (no separate query key to manage).

# ---- main -------------------------------------------------------------------

def main():
    import time
    ap = argparse.ArgumentParser(description="Send a Claude Code session to Honeycomb as an OTel trace.")
    ap.add_argument("session", nargs="?",
                    help="path, session-id (or prefix), slug, or title substring. "
                         "Omit to use the current session.")
    ap.add_argument("-l", "--list", action="store_true", help="list recent sessions and exit")
    ap.add_argument("--all", action="store_true", help="scan all projects, not just the current one")
    ap.add_argument("--send", action="store_true", help="POST to Honeycomb (default: dry run)")
    ap.add_argument("--dataset", default=os.environ.get("HONEYCOMB_DATASET", DEFAULT_DATASET))
    ap.add_argument("--no-messages", action="store_true", help="omit prompt/response/tool bodies")
    ap.add_argument("--out", help="where to write the OTLP payload (dry run)")
    args = ap.parse_args()
    now = time.time()

    if args.list:
        print_table(SessionCatalog(args.all, now).list(), now)
        return

    if args.session:
        res = SessionCatalog(args.all, now).resolve(args.session)
        if res.found:
            path = res.session.path
        else:
            if res.candidates:
                print(f"'{args.session}' matched {len(res.candidates)} sessions — be more specific:\n", file=sys.stderr)
                print_table(res.candidates, now)
            else:
                print(f"no session matched '{args.session}'. try --list (or --all).", file=sys.stderr)
            sys.exit(1)
    else:
        path = SessionCatalog(all_projects=False, now=now).current()
        if not path:
            print("no current session found. pass a path/slug, or --list.", file=sys.stderr); sys.exit(1)
        if project_dir_for_cwd() is None:
            # cwd isn't a recognized Claude Code project — "current" fell back to
            # the newest session across ALL projects, which may not be this one.
            print(f"note: this directory isn't a known project, so 'current' = the newest "
                  f"session anywhere ({os.path.basename(os.path.dirname(path)).rsplit('-',1)[-1]}). "
                  f"cd into your project, or pass a selector / --list.", file=sys.stderr)
        print(f"(current session: {os.path.basename(path)})\n")

    # fail fast: don't parse a huge transcript only to discover the key is missing
    if args.send and not os.environ.get("HONEYCOMB_API_KEY"):
        print("error: HONEYCOMB_API_KEY not set. Get an ingest key from Honeycomb → "
              "Environment settings → API keys, then: export HONEYCOMB_API_KEY=... "
              "(re-run adds --send).", file=sys.stderr)
        sys.exit(2)

    rows = load(path)
    payload, stats = build_spans(rows, args.dataset, include_messages=not args.no_messages)

    print("Session trace")
    print(f"  session   {stats['session_id']}")
    print(f"  trace_id  {stats['trace_id']}")
    print(f"  dataset   {stats['dataset']}")
    print(f"  models    {', '.join(stats['models'])}")
    print(f"  spans     {stats['spans']}  ({stats['turns']} turns, {stats['chats']} chat, {stats['tools']} tools)")
    print(f"  tokens    {stats['tokens_in']:,} in / {stats['tokens_out']:,} out")
    print(f"  duration  {stats['duration_s']}s")

    if args.send:
        status, resp = send(payload, os.environ["HONEYCOMB_API_KEY"], args.dataset)
        if status == 200:
            from datetime import datetime, timezone
            fmt = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"\n✓ sent {stats['spans']} spans to Honeycomb dataset '{stats['dataset']}'")
            print(f"  find it: dataset '{stats['dataset']}', filter trace.trace_id = {stats['trace_id']}")
            print(f"  heads-up: spans are backdated to when they happened "
                  f"({fmt(stats['oldest_epoch'])} → {fmt(stats['newest_epoch'])} UTC), so set your "
                  f"Honeycomb time range to cover that")
            print(f"           (the default 'last 2 hours' will look empty).")
            print(f"  send once only — re-sending appends duplicate spans to this trace.")
        else:
            hint = " (check the key / region — EU keys need the eu1 endpoint)" if status in (401, 403) else ""
            print(f"\n✗ send failed (HTTP {status}){hint}: {resp}", file=sys.stderr); sys.exit(3)
    else:
        out = args.out or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"session-trace-{stats['trace_id'][:8]}.json")
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n(dry run — nothing sent) OTLP payload written to:\n  {out}")
        print("  re-run with --send and HONEYCOMB_API_KEY set to ship it.")

if __name__ == "__main__":
    main()
