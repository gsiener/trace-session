"""Tests for the transcript -> SpanRecord seam (and the OTLP encoder).

Run:  python3 -m unittest -v   (from this directory)

These assert on SpanRecords — the point of the build_span_tree / to_otlp split.
No filesystem, no network: build a fixture transcript, check the tree.
"""
import json
import os
import shutil
import tempfile
import unittest
import convert


def row(**kw):
    kw.setdefault("sessionId", "sess-0001")
    return kw

def prompt(uuid, text, ts, parent=None):
    return row(type="user", uuid=uuid, parentUuid=parent, timestamp=ts,
               message={"role": "user", "content": text})

def assistant(uuid, parent, ts, content, model="claude-opus-4-8",
              stop_reason="end_turn", usage=None, **extra):
    msg = {"role": "assistant", "model": model, "stop_reason": stop_reason,
           "content": content, "usage": usage or {}}
    return row(type="assistant", uuid=uuid, parentUuid=parent, timestamp=ts,
               message=msg, **extra)

def tool_result(uuid, parent, ts, tool_use_id, text="ok", is_error=False):
    return row(type="user", uuid=uuid, parentUuid=parent, timestamp=ts,
               message={"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": tool_use_id,
                    "content": text, "is_error": is_error}]})

def attr(rec, key):
    return rec.attributes.get(key)

def by_op(records, op):
    return [r for r in records if attr(r, "gen_ai.operation.name") == op]


class TokenAccounting(unittest.TestCase):
    def test_input_tokens_is_total_context(self):
        rows = [
            prompt("u1", "do the thing", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:05.000Z",
                      [{"type": "text", "text": "done"}],
                      usage={"input_tokens": 10, "cache_read_input_tokens": 100,
                             "cache_creation_input_tokens": 5, "output_tokens": 20}),
        ]
        records, stats = convert.build_span_tree(rows)
        chat = by_op(records, "chat")[0]
        self.assertEqual(attr(chat, "gen_ai.usage.input_tokens"), 115)        # 10+100+5
        self.assertEqual(attr(chat, "gen_ai.usage.uncached_input_tokens"), 10)
        self.assertEqual(stats["tokens_in"], 115)


class SpanTreeShape(unittest.TestCase):
    def setUp(self):
        self.rows = [
            prompt("u1", "list the files", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:05.000Z", [
                {"type": "text", "text": "sure"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ], stop_reason="tool_use",
               usage={"input_tokens": 3, "output_tokens": 8}),
            tool_result("r1", "a1", "2026-01-01T00:00:06.000Z", "t1", text="file.txt"),
        ]
        self.records, self.stats = convert.build_span_tree(self.rows)

    def test_one_root_invoke_agent(self):
        roots = [r for r in self.records if r.parent_id is None]
        self.assertEqual(len(roots), 1)
        self.assertEqual(attr(roots[0], "gen_ai.operation.name"), "invoke_agent")
        self.assertEqual(roots[0].name, "invoke_agent claude-code")

    def test_tool_span_nested_under_its_chat(self):
        chat = by_op(self.records, "chat")[0]
        tool = by_op(self.records, "execute_tool")[0]
        self.assertEqual(tool.parent_id, chat.span_id)
        self.assertEqual(attr(tool, "gen_ai.tool.name"), "Bash")
        self.assertEqual(tool.name, "execute_tool Bash")

    def test_messages_ride_as_attributes(self):
        chat = by_op(self.records, "chat")[0]
        self.assertEqual(attr(chat, "gen_ai.input.messages"), "list the files")
        self.assertEqual(attr(chat, "gen_ai.output.messages"), "sure")

    def test_tool_error_sets_span_status(self):
        rows = [
            prompt("u1", "break it", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:05.000Z",
                      [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
                      stop_reason="tool_use"),
            tool_result("r1", "a1", "2026-01-01T00:00:06.000Z", "t1", is_error=True),
        ]
        records, _ = convert.build_span_tree(rows)
        tool = by_op(records, "execute_tool")[0]
        self.assertIsNotNone(tool.status)
        self.assertEqual(tool.status["code"], 2)


class SubAgentLanes(unittest.TestCase):
    def test_task_spawn_becomes_named_invoke_agent(self):
        rows = [
            prompt("u1", "research it", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:05.000Z", [
                {"type": "tool_use", "id": "t1", "name": "Task",
                 "input": {"subagent_type": "general-purpose",
                           "description": "Analyze the logs"}},
            ], stop_reason="tool_use"),
            tool_result("r1", "a1", "2026-01-01T00:00:30.000Z", "t1"),
        ]
        records, _ = convert.build_span_tree(rows)
        inv = [r for r in records
               if attr(r, "gen_ai.operation.name") == "invoke_agent" and r.parent_id]
        self.assertEqual(len(inv), 1)
        self.assertEqual(attr(inv[0], "gen_ai.agent.name"), "general-purpose: Analyze the logs")
        self.assertEqual(inv[0].name, "invoke_agent general-purpose: Analyze the logs")
        self.assertEqual(attr(inv[0], "gen_ai.agent.type"), "general-purpose")


class SyntheticFiltered(unittest.TestCase):
    def test_synthetic_turns_are_skipped(self):
        rows = [
            prompt("u1", "hi", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:01.000Z",
                      [{"type": "text", "text": "x"}], model="<synthetic>"),
            assistant("a2", "u1", "2026-01-01T00:00:02.000Z",
                      [{"type": "text", "text": "real"}]),
        ]
        records, stats = convert.build_span_tree(rows)
        self.assertEqual(len(by_op(records, "chat")), 1)
        self.assertEqual(stats["chats"], 2)  # counts assistant rows; synthetic just isn't emitted as a span


class OtlpEncoding(unittest.TestCase):
    def test_encoder_shapes_payload(self):
        rows = [
            prompt("u1", "go", "2026-01-01T00:00:00.000Z"),
            assistant("a1", "u1", "2026-01-01T00:00:05.000Z",
                      [{"type": "text", "text": "ok"}],
                      usage={"input_tokens": 7, "output_tokens": 2}),
        ]
        records, stats = convert.build_span_tree(rows)
        payload = convert.to_otlp(records, stats["trace_id"], "my-dataset")
        rs = payload["resourceSpans"][0]
        res_attrs = {a["key"]: a["value"] for a in rs["resource"]["attributes"]}
        self.assertEqual(res_attrs["service.name"]["stringValue"], "my-dataset")
        spans = rs["scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), len(records))
        # ints encode as stringValue per OTLP JSON
        chat = next(s for s in spans if s["name"].startswith("chat"))
        toks = {a["key"]: a["value"] for a in chat["attributes"]}["gen_ai.usage.input_tokens"]
        self.assertEqual(toks, {"intValue": "7"})
        # root omits parentSpanId; children include it
        root = next(s for s in spans if s["name"] == "invoke_agent claude-code")
        self.assertNotIn("parentSpanId", root)
        self.assertIn("parentSpanId", chat)


def write_session(directory, sid, title=None, prompt=None, mtime=None):
    p = os.path.join(directory, f"{sid}.jsonl")
    lines = []
    if title:
        lines.append({"type": "ai-title", "aiTitle": title, "sessionId": sid})
    if prompt:
        lines.append({"type": "user", "uuid": "u1", "parentUuid": None,
                      "sessionId": sid, "message": {"role": "user", "content": prompt}})
    with open(p, "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    if mtime:
        os.utime(p, (mtime, mtime))
    return p


class Catalog(unittest.TestCase):
    """SessionCatalog resolves against a fixture directory — no ~/.claude, no network."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        write_session(self.tmp, "aaa10000-0000-0000-0000-000000000001",
                      title="Build the widget", prompt="build it", mtime=1000)
        write_session(self.tmp, "aaa20000-0000-0000-0000-000000000002",
                      title="Debug the pipeline", prompt="debug it", mtime=2000)
        write_session(self.tmp, "bbb30000-0000-0000-0000-000000000003",
                      title="Write the memo", prompt="write it", mtime=3000)
        self.cat = convert.SessionCatalog(dirs=[self.tmp], now=9999)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_newest_first_with_metadata(self):
        sessions = self.cat.list()
        self.assertEqual([s.id[:6] for s in sessions], ["bbb300", "aaa200", "aaa100"])
        self.assertEqual(sessions[0].title, "Write the memo")

    def test_current_is_newest(self):
        self.assertTrue(self.cat.current().endswith("000000000003.jsonl"))

    def test_resolve_unique_id_prefix(self):
        r = self.cat.resolve("bbb3")
        self.assertEqual(r.status, "found")
        self.assertTrue(r.found)
        self.assertEqual(r.session.id[:4], "bbb3")

    def test_resolve_ambiguous_id_reports_candidates(self):
        r = self.cat.resolve("aaa")
        self.assertEqual(r.status, "ambiguous")
        self.assertEqual(len(r.candidates), 2)

    def test_resolve_by_title_substring(self):
        r = self.cat.resolve("pipeline")
        self.assertEqual(r.status, "found")
        self.assertEqual(r.session.id[:6], "aaa200")

    def test_resolve_notfound(self):
        r = self.cat.resolve("nonexistent-xyz")
        self.assertEqual(r.status, "notfound")
        self.assertFalse(r.found)
        self.assertEqual(r.candidates, [])


if __name__ == "__main__":
    unittest.main()
