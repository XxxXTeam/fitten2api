import json
import threading
import unittest
import urllib.request
import urllib.error

from fitten2api.config import ModelConfig
from fitten2api.credentials import FittenCredentials
from fitten2api.fitten_client import FittenClient
from fitten2api.server import OpenAIProxyServer


class ServerTests(unittest.TestCase):
    def start_server(self, transport, *, models=None, api_key=""):
        client = FittenClient(FittenCredentials("a", "r", "u"), transport=transport)
        server = OpenAIProxyServer(
            ("127.0.0.1", 0),
            client,
            models or [ModelConfig(id="fitten-code")],
            quiet=True,
            api_key=api_key,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(cleanup)
        return f"http://127.0.0.1:{server.server_port}"

    def request_json(self, url, payload=None, headers=None):
        headers = headers or {}
        if payload is None:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def request_text(self, url, payload, headers=None):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode("utf-8")

    def test_models_endpoint(self):
        base = self.start_server(
            lambda url, headers, payload, stream: "unused",
            models=[ModelConfig(id="fitten-code"), ModelConfig(id="fitten-chat", upstream="chat")],
        )
        data = self.request_json(base + "/v1/models")
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "fitten-code")
        self.assertEqual(data["data"][1]["id"], "fitten-chat")

    def test_chat_completion_endpoint(self):
        seen = {}

        def transport(url, headers, payload, stream):
            seen.update(payload=payload)
            return {"content": "pong"}

        base = self.start_server(transport)
        data = self.request_json(
            base + "/v1/chat/completions",
            {"model": "fitten-code", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "pong")
        self.assertNotIn("model", seen["payload"]["meta_datas"])

    def test_chat_completion_tool_call_response(self):
        upstream = json.dumps({"tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}]})
        base = self.start_server(lambda url, headers, payload, stream: upstream)
        data = self.request_json(
            base + "/v1/chat/completions",
            {
                "model": "fitten-code",
                "messages": [{"role": "user", "content": "lookup"}],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            },
        )
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "lookup")

    def test_chat_completion_xml_tool_call_response(self):
        upstream = '<tool_call><name>lookup</name><arguments>{"q":"x"}</arguments></tool_call>'
        base = self.start_server(lambda url, headers, payload, stream: upstream)
        data = self.request_json(
            base + "/v1/chat/completions",
            {
                "model": "fitten-code",
                "messages": [{"role": "user", "content": "lookup"}],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            },
        )
        choice = data["choices"][0]
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            choice["message"]["tool_calls"][0],
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            },
        )

    def test_chat_completion_xml_mcp_tool_call_response(self):
        upstream = '<tool_call><name>mcp__CherryHub__list</name><arguments>{}</arguments></tool_call>'
        base = self.start_server(lambda url, headers, payload, stream: upstream)
        data = self.request_json(
            base + "/v1/chat/completions",
            {
                "model": "fitten-code",
                "messages": [{"role": "user", "content": "list"}],
                "tools": [{"type": "function", "function": {"name": "mcp__CherryHub__list", "parameters": {"type": "object"}}}],
            },
        )
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "mcp__CherryHub__list")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_stream_chat_completion_xml_mcp_tool_call_response(self):
        upstream = ['<tool_call><name>mcp__CherryHub__list</name>', '<arguments>{}</arguments></tool_call>']
        base = self.start_server(lambda url, headers, payload, stream: upstream)
        text = self.request_text(
            base + "/v1/chat/completions",
            {
                "model": "fitten-code",
                "stream": True,
                "messages": [{"role": "user", "content": "list"}],
                "tools": [{"type": "function", "function": {"name": "mcp__CherryHub__list", "parameters": {"type": "object"}}}],
            },
        )
        events = [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: {")]
        choice = events[-1]["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["delta"]["tool_calls"][0]["function"]["name"], "mcp__CherryHub__list")
        self.assertEqual(choice["delta"]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_stream_chat_completion_text_response(self):
        base = self.start_server(lambda url, headers, payload, stream: ["he", "llo"])
        text = self.request_text(
            base + "/v1/chat/completions",
            {"model": "fitten-code", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        events = [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: {")]
        self.assertEqual(events[1]["choices"][0]["delta"], {"content": "hello"})
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_chat_completion_maps_model_to_upstream(self):
        seen = {}

        def transport(url, headers, payload, stream):
            seen.update(payload=payload)
            return {"content": "pong"}

        base = self.start_server(transport, models=[ModelConfig(id="fitten-chat", upstream="chat-v2")])
        data = self.request_json(
            base + "/v1/chat/completions",
            {"model": "fitten-chat", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(data["model"], "fitten-chat")
        self.assertEqual(seen["payload"]["meta_datas"]["model"], "chat-v2")

    def test_chat_completion_maps_agent_model_field(self):
        seen = {}

        def transport(url, headers, payload, stream):
            seen.update(payload=payload)
            return {"content": "pong"}

        base = self.start_server(
            transport,
            models=[ModelConfig(id="S5", upstream="S5", owned_by="fitten-agent", upstream_field="agentModel")],
        )
        data = self.request_json(
            base + "/v1/chat/completions",
            {"model": "S5", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(data["model"], "S5")
        self.assertEqual(seen["payload"]["meta_datas"]["agentModel"], "S5")
        self.assertNotIn("model", seen["payload"]["meta_datas"])

    def test_api_key_auth(self):
        base = self.start_server(lambda url, headers, payload, stream: {"content": "pong"}, api_key="secret")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request_json(base + "/v1/models")
        self.assertEqual(ctx.exception.code, 401)

        data = self.request_json(base + "/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(data["object"], "list")


if __name__ == "__main__":
    unittest.main()
