import json
import unittest

from fitten2api.credentials import FittenCredentials
from fitten2api.fitten_client import FittenClient, FittenConfig, parse_models_response, parse_tool_call_text, render_prompt


class FittenClientTests(unittest.TestCase):
    def credentials(self):
        return FittenCredentials("access", "refresh", "user").validate()

    def test_render_prompt_adds_default_system_block(self):
        prompt = render_prompt([{"role": "user", "content": "hi"}])
        self.assertTrue(prompt.startswith("<|system|>\n"))
        self.assertIn("Reply same language as the user's input.", prompt)
        self.assertIn("<|user|>\nhi\n<|end|>", prompt)

    def test_render_prompt_includes_tools_and_tool_results(self):
        prompt = render_prompt(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
            ],
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            tool_choice="auto",
        )
        self.assertIn("<|system|>", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("<name>tool_name</name>", prompt)
        self.assertIn("<name>get_weather</name>", prompt)
        self.assertIn("<arguments>{&quot;city&quot;:&quot;北京&quot;}</arguments>", prompt)
        self.assertNotIn('"tool_calls"', prompt)
        self.assertIn("Tool result (call_0):", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_render_prompt_renders_multiple_assistant_tool_calls_as_xml(self):
        prompt = render_prompt(
            [
                {"role": "user", "content": "inspect workspace"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_0", "type": "function", "function": {"name": "mcp__CherryHub__list", "arguments": "{}"}},
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "mcp__CherryHub__inspect", "arguments": {"path": "/workspace/src"}},
                        },
                    ],
                },
            ]
        )
        self.assertIn("<name>mcp__CherryHub__list</name>", prompt)
        self.assertIn("<arguments>{}</arguments>", prompt)
        self.assertIn("<name>mcp__CherryHub__inspect</name>", prompt)
        self.assertIn("<arguments>{&quot;path&quot;: &quot;/workspace/src&quot;}</arguments>", prompt)
        self.assertNotIn('"tool_calls"', prompt)

    def test_complete_extracts_text_from_openai_like_upstream(self):
        seen = {}

        def transport(url, headers, payload, stream):
            seen.update(url=url, headers=headers, payload=payload, stream=stream)
            return {"choices": [{"message": {"content": "hello"}}]}

        client = FittenClient(
            self.credentials(),
            FittenConfig(
                base_url="https://example.test",
                chat_endpoint="/codeapi/chat_auth",
                ide="vsc",
                ide_version="1.2.3",
                extension_version="0.1.138",
                session_id="sid",
                os_name="Windows_NT",
                os_version="10.0.0",
            ),
            transport,
        )
        content, tool_calls = client.complete({"model": "fitten-code", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(content, "hello")
        self.assertIsNone(tool_calls)
        self.assertEqual(
            seen["url"],
            "https://example.test/codeapi/chat_auth?ft_token=user&ide=vsc&ide_v=1.2.3&os=Windows_NT&os_v=10.0.0&v=0.1.138&sessionId=sid",
        )
        self.assertEqual(seen["headers"]["Authorization"], "Bearer access")
        self.assertTrue(seen["payload"]["inputs"].startswith("<|system|>"))
        self.assertIn("<|user|>", seen["payload"]["inputs"])
        self.assertEqual(seen["payload"]["ft_token"], "user")
        self.assertEqual(seen["payload"]["meta_datas"]["model"], "fitten-code")

    def test_parse_tool_call_json(self):
        calls = parse_tool_call_text(
            json.dumps({"tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}]})
        )
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "lookup")
        self.assertEqual(calls[0]["function"]["arguments"], '{"q": "x"}')

    def test_parse_tool_call_xml_with_json_arguments(self):
        calls = parse_tool_call_text(
            '<tool_call><name>lookup</name><arguments>{"q":"x","limit":2}</arguments></tool_call>'
        )
        self.assertEqual(calls[0]["id"], "call_0")
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "lookup")
        self.assertEqual(calls[0]["function"]["arguments"], '{"q":"x","limit":2}')

    def test_parse_tool_call_xml_with_mcp_tool_name(self):
        calls = parse_tool_call_text(
            '<tool_call><name>mcp__CherryHub__list</name><arguments>{}</arguments></tool_call>'
        )
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "mcp__CherryHub__list")
        self.assertEqual(calls[0]["function"]["arguments"], '{}')

    def test_parse_tool_call_xml_with_argument_tags(self):
        calls = parse_tool_call_text(
            """
            I will call the tool.
            <tool_call>
              <name>get_weather</name>
              <arguments>
                <city>北京</city>
                <unit>celsius</unit>
                <days>3</days>
              </arguments>
            </tool_call>
            """
        )
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "北京", "unit": "celsius", "days": "3"})

    def test_stream_extracts_sse_chunks(self):
        def transport(url, headers, payload, stream):
            return [
                'data: {"choices":[{"delta":{"content":"he"}}]}',
                'data: {"choices":[{"delta":{"content":"llo"}}]}',
                "data: [DONE]",
            ]

        client = FittenClient(self.credentials(), transport=transport)
        self.assertEqual("".join(client.stream({"messages": [{"role": "user", "content": "hi"}]})), "hello")

    def test_stream_extracts_fitten_json_lines(self):
        def transport(url, headers, payload, stream):
            return [
                '{"delta":"he"}',
                '{"delta":"heartbeat"}',
                '{"reasoning_delta":"think"}',
                '{"delta":"llo"}',
            ]

        client = FittenClient(self.credentials(), transport=transport)
        self.assertEqual("".join(client.stream({"messages": [{"role": "user", "content": "hi"}]})), "hethinkllo")

    def test_fetch_models_uses_fitten_models_endpoint(self):
        seen = {}

        def models_transport(url, headers):
            seen.update(url=url, headers=headers)
            return {
                "chat": [
                    {"selection_name": "S2", "display_name": "DeepSeek V3", "is_default": False, "order": 95},
                    {"selection_name": "S1", "display_name": "Default", "is_default": True, "order": 100},
                ],
                "agent": [{"selection_name": "S5", "display_name": "Default", "is_default": True, "order": 100}],
            }

        client = FittenClient(
            self.credentials(),
            FittenConfig(models_base_url="https://api.example.test", models_endpoint="/codeapi/chat/models"),
            transport=lambda url, headers, payload, stream: "unused",
            models_transport=models_transport,
        )
        models = client.fetch_models()
        self.assertEqual(seen["url"], "https://api.example.test/codeapi/chat/models")
        self.assertEqual(seen["headers"]["X-Api-Version"], "v2")
        self.assertEqual([model.id for model in models], ["Default (Chat)", "DeepSeek V3", "Default (Agent)"])
        self.assertEqual([model.upstream for model in models], ["S1", "S2", "S5"])
        self.assertEqual(models[0].upstream_field, "model")
        self.assertEqual(models[2].upstream_field, "agentModel")

    def test_parse_models_response_accepts_json_string(self):
        models = parse_models_response(
            json.dumps(
                {
                    "chat": [{"selection_name": "S1", "display_name": "Default", "is_default": True}],
                    "agent": [{"selection_name": "S5", "display_name": "Default", "is_default": True}],
                }
            )
        )
        self.assertEqual([model.id for model in models], ["Default (Chat)", "Default (Agent)"])
        self.assertEqual([model.upstream for model in models], ["S1", "S5"])


if __name__ == "__main__":
    unittest.main()
