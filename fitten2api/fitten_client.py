from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .config import FittenUpstreamConfig, ModelConfig
from .credentials import FittenCredentials


Transport = Callable[[str, dict[str, str], dict[str, Any], bool], Iterable[str] | str | dict[str, Any]]
ModelTransport = Callable[[str, dict[str, str]], Any]


@dataclass(frozen=True)
class FittenConfig:
    base_url: str = "https://fc.fittenlab.cn"
    chat_endpoint: str = "/codeapi/chat_auth"
    models_base_url: str = "https://api.fittentech.com"
    models_endpoint: str = "/codeapi/chat/models"
    fetch_models: bool = True
    timeout: int = 120
    ide: str = "vsc"
    ide_version: str = "1.125.1"
    extension_version: str = "1.0.6"
    session_id: str = ""
    os_name: str = ""
    os_version: str = ""
    extra_query: str = ""
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Code/1.125.1 Chrome/148.0.7778.97 "
        "Electron/42.2.0 Safari/537.36"
    )
    api_version: str = "v2"
    origin: str = "https://cs.fittentech.com"
    referer: str = "https://cs.fittentech.com/"
    accept_language: str = "zh-CN"

    @classmethod
    def from_env(cls) -> "FittenConfig":
        return cls(
            base_url=os.getenv("FITTEN_BASE_URL", cls.base_url).rstrip("/"),
            chat_endpoint=os.getenv("FITTEN_CHAT_ENDPOINT", cls.chat_endpoint),
            models_base_url=os.getenv("FITTEN_MODELS_BASE_URL", cls.models_base_url).rstrip("/"),
            models_endpoint=os.getenv("FITTEN_MODELS_ENDPOINT", cls.models_endpoint),
            fetch_models=os.getenv("FITTEN_FETCH_MODELS", "true").strip().lower() not in {"0", "false", "no", "off"},
            timeout=int(os.getenv("FITTEN_TIMEOUT", str(cls.timeout))),
        )

    @classmethod
    def from_app_config(cls, config: FittenUpstreamConfig) -> "FittenConfig":
        return cls(
            base_url=config.base_url.rstrip("/"),
            chat_endpoint=config.chat_endpoint,
            models_base_url=config.models_base_url.rstrip("/"),
            models_endpoint=config.models_endpoint,
            fetch_models=config.fetch_models,
            timeout=config.timeout,
            ide=config.ide,
            ide_version=config.ide_version,
            extension_version=config.extension_version,
            session_id=config.session_id,
            os_name=config.os_name,
            os_version=config.os_version,
            extra_query=config.extra_query,
            user_agent=config.user_agent,
            api_version=config.api_version,
            origin=config.origin,
            referer=config.referer,
            accept_language=config.accept_language,
        )


class FittenClient:
    def __init__(
        self,
        credentials: FittenCredentials,
        config: FittenConfig | None = None,
        transport: Transport | None = None,
        models_transport: ModelTransport | None = None,
    ) -> None:
        self.credentials = credentials
        self.config = config or FittenConfig.from_env()
        self.transport = transport or self._urlopen_transport
        self.models_transport = models_transport or self._urlopen_models_transport

    def complete(self, request: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
        raw = self._send(request, stream=True)
        if isinstance(raw, str | dict):
            text = self._coerce_text(raw)
        else:
            text = "".join(self._coerce_text(chunk) for chunk in raw)
        return self._split_tool_calls(text)

    def stream(self, request: dict[str, Any]) -> Iterator[str]:
        raw = self._send(request, stream=True)
        if isinstance(raw, str | dict):
            text = self._coerce_text(raw)
            if text:
                yield text
            return
        for chunk in raw:
            text = self._coerce_text(chunk)
            if text:
                yield text

    def fetch_models(self) -> list[ModelConfig]:
        data = self.models_transport(self._models_url(), self._browser_headers())
        return parse_models_response(data)

    def _send(self, request: dict[str, Any], *, stream: bool) -> Iterable[str] | str | dict[str, Any]:
        payload = self._build_payload(request, stream=stream)
        headers = {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json, text/plain",
            "X-Fitten-User-Id": self.credentials.user_id,
        }
        return self.transport(self._chat_url(), headers, payload, stream)

    def _chat_url(self) -> str:
        endpoint = self.config.chat_endpoint
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        query = {
            "ft_token": self.credentials.user_id,
            "ide": self.config.ide,
            "ide_v": self.config.ide_version,
            "os": self.config.os_name,
            "os_v": self.config.os_version,
            "v": self.config.extension_version,
            "sessionId": self.config.session_id,
        }
        url = f"{self.config.base_url}{endpoint}?{urllib.parse.urlencode(query)}"
        if self.config.extra_query:
            url = f"{url}&{self.config.extra_query.lstrip('&?')}"
        return url

    def _models_url(self) -> str:
        endpoint = self.config.models_endpoint
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{self.config.models_base_url}{endpoint}"

    def _build_payload(self, request: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        prompt = render_prompt(request.get("messages") or [], request.get("tools"), request.get("tool_choice"))
        model = request.get("model")
        meta_datas = {
            "project_id": request.get("project_id"),
            "enable_search": request.get("enable_search"),
            "user_raw_input": _last_user_text(request.get("messages") or []),
            "enable_terminal": request.get("enable_terminal"),
            "user_id": self.credentials.user_id,
            "source": "fitten2api",
        }
        if model:
            meta_datas[str(request.get("_fitten_model_field") or "model")] = model
        return {
            "inputs": prompt,
            "ft_token": self.credentials.user_id,
            "meta_datas": {key: value for key, value in meta_datas.items() if value is not None},
        }

    def _browser_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": self.config.user_agent,
            "X-Api-Version": self.config.api_version,
            "Origin": self.config.origin,
            "Referer": self.config.referer,
            "Accept-Language": self.config.accept_language,
        }

    def _urlopen_transport(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], stream: bool
    ) -> Iterable[str] | str | dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=self.config.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fitten upstream returned HTTP {exc.code}: {detail}") from exc
        if stream:
            return _iter_sse_response(response)
        content_type = response.headers.get("content-type", "")
        body = response.read().decode("utf-8", errors="replace")
        if "json" in content_type:
            return json.loads(body)
        return body

    def _urlopen_models_transport(self, url: str, headers: dict[str, str]) -> Any:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            response = urllib.request.urlopen(request, timeout=self.config.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fitten models endpoint returned HTTP {exc.code}: {detail}") from exc
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def _coerce_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            event = _parse_sse_data(raw)
            if event is None:
                try:
                    return self._coerce_text(json.loads(raw))
                except json.JSONDecodeError:
                    return raw
            if event == "[DONE]":
                return ""
            try:
                return self._coerce_text(json.loads(event))
            except json.JSONDecodeError:
                return event
        if isinstance(raw, dict):
            if "tool_calls" in raw and "content" not in raw:
                return json.dumps(raw, ensure_ascii=False)
            return _extract_text(raw)
        return str(raw)

    def _split_tool_calls(self, text: str) -> tuple[str | None, list[dict[str, Any]] | None]:
        parsed = parse_tool_call_text(text)
        if parsed:
            return None, parsed
        return text, None


DEFAULT_SYSTEM_PROMPT = "Reply same language as the user's input."


def render_prompt(messages: list[dict[str, Any]], tools: Any = None, tool_choice: Any = None) -> str:
    system_parts = [_message_content(message) for message in messages if message.get("role") == "system"]
    if tools:
        system_parts.append(_render_tools(tools, tool_choice))
    if not system_parts:
        system_parts.append(DEFAULT_SYSTEM_PROMPT)

    parts: list[str] = ["<|system|>\n" + "\n\n".join(filter(None, system_parts)) + "\n<|end|>"]
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        content = _message_content(message)
        if role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "tool"
            content = f"Tool result ({name}):\n{content}"
            role = "user"
        parts.append(f"<|{role}|>\n{content}\n<|end|>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def parse_tool_call_text(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        calls = data.get("tool_calls") if isinstance(data, dict) else data
        if isinstance(calls, dict):
            calls = [calls]
        if isinstance(calls, list) and calls:
            return [_normalize_tool_call(call, index) for index, call in enumerate(calls)]

    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        return [_normalize_tool_call(call, index) for index, call in enumerate(xml_calls)]
    return None


def parse_models_response(data: Any) -> list[ModelConfig]:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        return []

    raw_models: list[tuple[str, str, str, str]] = []
    for section, field in (("chat", "model"), ("agent", "agentModel")):
        items = data.get(section)
        if not isinstance(items, list):
            continue
        for item in sorted(items, key=_model_sort_key):
            if not isinstance(item, dict) or not item.get("selection_name"):
                continue
            selection_name = str(item["selection_name"])
            display_name = str(item.get("display_name") or selection_name)
            raw_models.append((display_name, selection_name, f"fitten-{section}", field))

    name_counts: dict[str, int] = {}
    for display_name, _, _, _ in raw_models:
        name_counts[display_name] = name_counts.get(display_name, 0) + 1

    models: list[ModelConfig] = []
    for display_name, selection_name, owned_by, field in raw_models:
        model_id = display_name
        if name_counts[display_name] > 1:
            section = owned_by.removeprefix("fitten-").title()
            model_id = f"{display_name} ({section})"
        models.append(
            ModelConfig(
                id=model_id,
                upstream=selection_name,
                owned_by=owned_by,
                upstream_field=field,
            )
        )
    return models


def _model_sort_key(item: Any) -> tuple[int, int, str]:
    if not isinstance(item, dict):
        return (1, 0, "")
    default_rank = 0 if item.get("is_default") else 1
    try:
        order = int(item.get("order") or 0)
    except (TypeError, ValueError):
        order = 0
    return (default_rank, -order, str(item.get("selection_name") or ""))


def _normalize_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name") or call.get("name") or f"tool_{index}"
    arguments = function.get("arguments") or call.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call.get("id") or f"call_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _parse_xml_tool_calls(text: str) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []
    for raw_call in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        name = _xml_tag_text(raw_call, "name") or _xml_tag_text(raw_call, "tool")
        arguments_text = _xml_tag_text(raw_call, "arguments") or _xml_tag_text(raw_call, "args")
        if not name:
            continue
        call: dict[str, Any] = {"name": name}
        if arguments_text:
            call["arguments"] = _coerce_xml_arguments(arguments_text)
        calls.append(call)
    return calls or None


def _xml_tag_text(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.S)
    if not match:
        return None
    return _unescape_xml(match.group(1).strip())


def _coerce_xml_arguments(arguments_text: str) -> str | dict[str, Any]:
    stripped = arguments_text.strip()
    if not stripped:
        return "{}"
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    args: dict[str, Any] = {}
    for match in re.finditer(r"<([A-Za-z_][\w.-]*)\b[^>]*>(.*?)</\1>", stripped, flags=re.S):
        args[match.group(1)] = _unescape_xml(match.group(2).strip())
    if args:
        return args
    return stripped


def _unescape_xml(value: str) -> str:
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def _message_content(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        tool_calls_xml = _tool_calls_to_xml(message["tool_calls"])
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return f"{content}\n{tool_calls_xml}"
        return tool_calls_xml

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
                elif "text" in item:
                    texts.append(str(item["text"]))
                else:
                    texts.append(json.dumps(item, ensure_ascii=False))
            else:
                texts.append(str(item))
        return "\n".join(filter(None, texts))
    return "" if content is None else str(content)


def _tool_calls_to_xml(tool_calls: Any) -> str:
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return ""

    blocks: list[str] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        normalized = _normalize_tool_call(call, index)
        function = normalized["function"]
        blocks.append(
            "<tool_call>"
            f"<name>{_escape_xml(function['name'])}</name>"
            f"<arguments>{_escape_xml(function['arguments'])}</arguments>"
            "</tool_call>"
        )
    return "\n".join(blocks)


def _escape_xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _render_tools(tools: Any, tool_choice: Any) -> str:
    return (
        "You may call tools by returning XML only, with one <tool_call> block per call.\n"
        "Use this exact shape:\n"
        "<tool_call><name>tool_name</name><arguments>{\"arg\":\"value\"}</arguments></tool_call>\n"
        "The <arguments> value must be a JSON object string matching the selected tool schema.\n"
        "Do not wrap tool calls in Markdown and do not add assistant prose when calling a tool.\n"
        f"tool_choice: {json.dumps(tool_choice, ensure_ascii=False)}\n"
        f"tools: {json.dumps(tools, ensure_ascii=False)}"
    )


def _extract_text(data: dict[str, Any]) -> str:
    for key in ("delta", "reasoning_delta", "thinking_delta"):
        value = data.get(key)
        if isinstance(value, str):
            if value.startswith("heartbeat"):
                return ""
            return value
    if isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            for container in (delta, message, choice):
                if isinstance(container, dict) and isinstance(container.get("content"), str):
                    return container["content"]
    for key in ("content", "text", "answer", "output", "response", "data"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = _extract_text(value)
            if nested:
                return nested
    return ""


def _iter_sse_response(response: Any) -> Iterator[str]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            yield line


def _parse_sse_data(chunk: str) -> str | None:
    stripped = chunk.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[5:].strip()


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_content(message)
    return ""
