from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import AppConfig, ModelConfig, load_config
from .credentials import CredentialError, credentials_path, load_credentials
from .fitten_client import FittenClient, FittenConfig, parse_tool_call_text
from .openai_types import DEFAULT_MODEL, chat_chunk, chat_completion, error_response, model_card, new_chat_id, now_ts


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    server: "OpenAIProxyServer"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._send_json({"object": "list", "data": [model_card(model.id, model.owned_by) for model in self.server.models]})
            return
        if self.path.startswith("/v1/models/"):
            model = self.server.find_model(self.path.rsplit("/", 1)[-1])
            if not model:
                self._send_error("Model not found", HTTPStatus.NOT_FOUND)
                return
            self._send_json(model_card(model.id, model.owned_by))
            return
        self._send_error("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if not self._authorized():
                return
            body = self._read_json()
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
                return
            if self.path == "/v1/completions":
                self._handle_completion(body)
                return
            self._send_error("Not found", HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self._send_error("Request body must be valid JSON", HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._send_error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_chat(self, body: dict[str, Any]) -> None:
        model = str(body.get("model") or self.server.default_model.id)
        body = self.server.with_upstream_model(body, model)
        if not isinstance(body.get("messages"), list):
            self._send_error("'messages' must be an array", HTTPStatus.BAD_REQUEST)
            return
        if body.get("stream"):
            self._stream_chat(body, model)
            return
        content, tool_calls = self.server.client.complete(body)
        self._send_json(chat_completion(model=model, content=content, tool_calls=tool_calls))

    def _handle_completion(self, body: dict[str, Any]) -> None:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(str(item) for item in prompt)
        exposed_model = str(body.get("model") or self.server.default_model.id)
        chat_body = {
            **body,
            "messages": [{"role": "user", "content": str(prompt)}],
            "model": exposed_model,
        }
        chat_body = self.server.with_upstream_model(chat_body, str(chat_body["model"]))
        content, _ = self.server.client.complete(chat_body)
        self._send_json(
            {
                "id": new_chat_id("cmpl"),
                "object": "text_completion",
                "created": now_ts(),
                "model": exposed_model,
                "choices": [{"text": content or "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    def _stream_chat(self, body: dict[str, Any], model: str) -> None:
        completion_id = new_chat_id()
        created = now_ts()
        self.send_response(HTTPStatus.OK)
        self._send_common_headers("text/event-stream; charset=utf-8")
        self.end_headers()
        self._write_sse(chat_chunk(model=model, completion_id=completion_id, created=created, delta={"role": "assistant"}))
        content_parts: list[str] = []
        for text in self.server.client.stream(body):
            content_parts.append(text)
        content = "".join(content_parts)
        tool_calls = parse_tool_call_text(content)
        if tool_calls:
            self._write_sse(
                chat_chunk(
                    model=model,
                    completion_id=completion_id,
                    created=created,
                    delta={"tool_calls": tool_calls},
                    finish_reason="tool_calls",
                )
            )
        else:
            if content:
                self._write_sse(chat_chunk(model=model, completion_id=completion_id, created=created, delta={"content": content}))
            self._write_sse(chat_chunk(model=model, completion_id=completion_id, created=created, delta={}, finish_reason="stop"))
        self.wfile.write(b"data: [DONE]\n\n")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8") if raw else "{}")
        if not isinstance(data, dict):
            raise json.JSONDecodeError("root must be object", "", 0)
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, message: str, status: HTTPStatus) -> None:
        self._send_json(error_response(message, int(status)), status)

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_common_headers("text/plain")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _authorized(self) -> bool:
        if not self.server.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.api_key}"
        if auth == expected:
            return True
        self._send_error("Unauthorized", HTTPStatus.UNAUTHORIZED)
        return False

    def _write_sse(self, payload: dict[str, Any]) -> None:
        line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.quiet:
            return
        super().log_message(format, *args)


class OpenAIProxyServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        client: FittenClient,
        model: str | list[ModelConfig],
        quiet: bool = False,
        api_key: str = "",
    ) -> None:
        super().__init__(address, OpenAIProxyHandler)
        self.client = client
        self.models = model if isinstance(model, list) else [ModelConfig(id=model)]
        self.default_model = self.models[0] if self.models else ModelConfig(id=DEFAULT_MODEL)
        self.quiet = quiet
        self.api_key = api_key

    def find_model(self, model_id: str) -> ModelConfig | None:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def with_upstream_model(self, body: dict[str, Any], model_id: str) -> dict[str, Any]:
        model = self.find_model(model_id)
        if not model:
            raise ValueError(f"Unknown model: {model_id}")
        upstream_model = model.upstream or None
        next_body = dict(body)
        next_body["model"] = upstream_model
        next_body["_fitten_model_field"] = model.upstream_field
        next_body["openai_model"] = model.id
        return next_body


def build_server(host: str | None = None, port: int | None = None, *, quiet: bool = False, config_path: str | Path | None = None) -> OpenAIProxyServer:
    app_config = load_config(config_path)
    credential_file = credentials_path()
    credentials = load_credentials(credential_file, export=True)
    client = FittenClient(credentials, FittenConfig.from_app_config(app_config.fitten))
    models = app_config.models
    if app_config.fitten.fetch_models:
        try:
            models = client.fetch_models() or models
        except Exception as exc:
            if not quiet:
                print(f"Could not fetch Fitten models, using config fallback: {exc}")
    bind_host = host or app_config.server.host
    bind_port = port if port is not None else app_config.server.port
    return OpenAIProxyServer(
        (bind_host, bind_port),
        client,
        models,
        quiet=quiet,
        api_key=app_config.server.api_key,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose FittenCode as an OpenAI-compatible API.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        server = build_server(args.host, args.port, quiet=args.quiet, config_path=args.config)
    except CredentialError as exc:
        print(f"Credential error: {exc}")
        return 2
    host, port = server.server_address
    print(f"OpenAI-compatible Fitten proxy listening on http://{host}:{port}/v1")
    server.serve_forever()
    return 0
