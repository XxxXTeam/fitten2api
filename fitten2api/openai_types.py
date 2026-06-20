from __future__ import annotations

import time
import uuid
from typing import Any


DEFAULT_MODEL = "fitten-code"


def now_ts() -> int:
    return int(time.time())


def new_chat_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def model_card(model: str = DEFAULT_MODEL, owned_by: str = "fitten") -> dict[str, Any]:
    return {
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
    }


def chat_completion(
    *,
    model: str,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    completion_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
        if content is None:
            message["content"] = None
    return {
        "id": completion_id or new_chat_id(),
        "object": "chat.completion",
        "created": created or now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def chat_chunk(
    *,
    model: str,
    completion_id: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def error_response(message: str, status: int = 400, code: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error" if status < 500 else "server_error",
            "param": None,
            "code": code,
        }
    }
