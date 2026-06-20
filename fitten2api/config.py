from __future__ import annotations

import os
import platform
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


DEFAULT_CONFIG_PATH = Path.cwd() / "config.toml"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""


@dataclass(frozen=True)
class FittenUpstreamConfig:
    base_url: str = "https://fc.fittenlab.cn"
    chat_endpoint: str = "/codeapi/chat_auth"
    models_base_url: str = "https://api.fittentech.com"
    models_endpoint: str = "/codeapi/chat/models"
    fetch_models: bool = True
    timeout: int = 120
    ide: str = "vsc"
    ide_version: str = "1.125.1"
    extension_version: str = "1.0.6"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    os_name: str = field(default_factory=platform.system)
    os_version: str = field(default_factory=platform.release)
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


@dataclass(frozen=True)
class ModelConfig:
    id: str
    upstream: str = ""
    owned_by: str = "fitten"
    upstream_field: str = "model"


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    fitten: FittenUpstreamConfig = field(default_factory=FittenUpstreamConfig)
    models: list[ModelConfig] = field(default_factory=list)

    @property
    def default_model(self) -> ModelConfig:
        return self.models[0] if self.models else default_models()[0]


def default_models() -> list[ModelConfig]:
    return [
        ModelConfig(id="Default", upstream="S1", owned_by="fitten-chat", upstream_field="model"),
        ModelConfig(id="DeepSeek V3", upstream="S2", owned_by="fitten-chat", upstream_field="model"),
        ModelConfig(id="DeepSeek R1", upstream="S3", owned_by="fitten-chat", upstream_field="model"),
        ModelConfig(id="Default (Agent)", upstream="S5", owned_by="fitten-agent", upstream_field="agentModel"),
    ]


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("FITTEN2API_CONFIG") or DEFAULT_CONFIG_PATH)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    server_data = data.get("server", {})
    fitten_data = data.get("fitten", {})

    server = ServerConfig(
        host=str(os.getenv("FITTEN2API_HOST") or server_data.get("host") or ServerConfig.host),
        port=int(os.getenv("FITTEN2API_PORT") or server_data.get("port") or ServerConfig.port),
        api_key=str(os.getenv("FITTEN2API_API_KEY") or server_data.get("api_key") or ""),
    )
    fitten = FittenUpstreamConfig(
        base_url=str(os.getenv("FITTEN_BASE_URL") or fitten_data.get("base_url") or FittenUpstreamConfig.base_url),
        chat_endpoint=str(
            os.getenv("FITTEN_CHAT_ENDPOINT") or fitten_data.get("chat_endpoint") or FittenUpstreamConfig.chat_endpoint
        ),
        models_base_url=str(
            os.getenv("FITTEN_MODELS_BASE_URL")
            or fitten_data.get("models_base_url")
            or FittenUpstreamConfig.models_base_url
        ),
        models_endpoint=str(
            os.getenv("FITTEN_MODELS_ENDPOINT")
            or fitten_data.get("models_endpoint")
            or FittenUpstreamConfig.models_endpoint
        ),
        fetch_models=_as_bool(
            os.getenv("FITTEN_FETCH_MODELS") if os.getenv("FITTEN_FETCH_MODELS") is not None else fitten_data.get("fetch_models"),
            FittenUpstreamConfig.fetch_models,
        ),
        timeout=int(os.getenv("FITTEN_TIMEOUT") or fitten_data.get("timeout") or FittenUpstreamConfig.timeout),
        ide=str(fitten_data.get("ide") or FittenUpstreamConfig.ide),
        ide_version=str(fitten_data.get("ide_version") or FittenUpstreamConfig.ide_version),
        extension_version=str(fitten_data.get("extension_version") or FittenUpstreamConfig.extension_version),
        session_id=str(fitten_data.get("session_id") or uuid.uuid4().hex),
        os_name=str(fitten_data.get("os") or platform.system()),
        os_version=str(fitten_data.get("os_version") or platform.release()),
        extra_query=str(fitten_data.get("extra_query") or ""),
        user_agent=str(fitten_data.get("user_agent") or FittenUpstreamConfig.user_agent),
        api_version=str(fitten_data.get("api_version") or FittenUpstreamConfig.api_version),
        origin=str(fitten_data.get("origin") or FittenUpstreamConfig.origin),
        referer=str(fitten_data.get("referer") or FittenUpstreamConfig.referer),
        accept_language=str(fitten_data.get("accept_language") or FittenUpstreamConfig.accept_language),
    )
    return AppConfig(server=server, fitten=fitten, models=default_models())


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
