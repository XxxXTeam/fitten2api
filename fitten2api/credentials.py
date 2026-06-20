from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ACCESS_KEY = "fittencode.fittenAccess.token"
REFRESH_KEY = "fittencode.fittenRefresh.token"
USER_KEY = "fittencode.fittenUser.id"


class CredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class FittenCredentials:
    access_token: str
    refresh_token: str
    user_id: str

    def validate(self) -> "FittenCredentials":
        missing = [
            name
            for name, value in (
                ("access_token", self.access_token),
                ("refresh_token", self.refresh_token),
                ("user_id", self.user_id),
            )
            if not value
        ]
        if missing:
            raise CredentialError(f"Missing Fitten credential fields: {', '.join(missing)}")
        return self


def credentials_path(base_dir: Path | None = None) -> Path:
    return (base_dir or Path.cwd()) / "credentials.json"


def load_credentials(path: Path | None = None, *, export: bool = True) -> FittenCredentials:
    env_creds = _load_from_env()
    if env_creds:
        if export and path:
            save_credentials(env_creds, path)
        return env_creds

    file_path = path or credentials_path()
    if file_path.exists():
        return _load_from_file(file_path)

    vscode_creds = _load_from_vscode_state()
    if vscode_creds:
        if export:
            save_credentials(vscode_creds, file_path)
        return vscode_creds

    raise CredentialError(
        "No Fitten credentials found. Set FITTEN_ACCESS_TOKEN, FITTEN_REFRESH_TOKEN, "
        "FITTEN_USER_ID, or sign in to FittenCode in VS Code first."
    )


def save_credentials(credentials: FittenCredentials, path: Path | None = None) -> Path:
    target = path or credentials_path()
    target.write_text(json.dumps(asdict(credentials), indent=2), encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


def _load_from_env() -> FittenCredentials | None:
    access = os.getenv("FITTEN_ACCESS_TOKEN") or os.getenv("FITTEN_ACCESS")
    refresh = os.getenv("FITTEN_REFRESH_TOKEN") or os.getenv("FITTEN_REFRESH")
    user_id = os.getenv("FITTEN_USER_ID") or os.getenv("FITTEN_USER")
    if not any((access, refresh, user_id)):
        return None
    return FittenCredentials(access or "", refresh or "", user_id or "").validate()


def _load_from_file(path: Path) -> FittenCredentials:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FittenCredentials(
        access_token=data.get("access_token") or data.get("accessToken") or "",
        refresh_token=data.get("refresh_token") or data.get("refreshToken") or "",
        user_id=data.get("user_id") or data.get("userId") or "",
    ).validate()


def _load_from_vscode_state() -> FittenCredentials | None:
    state_db = _vscode_state_db()
    local_state = _vscode_local_state()
    if not state_db.exists() or not local_state.exists():
        return None

    encrypted = _read_vscode_secret_buffers(state_db)
    required = {ACCESS_KEY, REFRESH_KEY, USER_KEY}
    if not required.issubset(encrypted):
        return None

    decrypted = _decrypt_chromium_v10_values(local_state, encrypted)
    return FittenCredentials(
        access_token=decrypted.get(ACCESS_KEY, ""),
        refresh_token=decrypted.get(REFRESH_KEY, ""),
        user_id=decrypted.get(USER_KEY, ""),
    ).validate()


def _vscode_state_db() -> Path:
    user_data = os.getenv("VSCODE_USER_DATA_DIR")
    if user_data:
        return Path(user_data) / "User" / "globalStorage" / "state.vscdb"
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "Code" / "User" / "globalStorage" / "state.vscdb"
    return Path.home() / ".config" / "Code" / "User" / "globalStorage" / "state.vscdb"


def _vscode_local_state() -> Path:
    user_data = os.getenv("VSCODE_USER_DATA_DIR")
    if user_data:
        return Path(user_data) / "Local State"
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "Code" / "Local State"
    return Path.home() / ".config" / "Code" / "Local State"


def _read_vscode_secret_buffers(state_db: Path) -> dict[str, list[int]]:
    rows: list[tuple[str, Any]]
    with sqlite3.connect(state_db) as connection:
        rows = connection.execute(
            "select key, value from ItemTable where key like ?",
            ('secret://{"extensionId":"fittentech.fitten-code"%',),
        ).fetchall()

    values: dict[str, list[int]] = {}
    for key, raw in rows:
        try:
            key_data = json.loads(key.removeprefix("secret://"))
            secret_name = key_data.get("key")
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            data = payload.get("data")
            if secret_name and isinstance(data, list):
                values[secret_name] = [int(byte) for byte in data]
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            continue
    return values


def _decrypt_chromium_v10_values(local_state: Path, values: dict[str, list[int]]) -> dict[str, str]:
    if sys.platform != "win32":
        raise CredentialError("VS Code SecretStorage decryption is only implemented for Windows.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise CredentialError("Install cryptography to decrypt VS Code SecretStorage.") from exc

    local_state_data = json.loads(local_state.read_text(encoding="utf-8"))
    encrypted_key = base64.b64decode(local_state_data["os_crypt"]["encrypted_key"])
    if not encrypted_key.startswith(b"DPAPI"):
        raise CredentialError("Unsupported VS Code os_crypt key format.")
    master_key = _windows_unprotect(encrypted_key[5:])
    aesgcm = AESGCM(master_key)
    decrypted: dict[str, str] = {}
    for name, byte_values in values.items():
        blob = bytes(byte_values)
        if len(blob) < 32 or not blob.startswith(b"v10"):
            raise CredentialError(f"Unsupported secret blob format for {name}.")
        nonce = blob[3:15]
        ciphertext_and_tag = blob[15:]
        plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, None)
        decrypted[name] = plaintext.decode("utf-8")
    return decrypted


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _windows_unprotect(data: bytes) -> bytes:
    input_buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(len(data), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output_blob = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output_blob.pbData)
