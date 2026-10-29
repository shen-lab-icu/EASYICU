"""Desktop credentials belong to the native OS credential store, never a file.

The desktop index contains only connection coordinates. Desktop legacy keys
migrate to the OS credential store before their plaintext entry is removed.
Hosted/server instances retain their administrator-provisioned connection file.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from easyicu.webserver.hosted import deployment_kind

from .contracts import PiCopilotError

KEY = "EASYICU_PI_API_KEY"
IDENTITY_KEYS = ("EASYICU_PI_PROVIDER", "EASYICU_PI_BASE_URL", "EASYICU_PI_API")


class CredentialBackend(Protocol):
    def get_password(self, service: str, account: str) -> str | None: ...
    def set_password(self, service: str, account: str, password: str) -> None: ...
    def delete_password(self, service: str, account: str) -> None: ...


def credential_error(code: str = "store_unavailable") -> PiCopilotError:
    messages = {
        "desktop_required": "Remembering a key is available only in the desktop app.",
        "store_unavailable": "无法访问系统凭据库。请解锁系统凭据库，或取消「在本机记住此密钥」后重试。",
        "not_found": "The remembered connection no longer exists.",
    }
    return PiCopilotError(f"pi_credential_{code}", messages[code], status_code=409)


class MacOSCredentials:
    """Use Apple's fixed system binary; never surface subprocess diagnostics.

    security's supported non-interactive write takes the secret on argv. It is
    briefly visible to local process inspection, but is never echoed or logged.
    """

    def __init__(self) -> None:
        binary = Path("/usr/bin/security")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise credential_error()

    @staticmethod
    def _run(arguments: list[str], *, absent_ok: bool = False) -> str | None:
        try:
            result = subprocess.run(
                ["/usr/bin/security", *arguments], capture_output=True,
                text=True, check=False, timeout=15,
            )
            # security maps errSecItemNotFound (-25300) to exit status 44.
            if absent_ok and result.returncode == 44:
                return None
            if result.returncode != 0:
                raise credential_error()
            return result.stdout.removesuffix("\n")
        except Exception:
            raise credential_error() from None

    def set_password(self, service: str, account: str, password: str) -> None:
        self._run(["add-generic-password", "-U", "-s", service, "-a", account, "-w", password])

    def get_password(self, service: str, account: str) -> str | None:
        return self._run(["find-generic-password", "-s", service, "-a", account, "-w"], absent_ok=True)

    def delete_password(self, service: str, account: str) -> None:
        self._run(["delete-generic-password", "-s", service, "-a", account], absent_ok=True)


class _WindowsFileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _WindowsCredential(ctypes.Structure):
    # Fixed-width DWORDs also let the ABI boundary be tested on other OSes.
    _fields_ = [
        ("Flags", ctypes.c_uint32), ("Type", ctypes.c_uint32),
        ("TargetName", ctypes.c_wchar_p), ("Comment", ctypes.c_wchar_p),
        ("LastWritten", _WindowsFileTime), ("CredentialBlobSize", ctypes.c_uint32),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", ctypes.c_uint32), ("AttributeCount", ctypes.c_uint32),
        ("Attributes", ctypes.c_void_p), ("TargetAlias", ctypes.c_wchar_p),
        ("UserName", ctypes.c_wchar_p),
    ]


class WindowsCredentials:
    """Generic local-machine credentials via advapi32, with UTF-16LE blobs."""

    def __init__(self) -> None:
        self.api = ctypes.WinDLL("advapi32", use_last_error=True)
        pointer = ctypes.POINTER(_WindowsCredential)
        self.api.CredWriteW.argtypes = [pointer, ctypes.c_uint32]
        self.api.CredReadW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(pointer)]
        self.api.CredDeleteW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        for name in ("CredWriteW", "CredReadW", "CredDeleteW"):
            getattr(self.api, name).restype = ctypes.c_int32
        self.api.CredFree.argtypes = [ctypes.c_void_p]
        self.api.CredFree.restype = None

    def set_password(self, service: str, account: str, password: str) -> None:
        try:
            encoded = password.encode("utf-16-le")
            blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
            row = _WindowsCredential()
            row.Type = 1  # CRED_TYPE_GENERIC
            row.TargetName = "EasyICU/" + account
            row.UserName = account
            row.CredentialBlobSize = len(encoded)
            row.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
            row.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE (this user's local machine)
            if not self.api.CredWriteW(ctypes.byref(row), 0):
                raise credential_error()
        except Exception:
            raise credential_error() from None

    def get_password(self, service: str, account: str) -> str | None:
        pointer = ctypes.POINTER(_WindowsCredential)()
        try:
            try:
                if not self.api.CredReadW("EasyICU/" + account, 1, 0, ctypes.byref(pointer)):
                    if ctypes.get_last_error() == 1168:  # ERROR_NOT_FOUND
                        return None
                    raise credential_error()
                row = pointer.contents
                return ctypes.string_at(row.CredentialBlob, row.CredentialBlobSize).decode("utf-16-le")
            finally:
                if pointer:
                    self.api.CredFree(pointer)
        except Exception:
            raise credential_error() from None

    def delete_password(self, service: str, account: str) -> None:
        try:
            if not self.api.CredDeleteW("EasyICU/" + account, 1, 0) and ctypes.get_last_error() != 1168:
                raise credential_error()
        except Exception:
            raise credential_error() from None


def _native_backend() -> CredentialBackend:
    try:
        if sys.platform == "darwin":
            return MacOSCredentials()
        if sys.platform == "win32":
            return WindowsCredentials()
        raise credential_error()
    except Exception:
        raise credential_error() from None


class ProviderCredentials:
    def __init__(self, path: Path, write: Callable[[Path, str], None], *, backend: CredentialBackend | None = None):
        self.path = path
        self.index_path = path.with_name(path.name + ".keyring.json")
        self.write = write
        self.backend = backend
        self.service = "EasyICU model connections " + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
        self.session: dict[str, str] = {}
        self.remember: bool | None = None

    @staticmethod
    def is_desktop() -> bool:
        return deployment_kind() == "desktop"

    def supported(self) -> bool:
        if not self.is_desktop():
            return False
        try:
            self._backend()
            return True
        except PiCopilotError:
            return False

    def unavailable_reason(self) -> str:
        return "系统凭据库不可用；密钥仅用于本次启动。" if self.is_desktop() and not self.supported() else ""

    def check_remember(self, remember: bool | None) -> bool:
        choice = self.remember if remember is None and self.remember is not None else (
            self.supported() if remember is None else remember
        )
        if choice and not self.is_desktop():
            raise credential_error("desktop_required")
        if choice:
            self._backend()
        return choice

    def _backend(self) -> CredentialBackend:
        if not self.is_desktop():
            raise credential_error("desktop_required")
        try:
            if self.backend is None:
                self.backend = _native_backend()
            return self.backend
        except Exception:
            # Some backends include their arguments in exceptions. Neither
            # logs nor chained tracebacks may carry them to the HTTP boundary.
            raise credential_error() from None

    def _call(self, operation: str, account: str, password: str | None = None) -> Any:
        backend = self._backend()
        try:
            if operation == "set":
                return backend.set_password(self.service, account, password)
            if operation == "delete":
                return backend.delete_password(self.service, account)
            return backend.get_password(self.service, account)
        except Exception:
            raise credential_error() from None

    @staticmethod
    def identity(values: Mapping[str, str]) -> str:
        canonical = [values.get(key, "").rstrip("/") for key in IDENTITY_KEYS]
        return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()

    def _index(self) -> dict[str, dict[str, str]]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            raise credential_error() from None
        if not isinstance(payload, dict):
            raise credential_error()
        # Project only the declared non-secret fields, even from an old index.
        return {str(key): {name: str(row.get(name, "")) for name in IDENTITY_KEYS}
                for key, row in payload.items() if isinstance(row, dict)}

    def _metadata(self, values: Mapping[str, str]) -> None:
        desktop = self.is_desktop()
        lines = ["# EasyICU connection coordinates; credentials live in the OS credential store."
                 if desktop else "# EasyICU administrator-provisioned model connection."]
        lines.extend(f"{key}={json.dumps(value, ensure_ascii=False)}"
                     for key, value in values.items() if key != KEY or not desktop)
        self.write(self.path, "\n".join(lines) + "\n")

    def save(self, values: Mapping[str, str], *, remember: bool | None = None) -> None:
        choice = self.check_remember(remember)
        account = self.identity(values)
        if self.is_desktop():
            index = self._index()
            if choice:
                self._call("set", account, values[KEY])
                index[account] = {key: values.get(key, "") for key in IDENTITY_KEYS}
            elif account in index:
                self._call("delete", account)
                index.pop(account)
            if choice or self.index_path.exists():
                self.write(self.index_path, json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        self._metadata(values)
        self.session = dict(values)
        self.remember = choice

    def resolve(self, values: Mapping[str, str]) -> dict[str, str]:
        result = dict(values)
        if not self.is_desktop():
            return result
        if self.session and self.identity(result) == self.identity(self.session):
            result[KEY] = self.session[KEY]
            return result
        if result.get(KEY):
            # Store first. On a native-store failure the original bytes stay intact.
            self.save(result, remember=True)
            return result
        account = self.identity(result)
        if self.supported() and account in self._index():
            password = self._call("get", account)
            if password:
                result[KEY] = password
                self.session = dict(result)
        return result

    def is_remembered(self, values: Mapping[str, str]) -> bool:
        """Read storage metadata without unlocking every remembered account."""
        return self.is_desktop() and self.identity(values) in self._index()

    def remembered(self) -> list[dict[str, str]]:
        if not self.is_desktop():
            return []
        return [{"id": account, "provider": row["EASYICU_PI_PROVIDER"],
                 "base_url": row["EASYICU_PI_BASE_URL"]}
                for account, row in self._index().items() if self._call("get", account)]

    def delete(self, account: str) -> None:
        if not self.is_desktop():
            raise credential_error("desktop_required")
        index = self._index()
        if account not in index:
            raise credential_error("not_found")
        self._call("delete", account)
        index.pop(account)
        self.write(self.index_path, json.dumps(index, ensure_ascii=False, indent=2) + "\n")
        if self.session and self.identity(self.session) == account:
            self.session = {}
            self.remember = False
