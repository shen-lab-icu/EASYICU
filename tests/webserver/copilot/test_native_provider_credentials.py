"""Native credential boundaries use synthetic secrets and never touch the OS store."""

import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import provider_credentials as credentials
from easyicu.webserver.pi_copilot.contracts import PiCopilotError

SECRET = "synthetic-密钥-🔑 with spaces "
SERVICE = "EasyICU synthetic service"
ACCOUNT = "synthetic-connection"


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/usr/bin/security")
    monkeypatch.setattr(credentials.os, "access", lambda path, mode: str(path) == "/usr/bin/security")
    calls = []
    response = SimpleNamespace(returncode=0, stdout=SECRET + "\n", stderr=SECRET)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return response

    monkeypatch.setattr(credentials.subprocess, "run", run)
    return calls, response


def test_macos_uses_only_the_absolute_security_binary_and_exact_arguments(mac):
    calls, _ = mac
    backend = credentials._native_backend()
    backend.set_password(SERVICE, ACCOUNT, SECRET)
    assert backend.get_password(SERVICE, ACCOUNT) == SECRET
    backend.delete_password(SERVICE, ACCOUNT)
    assert [argv for argv, _ in calls] == [
        ["/usr/bin/security", "add-generic-password", "-U", "-s", SERVICE, "-a", ACCOUNT, "-w", SECRET],
        ["/usr/bin/security", "find-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w"],
        ["/usr/bin/security", "delete-generic-password", "-s", SERVICE, "-a", ACCOUNT],
    ]
    for _, options in calls:
        assert options == {"capture_output": True, "text": True, "check": False, "timeout": 15}


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_macos_item_not_found_is_absent(mac, operation):
    _, response = mac
    response.returncode = 44
    backend = credentials._native_backend()
    assert getattr(backend, operation + "_password")(SERVICE, ACCOUNT) is None


@pytest.mark.parametrize("operation", ["set", "get", "delete"])
@pytest.mark.parametrize("failure", ["exit", "exception"])
def test_macos_errors_do_not_expose_argv_stdout_or_stderr(mac, monkeypatch, caplog, operation, failure):
    _, response = mac
    response.returncode = 1
    if failure == "exception":
        def fail(*args, **kwargs):
            raise RuntimeError(SECRET)
        monkeypatch.setattr(credentials.subprocess, "run", fail)
    backend = credentials._native_backend()
    args = (SERVICE, ACCOUNT, SECRET) if operation == "set" else (SERVICE, ACCOUNT)
    with pytest.raises(PiCopilotError) as error:
        getattr(backend, operation + "_password")(*args)
    assert error.value.code == "pi_credential_store_unavailable"
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert SECRET not in caplog.text


def test_macos_write_not_found_is_an_error(mac):
    _, response = mac
    response.returncode = 44
    with pytest.raises(PiCopilotError):
        credentials._native_backend().set_password(SERVICE, ACCOUNT, SECRET)


@pytest.mark.parametrize("missing", ["file", "executable"])
def test_macos_requires_the_exact_binary_to_exist_and_be_executable(mac, monkeypatch, missing):
    calls, _ = mac
    if missing == "file":
        monkeypatch.setattr(Path, "is_file", lambda _: False)
    else:
        monkeypatch.setattr(credentials.os, "access", lambda *_: False)
    with pytest.raises(PiCopilotError):
        credentials._native_backend()
    assert calls == []


class NativeCall:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class FakeAdvapi:
    def __init__(self):
        self.values = {}
        self.freed = []
        self.last_error = 1168
        self.failure = False
        self.read_flags = []
        self.delete_flags = []
        self.CredWriteW = NativeCall(self.write)
        self.CredReadW = NativeCall(self.read)
        self.CredDeleteW = NativeCall(self.delete)
        self.CredFree = NativeCall(self.free)

    def write(self, pointer, flags):
        if self.failure:
            return 0
        row = ctypes.cast(pointer, ctypes.POINTER(credentials._WindowsCredential)).contents
        self.values[row.TargetName] = {
            "blob": ctypes.string_at(row.CredentialBlob, row.CredentialBlobSize),
            "type": row.Type, "persist": row.Persist, "flags": row.Flags,
            "call_flags": flags, "username": row.UserName,
        }
        return 1

    def read(self, target, kind, flags, output):
        self.read_flags.append((target, kind, flags))
        if self.failure or target not in self.values:
            return 0
        blob = self.values[target]["blob"]
        self.buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        self.row = credentials._WindowsCredential()
        self.row.Type = kind
        self.row.CredentialBlobSize = len(blob)
        self.row.CredentialBlob = ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_ubyte))
        ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(credentials._WindowsCredential)))[0] = ctypes.pointer(self.row)
        return 1

    def delete(self, target, kind, flags):
        self.delete_flags.append((target, kind, flags))
        if self.failure or target not in self.values:
            return 0
        del self.values[target]
        return 1

    def free(self, pointer):
        self.freed.append(ctypes.cast(pointer, ctypes.c_void_p).value)


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(credentials.sys, "platform", "win32")
    api = FakeAdvapi()
    loaded = []

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return api

    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: api.last_error, raising=False)
    return api, loaded


def test_windows_utf16_round_trip_namespaces_target_and_frees_read_buffer(windows):
    api, loaded = windows
    backend = credentials._native_backend()
    backend.set_password(SERVICE, ACCOUNT, SECRET)
    assert loaded == [("advapi32", {"use_last_error": True})]
    assert api.values == {"EasyICU/" + ACCOUNT: {
        "blob": SECRET.encode("utf-16-le"), "type": 1, "persist": 2,
        "flags": 0, "call_flags": 0, "username": ACCOUNT,
    }}
    assert backend.get_password(SERVICE, ACCOUNT) == SECRET
    assert api.read_flags == [("EasyICU/" + ACCOUNT, 1, 0)]
    assert len(api.freed) == 1
    assert api.freed[0] == ctypes.addressof(api.row)
    backend.delete_password(SERVICE, ACCOUNT)
    assert api.delete_flags == [("EasyICU/" + ACCOUNT, 1, 0)]
    assert api.values == {}
    for name in ("CredWriteW", "CredReadW", "CredDeleteW", "CredFree"):
        assert getattr(api, name).argtypes
    assert api.CredFree.restype is None


def test_windows_native_record_matches_the_credentialw_abi():
    pointer_bytes = ctypes.sizeof(ctypes.c_void_p)
    assert ctypes.sizeof(credentials._WindowsCredential) == (80 if pointer_bytes == 8 else 52)
    expected = {
        "Flags": 0, "Type": 4, "TargetName": 8,
        "Comment": 16 if pointer_bytes == 8 else 12,
        "LastWritten": 24 if pointer_bytes == 8 else 16,
        "CredentialBlobSize": 32 if pointer_bytes == 8 else 24,
        "CredentialBlob": 40 if pointer_bytes == 8 else 28,
        "Persist": 48 if pointer_bytes == 8 else 32,
        "AttributeCount": 52 if pointer_bytes == 8 else 36,
        "Attributes": 56 if pointer_bytes == 8 else 40,
        "TargetAlias": 64 if pointer_bytes == 8 else 44,
        "UserName": 72 if pointer_bytes == 8 else 48,
    }
    assert {name: getattr(credentials._WindowsCredential, name).offset for name in expected} == expected


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_windows_item_not_found_is_absent(windows, operation):
    api, _ = windows
    backend = credentials._native_backend()
    assert getattr(backend, operation + "_password")(SERVICE, ACCOUNT) is None
    assert api.freed == []


def test_windows_write_not_found_is_an_error(windows):
    api, _ = windows
    api.failure = True
    with pytest.raises(PiCopilotError):
        credentials._native_backend().set_password(SERVICE, ACCOUNT, SECRET)


@pytest.mark.parametrize("operation", ["set", "get", "delete"])
@pytest.mark.parametrize("failure", ["return", "exception"])
def test_windows_errors_are_redacted(windows, caplog, operation, failure):
    api, _ = windows
    backend = credentials._native_backend()
    api.failure = True
    api.last_error = 5
    if failure == "exception":
        def fail(*args):
            raise RuntimeError(SECRET)
        getattr(api, {"set": "CredWriteW", "get": "CredReadW", "delete": "CredDeleteW"}[operation]).call = fail
    args = (SERVICE, ACCOUNT, SECRET) if operation == "set" else (SERVICE, ACCOUNT)
    with pytest.raises(PiCopilotError) as error:
        getattr(backend, operation + "_password")(*args)
    assert error.value.code == "pi_credential_store_unavailable"
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert SECRET not in caplog.text


def test_windows_always_frees_a_read_buffer_even_when_decoding_fails(windows):
    api, _ = windows
    api.values["EasyICU/" + ACCOUNT] = {"blob": b"\xff"}
    with pytest.raises(PiCopilotError) as error:
        credentials._native_backend().get_password(SERVICE, ACCOUNT)
    assert len(api.freed) == 1
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("platform", ["linux", "freebsd"])
def test_other_platforms_never_select_a_file_backend(monkeypatch, platform, windows):
    _, loaded = windows
    monkeypatch.setattr(credentials.sys, "platform", platform)
    with pytest.raises(PiCopilotError) as error:
        credentials._native_backend()
    assert error.value.code == "pi_credential_store_unavailable"
    assert loaded == []


def test_missing_windows_native_api_is_redacted(windows, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(ctypes, "WinDLL", fail)
    with pytest.raises(PiCopilotError) as error:
        credentials._native_backend()
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__ is True
