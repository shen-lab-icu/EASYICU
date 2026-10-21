"""Administrator tooling uses synthetic keys and mocked processes/network."""

import importlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore


@pytest.fixture
def tooling(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    return importlib.import_module("tools.hosted_instances")


def create(tooling, tmp_path):
    users = tmp_path / "users"
    data = tmp_path / "shared-data"
    data.mkdir()
    assert tooling.main(["--users-root", str(users), "create", "doctor", "--display-name", "王医生",
                         "--data-root", str(data), "--runner-image", "easyicu-runner:synthetic",
                         "--support-contact", "Hospital help desk", "--public-origin", "https://clinic.example"]) == 0
    return users, users / "doctor"


def test_create_is_private_and_provisions_settings_via_owner(tooling, tmp_path):
    users, instance = create(tooling, tmp_path)
    for directory in (users, instance, instance / "home", instance / "run"):
        assert directory.stat().st_mode & 0o777 == 0o700
    env_file = instance / "instance.env"
    assert env_file.stat().st_mode & 0o777 == 0o600
    env = tooling.instance_environment(instance)
    assert env["EASYICU_HOSTED"] == "1"
    assert env["EASYICU_USER_DISPLAY_NAME"] == "王医生"
    assert env["EASYICU_DATA_ROOTS"] == str(tmp_path / "shared-data")
    assert env["EASYICU_WEB_UNIX_SOCKET"] == str(instance / "run" / "web.sock")
    assert env["EASYICU_WEB_TRUST_PROXY"] == "1"
    assert env["EASYICU_RUNNER_IMAGE"] == "easyicu-runner:synthetic"
    saved = json.loads((instance / "home" / ".easyicu" / "webserver_settings.json").read_text())
    assert saved["support_contact"] == "Hospital help desk"


@pytest.mark.parametrize("name", ["../other", "a/b", ".", "..", "bad name", "-option"])
def test_create_refuses_unsafe_names(tooling, tmp_path, name):
    try:
        code = tooling.main(["--users-root", str(tmp_path / "users"), "create", name, "--display-name", "Synthetic",
                             "--data-root", str(tmp_path), "--runner-image", "synthetic"])
    except SystemExit as exc:
        code = exc.code
    assert code != 0


def test_create_never_overwrites_an_instance(tooling, tmp_path):
    users, instance = create(tooling, tmp_path)
    before = (instance / "instance.env").read_bytes()
    assert tooling.main(["--users-root", str(users), "create", "doctor", "--display-name", "Other",
                         "--data-root", str(tmp_path), "--runner-image", "other"]) != 0
    assert (instance / "instance.env").read_bytes() == before


def test_create_refuses_a_filesystem_root_grant(tooling, tmp_path):
    assert tooling.main(["--users-root", str(tmp_path / "users"), "create", "doctor", "--display-name", "Synthetic",
                         "--data-root", "/", "--runner-image", "synthetic"]) != 0


def fake_verify(method, url, headers, body, timeout):
    assert headers["Authorization"] == "Bearer synthetic-test-key"
    if method == "GET":
        return 200, {"data": [{"id": "deepseek-chat"}]}
    assert body["max_tokens"] == 8
    return 200, {"choices": [{"message": {"content": "READY"}}]}


def test_provision_uses_existing_verifier_and_changes_capability(tooling, tmp_path, monkeypatch, capsys):
    from easyicu.webserver import sources

    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key\n")
    key.chmod(0o600)
    store = PiProviderConfigStore(config_path=instance / "home/.easyicu/pi-provider.env",
                                  receipt_path=instance / "home/.easyicu/pi-provider-verification.json")
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": []})
    original = PiProviderConfigStore.public_status
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda self, **kw: original(store, environ={}))
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    assert TestClient(app).get("/api/capabilities").json()["deployment"]["provider_provisioned"] is False
    tooling.provision_model(instance, key, verifier=fake_verify)
    status = store.public_status()
    assert status["connection_verified"] and status["inference_verified"]
    deployment = TestClient(app).get("/api/capabilities").json()["deployment"]
    assert deployment == {"hosted": True, "provider_provisioned": True}
    output = capsys.readouterr()
    assert "synthetic-test-key" not in output.out + output.err
    saved = json.loads((instance / "home/.easyicu/webserver_settings.json").read_text())
    assert saved["ai_enabled"] is True
    assert saved["support_contact"] == "Hospital help desk"


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o666])
def test_provision_refuses_readable_key_before_read_or_network(tooling, tmp_path, mode):
    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key")
    key.chmod(mode)
    calls = []
    with pytest.raises(ValueError, match="private"):
        tooling.provision_model(instance, key, verifier=lambda *a: calls.append(a))
    assert calls == []
    assert not (instance / "home/.easyicu/pi-provider.env").exists()


def test_failed_verification_does_not_persist_or_enable_ai(tooling, tmp_path):
    _, instance = create(tooling, tmp_path)
    key = tmp_path / "key"
    key.write_text("synthetic-test-key")
    key.chmod(0o600)
    with pytest.raises(Exception, match="credential"):
        tooling.provision_model(instance, key, verifier=lambda *a: (401, {}))
    assert not (instance / "home/.easyicu/pi-provider.env").exists()
    settings = json.loads((instance / "home/.easyicu/webserver_settings.json").read_text())
    assert settings["ai_enabled"] is False


def test_start_uses_shared_checkout_image_and_socket_and_records_identity(tooling, tmp_path, monkeypatch):
    users, instance = create(tooling, tmp_path)
    calls = []
    monkeypatch.setattr(tooling.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(pid=12345, poll=lambda: None))
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42.0))
    assert tooling.main(["--users-root", str(users), "start", "doctor"]) == 0
    cmd, kwargs = calls[0]
    assert "--uds" in cmd and "--host" not in cmd and "--port" not in cmd
    assert cmd[cmd.index("--uds") + 1] == str(instance / "run/web.sock")
    assert "--proxy-headers" in cmd
    assert kwargs["env"]["EASYICU_RUNNER_IMAGE"] == "easyicu-runner:synthetic"
    assert Path(kwargs["cwd"]) == Path(__file__).resolve().parents[2]
    assert "EASYICU_PI_API_KEY" not in kwargs["env"]
    record = json.loads((instance / "run/instance.pid.json").read_text())
    assert record["pid"] == 12345 and record["create_time"] == 42.0


@pytest.mark.parametrize("mismatch", [None, "time", "application", "socket"])
def test_stop_only_signals_matching_recorded_process(tooling, tmp_path, monkeypatch, mismatch):
    users, instance = create(tooling, tmp_path)
    socket = str(instance / "run/web.sock")
    (instance / "run/instance.pid.json").write_text(json.dumps({"pid": 12345, "create_time": 42.0}))
    signals = []
    command = ["python", "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", socket]
    if mismatch == "application":
        command[3] = "another.application:app"
    if mismatch == "socket":
        command[-1] = str(instance / "run/another.sock")
    process = SimpleNamespace(create_time=lambda: 99.0 if mismatch == "time" else 42.0,
                              cmdline=lambda: command,
                              terminate=lambda: signals.append(12345), wait=lambda timeout: None)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(tooling.psutil, "process_iter", lambda *a, **kw: pytest.fail("No process sweep"))
    assert tooling.main(["--users-root", str(users), "stop", "doctor"]) == 0
    assert signals == ([] if mismatch else [12345])


def test_list_status_do_not_read_provider_keys(tooling, tmp_path, monkeypatch, capsys):
    users, instance = create(tooling, tmp_path)
    original = Path.read_text

    def read(path, *args, **kw):
        assert path.name != "pi-provider.env"
        return original(path, *args, **kw)

    monkeypatch.setattr(Path, "read_text", read)
    assert tooling.main(["--users-root", str(users), "list"]) == 0
    assert tooling.main(["--users-root", str(users), "status", "doctor"]) == 0
    assert "doctor" in capsys.readouterr().out
    assert not (instance / "run/instance.pid.json").exists()
    assert os.environ.get("EASYICU_HOME") != str(instance / "home")
