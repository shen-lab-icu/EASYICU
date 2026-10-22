"""A stopped hosted instance can move to another model on its stored connection.

A hosted page refuses model changes (the administrator manages them), and
``provision-model`` takes a new key file.  When the provider stops serving
one model (pilot1, 10-22: the local relay put the configured model in a
week-long cooldown), the administrator moves the instance to another model
the same connection reports, without handling the key again.  The provider
owner re-verifies with one bounded inference and saves only after it
answers; a running instance, a model the connection does not report, or a
refused inference changes nothing, and no output carries the key.
"""

from __future__ import annotations

import importlib
import json
import socket
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore

_KEY = "synthetic-test-key"


@pytest.fixture
def tooling(monkeypatch):
    # The provider owner's address policy resolves the endpoint; keep it offline.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    return importlib.import_module("tools.hosted_instances")


def _verifier(*, answering: bool = True, calls: list | None = None):
    def verify(method, url, headers, body, timeout):
        assert headers["Authorization"] == f"Bearer {_KEY}"
        if calls is not None:
            calls.append((method, body))
        if method == "GET":
            return 200, {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        if not answering:
            return 429, {"error": {"code": "model_cooldown"}}
        return 200, {"choices": [{"message": {"content": "READY"}}]}
    return verify


def _provisioned(tooling, tmp_path):
    users = tmp_path / "users"
    data = tmp_path / "shared-data"
    data.mkdir()
    assert tooling.main([
        "--users-root", str(users), "create", "doctor", "--display-name", "王医生",
        "--data-root", str(data), "--runner-image", "easyicu-runner:synthetic",
    ]) == 0
    instance = users / "doctor"
    store = PiProviderConfigStore(config_path=instance / "home/.easyicu/pi-provider.env",
                                  receipt_path=instance / "home/.easyicu/pi-provider-verification.json")
    store.verify_and_save(provider="deepseek", api_key=_KEY, base_url="https://relay.example/v1",
                          model="model-a", api_transport="openai-completions", verifier=_verifier())
    return instance, store


def _model(store):
    return store.resolved_config(environ={}).model


def test_the_model_moves_and_the_connection_stays(tooling, tmp_path, monkeypatch, capsys):
    instance, store = _provisioned(tooling, tmp_path)
    before = store.resolved_config(environ={})
    # The administrator's own shell does not override the instance's connection.
    monkeypatch.setenv("EASYICU_PI_BASE_URL", "https://elsewhere.example/v1")
    calls: list = []

    tooling.set_model(instance, "model-b", verifier=_verifier(calls=calls))

    after = store.resolved_config(environ={})
    assert after.model == "model-b"
    assert (after.provider, after.api_key, after.base_url) == (before.provider, before.api_key, before.base_url)
    assert [body["model"] for method, body in calls if method == "POST"] == ["model-b"]
    output = capsys.readouterr()
    assert "start doctor" in output.out and _KEY not in output.out + output.err


@pytest.mark.parametrize(
    "model, verifier",
    [("model-c", _verifier()), ("model-b", _verifier(answering=False))],
    ids=["not_reported", "inference_refused"],
)
def test_a_model_that_does_not_answer_changes_nothing(tooling, tmp_path, model, verifier):
    instance, store = _provisioned(tooling, tmp_path)
    config = (instance / "home/.easyicu/pi-provider.env").read_bytes()

    with pytest.raises(Exception):
        tooling.set_model(instance, model, verifier=verifier)

    assert (instance / "home/.easyicu/pi-provider.env").read_bytes() == config
    assert _model(store) == "model-a"


def test_a_running_instance_keeps_its_model(tooling, tmp_path, monkeypatch):
    instance, store = _provisioned(tooling, tmp_path)
    (instance / "run/instance.pid.json").write_text(json.dumps({"pid": 12345, "create_time": 42.0}))
    command = ["python", "-m", "uvicorn", "easyicu.webserver.app:app", "--uds", str(instance / "run/web.sock")]
    process = SimpleNamespace(create_time=lambda: 42.0, cmdline=lambda: command)
    monkeypatch.setattr(tooling.psutil, "Process", lambda pid: process)
    calls: list = []

    with pytest.raises(ValueError, match="stop the instance"):
        tooling.set_model(instance, "model-b", verifier=_verifier(calls=calls))

    assert calls == [] and _model(store) == "model-a"


def test_the_command_is_documented_and_reports_only_a_code(tooling, tmp_path, monkeypatch, capsys):
    from easyicu.webserver.pi_copilot import provider_config

    instance, store = _provisioned(tooling, tmp_path)
    with pytest.raises(SystemExit):
        tooling.main(["set-model", "--help"])
    assert "stored connection" in capsys.readouterr().out

    def failing_transport(*_args, **_kwargs):
        raise RuntimeError(f"upstream echoed Bearer {_KEY}")

    monkeypatch.setattr(provider_config, "_default_verification_transport", failing_transport)
    status = tooling.main(["--users-root", str(instance.parent), "set-model", "doctor", "model-b"])
    output = capsys.readouterr()
    assert status != 0 and _KEY not in output.out + output.err
    assert _model(store) == "model-a"
