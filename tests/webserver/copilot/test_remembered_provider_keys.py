"""Only a fake OS keyring sees these synthetic credentials."""

import json
import ast
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeCredentialBackend

SYNTHETIC_KEY = "synthetic-regression-private-value"
CONNECTION = dict(provider="fixture", api_key=SYNTHETIC_KEY,
                  base_url="http://127.0.0.1:8317/v1", model="fixture-model",
                  api_transport="openai-completions")


def verified(method, *args):
    if method == "GET":
        return 200, {"data": [{"id": "fixture-model"}]}
    return 200, {"choices": [{"message": {"content": "READY"}}]}


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    backend = FakeCredentialBackend()
    def store():
        return PiProviderConfigStore(config_path=tmp_path / "connection.txt",
                                     receipt_path=tmp_path / "verification.json",
                                     credential_backend=backend)
    return backend, store


def test_desktop_remembers_and_reloads_without_plaintext(desktop, tmp_path, caplog):
    backend, store = desktop
    config, status = store().verify_and_save(**CONNECTION, verifier=verified, environ={})
    assert list(backend.values.values()) == [SYNTHETIC_KEY]
    assert status["credential_remembered"] is True
    loaded = store()
    assert loaded.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert loaded.public_status(environ={})["connection_verified"] is True
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SYNTHETIC_KEY not in path.read_text()
    assert SYNTHETIC_KEY not in json.dumps(status)
    assert SYNTHETIC_KEY not in repr(config)
    assert SYNTHETIC_KEY not in caplog.text
    with pytest.raises(PiCopilotError, match="authorization"):
        loaded.research_agent_environment(external_llm_opt_in=False, environ={})


def test_unchecked_key_lives_only_for_this_process(desktop, tmp_path):
    backend, factory = desktop
    current = factory()
    current.verify_and_save(**CONNECTION, remember_key=False, verifier=verified, environ={})
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert backend.values == {}
    assert factory().resolved_config(environ={}) is None
    assert all(SYNTHETIC_KEY not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_locked_store_keeps_status_readable_but_execution_blocked(desktop, monkeypatch):
    from easyicu.webserver.pi_copilot.provider_credentials import credential_error

    _, factory = desktop
    factory().verify_and_save(**CONNECTION, verifier=verified, environ={})
    loaded = factory()
    before = loaded.config_path.read_bytes()
    def locked(*_args):
        raise credential_error()
    monkeypatch.setattr(loaded.credentials, "_call", locked)
    status = loaded.public_status(environ={})
    assert status["connection_error"] == "pi_credential_store_unavailable"
    assert not status["connection_verified"] and not status["credential_present"]
    assert SYNTHETIC_KEY not in json.dumps(status)
    assert loaded.config_path.read_bytes() == before
    with pytest.raises(PiCopilotError, match="系统凭据库"):
        loaded.resolved_config(environ={})


@pytest.mark.parametrize("damage", ["missing", "changed_connection"])
def test_session_key_requires_matching_connection_coordinates(desktop, damage):
    _, factory = desktop
    store = factory()
    store.verify_and_save(**CONNECTION, remember_key=False, verifier=verified, environ={})
    if damage == "missing":
        store.config_path.unlink()
    else:
        store.config_path.write_text(store.config_path.read_text().replace('EASYICU_PI_PROVIDER="fixture"', 'EASYICU_PI_PROVIDER="other"'))
    assert store.resolved_config(environ={}) is None


def test_delete_remembered_key_disables_reload(desktop):
    backend, factory = desktop
    current = factory()
    current.verify_and_save(**CONNECTION, verifier=verified, environ={})
    remembered = current.remembered_credentials()
    assert remembered[0]["provider"] == "fixture"
    current.delete_remembered_credential(remembered[0]["id"])
    assert backend.values == {}
    assert factory().resolved_config(environ={}) is None
    assert current.remembered_credentials() == []


@pytest.mark.parametrize("deployment", ["hosted", "server"])
def test_other_deployments_refuse_native_key_remembering(desktop, monkeypatch, deployment, tmp_path):
    backend, factory = desktop
    monkeypatch.setenv("EASYICU_DEPLOYMENT", deployment)
    if deployment == "hosted":
        monkeypatch.setenv("EASYICU_HOSTED", "1")
    with pytest.raises(PiCopilotError) as error:
        factory().verify_and_save(**CONNECTION, remember_key=True, verifier=verified, environ={})
    assert error.value.code == "pi_credential_desktop_required"
    assert backend.values == {}
    assert list(tmp_path.iterdir()) == []


def test_plaintext_connection_is_migrated_only_after_keyring_write(desktop, tmp_path):
    backend, factory = desktop
    current = factory()
    config = current.make_config(**CONNECTION)
    legacy = "\n".join(f"{k}={json.dumps(v)}" for k, v in config.as_environment().items())
    current.config_path.write_text(legacy)
    current.config_path.chmod(0o600)
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert list(backend.values.values()) == [SYNTHETIC_KEY]
    assert SYNTHETIC_KEY not in current.config_path.read_text()
    assert factory().resolved_config(environ={}).api_key == SYNTHETIC_KEY


def test_unavailable_native_store_never_falls_back_to_plaintext(desktop, tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot import provider_credentials

    _, factory = desktop
    current = factory()
    current.credentials.backend = None
    def unavailable():
        raise RuntimeError(SYNTHETIC_KEY)
    monkeypatch.setattr(provider_credentials, "_native_backend", unavailable, raising=False)
    with pytest.raises(PiCopilotError) as error:
        current.verify_and_save(**CONNECTION, remember_key=True, verifier=verified, environ={})
    assert error.value.code == "pi_credential_store_unavailable"
    assert SYNTHETIC_KEY not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_desktop_without_a_native_store_disables_remembering_with_a_reason(desktop, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    current = PiProviderConfigStore(config_path=tmp_path / "connection.txt", receipt_path=tmp_path / "verification.json")
    status = current.public_status(environ={})
    assert status["remember_key_supported"] is False
    assert status["remember_key_unavailable_reason"]
    current.verify_and_save(**CONNECTION, verifier=verified, environ={})
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert all(SYNTHETIC_KEY not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_unavailable_store_does_not_erase_desktop_legacy_key(desktop, monkeypatch):
    from easyicu.webserver.pi_copilot import provider_credentials

    _, factory = desktop
    current = factory()
    current.credentials.backend = None
    values = current.make_config(**CONNECTION).as_environment()
    legacy = "\n".join(f"{key}={json.dumps(value)}" for key, value in values.items())
    current.config_path.write_text(legacy)
    current.config_path.chmod(0o600)
    def unavailable():
        raise RuntimeError(SYNTHETIC_KEY)
    monkeypatch.setattr(provider_credentials, "_native_backend", unavailable, raising=False)
    with pytest.raises(PiCopilotError):
        current.resolved_config(environ={})
    assert current.config_path.read_text() == legacy


def test_keyring_error_is_redacted_and_does_not_erase_legacy(desktop, tmp_path, caplog):
    backend, factory = desktop
    current = factory()
    config = current.make_config(**CONNECTION)
    legacy = "\n".join(f"{k}={json.dumps(v)}" for k, v in config.as_environment().items())
    current.config_path.write_text(legacy)
    current.config_path.chmod(0o600)
    def fail(*args):
        raise RuntimeError(SYNTHETIC_KEY)
    backend.set_password = fail
    with pytest.raises(PiCopilotError) as error:
        current.resolved_config(environ={})
    assert SYNTHETIC_KEY not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert current.config_path.read_text() == legacy
    assert SYNTHETIC_KEY not in caplog.text


def test_credential_owners_have_no_secret_logging_sinks():
    root = Path(__file__).parents[3] / "src/easyicu/webserver"
    for filename in ("provider_config.py", "provider_credentials.py"):
        tree = ast.parse((root / "pi_copilot" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                assert name != "print"
                assert name.rsplit(".", 1)[-1] not in {"debug", "info", "warning", "error", "exception", "critical", "log"}
    control = (root / "static/js/screens-guided-pi-provider-control.js").read_text()
    assert "console." not in control


def test_multiple_providers_keep_separate_keys_and_delete_only_one(desktop):
    backend, factory = desktop
    current = factory()
    current.verify_and_save(**CONNECTION, verifier=verified, environ={})
    current.verify_and_save(**{**CONNECTION, "provider": "second", "api_key": "second-synthetic"}, verifier=verified, environ={})
    rows = current.remembered_credentials()
    assert {row["provider"] for row in rows} == {"fixture", "second"}
    current.delete_remembered_credential(next(row["id"] for row in rows if row["provider"] == "fixture"))
    assert list(backend.values.values()) == ["second-synthetic"]
    assert factory().resolved_config(environ={}).api_key == "second-synthetic"


@pytest.mark.parametrize("remember", [True, False])
def test_connection_route_forwards_the_remember_choice(monkeypatch, remember):
    from types import SimpleNamespace
    from easyicu.webserver.routes import pi_copilot as routes

    captured = []
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(configure_provider=lambda **kwargs: captured.append(kwargs) or {"ok": True}))
    monkeypatch.setattr(routes, "_connection_api_key", lambda body, request: body.api_key)
    body = routes.PiProviderConfigRequest(**CONNECTION, enable_ai=True, remember_key=remember)
    assert routes.post_pi_copilot_provider_config(body, None, None)["ok"]
    assert captured[0]["remember_key"] is remember


def test_settings_credential_routes_list_and_delete_without_returning_key(desktop, monkeypatch):
    from types import SimpleNamespace
    from threading import RLock
    from easyicu.webserver.routes import pi_copilot as routes

    _, factory = desktop
    store = factory()
    store.verify_and_save(**CONNECTION, verifier=verified, environ={})
    service = SimpleNamespace(provider_store=store, _lock=RLock(), _busy_sessions=set())
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    listed = routes.get_pi_remembered_credentials()
    assert SYNTHETIC_KEY not in json.dumps(listed)
    assert listed["credentials"][0]["provider"] == "fixture"
    account = listed["credentials"][0]["id"]
    result = routes.delete_pi_remembered_credential(account)
    assert result == {"ok": True, "credentials": []}
    assert factory().resolved_config(environ={}) is None


@pytest.mark.parametrize("deployment", ["hosted", "server"])
@pytest.mark.parametrize("remember", [None, False])
def test_other_deployments_persist_the_administrators_connection(desktop, monkeypatch, deployment, remember, tmp_path):
    backend, factory = desktop
    monkeypatch.setenv("EASYICU_DEPLOYMENT", deployment)
    if deployment == "hosted":
        monkeypatch.setenv("EASYICU_HOSTED", "1")
    current = factory()
    _, status = current.verify_and_save(**CONNECTION, remember_key=remember, verifier=verified, environ={})
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    loaded = factory()
    assert loaded.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert loaded.public_status(environ={})["inference_verified"] is True
    assert status["remember_key_supported"] is False
    assert current.config_path.stat().st_mode & 0o777 == 0o600
    assert not current.credentials.index_path.exists()
    assert backend.values == {}
    assert SYNTHETIC_KEY not in json.dumps(status)
    assert all(SYNTHETIC_KEY not in path.read_text() for path in tmp_path.rglob("*")
               if path.is_file() and path != current.config_path)


@pytest.mark.parametrize("deployment", ["hosted", "server"])
def test_reading_an_administrators_connection_preserves_its_bytes(desktop, monkeypatch, deployment):
    backend, factory = desktop
    monkeypatch.setenv("EASYICU_DEPLOYMENT", deployment)
    if deployment == "hosted":
        monkeypatch.setenv("EASYICU_HOSTED", "1")
    current = factory()
    values = current.make_config(**CONNECTION).as_environment()
    current.config_path.write_text("\n".join(f"{key}={json.dumps(value)}" for key, value in values.items()))
    current.config_path.chmod(0o600)
    before = current.config_path.read_bytes()
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert current.config_path.read_bytes() == before
    assert current.resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert factory().resolved_config(environ={}).api_key == SYNTHETIC_KEY
    assert backend.values == {}


@pytest.mark.parametrize("deployment", ["hosted", "server"])
def test_an_existing_server_store_reads_the_reprovisioned_key(desktop, monkeypatch, deployment):
    backend, factory = desktop
    monkeypatch.setenv("EASYICU_DEPLOYMENT", deployment)
    if deployment == "hosted":
        monkeypatch.setenv("EASYICU_HOSTED", "1")
    current = factory()
    current.verify_and_save(**CONNECTION, verifier=verified, environ={})
    replacement = "replacement-synthetic-value"
    factory().verify_and_save(**{**CONNECTION, "api_key": replacement}, verifier=verified, environ={})
    assert current.resolved_config(environ={}).api_key == replacement
    assert current.public_status(environ={})["connection_verified"] is True
    assert backend.values == {}
