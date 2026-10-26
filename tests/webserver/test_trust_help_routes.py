import io
import json
import zipfile
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from easyicu.webserver import run_refs
from easyicu.webserver.routes import trust_help
from tests.support.trust_help import call, ledger


def test_read_routes_resolve_only_issued_run_references(tmp_path, monkeypatch):
    checks = []
    monkeypatch.setattr(trust_help, "require_instance_path", checks.append)
    ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call()])
    ref = run_refs.issue(str(tmp_path))
    result = trust_help.get_run_egress(ref)
    assert json.loads(result.body)["requests"][0]["byte_size"] == 42
    assert result.headers["cache-control"] == "no-store"
    bundle = trust_help.get_problem_bundle(ref)
    assert bundle.media_type == "application/zip"
    assert "attachment" in bundle.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(bundle.body)) as archive:
        assert "manifest.json" in archive.namelist()
    assert checks == [str(tmp_path), str(tmp_path)]
    with pytest.raises(HTTPException) as caught:
        trust_help.get_run_egress(str(tmp_path))
    assert caught.value.status_code == 404


def test_whats_new_read_does_not_acknowledge_an_update():
    first = trust_help.get_whats_new()
    assert first.body == trust_help.get_whats_new().body
    latest = json.loads(first.body)["latest_version"]
    assert not json.loads(trust_help.get_whats_new(latest).body)["show_after_update"]
    with pytest.raises(HTTPException) as caught:
        trust_help.get_whats_new("invalid")
    assert caught.value.status_code == 400


def test_conversation_read_uses_only_existing_store_and_receipts(tmp_path, monkeypatch):
    service = SimpleNamespace(gateway=SimpleNamespace(session_dir=tmp_path),
        _get_record=lambda identity: SimpleNamespace(session_id=identity))
    monkeypatch.setattr(trust_help, "get_pi_copilot_service", lambda: service)
    response = trust_help.get_conversation_egress("synthetic")
    assert json.loads(response.body)["status"] == "not_recorded"
    assert not list(tmp_path.iterdir())


def test_router_is_mounted_before_static_files():
    from easyicu.webserver.app import app
    from pathlib import Path
    import easyicu.webserver.app as owner
    paths = app.openapi()["paths"]
    for route in trust_help.router.routes:
        assert route.path in paths
    source = Path(owner.__file__).read_text()
    assert source.index("app.include_router(trust_help_router)") < source.index("app.mount(")
