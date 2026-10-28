"""Exercise the copy HTTP adapter directly, without a TestClient transport."""

import asyncio
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.webserver import hosted
from easyicu.webserver.routes import extensions as routes


@pytest.fixture
def registry(tmp_path, monkeypatch):
    registry = ExtensionRegistry(tmp_path / "extensions")
    monkeypatch.setattr(routes, "_registry", lambda: registry)
    monkeypatch.setattr(routes.settings_store, "load_settings", lambda: {})
    return registry


def test_copy_adapter_returns_disabled_skill_and_current_catalog(registry):
    request = routes.SkillCopyRequest(
        builtin_id="cohort-characterization-table-one",
        name="personal-cohort",
        stages=["conversation", "writing"],
        expected_sha256=registry.snapshot().activation_sha256,
    )
    response = routes.post_copy_skill(request)
    assert response["ok"] is True
    assert response["skill"]["name"] == "personal-cohort"
    assert response["skill"]["enabled"] is False
    assert response["skill"]["stages"] == ["conversation", "writing"]
    assert response["extensions"]["skills"] == [response["skill"]]
    assert any(
        route.path == "/api/extensions/skills/copy" and "POST" in route.methods
        for route in routes.router.routes
    )


def test_copy_adapter_reports_stale_revision_and_name_conflict(registry):
    request = routes.SkillCopyRequest(
        builtin_id="survival-time-to-event",
        expected_sha256=registry.snapshot().activation_sha256,
    )
    routes.post_copy_skill(request)
    with pytest.raises(HTTPException) as stale:
        routes.post_copy_skill(request)
    assert stale.value.status_code == 409
    assert stale.value.detail["error"] == "extension_revision_mismatch"
    assert (
        stale.value.detail["activation_sha256"] == registry.snapshot().activation_sha256
    )
    request.expected_sha256 = registry.snapshot().activation_sha256
    with pytest.raises(HTTPException) as conflict:
        routes.post_copy_skill(request)
    assert conflict.value.status_code == 409
    assert conflict.value.detail["error"] == "extension_skill_name_conflict"


@pytest.mark.parametrize(
    "extra", [{"enabled": True}, {"stages": ["execution"]}, {"source": {}}]
)
def test_copy_request_cannot_enable_or_grant_host_authority(extra):
    with pytest.raises(ValidationError):
        routes.SkillCopyRequest(
            builtin_id="survival-time-to-event", expected_sha256="a" * 64, **extra
        )


def test_copy_request_requires_revision():
    with pytest.raises(ValidationError):
        routes.SkillCopyRequest(builtin_id="survival-time-to-event")


def test_copy_is_allowed_in_the_hosted_users_workspace(monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")

    async def payload():
        return {}

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/extensions/skills/copy"),
        query_params={},
        json=payload,
    )
    asyncio.run(hosted.guard_request(request))
