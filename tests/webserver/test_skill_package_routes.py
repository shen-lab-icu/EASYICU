"""Call upload/export adapters and the hosted guard directly, without ASGI."""

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
import zipfile

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from easyicu.webserver import hosted
from easyicu.webserver.routes import extensions as routes


SKILL = "---\nname: synthetic-upload\ndescription: Explain a synthetic example.\n---\nRead references/a.md.\n"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "user-a"))
    monkeypatch.delenv("EASYICU_EXTENSION_HOME", raising=False)
    monkeypatch.setattr(routes.settings_store, "load_settings", lambda: {})
    return tmp_path


def guard(path, body):
    async def payload():
        return body

    asyncio.run(hosted.guard_request(SimpleNamespace(
        method="POST", url=SimpleNamespace(path=path), query_params={}, json=payload,
    )))


@pytest.mark.parametrize("hosted_mode", ["0", "1"])
def test_uploaded_folder_is_editable_exportable_and_removable_in_own_workspace(workspace, monkeypatch, hosted_mode):
    monkeypatch.setenv("EASYICU_HOSTED", hosted_mode)
    files = {"local/SKILL.md": SKILL, "local/references/a.md": "Explain each assumption."}
    guard("/api/extensions/skills/import", {"files": files})
    registry = routes._registry()
    result = routes.post_import_skill_package(routes.SkillPackageImportRequest(
        files=files, expected_sha256=registry.snapshot().activation_sha256,
    ))
    assert result["extensions"]["skills"][0]["name"] == "synthetic-upload"
    current = routes.get_installed_skill("synthetic-upload")
    guard("/api/extensions/skills/install", {"skill_md": current["skill_md"]})
    routes.post_install_skill(routes.SkillInstallRequest(
        skill_md=current["skill_md"].replace("Read references", "First read references"),
        reference_files=current["reference_files"], expected_sha256=result["extensions"]["activation_sha256"],
    ))
    response = routes.get_skill_package("synthetic-upload")
    assert response.media_type == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="synthetic-upload.zip"'
    with zipfile.ZipFile(BytesIO(response.body)) as package:
        assert b"First read" in package.read("SKILL.md")
        assert package.read("references/a.md") == b"Explain each assumption."
    revision = registry.snapshot().activation_sha256
    guard("/api/extensions/state", {"kind": "skill"})
    routes.post_extension_state(routes.ExtensionStateRequest(kind="skill", name="synthetic-upload", enabled=False, expected_sha256=revision))
    guard("/api/extensions/remove", {"kind": "skill"})
    routes.post_remove_extension(routes.ExtensionRemoveRequest(kind="skill", name="synthetic-upload", expected_sha256=registry.snapshot().activation_sha256))
    assert registry.list_public()["skills"] == []
    monkeypatch.setenv("EASYICU_HOME", str(workspace / "user-b"))
    with pytest.raises(HTTPException):
        routes.get_skill_package("synthetic-upload")


def test_zip_route_bounds_payload_and_returns_schema_and_revision_failures(workspace):
    from easyicu.extensions.skill_package import write_archive

    registry = routes._registry()
    body = routes.SkillPackageImportRequest(
        archive_base64=base64.b64encode(write_archive(SKILL, {})).decode(),
        expected_sha256=registry.snapshot().activation_sha256,
    )
    assert routes.post_import_skill_package(body)["ok"]
    with pytest.raises(HTTPException) as stale:
        routes.post_import_skill_package(body)
    assert stale.value.status_code == 409
    for patch in ({"archive_base64": "bad%%"}, {"files": {"SKILL.md": "invalid"}, "archive_base64": None}, {"files": {"../SKILL.md": SKILL}, "archive_base64": None}):
        invalid = body.model_copy(update=patch)
        with pytest.raises(HTTPException) as failure:
            routes.post_import_skill_package(invalid)
        assert failure.value.status_code == 400


@pytest.mark.parametrize("extra", [{"path": "/server/skill.zip"}, {"files": {}, "stages": ["execution"]}, {"archive_base64": "x" * 341_337}])
def test_import_request_accepts_no_server_path_or_execution_stage(extra):
    with pytest.raises(ValidationError):
        routes.SkillPackageImportRequest(expected_sha256="a" * 64, **extra)


def test_hosted_user_cannot_mutate_mcp_or_escape_workspace(workspace, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    for path in ("/api/extensions/state", "/api/extensions/remove", "/api/extensions/mcp/install", "/api/extensions/mcp/test"):
        with pytest.raises(HTTPException) as failure:
            guard(path, {"kind": "mcp"})
        assert failure.value.status_code == 403
    monkeypatch.setenv("EASYICU_EXTENSION_HOME", str(workspace / "other-user"))
    with pytest.raises(HTTPException) as failure:
        routes.post_import_skill_package(routes.SkillPackageImportRequest(files={"SKILL.md": SKILL}, expected_sha256="a" * 64))
    assert failure.value.status_code == 403
    assert not (workspace / "other-user").exists()
