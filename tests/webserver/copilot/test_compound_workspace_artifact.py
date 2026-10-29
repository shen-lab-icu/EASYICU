"""Compound file delivery spends one mutation and preserves each stage's truth."""
from __future__ import annotations

import hashlib
import json
from unittest.mock import Mock

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.tools import execute_tool
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace

HTML = "<!doctype html><p>before</p><script>throw new Error('RAW_HTML_SENTINEL')</script>"


def setup_artifact(tmp_path, mode):
    workspace = ProjectWorkspace(tmp_path / "workspace")
    context = ToolExecutionContext(session=PiSessionRecord(
        session_id="compound-session", project_id="compound-project", agent_mode="workspace"),
        workspace=workspace, allowed_actions=frozenset({"workspace_write"}))
    if mode == "write":
        params = {"file": "page.html", "content": HTML}
    else:
        before = workspace.write_file("compound-project", "page.html", HTML)
        params = {"file": "page.html", "old_text": ">before<", "new_text": ">after<", "expected_sha256": before["sha256"]}
    return workspace, context, params


def run(context, mode, params, **flags):
    return execute_tool(f"easyicu_{mode}_project_file", {**params, **flags}, context)


@pytest.mark.parametrize("mode", ["write", "edit"])
def test_compound_mutation_checks_and_prepares_one_exact_webpage_without_executing_or_projecting_html(tmp_path, monkeypatch, mode):
    workspace, context, params = setup_artifact(tmp_path, mode)
    order = []
    for name in (f"{mode}_file", "check_file", "preview_file"):
        original = getattr(workspace, name)
        def call(*args, _name=name, _original=original, **kwargs):
            order.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(workspace, name, call)
    atomic = Mock(wraps=workspace._atomic_write)
    monkeypatch.setattr(workspace, "_atomic_write", atomic)
    monkeypatch.setattr("easyicu.webserver.pi_copilot.workspace.subprocess.run", lambda *args, **kwargs: pytest.fail("Host executed page JS"))
    result = run(context, mode, params, check_after=True, preview_after=True)
    details = result["details"]
    assert order == [f"{mode}_file", "check_file", "preview_file"]
    assert result["status"] == "ok" and details["mutation_receipt"]["ordinal"] == 1
    assert atomic.call_count == 1
    assert details["resource"]["kind"] == "webpage"
    assert details["resource"]["sha256"] == details["resource"]["checked_sha256"] == details["sha256"]
    assert details["static_check"]["valid"] is True and details["preview"]["status"] == "ready"
    assert details["runtime_status"] == "pending" and details["verification_scope"] == "preview_prepared"
    assert details["scientific_evidence"] is False and details["claim_ceiling"] == "unsupported"
    assert "RAW_HTML_SENTINEL" not in json.dumps(result)
    assert not any(key in details for key in ("text", "content", "allow_downloads", "download_nonce"))


@pytest.mark.parametrize("mode", ["write", "edit"])
@pytest.mark.parametrize("flags", [
    {"check_after": "true"}, {"check_after": 1}, {"check_after": None},
    {"preview_after": "true", "check_after": True}, {"preview_after": 1},
    {"preview_after": True}, {"preview_after": True, "check_after": False},
])
def test_invalid_flags_are_rejected_before_mutation_reservation_or_file_change(tmp_path, mode, flags):
    workspace, context, params = setup_artifact(tmp_path, mode)
    before = workspace.list_files("compound-project")
    with pytest.raises(PiCopilotError):
        run(context, mode, params, **flags)
    assert workspace.list_files("compound-project") == before
    assert context.grant.reserve_workspace_mutation(mode).ordinal == 1
    if mode == "edit":
        assert workspace.read_file("compound-project", "page.html")["sha256"] == params["expected_sha256"]


@pytest.mark.parametrize("mode", ["write", "edit"])
def test_legacy_default_and_check_only_keep_their_original_resource_behavior(tmp_path, mode):
    workspace, context, params = setup_artifact(tmp_path, mode)
    default = run(context, mode, params)
    assert default["details"]["resource"]["kind"] == "file"
    assert "preview" not in default["details"] and "static_check" not in default["details"]
    assert "checked_sha256" not in default["details"]
    assert default["summary"] == ("Created page.html." if mode == "write" else "Edited page.html with one exact replacement.")
    other, other_context, other_params = setup_artifact(tmp_path / "other", mode)
    checked = run(other_context, mode, other_params, check_after=True)
    assert checked["details"]["resource"]["kind"] == "file"
    assert checked["details"]["resource"]["checked_sha256"] == checked["details"]["sha256"]
    assert "preview" not in checked["details"]


@pytest.mark.parametrize("mode", ["write", "edit"])
def test_static_failure_keeps_saved_write_fact_and_never_calls_preview(tmp_path, monkeypatch, mode):
    workspace, context, params = setup_artifact(tmp_path, mode)
    invalid = '<script type="application/json" data-easyicu-runtime-qa>{bad json}</script>'
    params["content" if mode == "write" else "new_text"] = invalid
    if mode == "edit":
        params["old_text"] = "<p>before</p>"
    preview = Mock(side_effect=AssertionError("Preview must not run"))
    monkeypatch.setattr(workspace, "preview_file", preview)
    result = run(context, mode, params, check_after=True, preview_after=True)
    details = result["details"]
    assert result["status"] == "ok" and details["mutation_receipt"]["ordinal"] == 1
    assert workspace.read_file("compound-project", "page.html")["sha256"] == details["sha256"]
    assert details["static_check"]["valid"] is False and details["preview"]["status"] == "not_prepared"
    assert "resource" not in details and "checked_sha256" not in details
    preview.assert_not_called()


@pytest.mark.parametrize("mode", ["write", "edit"])
@pytest.mark.parametrize("stage", ["check", "preview", "preview_error", "preview_bad_identity"])
def test_postwrite_races_or_preview_failures_emit_no_resource_and_keep_the_static_history(tmp_path, monkeypatch, mode, stage):
    workspace, context, params = setup_artifact(tmp_path, mode)
    candidate = workspace.project_root("compound-project") / "page.html"
    if stage == "check":
        original = workspace.check_file
        def check(*args):
            candidate.write_text("<p>changed between write and check</p>")
            return original(*args)
        monkeypatch.setattr(workspace, "check_file", check)
    else:
        original = workspace.preview_file
        def preview(*args, **kwargs):
            if stage == "preview_error":
                raise RuntimeError("RAW_HTML_SENTINEL")
            if stage == "preview_bad_identity":
                return {"file": "page.html", "sha256": "a" * 64, "checked_sha256": "a" * 64, "text": "RAW_HTML_SENTINEL"}
            candidate.write_text("<p>changed between check and preview</p>")
            return original(*args, **kwargs)
        monkeypatch.setattr(workspace, "preview_file", preview)
    result = run(context, mode, params, check_after=True, preview_after=True)
    details = result["details"]
    assert result["status"] == "ok" and details["mutation_receipt"]["ordinal"] == 1
    assert "resource" not in details and "checked_sha256" not in details
    assert details["preview"]["status"] != "ready"
    assert details["static_check"]["valid"] is (stage != "check")
    assert "RAW_HTML_SENTINEL" not in json.dumps(result)
    if stage in {"check", "preview"}:
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() != details["sha256"]


def test_non_html_write_can_be_saved_but_is_not_issued_a_webpage(tmp_path, monkeypatch):
    workspace, context, _ = setup_artifact(tmp_path, "write")
    monkeypatch.setattr(workspace, "preview_file", Mock(side_effect=AssertionError("Not HTML")))
    result = run(context, "write", {"file": "synthetic.csv", "content": "synthetic_id\nSYN-001\n"}, check_after=True, preview_after=True)
    assert result["status"] == "ok"
    assert result["details"]["static_check"]["status"] == "not_applicable"
    assert result["details"]["preview"]["status"] == "not_prepared"
    assert "resource" not in result["details"]
    assert workspace.download_file("compound-project", "synthetic.csv", expected_sha256=result["details"]["sha256"])["content"] == b"synthetic_id\nSYN-001\n"
