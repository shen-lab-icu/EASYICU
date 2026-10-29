"""The workspace file-text channel is larger than metadata, never unbounded."""
from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.projections import ensure_safe_projection
from easyicu.webserver.pi_copilot.tools import execute_tool
from easyicu.webserver.pi_copilot.workspace import MAX_READ_CHARS, ProjectWorkspace
from easyicu.webserver.pi_copilot.workspace_read_projection import MAX_READ_RESULT_BYTES, project_workspace_read_result


def context_for(tmp_path):
    return ToolExecutionContext(session=PiSessionRecord(session_id="read-test", project_id="synthetic-read", agent_mode="workspace"),
                                workspace=ProjectWorkspace(tmp_path / "workspace"))


@pytest.mark.parametrize("text", [
    "<html><script>const caption = '" + "x" * 10_400 + "';</script></html>",
    "x" * MAX_READ_CHARS,
    "中" * MAX_READ_CHARS,
    "😀" * MAX_READ_CHARS,
    "\x01" * MAX_READ_CHARS,
    '\\"' * (MAX_READ_CHARS // 2),
])
def test_legal_file_text_survives_full_tool_projection_with_sha_and_bounded_utf8_json(tmp_path, text):
    context = context_for(tmp_path)
    context.workspace.write_file("synthetic-read", "page.html", text)
    result = execute_tool("easyicu_read_project_file", {"file": "page.html"}, context)
    assert result["details"]["text"] == text
    assert result["details"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert result["details"]["truncated"] is False
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= MAX_READ_RESULT_BYTES
    assert result["code"] == "pi_workspace_file_read"


def test_generic_projection_string_limit_remains_unchanged():
    with pytest.raises(PiCopilotError) as failure:
        ensure_safe_projection({"text": "x" * 2001})
    assert failure.value.code == "pi_projection_too_large"


@pytest.mark.parametrize("suffix", ["stay_id=123456789", "Bearer syntheticSecret123", "/Users/someone/private.txt"])
def test_complete_file_text_privacy_scan_covers_the_tail_after_two_thousand_characters(tmp_path, suffix):
    context = context_for(tmp_path)
    context.workspace.write_file("synthetic-read", "page.html", "x" * 10_000 + "\n" + suffix)
    with pytest.raises(PiCopilotError) as failure:
        execute_tool("easyicu_read_project_file", {"file": "page.html"}, context)
    assert failure.value.code == "pi_projection_blocked"


@pytest.mark.parametrize("extra", [{"caption": "x" * 2001}, {"credentials": "synthetic"}, {"file": "/Users/someone/file.html"}])
def test_text_channel_does_not_exempt_metadata_from_ordinary_guards(tmp_path, extra):
    context = context_for(tmp_path)
    payload = {"file": "page.html", "text": "x" * 10_000, "sha256": "a" * 64, **extra}
    with pytest.raises(PiCopilotError):
        project_workspace_read_result(context, payload, {"kind": "file", "file": "page.html"})


def test_file_text_budget_is_an_independent_hard_limit(tmp_path):
    context = context_for(tmp_path)
    with pytest.raises(PiCopilotError) as failure:
        project_workspace_read_result(context, {"file": "page.html", "text": "x" * (MAX_READ_CHARS + 1)}, {"kind": "file", "file": "page.html"})
    assert failure.value.code == "pi_workspace_read_projection_too_large"


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("file", ["dataset.csv", "data/page.html", "exports/page.html", "source.parquet"])
def test_data_files_are_still_rejected_before_bytes_are_read(tmp_path, monkeypatch, hosted, file):
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    context = context_for(tmp_path)
    monkeypatch.setattr(context.workspace, "_read_complete_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data bytes were read")))
    with pytest.raises(PiCopilotError) as failure:
        execute_tool("easyicu_read_project_file", {"file": file}, context)
    assert failure.value.code == "tabular_read_unavailable_when_hosted"


def test_large_file_read_keeps_exact_edit_sha_contract(tmp_path):
    context = context_for(tmp_path)
    text = "<html>" + "x" * 10_000 + "<p>old-value</p></html>"
    context.workspace.write_file("synthetic-read", "page.html", text)
    read = execute_tool("easyicu_read_project_file", {"file": "page.html"}, context)["details"]
    context.workspace.edit_file("synthetic-read", "page.html", old_text="old-value", new_text="new-value", expected_sha256=read["sha256"])
    with pytest.raises(PiCopilotError):
        context.workspace.edit_file("synthetic-read", "page.html", old_text="new-value", new_text="third-value", expected_sha256=read["sha256"])
