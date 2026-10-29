"""Plan exact edits against one original text before any workspace write."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

from .contracts import PiCopilotError

MISSING_EDIT = object()
MAX_BATCH_EDITS = 16
MAX_EDIT_TEXT_CHARS = 120_000


def validate_edit_request(params: Mapping[str, Any]) -> None:
    if "edits" in params and ("old_text" in params or "new_text" in params):
        raise PiCopilotError("pi_workspace_edit_modes_conflict", "Use either old_text/new_text or edits, not both.")
    if "check_after" in params and type(params["check_after"]) is not bool:
        raise PiCopilotError("pi_workspace_check_after_invalid", "check_after must be a boolean.")
    if "edits" not in params and ("old_text" not in params or not str(params.get("old_text") or "").strip()):
        raise PiCopilotError("pi_tool_arguments_required", "The EasyICU tool is missing required arguments.", details={"fields": ["old_text"]})


def apply_exact_edits(
    source: str, relative: str, *, old_text: Any = MISSING_EDIT,
    new_text: Any = MISSING_EDIT, edits: Any = MISSING_EDIT,
) -> tuple[str, int]:
    if edits is MISSING_EDIT:
        needle = str(old_text if old_text is not MISSING_EDIT and old_text is not None else "")
        if not needle:
            raise PiCopilotError("pi_workspace_edit_target_required", "An exact non-empty edit target is required.")
        occurrences = source.count(needle)
        if occurrences != 1:
            raise PiCopilotError("pi_workspace_edit_target_not_unique", "The exact edit target must occur once in the project file.",
                                 details={"file": relative, "occurrences": occurrences})
        replacement = str(new_text if new_text is not MISSING_EDIT and new_text is not None else "")
        return source.replace(needle, replacement, 1), 1
    if old_text is not MISSING_EDIT or new_text is not MISSING_EDIT:
        raise PiCopilotError("pi_workspace_edit_modes_conflict", "Use either old_text/new_text or edits, not both.")
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_BATCH_EDITS:
        raise PiCopilotError("pi_workspace_edit_batch_invalid", "edits must contain between 1 and 16 exact replacements.")
    spans = []
    for index, item in enumerate(edits):
        if (not isinstance(item, Mapping) or set(item) != {"old_text", "new_text"}
                or not isinstance(item["old_text"], str) or not item["old_text"]
                or not isinstance(item["new_text"], str)
                or max(len(item["old_text"]), len(item["new_text"])) > MAX_EDIT_TEXT_CHARS):
            raise PiCopilotError("pi_workspace_edit_batch_invalid", "Each edit needs bounded old_text and new_text strings only.",
                                 details={"edit_index": index})
        needle = item["old_text"]
        start = source.find(needle)
        if start < 0 or source.find(needle, start + 1) >= 0:
            raise PiCopilotError("pi_workspace_edit_target_not_unique", "Every exact batch target must occur once in the original file.",
                                 details={"file": relative, "edit_index": index})
        spans.append((start, start + len(needle), item["new_text"]))
    spans.sort(key=lambda span: span[0])
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise PiCopilotError("pi_workspace_edit_targets_overlap", "Batch targets must not overlap in the original file.", details={"file": relative})
    updated = source
    for start, end, replacement in reversed(spans):
        updated = updated[:start] + replacement + updated[end:]
    return updated, len(spans)


def check_edited_html(workspace: Any, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Check the saved version without running page scripts or claiming runtime QA."""
    result = dict(payload)
    if PurePosixPath(payload["file"]).suffix.lower() not in {".html", ".htm"}:
        result["static_check"] = {"status": "not_applicable", "valid": False, "code": "pi_workspace_html_check_only"}
        return result
    try:
        checked = workspace.check_file(project_id, payload["file"])
    except PiCopilotError as exc:
        result["static_check"] = {"status": "failed", "valid": False, "code": exc.code, "details": dict(exc.details)}
        return result
    except Exception:
        result["static_check"] = {"status": "failed", "valid": False, "code": "pi_workspace_checker_unavailable"}
        return result
    if checked.get("valid") is not True or checked.get("checked_sha256") != payload["sha256"]:
        result["static_check"] = {"status": "unconfirmed", "valid": False, "code": "pi_workspace_file_changed"}
        return result
    result["static_check"] = {**checked, "status": "passed", "code": "pi_workspace_file_checked"}
    result.update(checked_sha256=payload["sha256"], check_scope=checked.get("check_scope"), verification_scope="static_only", runtime_status="pending")
    return result
