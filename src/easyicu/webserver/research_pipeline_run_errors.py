"""Stable Web-facing errors shared by Research Agent launch owners."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from easyicu.research_agent.orchestration.progress import ProgressControlSignal


WINDOWED_SUMMARY_STOP_MESSAGES = {
    "windowed_summary_statistic_unresolved": (
        "The question names a measurement but not which summary to report, such as its highest, "
        "lowest or mean value over a time window. State it, for example the highest lactate in "
        "the first 24 hours after ICU admission, then generate the plan again."
    ),
    "windowed_summary_statistic_unsupported": (
        "EasyICU can summarize a measurement over a time window only as its highest, lowest, "
        "mean, first or last value. Choose one of these, then generate the plan again."
    ),
    "windowed_summary_window_required": (
        "A summary of a measurement needs a time window, for example the first 24 hours after "
        "ICU admission. State the window in the question, then generate the plan again."
    ),
    "windowed_summary_window_unresolved": (
        "EasyICU could not read a single time window for this measurement from the question. "
        "State one window, for example the first 24 hours after ICU admission, then generate "
        "the plan again."
    ),
    "windowed_summary_window_unsupported": (
        "EasyICU can summarize a measurement only over a window that starts at ICU admission, "
        "such as the first 24 hours. Change the window, then generate the plan again."
    ),
    "windowed_summary_concept_not_continuous": (
        "This variable is not a continuous measurement, so a highest, lowest or mean value "
        "over a time window does not apply. Ask about it another way, for example whether it "
        "occurred, then generate the plan again."
    ),
    "windowed_summary_materialization_window_mismatch": (
        "The prepared data covers a different time window from the one the question states, "
        "so the summary would not match the question. No analysis was run. Prepare the data "
        "again for the stated window."
    ),
}


class ResearchPipelineRunError(ProgressControlSignal):
    """Stable Web-facing failure from the true Research Agent bridge."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def revision_scope_diagnostic(error: BaseException) -> dict[str, Any]:
    """Project bounded coordinates of a refused revision, never free text.

    Review codes describe the recorded revision requests, not authorization
    for the refused step. The guard has already found no applicable request.
    """
    if not isinstance(error, ResearchPipelineRunError) or error.code != "revision_added_unrequested_analysis":
        return {}
    details = error.details
    diagnostic: dict[str, Any] = {
        "owner": "easyicu.planning.revision_scope_v1",
        "reason_code": error.code,
        "authorization": "no_question_or_applicable_review_item",
    }
    for key in ("step_id", "module"):
        value = details.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value):
            diagnostic[key] = value
    role = details.get("role")
    if isinstance(role, str) and role in {"primary", "secondary", "sensitivity", "auxiliary"}:
        diagnostic["role"] = role
    for key, pattern in (
        ("added_inputs", r"[a-zA-Z][a-zA-Z0-9_]{0,63}"),
        ("review_item_codes", r"[A-Z][A-Z0-9_]{0,79}"),
    ):
        values = details.get(key)
        diagnostic[key] = [
            value for value in values[:16]
            if isinstance(value, str) and re.fullmatch(pattern, value)
        ] if isinstance(values, (list, tuple)) else []
    return diagnostic
