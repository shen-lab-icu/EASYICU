"""The host starts the extraction a raw Demo study's researcher asked for (B19).

On the raw-source plan card (B18) a reviewed candidate cannot prepare its data
until this study's own data package exists, and the researcher asks for that
extraction in the conversation. The model then wandered through read tools and
described a host control that does not exist, so the request never reached the
extraction owner.

When the researcher's own sent turn is granted ``extract`` by the existing turn
inference and the workflow is exactly in that state for a verified official
Demo raw source, the host calls the extraction tool once before the model turn,
through the same tool owner and the same one-use grant, and hands the model the
owner's receipt to report. The grant's source is unchanged: the card only
drafts the text; the researcher sends it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping

from .contracts import PiCopilotError, ToolExecutionContext, utc_now
from .extraction_handoff import official_demo_raw_source

HOST_STUDY_EXTRACTION_INTENT = "report_host_study_extraction"
HOST_STUDY_EXTRACTION_TOOL = "easyicu_start_extraction"
_RAW_PLAN_STATE = "plan_execution_upgrade_required"
_LIVE_JOB_STATUSES = frozenset({"queued", "running"})


def raw_demo_extraction_due(
    *,
    granted_actions: Iterable[str],
    message_origin: str,
    workflow: Any,
    study: Mapping[str, Any],
) -> bool:
    """Whether the host starts this study's extraction before the model turn.

    ``workflow`` is the project workflow projection (``workflow`` snapshot plus
    ``active_job``). Every condition must hold; any missing one leaves the turn
    to the model unchanged.
    """

    if "extract" not in set(granted_actions) or message_origin != "typed":
        return False
    snapshot = getattr(workflow, "workflow", None)
    if snapshot is None:
        return False
    if snapshot.next_action_code != _RAW_PLAN_STATE:
        return False
    if snapshot.execution_input_available is not False:
        return False
    active_job = getattr(workflow, "active_job", None) or {}
    if str(active_job.get("status") or "") in _LIVE_JOB_STATUSES:
        return False
    return official_demo_raw_source(study)


def submit_host_study_extraction(
    context: ToolExecutionContext,
    *,
    call_id: str,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Run the extraction tool once for the turn and return its owner receipt.

    The tool events go through the turn's own event sink, so replay and the
    child-job watch see this call as they see a model-requested one.
    """

    from . import tools

    emit(
        {
            "type": "tool_start",
            "at": utc_now(),
            "tool_call_id": call_id,
            "tool_name": HOST_STUDY_EXTRACTION_TOOL,
        }
    )
    try:
        result = tools.execute_tool(HOST_STUDY_EXTRACTION_TOOL, {}, context)
    except PiCopilotError as exc:
        result = {
            "status": "blocked",
            "code": exc.code,
            "summary": exc.message,
            "owner": "easyicu.webserver.pi_copilot",
            "details": {},
            "authority": {},
        }
    details = (
        result.get("details") if isinstance(result.get("details"), Mapping) else {}
    )
    status = str(result.get("status") or "blocked")
    event: Dict[str, Any] = {
        "type": "tool_end",
        "at": utc_now(),
        "tool_call_id": call_id,
        "tool_name": HOST_STUDY_EXTRACTION_TOOL,
        "status": status,
        "code": str(result.get("code") or ""),
        "owner": str(result.get("owner") or ""),
        "is_error": status in {"blocked", "failed", "error"},
    }
    if details.get("job_id"):
        event["job_id"] = str(details["job_id"])
    if isinstance(details.get("resource"), Mapping):
        event["resource"] = dict(details["resource"])
    emit(event)
    return result


__all__ = [
    "HOST_STUDY_EXTRACTION_INTENT",
    "HOST_STUDY_EXTRACTION_TOOL",
    "raw_demo_extraction_due",
    "submit_host_study_extraction",
]
