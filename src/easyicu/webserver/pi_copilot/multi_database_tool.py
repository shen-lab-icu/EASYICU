"""Model-selected same-study proposals; no text routing or approval authority."""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from easyicu.webserver.research_multi_contracts import MultiDatabaseError, MultiDatabaseRequest
from easyicu.webserver.research_multi_sources import select_source
from .contracts import PiCopilotError, ToolExecutionContext


def propose_same_study(context: ToolExecutionContext, params: Mapping[str, Any]) -> dict:
    """Resolve explicit registry choices before asking the existing batch owner."""
    from .tools import _result

    current_turn = context.user_turns[-1] if context.user_turns else {}
    if current_turn.get("message_origin") != "typed" or not str(current_turn.get("text") or "").strip():
        raise PiCopilotError("multi_database_typed_turn_required", "Type a request to compare this study across databases.")
    binding = context.session.binding
    if not binding.study_context_id:
        raise PiCopilotError("multi_database_study_required", "Bind a study before requesting other databases.")
    if context.propose_multi_database is None:
        raise PiCopilotError("multi_database_turn_required", "Request this proposal in the current conversation turn.")
    try:
        request = MultiDatabaseRequest(project_id=context.session.project_id, session_id=context.session.session_id,
            study_context_id=binding.study_context_id, source_run_id=binding.run_id, targets=params["targets"])
        # Pin the registration the user/model selected. Preparation verifies it
        # again and issues independent per-source egress grants as before.
        targets = tuple(target.model_copy(update={"source_id": select_source(target)[0]["id"]}) for target in request.targets)
        request = request.model_copy(update={"targets": targets})
        receipt = context.propose_multi_database(request)
    except ValidationError as exc:
        raise PiCopilotError("multi_database_request_invalid", "Choose one to six distinct registered databases.") from exc
    except MultiDatabaseError as exc:
        raise PiCopilotError(exc.code, "The multi-database proposal could not be prepared.", status_code=409) from exc
    return _result(context, status="ok", code="multi_database_review_preparing",
        summary="Preparing one combined review card. Nothing runs until you approve that card.",
        owner="easyicu.webserver.research_multi_actions", details={**receipt, "approval_required": True})
