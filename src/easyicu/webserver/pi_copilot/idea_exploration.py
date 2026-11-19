"""An open idea exploration holds the automatic first plan.

Idea Mining runs inside a research conversation whose study question is the
mining request and whose outcome is still empty.  Confirming a data source
then made the workflow ready for a plan, and the conversation started one on
the mining request itself: no direction chosen, no outcome, and feasibility
and handoff skipped.  While this conversation has mined ideas, none has been
accepted, and the study names no outcome, the plan waits for the
researcher's own "generate plan" click instead (screens-guided-pi-plan-actions).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

IDEAS_MINED = "easyicu_idea_mined"
IDEA_ACCEPTED = "easyicu_idea_handoff_accepted"


def exploration_open(turns: Iterable[Any], study: Mapping[str, Any] | None) -> bool:
    """Whether the newest idea result of these turns still awaits a choice."""

    if str((study or {}).get("outcome") or "").strip():
        return False
    state = False
    for turn in turns or ():
        events = turn.get("events") if isinstance(turn, Mapping) else None
        for event in events if isinstance(events, list) else ():
            if not isinstance(event, Mapping) or event.get("type") != "tool_end" or event.get("status") != "ok":
                continue
            if event.get("code") == IDEAS_MINED:
                state = True
            elif event.get("code") == IDEA_ACCEPTED:
                state = False
    return state


__all__ = ["exploration_open"]
