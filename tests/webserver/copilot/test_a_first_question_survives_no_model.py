"""A question typed on the first screen survives a missing model connection.

The first screen creates a project from the question and asks it there. With no
model connected the conversation cannot start; the question used to be dropped,
so a clinician who chose "Not now" (or took longer than two minutes to sign in)
had to type it again. It now waits as the project's pending question, which the
activation panel shows and the next conversation opens with.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[3] / "src/easyicu/webserver/static/js"


def _function_body(source: str, name: str) -> str:
    start = source.index(f"async function {name}(")
    end = source.index("\n  }\n", start)
    return source[start:end]


def test_no_session_keeps_the_question_pending_instead_of_dropping_it():
    body = _function_body((JS / "screens-guided-pi.js").read_text(encoding="utf-8"), "askFirstQuestion")
    no_session = body.index("if (!state.session)")
    keep = body.index("sessionStorage.setItem('easyicu.skillHub.question', queued.text)")
    draft = body.index("state.draft = queued.text")
    assert no_session < keep < draft
    # The project check still guards both branches.
    assert body.index("projectId() !== expectedId") < no_session


def test_the_pending_question_is_shown_and_filled_into_the_next_conversation():
    view = (JS / "screens-guided-pi-session-view.js").read_text(encoding="utf-8")
    nav = (JS / "screens-guided-pi-session-navigation.js").read_text(encoding="utf-8")
    assert "getItem('easyicu.skillHub.question')" in view
    assert re.search(r"pendingQuestionNote\(\)", view)
    assert "if (pending) { state.draft = pending; window.sessionStorage.removeItem('easyicu.skillHub.question'); }" in nav
