from __future__ import annotations

from types import SimpleNamespace

from easyicu.webapp import llm_chat


def test_ai_assistant_starts_disabled_until_user_opt_in(monkeypatch) -> None:
    fake_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    llm_chat._init_chat_state()

    assert fake_streamlit.session_state["llm_enabled"] is False
