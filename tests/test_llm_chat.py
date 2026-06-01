from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.webapp import llm_config
from easyicu.webapp import llm_chat


class _State(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Panel:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RerunRequested(Exception):
    pass


class _FakeStreamlit:
    def __init__(self, state=None, *, toggle_value=None, clicked_key=None) -> None:
        self.session_state = _State(state or {})
        self.toggle_value = toggle_value
        self.clicked_key = clicked_key
        self.toggle_calls: list[dict] = []

    def markdown(self, *_args, **_kwargs) -> None:
        pass

    def caption(self, *_args, **_kwargs) -> None:
        pass

    def success(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def expander(self, *_args, **_kwargs):
        return _Panel()

    def container(self, *_args, **_kwargs):
        return _Panel()

    def columns(self, spec):
        return [_Panel() for _ in spec]

    def toggle(self, label, *, value=False, key=None, **_kwargs):
        if self.toggle_value is None:
            result = self.session_state.get(key, value)
        else:
            result = self.toggle_value
        if key:
            self.session_state[key] = result
        self.toggle_calls.append({"label": label, "value": value, "key": key, "result": result})
        return result

    def selectbox(self, _label, *, options, index=0, **_kwargs):
        return options[index]

    def text_input(self, _label, *, value="", **_kwargs):
        return value

    def button(self, _label, *, key=None, **_kwargs):
        return key == self.clicked_key

    def rerun(self):
        raise _RerunRequested()


def test_ai_assistant_starts_disabled_until_user_opt_in(monkeypatch) -> None:
    fake_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    llm_chat._init_chat_state()

    assert fake_streamlit.session_state["llm_enabled"] is False


def test_easyicu_hosted_is_available_to_web_assistant() -> None:
    assert llm_config.public_default_provider_key() == "easyicu_hosted"
    assert "easyicu_hosted" in llm_config.public_provider_keys()
    assert llm_config.coerce_public_provider("easyicu_hosted") == "easyicu_hosted"


def test_easyicu_hosted_is_configured_without_user_key(monkeypatch) -> None:
    fake_streamlit = SimpleNamespace(
        session_state={
            "llm_provider": "easyicu_hosted",
            "llm_api_key": "",
            "llm_base_url": "",
        }
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    assert llm_chat._is_configured() is True


def test_llm_reasoning_blocks_are_hidden_from_web_answers() -> None:
    assert llm_chat._strip_llm_reasoning(
        "<think>private scratchpad</think>\nSOFA features: sofa, sofa2"
    ) == "SOFA features: sofa, sofa2"
    assert llm_chat._strip_llm_reasoning(
        "<think>streaming scratchpad still open"
    ) == ""


def test_verification_parser_ignores_model_reasoning_prefix() -> None:
    parsed = llm_chat._parse_verification_report(
        "<think>check answer</think>\n"
        "STATUS: pass\n"
        "ISSUES:\n"
        "- none\n"
        "CORRECTED_ANSWER:\n"
        "Ready."
    )

    assert parsed["status"] == "pass"
    assert parsed["corrected_answer"] == "Ready."


def test_enabling_ai_toggle_keeps_legacy_floating_panel_closed(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": False,
            "llm_provider": "easyicu_hosted",
            "llm_base_url": "http://localhost:8000/v1",
            "llm_model": "hosted-default",
            "_floating_ai_open": False,
        },
        toggle_value=True,
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    llm_chat.render_llm_settings()

    assert fake_streamlit.session_state["llm_enabled"] is True
    assert fake_streamlit.session_state["_floating_ai_open"] is False
    assert fake_streamlit.session_state["_sidebar_ai_open"] is True
    assert fake_streamlit.session_state["llm_provider"] == "easyicu_hosted"
    assert fake_streamlit.session_state["llm_configured"] is True


def test_disabling_floating_ai_toggle_hides_panel(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": True,
            "_llm_toggle": True,
            "_floating_ai_open": True,
        },
        toggle_value=False,
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    llm_chat.render_llm_settings()

    assert fake_streamlit.session_state["llm_enabled"] is False
    assert fake_streamlit.session_state["_floating_ai_open"] is False


def test_closing_floating_ai_panel_disables_sidebar_toggle_on_next_render(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": True,
            "_llm_toggle": True,
            "_floating_ai_open": True,
            "_floating_ai_size": "m",
            "_ai_pending_question": "Explain SOFA",
        },
        clicked_key="_floating_ai_close_btn",
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    with pytest.raises(_RerunRequested):
        llm_chat.render_floating_chat_dock()

    assert fake_streamlit.session_state["llm_enabled"] is False
    assert fake_streamlit.session_state["_floating_ai_open"] is False
    assert fake_streamlit.session_state["_ai_pending_question"] is None
    assert fake_streamlit.session_state["_llm_toggle_sync_pending"] is True


def test_floating_ai_launcher_reopens_panel(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": True,
            "_floating_ai_open": False,
            "_floating_ai_size": "m",
        },
        clicked_key="_floating_ai_open_btn",
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    with pytest.raises(_RerunRequested):
        llm_chat.render_floating_chat_dock()

    assert fake_streamlit.session_state["_floating_ai_open"] is True


@pytest.mark.parametrize(
    ("clicked_key", "expected_size"),
    [
        ("_floating_ai_size_s_btn", "s"),
        ("_floating_ai_size_m_btn", "m"),
        ("_floating_ai_size_l_btn", "l"),
    ],
)
def test_floating_ai_size_buttons_update_panel_size(monkeypatch, clicked_key, expected_size) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": True,
            "_floating_ai_open": True,
            "_floating_ai_size": "m",
        },
        clicked_key=clicked_key,
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    with pytest.raises(_RerunRequested):
        llm_chat.render_floating_chat_dock()

    assert fake_streamlit.session_state["_floating_ai_open"] is True
    assert fake_streamlit.session_state["_floating_ai_size"] == expected_size


def test_minimizing_floating_ai_panel_keeps_assistant_enabled(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": True,
            "_llm_toggle": True,
            "_floating_ai_open": True,
            "_floating_ai_size": "m",
            "_ai_pending_question": "Explain cohort filters",
        },
        clicked_key="_floating_ai_minimize_btn",
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    with pytest.raises(_RerunRequested):
        llm_chat.render_floating_chat_dock()

    assert fake_streamlit.session_state["llm_enabled"] is True
    assert fake_streamlit.session_state["_floating_ai_open"] is False
    assert fake_streamlit.session_state["_ai_pending_question"] is None
    assert fake_streamlit.session_state.get("_llm_toggle_sync_pending") is None


def test_pending_sidebar_toggle_sync_is_applied_before_toggle_render(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": False,
            "_llm_toggle": True,
            "_llm_toggle_sync_pending": True,
            "_floating_ai_open": False,
        },
        toggle_value=None,
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)

    llm_chat.render_llm_settings()

    assert fake_streamlit.session_state["_llm_toggle"] is False
    assert fake_streamlit.session_state.get("_llm_toggle_sync_pending") is None
    assert fake_streamlit.toggle_calls[0]["value"] is False
