from __future__ import annotations

import sys
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
        self.errors: list[str] = []

    def markdown(self, *_args, **_kwargs) -> None:
        pass

    def caption(self, *_args, **_kwargs) -> None:
        pass

    def success(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, message, *_args, **_kwargs) -> None:
        self.errors.append(str(message))

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


def test_openrouter_env_prefills_shared_llm_config(monkeypatch) -> None:
    fake_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(llm_config, "st", fake_streamlit)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-env")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EASYICU_OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")

    llm_config.ensure_llm_config_state()

    assert fake_streamlit.session_state["llm_enabled"] is False
    assert fake_streamlit.session_state["llm_provider"] == "openrouter"
    assert fake_streamlit.session_state["llm_api_key"] == "sk-test-env"
    assert fake_streamlit.session_state["llm_base_url"] == "https://openrouter.ai/api/v1"
    assert fake_streamlit.session_state["llm_model"] == "z-ai/glm-4.5-air:free"
    assert fake_streamlit.session_state["llm_configured"] is True


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


def test_openrouter_client_uses_reference_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    fake_streamlit = _FakeStreamlit(
        {
            "llm_provider": "openrouter",
            "llm_api_key": "sk-test",
            "llm_base_url": "https://openrouter.ai/api/v1",
            "llm_model": "z-ai/glm-4.5-air:free",
        }
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    client = llm_chat._get_client()

    assert client is not None
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
        "X-Title": "EasyICU web copilot",
    }
    assert "sk-test" not in repr(captured["default_headers"])


def test_background_openrouter_call_uses_reference_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Choice:
        message = SimpleNamespace(content="ok")

    class _Completions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs
            self.chat = _Chat()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    session_id = "unit-openrouter-bg"
    llm_chat._bg_results.pop(session_id, None)

    llm_chat._bg_llm_call(
        [{"role": "user", "content": "ping"}],
        "en",
        "openrouter",
        "z-ai/glm-4.5-air:free",
        "https://openrouter.ai/api/v1",
        "sk-test",
        session_id,
    )

    assert captured["client"]["default_headers"] == {
        "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
        "X-Title": "EasyICU web copilot",
    }
    assert captured["create"]["model"] == "z-ai/glm-4.5-air:free"
    assert llm_chat._bg_results.pop(session_id)["answer"] == "ok"


def test_empty_openrouter_response_is_treated_as_actionable_error(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Choice:
        message = SimpleNamespace(content="")

    class _Completions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return SimpleNamespace(choices=[_Choice()])

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs
            self.chat = _Chat()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    session_id = "unit-openrouter-empty"
    llm_chat._bg_results.pop(session_id, None)

    llm_chat._bg_llm_call(
        [{"role": "user", "content": "ping"}],
        "en",
        "openrouter",
        "z-ai/glm-4.5-air:free",
        "https://openrouter.ai/api/v1",
        "sk-test",
        session_id,
    )

    result = llm_chat._bg_results.pop(session_id)
    assert result["status"] == "error"
    assert "empty response" in result["answer"].lower()
    message = llm_chat._handle_api_error(RuntimeError(result["answer"]), "en", render=False)
    assert "empty response" in message
    assert "OpenRouter free model" in message


def test_ai_assistant_page_render_does_not_enable_hosted_provider(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": False,
            "_llm_toggle": False,
            "llm_provider": "easyicu_hosted",
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "",
        }
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)
    monkeypatch.setattr(
        llm_chat,
        "_render_ai_assistant_workspace_page",
        lambda _lang, *, pending_prompt: None,
    )

    llm_chat.render_ai_assistant_page("en")

    assert fake_streamlit.session_state["llm_enabled"] is False
    assert fake_streamlit.session_state["_llm_toggle"] is False


def test_stream_response_enforces_opt_in_before_building_client(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": False,
            "llm_provider": "easyicu_hosted",
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "hosted-default",
            "llm_messages": [{"role": "user", "content": "Explain SOFA."}],
        }
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)
    monkeypatch.setattr(
        llm_chat,
        "_get_client",
        lambda: pytest.fail("_stream_response built a client without opt-in"),
    )

    llm_chat._stream_response([{"role": "user", "content": "Explain SOFA."}], "en")

    assert fake_streamlit.errors
    assert "AI features are disabled" in fake_streamlit.errors[0]


def test_background_response_enforces_opt_in_before_composing_messages(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit(
        {
            "language": "en",
            "llm_enabled": False,
            "llm_provider": "easyicu_hosted",
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "hosted-default",
            "llm_messages": [{"role": "user", "content": "Explain SOFA."}],
        }
    )
    monkeypatch.setattr(llm_chat, "st", fake_streamlit)
    monkeypatch.setattr(
        llm_chat,
        "_compose_agent_messages",
        lambda _prompt: pytest.fail("_start_bg_response composed messages without opt-in"),
    )

    assert llm_chat._start_bg_response("Explain SOFA.", "en") is None
    assert fake_streamlit.errors
    assert "AI features are disabled" in fake_streamlit.errors[0]


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


def test_copilot_retry_routing_action_survives_current_step_filter() -> None:
    actions = [
        {
            "id": "choice_retry_routing",
            "kind": "copilot_prompt",
            "label": "↻ Retry routing",
            "prompt": "what real data path should I use?",
        },
    ]
    state = {"_copilot_guided_study": {"step": "question", "branch": "predict"}}

    rendered = llm_chat._copilot_message_actions_for_current_step(
        actions,
        "en",
        state,
        is_latest=True,
    )

    assert rendered[0]["id"] == "choice_retry_routing"
    assert any(action["id"] == "choice_question_outcome_model" for action in rendered)


def test_copilot_real_data_path_help_uses_local_data_step(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_chat,
        "_copilot_route_with_llm",
        lambda *_args, **_kwargs: pytest.fail("path-help prompt should not route through LLM"),
    )
    state = {
        "entry_mode": "none",
        "use_mock_data": True,
        "database": "mock",
        "llm_provider": "easyicu_hosted",
        "llm_model": "hosted-default",
    }

    reply = llm_chat._handle_copilot_guided_prompt(
        "what real data path should I use?",
        "en",
        state,
    )

    assert reply is not None
    body, actions = reply
    study = state["_copilot_guided_study"]
    assert state["entry_mode"] == "real"
    assert state["use_mock_data"] is False
    assert study["step"] == "data"
    assert study["data_source_choice"] == "prepared_path"
    assert "path field below this conversation" in body
    assert any(action["id"] == "choice_data_prepared_path" for action in actions)


# ---------------------------------------------------------------------------
# Copilot history context-budget guard
# ---------------------------------------------------------------------------
#
# `_compose_agent_messages` appends the in-session conversation history to the
# LLM payload. The persistence cap (SAVE_LIMIT) and UI render cap never bounded
# that payload, so a long study session grew the prompt linearly until the
# provider rejected it. `_trim_history_for_budget` keeps only the most recent
# turns within COPILOT_HISTORY_CHAR_BUDGET.


def test_trim_history_empty_returns_empty():
    assert llm_chat._trim_history_for_budget([]) == []


def test_trim_history_under_budget_is_unchanged_and_ordered():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert llm_chat._trim_history_for_budget(history, 1000) == history


def test_trim_history_keeps_most_recent_tail_within_budget():
    history = [{"role": "user", "content": "X" * 100} for _ in range(10)]
    # 100 + 100 = 200 <= 250; adding a third (300) exceeds -> keep 2 newest
    kept = llm_chat._trim_history_for_budget(history, 250)
    assert kept == history[-2:]


def test_trim_history_always_keeps_current_prompt_even_if_oversized():
    history = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "Z" * 5000},
    ]
    assert llm_chat._trim_history_for_budget(history, 100) == [history[-1]]


def test_trim_history_tolerates_non_string_content():
    history = [{"role": "user", "content": None}]
    assert llm_chat._trim_history_for_budget(history, 10) == history


# ---------------------------------------------------------------------------
# Copilot rolling-summary auto-compaction
# ---------------------------------------------------------------------------


def test_plan_history_compaction_keeps_recent_tail_and_yields_fold_block():
    history = [{"role": "user", "content": "X" * 1000} for _ in range(40)]
    fold, kept = llm_chat._plan_history_compaction(
        history, summarized_count=0, summary="", budget=10000
    )
    # tail budget floor is 8000 chars -> ~8 of the 1000-char messages kept
    assert kept == history[-len(kept):]
    assert fold == history[: len(history) - len(kept)]
    assert fold and kept  # genuinely split


def test_plan_history_compaction_excludes_already_summarized_prefix():
    history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    fold, kept = llm_chat._plan_history_compaction(
        history, summarized_count=4, summary="prior", budget=10_000_000
    )
    # everything after the summarized prefix fits -> nothing to fold
    assert fold == []
    assert kept == history[4:]


def test_compacted_history_folds_oldest_into_summary(monkeypatch):
    state = _State({"llm_messages": [{"role": "user", "content": "Z" * 1000} for _ in range(80)],
                    "language": "en"})
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))
    calls = []

    def fake_summarizer(prior, block):
        calls.append((prior, len(block)))
        return "ROLLING SUMMARY"

    payload = llm_chat._compacted_history_for_payload(summarizer=fake_summarizer)

    assert calls and calls[0][0] == ""  # first fold sees empty prior summary
    assert payload[0] == {"role": "system",
                          "content": "Summary of earlier conversation (older turns compacted to fit context):\nROLLING SUMMARY"}
    # state advanced so the same turns are not re-summarized next time
    assert state[llm_chat.COPILOT_CTX_SUMMARY_KEY] == "ROLLING SUMMARY"
    assert state[llm_chat.COPILOT_CTX_SUMMARY_COUNT_KEY] == calls[0][1]
    # the kept tail is the most recent messages, in order
    assert payload[-1] == state["llm_messages"][-1]


def test_compacted_history_falls_back_to_truncation_when_summarizer_unavailable(monkeypatch):
    msgs = [{"role": "user", "content": "Z" * 1000} for _ in range(80)]
    state = _State({"llm_messages": msgs, "language": "en"})
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))

    # summarizer returns None (e.g. offline / failed call) -> no summary message,
    # tail is still truncated and the count is NOT advanced (no fabricated summary)
    payload = llm_chat._compacted_history_for_payload(summarizer=lambda prior, block: None)

    assert all(m.get("content", "").startswith("Summary of earlier") is False for m in payload)
    assert state.get(llm_chat.COPILOT_CTX_SUMMARY_COUNT_KEY, 0) == 0
    assert payload[-1] == msgs[-1]
    assert len(payload) < len(msgs)  # truncated


def test_compacted_history_resets_stale_summary_after_clear(monkeypatch):
    state = _State({
        "llm_messages": [{"role": "user", "content": "fresh"}],
        "language": "en",
        llm_chat.COPILOT_CTX_SUMMARY_KEY: "OLD SESSION SUMMARY",
        llm_chat.COPILOT_CTX_SUMMARY_COUNT_KEY: 99,  # > len(history) -> stale
    })
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))
    payload = llm_chat._compacted_history_for_payload(summarizer=lambda p, b: "SHOULD NOT FIRE")

    assert state[llm_chat.COPILOT_CTX_SUMMARY_COUNT_KEY] == 0
    assert state[llm_chat.COPILOT_CTX_SUMMARY_KEY] == ""
    assert payload == [{"role": "user", "content": "fresh"}]


# ---------------------------------------------------------------------------
# Token-aware budgeting (provider-window-derived history budget)
# ---------------------------------------------------------------------------


def test_estimate_tokens_nonzero_and_scales_with_length():
    assert llm_chat._estimate_tokens("") == 0
    short = llm_chat._estimate_tokens("hello world")
    long = llm_chat._estimate_tokens("hello world " * 200)
    assert short >= 1
    assert long > short


def test_provider_context_window_env_override(monkeypatch):
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=_State({})))
    monkeypatch.setenv("EASYICU_COPILOT_CONTEXT_WINDOW", "12345")
    assert llm_chat._provider_context_window() == 12345


def test_provider_context_window_matches_model_substring(monkeypatch):
    monkeypatch.delenv("EASYICU_COPILOT_CONTEXT_WINDOW", raising=False)
    state = _State({"llm_provider": "openai", "llm_model": "gpt-4o-mini"})
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))
    assert llm_chat._provider_context_window() == llm_chat.COPILOT_MODEL_CONTEXT_WINDOWS["gpt-4o"]


def test_provider_context_window_unknown_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("EASYICU_COPILOT_CONTEXT_WINDOW", raising=False)
    state = _State({"llm_provider": "openai", "llm_model": "some-unlisted-model-xyz"})
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))
    assert llm_chat._provider_context_window() == llm_chat.COPILOT_CONTEXT_WINDOW_DEFAULT


def test_history_token_budget_tracks_window(monkeypatch):
    monkeypatch.setenv("EASYICU_COPILOT_CONTEXT_WINDOW", "8192")
    monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=_State({})))
    small = llm_chat._copilot_history_token_budget()
    monkeypatch.setenv("EASYICU_COPILOT_CONTEXT_WINDOW", "200000")
    large = llm_chat._copilot_history_token_budget()
    assert large > small
    # never below the floor
    monkeypatch.setenv("EASYICU_COPILOT_CONTEXT_WINDOW", "100")
    assert llm_chat._copilot_history_token_budget() == llm_chat.COPILOT_HISTORY_TOKEN_BUDGET_FLOOR


def test_token_mode_compaction_keeps_more_history_for_larger_window(monkeypatch):
    msgs = [{"role": "user", "content": "word " * 200} for _ in range(60)]

    def run_with_window(win: int) -> int:
        state = _State({"llm_messages": list(msgs), "language": "en"})
        monkeypatch.setattr(llm_chat, "st", SimpleNamespace(session_state=state))
        monkeypatch.setenv("EASYICU_COPILOT_CONTEXT_WINDOW", str(win))
        payload = llm_chat._compacted_history_for_payload(
            summarizer=lambda prior, block: "S",
            budget=llm_chat._copilot_history_token_budget(),
            size_fn=llm_chat._message_token_size,
            summary_sizer=llm_chat._estimate_tokens,
            min_tail=llm_chat.COPILOT_HISTORY_MIN_TAIL_TOKENS,
            summary_cap=llm_chat.COPILOT_HISTORY_SUMMARY_MAX_TOKENS,
        )
        # count real (non-summary) turns retained
        return sum(1 for m in payload if not str(m.get("content", "")).startswith(("Summary of earlier", "早先对话摘要")))

    kept_small = run_with_window(8192)
    kept_large = run_with_window(200000)
    assert kept_large > kept_small  # bigger window -> more raw history retained
