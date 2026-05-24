"""Central opt-in gate for AI / LLM features in the web app.

This module is the **single chokepoint** for the invariant documented in
both workspace and EASYICU CLAUDE.md:

    AI assistant features are opt-in and start disabled until the user
    enables them in the sidebar.

Any code path in `easyicu.webapp.*` that may issue an *external* LLM call
(real OpenAI / OpenRouter / Anthropic / etc. — anything that reaches the
network) MUST go through `enforce_external_llm_opt_in()` before
instantiating its client.

Offline / deterministic clients (MockLLMClient) are intentionally NOT
gated: they are safe to run without user consent because no data leaves
the machine.

If you add a new page or workflow that talks to a hosted LLM, call
`enforce_external_llm_opt_in()` at the top of the resolution path.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


class AIOptInError(RuntimeError):
    """Raised when an external LLM call is attempted without sidebar opt-in."""


_MOCK_LLM_CHOICES = frozenset({
    "MockLLMClient",
    "MockLLMClient (offline, deterministic)",
    "mock",
    "offline",
})


def _is_offline_choice(llm_choice: Optional[str]) -> bool:
    if not llm_choice:
        return False
    if llm_choice in _MOCK_LLM_CHOICES:
        return True
    return "MockLLMClient" in llm_choice or "offline" in llm_choice.lower()


def is_ai_enabled() -> bool:
    """Return the canonical sidebar opt-in flag."""
    return bool(st.session_state.get("llm_enabled", False))


def enforce_external_llm_opt_in(
    llm_choice: Optional[str] = None,
    *,
    language: str = "zh",
) -> None:
    """Block external LLM calls when the sidebar AI toggle is off.

    Parameters
    ----------
    llm_choice:
        Identifier of the LLM client about to be instantiated. If it
        names an offline / mock client, the gate is bypassed.
    language:
        ``"zh"`` or ``"en"`` — controls the error message. Pages that
        already know the current language should pass it explicitly;
        otherwise we fall back to whatever is stored in
        ``st.session_state['language']``.

    Raises
    ------
    AIOptInError
        When ``llm_choice`` requires an external call and the sidebar
        ``llm_enabled`` flag is False.
    """
    if _is_offline_choice(llm_choice):
        return
    if is_ai_enabled():
        return

    resolved_language = language or st.session_state.get("language", "zh")
    if resolved_language == "en":
        raise AIOptInError(
            "AI features are disabled. Enable AI assistant in the sidebar, "
            "or tick the per-run external LLM opt-in before launching this "
            "run. You can also pick the offline MockLLMClient option."
        )
    raise AIOptInError(
        "AI 功能当前处于关闭状态。请先在侧边栏的 AI 助手设置中启用, "
        "或勾选本次运行的外部 LLM 调用授权后再启动。也可以选择离线的 "
        "MockLLMClient。"
    )


__all__ = [
    "AIOptInError",
    "enforce_external_llm_opt_in",
    "is_ai_enabled",
]
