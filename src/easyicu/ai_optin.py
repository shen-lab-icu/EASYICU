"""Pure AI / LLM opt-in policy shared by EasyICU runtimes.

This module deliberately avoids UI-framework imports so FastAPI and CLI paths
can enforce the same external-provider policy without pulling in UI
dependencies.
"""

from __future__ import annotations

from typing import Optional


class AIOptInError(RuntimeError):
    """Raised when an external LLM call is attempted without explicit opt-in."""


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


def is_offline_llm_choice(llm_choice: Optional[str]) -> bool:
    """Return True when a selected LLM is local/offline and needs no opt-in."""
    return _is_offline_choice(llm_choice)


def check_external_llm_opt_in(
    llm_choice: Optional[str] = None,
    *,
    ai_enabled: bool,
    language: str = "zh",
) -> None:
    """Fail closed when an external LLM is requested without explicit opt-in."""
    if _is_offline_choice(llm_choice):
        return
    if bool(ai_enabled):
        return

    resolved_language = language or "zh"
    if resolved_language == "en":
        raise AIOptInError(
            "AI features are disabled. Enable AI assistant in the sidebar, "
            "or tick the per-run external LLM opt-in before launching this "
            "run. You can also pick the offline MockLLMClient option."
        )
    raise AIOptInError(
        "AI 功能当前处于关闭状态。请先在侧边栏的 Copilot 设置中启用, "
        "或勾选本次运行的外部 LLM 调用授权后再启动。也可以选择离线的 "
        "MockLLMClient。"
    )


__all__ = [
    "AIOptInError",
    "check_external_llm_opt_in",
    "is_offline_llm_choice",
]
