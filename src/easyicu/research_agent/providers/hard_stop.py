"""Transparent Provider client wrapper for durable run/batch stop-loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..authority.provider_hard_stop import (
    TaskProviderHardStop,
    consume_active_provider_hard_stop_attempt,
    provider_hard_stop_call_scope,
)
from .llm import (
    clear_provider_call_receipt,
    client_counts_transport_attempts,
    current_provider_call_receipt,
)
from .protocol import LLMMessage


def _model_identity(client: Any) -> str:
    for name in ("_model", "model", "name"):
        value = getattr(client, name, None)
        if isinstance(value, str) and value:
            return value
    return type(client).__name__


class HardStopClient:
    """Wrap one reviewed role client and account every transport retry."""

    name = "provider_hard_stop"

    def __init__(
        self,
        inner: Any,
        *,
        role: Optional[str],
        task: TaskProviderHardStop,
    ) -> None:
        self._inner = inner
        self._role = role
        self._task = task
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    @property
    def supports_vision(self) -> bool:
        """Preserve the wrapped client's capability instead of widening it."""

        from .llm import llm_supports_vision

        return llm_supports_vision(self._inner)

    def complete(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        response, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response

    def complete_with_usage(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        with provider_hard_stop_call_scope(
            task=self._task,
            role=self._role,
            model=_model_identity(self._inner),
            messages=messages,
            max_tokens=max_tokens,
        ) as hard_stop:
            try:
                # Reviewed OpenAI/fallback/mock transports reserve immediately
                # before each raw request. A legacy reviewed client may not yet
                # expose that hook, so reserve one conservative attempt here
                # before invoking it. This makes the stop-loss pre-transport for
                # every accepted client instead of discovering the gap only
                # after a paid response returned.
                if not client_counts_transport_attempts(self._inner):
                    consume_active_provider_hard_stop_attempt()
                complete_with_usage = getattr(
                    self._inner, "complete_with_usage", None
                )
                if callable(complete_with_usage):
                    response, raw_usage = complete_with_usage(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    # Usage doubles as the call-scoped model-provenance
                    # carrier. Keep non-numeric metadata intact; the hard-stop
                    # ledger reads only its numeric token fields.
                    usage = dict(raw_usage) if isinstance(raw_usage, dict) else None
                else:
                    complete = self._inner.complete
                    kwargs: Dict[str, Any] = {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    response = complete(messages, **kwargs)
                    usage = None
            except BaseException as exc:
                hard_stop.fail(type(exc).__name__)
                raise
            hard_stop.complete(usage)
            return response, usage

    def complete_with_images(
        self,
        *,
        prompt: str,
        image_paths: Sequence[Path],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Account an authorized multimodal request before image transport."""

        messages = (LLMMessage(role="user", content=str(prompt)),)
        with provider_hard_stop_call_scope(
            task=self._task,
            role=self._role,
            model=_model_identity(self._inner),
            messages=messages,
            max_tokens=max_tokens,
        ) as hard_stop:
            clear_provider_call_receipt()
            try:
                if not client_counts_transport_attempts(self._inner):
                    consume_active_provider_hard_stop_attempt()
                response = self._inner.complete_with_images(
                    prompt=prompt,
                    image_paths=image_paths,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                receipt = current_provider_call_receipt()
                usage = dict(receipt.usage_summary) if receipt is not None else None
            except BaseException as exc:
                hard_stop.fail(type(exc).__name__)
                raise
            hard_stop.complete(usage)
            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def hard_stop_role_resolver(llm: Any, task: TaskProviderHardStop):
    """Return a role resolver whose calls share one task/batch ledger."""

    from .llm import resolve_role_client

    def resolver(role: str):
        base = resolve_role_client(llm, role)
        if base is None:
            return None
        if isinstance(base, HardStopClient):
            return base
        return HardStopClient(base, role=role, task=task)

    return resolver


__all__ = ["HardStopClient", "hard_stop_role_resolver"]
