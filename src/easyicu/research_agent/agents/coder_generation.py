"""Crash-safe transport for a Coder's initial executable candidate.

The Coder owns scientific prompt construction. This module owns only the paid
transport lifecycle: reserve, call, validate, persist, seal, and at most one
audited regeneration when the provider returns a non-executable response.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Mapping, Optional, Sequence

from ..authority.provider_budget import (
    StepProviderCallBudget,
    complete_with_provider_budget,
)
from ..authority.step_capsule import ContentRef
from ..providers.protocol import LLMMessage
from ..repairs.patch import looks_like_executable_python

MAX_INITIAL_GENERATION_ATTEMPTS = 2


class IncompleteCoderResponseError(ValueError):
    """The provider returned text that is not a complete Python candidate."""


def _fail_pending_transport(
    budget: StepProviderCallBudget,
    *,
    transport_id: str,
    error_type: str,
) -> None:
    if budget.initial_generation_resume_status() in {
        "unpaid_pending",
        "paid_pending",
    }:
        budget.fail_initial_generation_transport(
            provider_transport_id=transport_id,
            error_type=error_type,
        )


def generate_initial_coder_candidate(
    *,
    messages: Sequence[LLMMessage],
    provider_call: Callable[[Sequence[LLMMessage]], str],
    response_parser: Callable[[str], str],
    provider_budget: Optional[StepProviderCallBudget],
    initial_generation_binding: Optional[Mapping[str, object]],
    persist_candidate: Optional[Callable[[str], ContentRef]],
    on_initial_reserved: Optional[Callable[[str, str], None]],
    on_initial_candidate: Optional[Callable[[ContentRef, str], None]],
) -> str:
    """Return one executable candidate with append-only transport evidence."""

    initial_messages = list(messages)
    for attempt in range(MAX_INITIAL_GENERATION_ATTEMPTS):
        transport_id: Optional[str] = None
        if provider_budget is not None and initial_generation_binding is not None:
            transport_id = provider_budget.reserve_initial_generation(
                initial_generation_binding
            )
            binding_sha256 = hashlib.sha256(
                json.dumps(
                    dict(initial_generation_binding),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if on_initial_reserved is not None:
                on_initial_reserved(transport_id, binding_sha256)
        try:
            raw = complete_with_provider_budget(
                budget=provider_budget,
                category="initial_generation",
                call=lambda: provider_call(initial_messages),
            )
            code = response_parser(raw)
            if not looks_like_executable_python(code):
                raise IncompleteCoderResponseError(
                    "Initial coder response is not a complete executable Python "
                    "script; refusing to persist or seal it as candidate authority."
                )
            candidate_ref = (
                persist_candidate(code) if persist_candidate is not None else None
            )
            if transport_id is not None:
                if candidate_ref is None:
                    raise RuntimeError(
                        "initial-generation transport requires persisted code bytes"
                    )
                assert provider_budget is not None
                provider_budget.complete_initial_generation_transport(
                    provider_transport_id=transport_id,
                    after_code_sha256=candidate_ref.sha256,
                    after_code_size_bytes=candidate_ref.size_bytes,
                )
            if (
                candidate_ref is not None
                and transport_id is not None
                and on_initial_candidate is not None
            ):
                on_initial_candidate(candidate_ref, transport_id)
            return code
        except BaseException as exc:
            if transport_id is not None and provider_budget is not None:
                _fail_pending_transport(
                    provider_budget,
                    transport_id=transport_id,
                    error_type=type(exc).__name__,
                )
            may_retry = (
                isinstance(exc, IncompleteCoderResponseError)
                and attempt + 1 < MAX_INITIAL_GENERATION_ATTEMPTS
                and provider_budget is not None
                and initial_generation_binding is not None
                and provider_budget.authorize_failed_initial_generation_retry(
                    error_type=type(exc).__name__,
                    max_generation_epochs=MAX_INITIAL_GENERATION_ATTEMPTS,
                )
            )
            if not may_retry:
                raise
            initial_messages = [
                *messages,
                LLMMessage(
                    role="user",
                    content=(
                        "The previous response was rejected locally because it was "
                        "not a complete executable Python script. Regenerate the "
                        "entire script now. Return only one complete runnable Python "
                        "script, with no explanation or placeholders."
                    ),
                ),
            ]
    raise AssertionError("initial-generation attempt loop terminated unexpectedly")


__all__ = [
    "IncompleteCoderResponseError",
    "MAX_INITIAL_GENERATION_ATTEMPTS",
    "generate_initial_coder_candidate",
]
