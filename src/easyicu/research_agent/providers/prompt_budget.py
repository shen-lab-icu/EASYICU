"""Per-consumer prompt transport envelopes, enforced at the role client.

One role client serves more than one consumer. ``role_resolver("analyzer")``
hands the same transport to the Analyzer, to the LLM concept auditor and to
the VLM visual-QA fallback; ``role_resolver("planner")`` serves the Planner,
the Replanner and cohort extraction. Historically each *agent class* checked
its own prompt immediately before calling, so a consumer that was not one of
those classes went out unmeasured: real transport receipts from the E1
development diagnostic recorded ``analyzer`` prompts of 53,393 and 78,401
bytes against a declared 48,000-byte ceiling, and a ``planner`` prompt of
101,878 bytes against a declared 80,000, all delivered.

The envelope is a property of the transport, not of the class that happens to
build the messages, so it is enforced here: every consumer of a budgeted role
is resolved through :func:`budgeted_role_client`, which fails closed on a
consumer nobody declared. A new consumer therefore cannot silently inherit no
budget -- it has to be added to :data:`PROMPT_TRANSPORT_BUDGETS` with a
reviewed number.

This module deliberately does not truncate. A prompt over its envelope is an
error, because the payloads involved (evidence digests, typed bindings, concept
drafts) carry binding scientific coordinates that must not be silently dropped.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

__all__ = [
    "PROMPT_TRANSPORT_BUDGETS",
    "PromptBudgetClient",
    "PromptConsumerBudget",
    "PromptTransportBudgetError",
    "UndeclaredPromptConsumerError",
    "active_prompt_consumer",
    "budgeted_role_client",
    "prompt_payload_bytes",
]


@dataclass(frozen=True)
class PromptConsumerBudget:
    """One declared consumer of a budgeted role transport."""

    consumer: str
    role: str
    limit_bytes: int
    rationale: str


# Every consumer of a budgeted role, with the reviewed number it must fit in.
#
# The numbers are not new. Each one is the ceiling this repository had already
# reviewed for that transport; this table only makes them apply to every
# consumer instead of to one agent class:
#
#   * 48,000 -- ``_ANALYZER_PROMPT_BYTE_LIMIT``
#   * 80,000 -- ``_PLANNER_PROMPT_BYTE_LIMIT``
#
# ``concept_audit`` is the one consumer that does not take its role's 48,000.
# That number was sized for the Analyzer's fixed projection (a step summary,
# evidence ids, a scoped context, four sentences out). The concept auditor
# carries the concept draft and its audit findings -- a different prompt shape
# that was never measured against 48,000, and that real receipts show running
# larger. It is declared at 80,000, this repository's largest reviewed text
# envelope, so the check is real without breaking observed traffic.
#
# These ceilings are bytes, and bytes are a proxy for the tokens the provider
# actually meters (measured ~0.248 tokens/byte on this workload). Re-sizing
# them against a declared model context window is separate work; it needs the
# window, which the current provider does not report.
PROMPT_TRANSPORT_BUDGETS: Mapping[str, PromptConsumerBudget] = {
    budget.consumer: budget
    for budget in (
        PromptConsumerBudget(
            consumer="analyzer_interpretation",
            role="analyzer",
            limit_bytes=48_000,
            rationale="AnalyzerAgent step interpretation (_ANALYZER_PROMPT_BYTE_LIMIT).",
        ),
        PromptConsumerBudget(
            consumer="concept_audit",
            role="analyzer",
            limit_bytes=80_000,
            rationale=(
                "LLM concept auditor; carries the concept draft, so it is sized "
                "at the largest reviewed text envelope rather than the "
                "Analyzer's projection budget."
            ),
        ),
        PromptConsumerBudget(
            consumer="vlm_visual_qa",
            role="analyzer",
            limit_bytes=48_000,
            rationale=(
                "VLM figure review. Bounds the text prompt only -- attached "
                "image bytes are not text payload and are not counted here."
            ),
        ),
        PromptConsumerBudget(
            consumer="cohort_extraction",
            role="planner",
            limit_bytes=80_000,
            rationale="Cohort definition extraction (_PLANNER_PROMPT_BYTE_LIMIT).",
        ),
        PromptConsumerBudget(
            consumer="legacy_model_roster_migration",
            role="planner",
            limit_bytes=80_000,
            rationale=(
                "Legacy model-roster migration. Embeds the full serialised "
                "ResearchContext, so it is the planner-role consumer most "
                "likely to grow with the study rather than with the plan."
            ),
        ),
    )
}


# Roles whose consumers must all be declared above. A role client for one of
# these may only be handed out through ``budgeted_role_client``.
BUDGETED_ROLES = frozenset(budget.role for budget in PROMPT_TRANSPORT_BUDGETS.values())


class PromptTransportBudgetError(RuntimeError):
    """A lossless request exceeds the envelope declared for its consumer."""

    def __init__(
        self,
        *,
        consumer: str,
        role: Optional[str],
        actual_bytes: int,
        limit_bytes: int,
    ) -> None:
        self.consumer = str(consumer)
        self.role = str(role) if role else ""
        self.actual_bytes = int(actual_bytes)
        self.limit_bytes = int(limit_bytes)
        super().__init__(
            f"{self.consumer} prompt transport budget exceeded: "
            f"{self.actual_bytes} > {self.limit_bytes} bytes "
            f"(role {self.role or 'unknown'}). No evidence digest or binding "
            "scientific coordinate was truncated; reduce the consumer-scoped "
            "projection or split the payload."
        )


class UndeclaredPromptConsumerError(RuntimeError):
    """A budgeted role was requested for a consumer nobody declared."""

    def __init__(self, *, consumer: str, role: str) -> None:
        self.consumer = str(consumer)
        self.role = str(role)
        declared = ", ".join(sorted(PROMPT_TRANSPORT_BUDGETS)) or "(none)"
        super().__init__(
            f"consumer {self.consumer!r} of budgeted role {self.role!r} has no "
            f"declared prompt transport budget. Declared consumers: {declared}. "
            "Add the consumer to PROMPT_TRANSPORT_BUDGETS with a reviewed "
            "ceiling rather than sending an unmeasured prompt."
        )


def prompt_payload_bytes(messages: Sequence[Any]) -> int:
    """Measure a request the way the transport receipt measures it."""

    total = 0
    for message in messages or ():
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        total += len(str(content or "").encode("utf-8"))
    return total


# Set for the duration of one budgeted call so the transport receipt can record
# which consumer produced it. Receipts previously recorded only the role, which
# is why the over-limit calls above could not be attributed to a consumer.
active_prompt_consumer: ContextVar[Optional[str]] = ContextVar(
    "easyicu_active_prompt_consumer", default=None
)


class PromptBudgetClient:
    """Wrap one role client and hold its consumer to a declared envelope."""

    name = "prompt_transport_budget"

    def __init__(self, inner: Any, *, budget: PromptConsumerBudget) -> None:
        self._inner = inner
        self._budget = budget
        # Reviewed wrappers must bind their child graph, or the provider trust
        # machinery cannot see through them. Unlike the metering and stop-loss
        # wrappers -- which wrap the router and inherit its ``iter_clients`` by
        # delegation -- this one wraps an already-resolved leaf, so it has to
        # publish the child itself. Without both, a wrapped offline mock stops
        # being discoverable as a mock and the pipeline silently loses its
        # context binding.
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    def iter_clients(self):
        """Yield the wrapped client so mock discovery can walk through."""

        inner_iter = getattr(self._inner, "iter_clients", None)
        if callable(inner_iter):
            yield from inner_iter()
        else:
            yield self._inner

    @property
    def consumer(self) -> str:
        return self._budget.consumer

    @property
    def limit_bytes(self) -> int:
        return self._budget.limit_bytes

    def _enforce(self, messages: Sequence[Any]) -> None:
        actual_bytes = prompt_payload_bytes(messages)
        if actual_bytes > self._budget.limit_bytes:
            raise PromptTransportBudgetError(
                consumer=self._budget.consumer,
                role=self._budget.role,
                actual_bytes=actual_bytes,
                limit_bytes=self._budget.limit_bytes,
            )

    def _attributed(self, call: Callable[[], Any]) -> Any:
        token = active_prompt_consumer.set(self._budget.consumer)
        try:
            return call()
        finally:
            active_prompt_consumer.reset(token)

    def complete(
        self, messages: Sequence[Any], **kwargs: Any
    ) -> Any:  # noqa: D102 - protocol
        self._enforce(messages)
        return self._attributed(lambda: self._inner.complete(messages, **kwargs))

    def complete_with_usage(
        self, messages: Sequence[Any], **kwargs: Any
    ) -> Any:  # noqa: D102 - protocol
        self._enforce(messages)
        return self._attributed(
            lambda: self._inner.complete_with_usage(messages, **kwargs)
        )

    def complete_with_images(
        self, *args: Any, prompt: Any = None, **kwargs: Any
    ) -> Any:
        """Bound the text prompt only; image bytes are not text payload.

        A figure sent for review is an attachment, not prose that could be
        split or shortened, so counting its bytes against a text envelope
        would block visual QA for a reason the envelope was never about.
        """

        if prompt is not None:
            self._enforce([_TextOnly(prompt)])
            kwargs["prompt"] = prompt
        return self._attributed(
            lambda: self._inner.complete_with_images(*args, **kwargs)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TextOnly:
    """Adapt a bare prompt string to the measured-message shape."""

    __slots__ = ("content",)

    def __init__(self, content: Any) -> None:
        self.content = content


def budgeted_role_client(
    role_resolver: Callable[[str], Any],
    role: str,
    consumer: str,
) -> Any:
    """Resolve ``role`` for ``consumer``, holding it to its declared envelope.

    Fails closed when ``consumer`` was never declared, so adding a new user of
    a shared role transport is a decision someone has to make explicitly.
    """

    budget = PROMPT_TRANSPORT_BUDGETS.get(str(consumer))
    if budget is None:
        raise UndeclaredPromptConsumerError(consumer=str(consumer), role=str(role))
    if budget.role != str(role):
        raise UndeclaredPromptConsumerError(consumer=str(consumer), role=str(role))
    base = role_resolver(str(role))
    if base is None:
        return None
    if isinstance(base, PromptBudgetClient):
        return base
    return PromptBudgetClient(base, budget=budget)


def declared_consumers_for_role(role: str) -> Dict[str, PromptConsumerBudget]:
    """Every declared consumer of one budgeted role."""

    return {
        consumer: budget
        for consumer, budget in PROMPT_TRANSPORT_BUDGETS.items()
        if budget.role == str(role)
    }
