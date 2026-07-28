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

**The budget is a local design ceiling, not the model's context window.** It
answers "did this projection grow beyond what it was designed to carry", which
is a question about our own assembly. Only the provider knows its own limit,
and nothing here should pretend otherwise: no model context window is declared
anywhere in this package, and the current provider does not report one.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

__all__ = [
    "DEFAULT_MAX_PROMPT_TOKENS",
    "CONSERVATIVE_BYTES_PER_TOKEN",
    "OBSERVED_BYTES_PER_TOKEN",
    "PROMPT_TRANSPORT_BUDGETS",
    "PromptBudgetClient",
    "PromptConsumerBudget",
    "PromptTransportBudgetError",
    "UndeclaredPromptConsumerError",
    "active_prompt_consumer",
    "budgeted_client",
    "budgeted_role_client",
    "budgeted_vlm_client",
    "estimate_prompt_tokens",
    "prompt_payload_bytes",
]


# Tokens are what the provider meters; bytes are all we can see before the
# call. The conversion is not assumed -- it is measured. Every completed
# transport receipt records both ``prompt_bytes`` and the provider's own
# ``usage.prompt_tokens``, so the ratio is re-derivable from any run:
#
#     bytes/token over the 2026-07-23 E1 replay (8 real calls, all roles)
#       min 3.7685   max 4.3812   mean 3.99
#
# That sample is entirely English prose and JSON. It is not the whole story:
# ``PipelineConfig.manuscript_language`` allows a Chinese manuscript, and CJK
# text is ~3 UTF-8 bytes per character at roughly one token per character, so
# its bytes/token can fall to ~2-3. A constant taken from the English sample
# would quietly under-count tokens on exactly that content.
#
# So the estimator deliberately divides by a value *below* everything observed.
# Estimating high is the safe direction: it can refuse a prompt that would have
# fit, but it cannot let one through by under-counting. Re-derive from receipts
# rather than adjusting this by feel -- and if you lower it, re-check the
# default ceiling below, which is sized in terms of it.
CONSERVATIVE_BYTES_PER_TOKEN = 3.0

# The observed English/JSON minimum, kept separate so the margin above is
# visible rather than folded invisibly into one number.
OBSERVED_BYTES_PER_TOKEN = 3.7685


# The default ceiling, in tokens.
#
# The old ceilings were written in bytes and, converted at the observed ratio,
# landed at roughly 12,700-21,200 tokens. The largest prompt this system has
# ever actually produced is 26,040 tokens / 101,878 bytes (a planner call in
# the same replay). So the guard was set *below* normal operating traffic --
# which is exactly why it kept tripping, and why past work went into shrinking
# prompts to fit rather than questioning the number.
#
# A guard meant to catch runaway assembly belongs above normal traffic, not
# inside it. Under the conservative estimator that largest real payload scores
# 33,959 tokens, so the ceiling has to clear that: 40,000 does, with headroom,
# and still sits far below any current model's context window, so a projection
# that has genuinely run away is still caught.
#
# It is a default, not a decree: set ``PipelineConfig.max_prompt_tokens_per_call``
# to change it, and the change is recorded in the run authority digest because
# the config is hashed into it.
DEFAULT_MAX_PROMPT_TOKENS = 40_000


def estimate_prompt_tokens(payload_bytes: int) -> int:
    """Estimate provider-metered tokens from the bytes we can see."""

    return int(math.ceil(int(payload_bytes) / CONSERVATIVE_BYTES_PER_TOKEN))


@dataclass(frozen=True)
class PromptConsumerBudget:
    """One declared consumer of a budgeted role transport."""

    consumer: str
    role: str
    rationale: str
    limit_tokens: int = DEFAULT_MAX_PROMPT_TOKENS

    def with_limit_tokens(self, limit_tokens: Optional[int]) -> "PromptConsumerBudget":
        """Return this budget under an operator-supplied ceiling."""

        if limit_tokens is None:
            return self
        return PromptConsumerBudget(
            consumer=self.consumer,
            role=self.role,
            rationale=self.rationale,
            limit_tokens=max(1, int(limit_tokens)),
        )


# Every consumer of a budgeted role.
#
# They share one ceiling on purpose. The old per-class numbers (48,000 and
# 80,000 bytes) differed for no measured reason -- each was the size someone
# expected that particular projection to reach, not a property of the
# transport, and neither survived contact with real traffic. A single declared
# ceiling with a documented derivation is more honest than five numbers whose
# differences nobody can justify. A consumer that genuinely needs a different
# ceiling should get one here, with the evidence that says so.
PROMPT_TRANSPORT_BUDGETS: Mapping[str, PromptConsumerBudget] = {
    budget.consumer: budget
    for budget in (
        PromptConsumerBudget(
            consumer="analyzer_interpretation",
            role="analyzer",
            rationale="AnalyzerAgent step interpretation.",
        ),
        PromptConsumerBudget(
            consumer="concept_audit",
            role="analyzer",
            rationale=(
                "LLM concept auditor; carries the concept draft and its audit "
                "findings, so it grows with the concept rather than the step."
            ),
        ),
        PromptConsumerBudget(
            consumer="vlm_visual_qa",
            role="analyzer",
            rationale=(
                "VLM figure review. Bounds the text prompt only -- attached "
                "image bytes are not text payload and are not counted here."
            ),
        ),
        PromptConsumerBudget(
            consumer="cohort_extraction",
            role="planner",
            rationale="Cohort definition extraction.",
        ),
        PromptConsumerBudget(
            consumer="coder_initial_generation",
            role="coder",
            rationale=(
                "Initial analysis-script generation after step-scoped context "
                "and host authority are assembled."
            ),
        ),
        PromptConsumerBudget(
            consumer="legacy_model_roster_migration",
            role="planner",
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
        limit_tokens: int,
    ) -> None:
        self.consumer = str(consumer)
        self.role = str(role) if role else ""
        self.actual_bytes = int(actual_bytes)
        self.actual_tokens = estimate_prompt_tokens(actual_bytes)
        self.limit_tokens = int(limit_tokens)
        super().__init__(
            f"{self.consumer} prompt budget exceeded: about "
            f"{self.actual_tokens} tokens ({self.actual_bytes} bytes) against a "
            f"ceiling of {self.limit_tokens} (role {self.role or 'unknown'}). "
            "This ceiling is a local design budget for how large this "
            "projection is expected to grow -- it is NOT the model's context "
            "window, which this package does not know and does not guess. "
            "No evidence digest or binding scientific coordinate was "
            "truncated. Either reduce the consumer-scoped projection, or raise "
            "PipelineConfig.max_prompt_tokens_per_call if the payload is "
            "legitimately this large."
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
    def limit_tokens(self) -> int:
        return self._budget.limit_tokens

    @property
    def budget(self) -> PromptConsumerBudget:
        """The whole envelope, so a re-wrap can compare it rather than its name."""

        return self._budget

    @property
    def inner(self) -> Any:
        """The client this wrapper bounds, for re-wrapping under a new budget."""

        return self._inner

    def _enforce(self, messages: Sequence[Any]) -> None:
        actual_bytes = prompt_payload_bytes(messages)
        if estimate_prompt_tokens(actual_bytes) > self._budget.limit_tokens:
            raise PromptTransportBudgetError(
                consumer=self._budget.consumer,
                role=self._budget.role,
                actual_bytes=actual_bytes,
                limit_tokens=self._budget.limit_tokens,
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
    *,
    limit_tokens: Optional[int] = None,
) -> Any:
    """Resolve ``role`` for ``consumer``, holding it to its declared ceiling.

    Fails closed when ``consumer`` was never declared, so adding a new user of
    a shared role transport is a decision someone has to make explicitly.
    ``limit_tokens`` carries the operator's configured ceiling; omitting it
    uses the declared default.
    """

    # Resolve nothing until the consumer is known: an undeclared consumer must
    # not even reach the provider.
    _declared_budget(consumer=consumer, role=role)
    return budgeted_client(
        role_resolver(str(role)), role, consumer, limit_tokens=limit_tokens
    )


def budgeted_client(
    base: Any,
    role: str,
    consumer: str,
    *,
    limit_tokens: Optional[int] = None,
) -> Any:
    """Hold an already-resolved client to ``consumer``'s declared envelope.

    The resolver-based form above cannot cover a client that was injected
    rather than resolved -- ``pipeline._vlm_client or budgeted_role_client(...)``
    short-circuits, so an explicitly supplied client reached the provider
    unwrapped and unattributed, with no ceiling at all.
    """

    budget = _declared_budget(consumer=consumer, role=role)
    if base is None:
        return None
    effective = budget.with_limit_tokens(limit_tokens)
    if isinstance(base, PromptBudgetClient):
        if base.budget == effective:
            # Already this consumer under exactly this ceiling: nothing to do.
            return base
        # Otherwise the existing envelope is the wrong one, in one of two ways.
        # Wrapped for somebody else, it would hand this consumer the other
        # one's name and ceiling -- calls attributed to a consumer that did not
        # make them, measured against a limit never sized for them. Wrapped for
        # this consumer under a different ceiling, comparing names alone
        # returned it unchanged and silently discarded the ceiling this call
        # asked for, so whichever caller happened to run first set the limit for
        # everyone after. Re-wrap the client underneath rather than stacking a
        # second envelope on top: re-wrapping must land exactly where a first
        # wrap would.
        base = base.inner
    return PromptBudgetClient(base, budget=effective)


def budgeted_vlm_client(
    pipeline: Any,
    role_resolver: Callable[[str], Any],
    consumer: str,
) -> Any:
    """The one way visual QA obtains its client, injected or resolved.

    Both call sites wrote ``pipeline._vlm_client or budgeted_role_client(...)``,
    which means an injected client took neither the ceiling nor the consumer
    attribution. Having one function own the choice keeps the two sites from
    drifting apart again.
    """

    injected = getattr(pipeline, "_vlm_client", None)
    limit_tokens = getattr(pipeline, "_max_prompt_tokens_per_call", None)
    if injected is not None:
        return budgeted_client(
            injected, "analyzer", consumer, limit_tokens=limit_tokens
        )
    return budgeted_role_client(
        role_resolver, "analyzer", consumer, limit_tokens=limit_tokens
    )


def _declared_budget(*, consumer: str, role: str) -> PromptConsumerBudget:
    """Fail closed unless ``consumer`` is a declared user of ``role``."""

    budget = PROMPT_TRANSPORT_BUDGETS.get(str(consumer))
    if budget is None or budget.role != str(role):
        raise UndeclaredPromptConsumerError(consumer=str(consumer), role=str(role))
    return budget


def declared_consumers_for_role(role: str) -> Dict[str, PromptConsumerBudget]:
    """Every declared consumer of one budgeted role."""

    return {
        consumer: budget
        for consumer, budget in PROMPT_TRANSPORT_BUDGETS.items()
        if budget.role == str(role)
    }
