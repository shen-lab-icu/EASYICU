"""Structured-response retry with feedback for LLM calls.

Generalises the pattern used in baselines (data-to-paper, HealthFlow):
when a structured output from an LLM cannot be parsed (bad JSON, pydantic
validation error, missing required field), the parse exception is
converted into a natural-language feedback message and fed back to the
*same* conversation, so the LLM gets another shot with a concrete hint.

Why this lives in its own module
--------------------------------

* It is used by ``PlannerAgent``, ``ReplannerAgent`` and any future agent
  that expects a structured output. Keeping it role-agnostic means the
  retry policy and the failure shape stay uniform across agents — a
  reviewer who reads how the planner recovers from a bad-JSON event
  reads the same code path as the writer or the reviewer.
* It is intentionally **not** ICU-aware. The error feedback is a
  shape-level hint ("your previous response could not be parsed as
  JSON"), not a domain hint. Domain reminders belong in the role's own
  system prompt, not here.
* The wrapper records every attempt in
  ``StructuredResponseFailure.attempts`` so a reviewer can inspect the
  exact LLM output that triggered each retry — important for the
  reproducibility-envelope claim.

Composability with the reproducibility envelope
-----------------------------------------------

Each retry calls ``llm.complete`` separately. The
``ReproRecordingClient`` wrapper records each call as its own
``ReproCallRecord``, so a 3-retry recovery shows up as three entries in
``reproducibility_envelope.json`` with distinct prompt SHA256s. This is
the intended behaviour: a reviewer can see whether the retry pattern
was triggered and whether it converged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from .protocol import LLMMessage
from .factory import authorized_complete

T = TypeVar("T")


_DEFAULT_FEEDBACK_PREAMBLE = (
    "Your previous response could not be parsed into the required structured "
    "output. The validator reported:"
)


_DEFAULT_FEEDBACK_INSTRUCTIONS = (
    "Please return ONLY a single valid JSON object matching the schema "
    "described in the original instructions. Do not include prose before or "
    "after the JSON. Do not wrap the JSON in a Markdown code fence. Do not "
    "include trailing commas. Do not include comments."
)


@dataclass
class StructuredAttempt:
    """One LLM call attempt during a structured-retry loop."""

    attempt: int
    raw_head: str  # first ~400 chars of the raw response
    raw_chars: int
    error_class: Optional[str]  # None when the attempt succeeded
    error_message: Optional[str]


class StructuredResponseFailure(RuntimeError):
    """All retries exhausted without producing a parseable structured response.

    The ``attempts`` field records every call (including the final failed
    one). Callers that previously crashed on the first parse failure
    should now catch this and either fall back to a deterministic plan,
    surface the audit trail, or re-raise after recording the failure.
    """

    def __init__(self, attempts: Sequence[StructuredAttempt], role: str) -> None:
        self.attempts = list(attempts)
        self.role = role
        last = self.attempts[-1] if self.attempts else None
        last_msg = (last.error_message if last else None) or "(no attempts recorded)"
        super().__init__(
            f"StructuredResponseFailure[role={role}, "
            f"n_attempts={len(self.attempts)}]: {last_msg}"
        )


def call_llm_with_structured_retry(
    llm: Any,
    messages: Sequence[LLMMessage],
    parser: Callable[[str], T],
    *,
    role: str = "structured",
    max_retries: int = 2,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    format_reminder: str = "",
    feedback_preamble: str = _DEFAULT_FEEDBACK_PREAMBLE,
    feedback_instructions: str = _DEFAULT_FEEDBACK_INSTRUCTIONS,
) -> T:
    """Call ``llm.complete`` and parse the result; retry with feedback on parse failure.

    Parameters
    ----------
    llm:
        Any object with a ``complete(messages, max_tokens, temperature)``
        method (``LLMClient``, ``LLMRouter.for_role(...)``,
        ``ReproRecordingClient`` etc.).
    messages:
        The full prompt to send. The wrapper does not modify the
        original list; on retry it constructs a new list that includes
        the failed assistant turn and a new user-feedback turn.
    parser:
        A function that takes the raw string and returns ``T`` or
        raises any subclass of ``Exception``. Common parsers raise
        ``json.JSONDecodeError``, ``pydantic.ValidationError`` or
        ``ValueError``; all are caught here.
    role:
        Free-form label for the agent role using this retry. Recorded in
        ``StructuredResponseFailure.role`` and used in logs.
    max_retries:
        Number of feedback retries after the initial call. ``max_retries=0``
        is equivalent to the legacy "one shot, then raise" behaviour;
        the default of ``2`` allows up to three total attempts.
    format_reminder:
        Optional role-specific format reminder appended to the feedback
        user message on every retry. Use this to remind the LLM of
        required keys (e.g. ``"The JSON must include keys: research_question,
        steps, rationale."``). Domain-specific clinical guidance should
        live in the role's *system* prompt, not here.
    feedback_preamble, feedback_instructions:
        Override the default natural-language feedback wrapping if a
        role needs different phrasing.

    Returns
    -------
    The parsed object on the first successful attempt.

    Raises
    ------
    StructuredResponseFailure
        When every attempt up to ``max_retries`` raised during parsing.
    """
    attempts: List[StructuredAttempt] = []
    base_messages: List[LLMMessage] = list(messages)
    current: List[LLMMessage] = list(base_messages)
    last_exc: Optional[BaseException] = None
    for i in range(max_retries + 1):
        raw = authorized_complete(
            llm, current, max_tokens=max_tokens, temperature=temperature
        )
        head = (raw or "").strip().replace("\n", " ⏎ ")[:400]
        try:
            value = parser(raw)
        except Exception as exc:  # noqa: BLE001 — parser may raise anything
            attempts.append(
                StructuredAttempt(
                    attempt=i,
                    raw_head=head,
                    raw_chars=len(raw or ""),
                    error_class=exc.__class__.__name__,
                    error_message=str(exc)[:600],
                )
            )
            last_exc = exc
            if i >= max_retries:
                break
            # Keep only the latest failed response beside the immutable base
            # prompt.  Accumulating every full JSON attempt grows Planner
            # retries quadratically (and can exceed the original prompt by
            # tens of kilobytes) without adding useful correction context.
            feedback_parts = [
                feedback_preamble,
                f"  {exc.__class__.__name__}: {str(exc)[:400]}",
                "",
                feedback_instructions,
            ]
            if format_reminder:
                feedback_parts.extend(["", format_reminder])
            feedback_message = "\n".join(feedback_parts)
            current = base_messages + [
                LLMMessage(role="assistant", content=raw or ""),
                LLMMessage(role="user", content=feedback_message),
            ]
            continue
        else:
            attempts.append(
                StructuredAttempt(
                    attempt=i,
                    raw_head=head,
                    raw_chars=len(raw or ""),
                    error_class=None,
                    error_message=None,
                )
            )
            return value
    # All attempts exhausted.
    failure = StructuredResponseFailure(attempts, role=role)
    if last_exc is not None:
        raise failure from last_exc
    raise failure


__all__ = [
    "StructuredAttempt",
    "StructuredResponseFailure",
    "call_llm_with_structured_retry",
]
