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


def distinct_failures(
    attempts: Sequence[StructuredAttempt],
) -> List[tuple]:
    """The distinct ``(error_class, message)`` rejections, in first-seen order."""

    seen: List[tuple] = []
    for attempt in attempts:
        if attempt.error_class is None:
            continue
        signature = (attempt.error_class, attempt.error_message or "")
        if signature not in seen:
            seen.append(signature)
    return seen


def summarise_attempt_history(
    attempts: Sequence[StructuredAttempt],
    *,
    role: str,
    per_failure_chars: int = 300,
) -> str:
    """Describe the *shape* of a retry history, not just its last entry.

    Whether the retries all failed the same way or each failed differently
    is the first thing an operator needs, and the two mean opposite things:
    identical failures say the feedback loop changed nothing, which points
    at the host or the contract; differing failures say the model was
    converging or thrashing. Reporting only the final message hides that
    distinction, and hid it twice in one evening -- once when five Planner
    attempts were rejected by a host check the model could not satisfy, and
    once when a transport error aborted the loop and took two already
    recorded parse failures with it.
    """

    failures = [attempt for attempt in attempts if attempt.error_class is not None]
    if not failures:
        if not attempts:
            return f"no {role} response was received before this failure"
        return f"{len(attempts)} {role} response(s) parsed cleanly"

    distinct = distinct_failures(attempts)

    if len(distinct) == 1:
        error_class, message = distinct[0]
        if len(failures) == 1:
            return f"1 {role} attempt failed ({error_class}: {message})"
        return (
            f"all {len(failures)} {role} attempts failed identically "
            f"({error_class}: {message}) -- the retry feedback did not change "
            "the outcome"
        )

    rendered = "; ".join(
        f"[{index}] {error_class}: {message[:per_failure_chars]}"
        for index, (error_class, message) in enumerate(distinct)
    )
    return (
        f"{len(failures)} {role} attempt(s) failed in {len(distinct)} "
        f"distinct ways: {rendered}"
    )


def annotate_with_attempt_history(
    exc: BaseException,
    attempts: Sequence[StructuredAttempt],
    *,
    role: str,
) -> None:
    """Attach a retry history to an exception that is about to abort the loop.

    A transport failure escapes the parser ``try`` entirely, so the attempts
    recorded before it would otherwise be discarded along with the loop. The
    operator then sees only the transport error and cannot tell that earlier
    responses had already been rejected -- which is a different problem with
    a different fix. The note rides along on the existing exception, so the
    type callers catch is unchanged.
    """

    if not attempts:
        return
    add_note = getattr(exc, "add_note", None)
    if not callable(add_note):  # pragma: no cover - Python < 3.11
        return
    add_note(
        "structured-retry history before this failure: "
        + summarise_attempt_history(attempts, role=role)
    )


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
        super().__init__(
            f"StructuredResponseFailure[role={role}, "
            f"n_attempts={len(self.attempts)}]: "
            + summarise_attempt_history(self.attempts, role=role)
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
    include_failed_response_on_retry: bool = True,
    failed_response_transform: Optional[Callable[[str], str]] = None,
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
    include_failed_response_on_retry:
        Preserve the latest failed assistant response in the retry
        conversation. Disable this for large, self-contained structured
        outputs when the immutable base prompt plus validator feedback is
        sufficient to regenerate the object without inflating every retry.
    failed_response_transform:
        Optional host-owned projection applied to the failed response before
        it is included in the next request. This lets callers preserve
        corrective structure while removing bulky prose. The original
        response remains unchanged in attempt and reproducibility records.
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
        try:
            raw = authorized_complete(
                llm, current, max_tokens=max_tokens, temperature=temperature
            )
        except BaseException as exc:
            # Transport failures abort the loop here, outside the parser
            # guard below. Carry the attempts recorded so far out with the
            # exception rather than letting them die with the frame.
            annotate_with_attempt_history(exc, attempts, role=role)
            raise
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
            # When retained, keep only the latest failed response beside the
            # immutable base prompt. Accumulating every full JSON attempt grows
            # structured retries quadratically. Large self-contained outputs
            # can opt out and regenerate from the base plus validator feedback.
            feedback_parts = [
                feedback_preamble,
                f"  {exc.__class__.__name__}: {str(exc)[:400]}",
                "",
                feedback_instructions,
            ]
            # Only the newest rejection used to be shown, so each attempt
            # satisfied the last complaint and re-broke an earlier one: three
            # consecutive real Planner runs burned all five attempts on three
            # to five *different* violations and never converged. Carrying the
            # earlier distinct rejections forward states the whole constraint
            # set at once. Bounded by construction -- distinct messages only,
            # each truncated -- so it cannot grow with the attempt count the
            # way retaining every full response would.
            earlier = [
                signature
                for signature in distinct_failures(attempts[:-1])
                if signature != (exc.__class__.__name__, str(exc)[:600])
            ]
            if earlier:
                feedback_parts.extend(
                    [
                        "",
                        "Earlier attempts were rejected for these other "
                        "reasons. They are not repeated above, and a response "
                        "that fixes only the latest one will be rejected "
                        "again. Satisfy all of them together:",
                        *(
                            f"  - {error_class}: {message[:250]}"
                            for error_class, message in earlier
                        ),
                    ]
                )
            if format_reminder:
                feedback_parts.extend(["", format_reminder])
            feedback_message = "\n".join(feedback_parts)
            retry_messages: List[LLMMessage] = []
            if include_failed_response_on_retry:
                failed_response = raw or ""
                if failed_response_transform is not None:
                    failed_response = failed_response_transform(failed_response)
                if failed_response:
                    retry_messages.append(
                        LLMMessage(role="assistant", content=failed_response)
                    )
            retry_messages.append(LLMMessage(role="user", content=feedback_message))
            current = base_messages + retry_messages
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
    "annotate_with_attempt_history",
    "call_llm_with_structured_retry",
    "distinct_failures",
    "summarise_attempt_history",
]
