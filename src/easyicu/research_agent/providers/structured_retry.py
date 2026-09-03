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

import json
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from .protocol import LLMMessage
from .factory import authorized_complete
from .llm import (
    clear_provider_call_receipt,
    current_provider_call_receipt,
    safe_provider_finish_reason,
)
from .structured_diagnostics import (
    infer_validation_stage,
    safe_projected_validation_issues,
    safe_validation_issues,
    safe_validation_stage,
    violation_sha256,
)

T = TypeVar("T")


@dataclass(frozen=True)
class StructuredRetryProgress:
    """Safe lifecycle projection for one structured model attempt.

    The projection deliberately excludes the model response and validator
    message.  Product surfaces may show that a role is generating, retrying,
    accepted, or exhausted without exposing private reasoning or unbounded
    scientific payloads.
    """

    role: str
    phase: Literal["started", "rejected", "accepted"]
    attempt: int
    total_attempts: int
    error_class: Optional[str] = None
    validation_stage: Optional[str] = None
    validation_issues: Optional[List[Dict[str, Any]]] = None
    violation_sha256: Optional[str] = None
    reason_code: Optional[str] = None


def _notify_progress(
    callback: Optional[Callable[[StructuredRetryProgress], None]],
    event: StructuredRetryProgress,
) -> None:
    """Treat UI progress as advisory; it must never change model execution."""

    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - observers cannot own retry authority
        return


_DEFAULT_FEEDBACK_PREAMBLE = (
    "Your previous response could not be parsed into the required structured "
    "output. The validator reported:"
)


#: Shown instead when the response WAS well-formed JSON and the rejection was
#: about what it said, not how it was written.
#:
#: MEASURED: a canonical-9 task (m2 mortality prediction) lost its whole run to
#: ``5 planner attempt(s) failed in 5 distinct ways``, and 27 such exhaustions
#: are recorded across the corpus. One of m2's rejections was "analysis plan may
#: declare at most one step with planned_analysis_role='primary'" -- a perfectly
#: valid JSON document describing an invalid study. It was told its response
#: "could not be parsed", and then, in the most salient position, not to use
#: trailing commas or comments. The retry already carries every distinct earlier
#: rejection forward; what it did not do was say which kind of thing was wrong.
_VALIDATION_FEEDBACK_PREAMBLE = (
    "Your previous response was well-formed JSON, but it did not satisfy the "
    "required contract. Nothing is wrong with the formatting. The validator "
    "rejected what the response said:"
)


_VALIDATION_FEEDBACK_INSTRUCTIONS = (
    "Return a corrected JSON object in the same format. The formatting was "
    "already accepted -- change the content so it satisfies every rule above, "
    "and do not reformat, restate or abbreviate parts that were not rejected."
)


_DEFAULT_FEEDBACK_INSTRUCTIONS = (
    "Please return ONLY a single valid JSON object matching the schema "
    "described in the original instructions. Do not include prose before or "
    "after the JSON. Do not wrap the JSON in a Markdown code fence. Do not "
    "include trailing commas. Do not include comments."
)


_FEEDBACK_MAX_VIOLATIONS = 40
_FEEDBACK_MAX_CHARS = 4000
_FEEDBACK_MAX_INPUT_ECHO = 80
_EARLIER_FAILURE_MAX_CHARS = 1200


def _violation_lines(exc: BaseException) -> Optional[List[str]]:
    """One compact line per structured-validation violation, or ``None``.

    Duck-typed on ``.errors()`` rather than imported from a validation
    library: this module is deliberately library-agnostic, and anything that
    can enumerate its own ``loc``/``msg`` records renders the same way.
    Anything that cannot is reported through the plain-string path.
    """

    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        records = list(errors())
    except Exception:  # noqa: BLE001 — an un-enumerable validator falls back
        return None
    if not records or not all(isinstance(record, dict) for record in records):
        return None
    lines: List[str] = []
    for record in records:
        location = ".".join(str(part) for part in record.get("loc", ())) or "<root>"
        message = str(record.get("msg", "")).strip()
        line = f"{location}: {message}" if message else location
        if "input" in record:
            # Omitted rather than truncated when long. A clipped container
            # repr is not the value the location names -- on a missing-field
            # violation the reported input is the *enclosing* object, so
            # "you sent: {...}" would attribute the whole payload to the one
            # field that was absent from it.
            echo = repr(record["input"])
            if len(echo) <= _FEEDBACK_MAX_INPUT_ECHO:
                line = f"{line} (you sent: {echo})"
        lines.append(line)
    return lines


def clip_to_whole_lines(text: str, max_chars: int) -> str:
    """Truncate on a line boundary, never inside a line.

    A violation list cut mid-line reads as a shorter, different constraint
    than the one the validator raised, which is the failure this whole
    module exists to avoid.
    """

    if len(text) <= max_chars:
        return text
    kept: List[str] = []
    used = 0
    for line in text.splitlines():
        if kept and used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


def render_parse_failure(
    exc: BaseException, *, max_chars: int = _FEEDBACK_MAX_CHARS
) -> str:
    """Render a parse failure so a retry can fix *all* of it, not just its head.

    A validation error prints roughly 240 characters per violation, a third
    of it a documentation URL the model cannot visit. Truncating that prose
    at a fixed character budget therefore states the first violation and
    silently drops the rest: measured on a real rejection, a 400-character
    window showed 2 of 6 forbidden fields, and 1 of 20.

    The model then fixes what it was shown, resubmits, and is rejected for
    the violations it was never told about. A real Planner run recorded
    exactly that: attempt 0 was told only that one field was missing, and
    attempt 4 -- having supplied it -- died on the same six forbidden fields
    that had been present, and unreported, from the start. Across the
    recorded runs 14 of 18 rejections carry more than one violation.

    Enumerating the violations compactly and budgeting by violation count
    states the whole constraint set in less space than the truncated prose
    used.
    """

    lines = _violation_lines(exc)
    if lines is None:
        text = str(exc)
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
    total = len(lines)
    kept: List[str] = []
    used = 0
    for line in lines[:_FEEDBACK_MAX_VIOLATIONS]:
        rendered_length = len(line) + 7  # "    - " plus the newline
        if kept and used + rendered_length > max_chars:
            break
        kept.append(line)
        used += rendered_length
    body = "\n".join(f"    - {line}" for line in kept)
    if len(kept) < total:
        body += (
            f"\n    - ...and {total - len(kept)} further problem(s) not listed "
            "here; re-check the whole object against the schema."
        )
    return (
        f"{total} problem(s), all of which must be fixed together in one "
        f"corrected response:\n{body}"
    )


@dataclass
class StructuredAttempt:
    """One LLM call attempt during a structured-retry loop."""

    attempt: int
    raw_head: str  # first ~400 chars of the raw response
    raw_chars: int
    error_class: Optional[str]  # None when the attempt succeeded
    error_message: Optional[str]
    finish_reason: Optional[str] = None
    usage_summary: Optional[Dict[str, int]] = None
    transport_attempts: int = 1
    validation_stage: Optional[str] = None
    validation_issues: Optional[List[Dict[str, Any]]] = None
    violation_sha256: Optional[str] = None
    reason_code: Optional[str] = None


def _safe_declared_reason_code(exc: BaseException) -> Optional[str]:
    """Project only an owner-declared, closed-form diagnostic reason code."""

    diagnostic = getattr(exc, "easyicu_safe_diagnostic", None)
    if not isinstance(diagnostic, Mapping):
        return None
    owner = str(diagnostic.get("owner") or "").strip()
    if owner not in {
        "easyicu.planning.progressive_compiler_v1",
        "easyicu.schema_validation_v1",
    }:
        return None
    reason_code = str(diagnostic.get("reason_code") or "").strip()
    if re.fullmatch(r"[a-z][a-z0-9_]{2,79}", reason_code) is None:
        return None
    return reason_code


def safe_provider_error_category(value: Any) -> Optional[str]:
    """Map an exception/type label onto a closed, non-secret category set."""

    if value is None:
        return None
    name = type(value).__name__ if isinstance(value, BaseException) else str(value)
    folded = name.strip().casefold()
    if not folded:
        return "error"
    if "providerhardstop" in folded or "budget" in folded:
        return "provider_budget"
    if "providerrefusal" in folded or "provider_refusal" in folded:
        return "provider_refusal"
    if "structuredresponse" in folded:
        return "structured_response"
    if "validation" in folded or folded in {"valueerror", "assertionerror"}:
        return "validation"
    if "json" in folded or "decode" in folded or "parse" in folded:
        return "parse"
    if "timeout" in folded:
        return "timeout"
    if "ratelimit" in folded or "rate_limit" in folded:
        return "rate_limit"
    if any(token in folded for token in ("connection", "connect", "protocol")):
        return "connection"
    if any(token in folded for token in ("permission", "authorization", "configuration")):
        return "authorization"
    if any(token in folded for token in ("http", "apierror", "status")):
        return "provider_http"
    return "error"


def _bounded_int(value: Any, *, minimum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, parsed)


def _safe_provider_call_metadata(
    llm: Any,
) -> tuple[Optional[str], Dict[str, int], int]:
    """Project bounded response metadata without prompt or response content."""

    receipt = current_provider_call_receipt()
    if receipt is not None:
        return (
            receipt.finish_reason,
            dict(receipt.usage_summary),
            receipt.transport_attempts,
        )

    try:
        raw_finish_reason = getattr(llm, "last_finish_reason", None)
    except Exception:
        raw_finish_reason = None
    finish_reason = safe_provider_finish_reason(raw_finish_reason)

    try:
        raw_usage = getattr(llm, "last_usage", None)
    except Exception:
        raw_usage = None
    usage_summary: Dict[str, int] = {}
    if isinstance(raw_usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage_summary[key] = int(value)

    try:
        raw_transport_attempts = getattr(llm, "last_transport_attempts", 1)
        transport_attempts = max(1, int(raw_transport_attempts))
    except (TypeError, ValueError):
        transport_attempts = 1
    return finish_reason, usage_summary, transport_attempts


def safe_structured_attempt_metadata(
    attempts: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Return the only structured-attempt shape permitted in artifacts.

    ``raw_head`` and ``error_message`` intentionally never cross this boundary:
    both can contain response text, patient free text, a prompt fragment, or a
    provider echo of a credential.  Operators still get the response length,
    failure class, finish reason, token counts, and physical transport count.
    """

    projected: List[Dict[str, Any]] = []
    for item in list(attempts)[:16]:
        if isinstance(item, Mapping):
            source = item
            attempt_index = _bounded_int(
                source.get("attempt"), minimum=1, default=1
            )
            raw_chars = source.get("raw_chars")
            raw_error = source.get("error_class")
            raw_finish = source.get("finish_reason")
            raw_usage = source.get("usage")
            raw_transport_attempts = source.get("transport_attempts")
            raw_validation_stage = source.get("validation_stage")
            raw_validation_issues = source.get("validation_issues")
            raw_violation_sha256 = source.get("violation_sha256")
            raw_reason_code = source.get("reason_code")
        else:
            attempt_index = _bounded_int(item.attempt, minimum=0, default=0) + 1
            raw_chars = item.raw_chars
            raw_error = item.error_class
            raw_finish = item.finish_reason
            raw_usage = item.usage_summary
            raw_transport_attempts = item.transport_attempts
            raw_validation_stage = item.validation_stage
            raw_validation_issues = item.validation_issues
            raw_violation_sha256 = item.violation_sha256
            raw_reason_code = item.reason_code
        error_class = safe_provider_error_category(raw_error)
        finish_reason = safe_provider_finish_reason(raw_finish)
        usage = {
            key: int(value)
            for key, value in dict(raw_usage or {}).items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
        row: Dict[str, Any] = {
            "attempt": attempt_index,
            "raw_chars": _bounded_int(raw_chars, minimum=0, default=0),
            "error_class": error_class,
            "finish_reason": finish_reason,
            "usage": usage,
            "transport_attempts": _bounded_int(
                raw_transport_attempts, minimum=1, default=1
            ),
        }
        validation_stage = safe_validation_stage(raw_validation_stage)
        if validation_stage is not None:
            row["validation_stage"] = validation_stage
        validation_issues = safe_projected_validation_issues(raw_validation_issues)
        if validation_issues:
            row["validation_issues"] = validation_issues
        violation_sha256 = str(raw_violation_sha256 or "").strip().lower()
        if len(violation_sha256) == 64 and all(
            char in "0123456789abcdef" for char in violation_sha256
        ):
            row["violation_sha256"] = violation_sha256
        reason_code = str(raw_reason_code or "").strip()
        if re.fullmatch(r"[a-z][a-z0-9_]{2,79}", reason_code):
            row["reason_code"] = reason_code
        projected.append(row)
    return projected


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


def _clip_failure_for_summary(message: str, max_chars: int) -> str:
    """Bound one failure for the post-mortem without implying it ended there.

    This summary is what a human reads when a task dies at planning and
    produces nothing else. It used to be a bare ``message[:300]``, which cuts
    mid-sentence with no signal that anything followed. On the 2026-08-02
    nine-task run that rendered E3's rejection as::

        model_requirements are currently supported only on
        method='adjusted_association_models' steps ...; other analysis
        families must use

    -- a sentence that stops exactly before it says what they must use. The
    constraint reads as incomplete host guidance rather than as a clipped log
    line, and the first thing that gets investigated is the wrong thing.

    The module already states the principle for the retry path
    (:func:`clip_to_whole_lines`: "a violation list cut mid-line reads as a
    shorter, different constraint than the one the validator raised"). It just
    was not applied here.

    A pure line-boundary clip is wrong for *this* payload, though. A pydantic
    rejection is one short header line plus one long line per violation, so
    clipping to whole lines keeps "1 problem(s), all of which must be fixed
    together in one corrected response:" and drops every actual violation --
    honest, and useless. So take whichever clip preserves more text and mark
    the cut either way: the marker is what stops a clipped line being read as
    the whole constraint, and that is the property worth guaranteeing.
    """

    if len(message) <= max_chars:
        return message
    by_line = clip_to_whole_lines(message, max_chars)
    by_char = message[:max_chars]
    clipped = by_line if len(by_line) >= len(by_char) else by_char
    return f"{clipped.rstrip()} [...truncated]"


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
        f"[{index}] {error_class}: "
        f"{_clip_failure_for_summary(message, per_failure_chars)}"
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
    # This machine-readable attachment is deliberately a safe projection, not
    # the ``StructuredAttempt`` objects themselves.  A Web failure serializer
    # can therefore preserve useful per-attempt diagnostics without ever
    # touching raw model text or parser messages.
    try:
        setattr(
            exc,
            "easyicu_structured_attempt_metadata",
            safe_structured_attempt_metadata(attempts),
        )
    except Exception:
        pass
    note = (
        "structured-retry history before this failure: "
        + summarise_attempt_history(attempts, role=role)
    )
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)
        return

    # ``BaseException.add_note`` is available only from Python 3.11, while
    # EasyICU supports 3.10.  Keep the same public ``__notes__`` contract on
    # 3.10 so callers and post-mortem tooling do not silently lose the retry
    # history on the oldest supported interpreter.
    notes = getattr(exc, "__notes__", None)
    if isinstance(notes, list):
        notes.append(note)
    else:
        setattr(exc, "__notes__", [note])


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
        self.easyicu_structured_attempt_metadata = safe_structured_attempt_metadata(
            self.attempts
        )
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
    progress_callback: Optional[Callable[[StructuredRetryProgress], None]] = None,
    structured_output: Any = None,
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
        total_attempts = max_retries + 1
        _notify_progress(
            progress_callback,
            StructuredRetryProgress(
                role=role,
                phase="started",
                attempt=i + 1,
                total_attempts=total_attempts,
            ),
        )
        try:
            clear_provider_call_receipt()
            complete_kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if structured_output is not None:
                complete_kwargs["structured_output"] = structured_output
            raw = authorized_complete(llm, current, **complete_kwargs)
        except BaseException as exc:
            # Transport failures abort the loop here, outside the parser
            # guard below. Carry the attempts recorded so far out with the
            # exception rather than letting them die with the frame.
            finish_reason, usage_summary, receipt_transport_attempts = (
                _safe_provider_call_metadata(llm)
            )
            try:
                exception_transport_attempts = max(
                    1, int(getattr(exc, "easyicu_transport_attempts", 1))
                )
            except (TypeError, ValueError):
                exception_transport_attempts = 1
            transport_attempts = max(
                receipt_transport_attempts,
                exception_transport_attempts,
            )
            terminal_reason_code = str(getattr(exc, "reason_code", "") or "").strip()
            terminal_attempt = StructuredAttempt(
                attempt=i,
                raw_head="",
                raw_chars=0,
                error_class=type(exc).__name__,
                error_message=(
                    terminal_reason_code
                    or "transport failed before a structured response"
                ),
                finish_reason=finish_reason,
                usage_summary=usage_summary,
                transport_attempts=transport_attempts,
            )
            all_attempts = [*attempts, terminal_attempt]
            # Keep the human note's established meaning: when parser failures
            # preceded the transport abort, summarize those responses rather
            # than counting a no-response event as another parser rejection.
            # The machine-readable projection still includes the terminal
            # transport failure so the Web artifact has the complete sequence.
            annotate_with_attempt_history(exc, attempts or [terminal_attempt], role=role)
            try:
                setattr(
                    exc,
                    "easyicu_structured_attempt_metadata",
                    safe_structured_attempt_metadata(all_attempts),
                )
            except Exception:
                pass
            raise
        head = (raw or "").strip().replace("\n", " ⏎ ")[:400]
        finish_reason, usage_summary, transport_attempts = _safe_provider_call_metadata(
            llm
        )
        try:
            value = parser(raw)
        except Exception as exc:  # noqa: BLE001 — parser may raise anything
            # Rendered once, then reused for the record, the feedback message
            # and the carry-forward signature -- three readers of one string,
            # so the retry cannot be shown a different rejection from the one
            # that was recorded.
            rendered_failure = render_parse_failure(exc)
            validation_stage = infer_validation_stage(exc)
            validation_issues = safe_validation_issues(exc)
            reason_code = _safe_declared_reason_code(exc)
            attempts.append(
                StructuredAttempt(
                    attempt=i,
                    raw_head=head,
                    raw_chars=len(raw or ""),
                    error_class=exc.__class__.__name__,
                    error_message=rendered_failure,
                    finish_reason=finish_reason,
                    usage_summary=usage_summary,
                    transport_attempts=transport_attempts,
                    validation_stage=validation_stage,
                    validation_issues=validation_issues,
                    violation_sha256=violation_sha256(rendered_failure),
                    reason_code=reason_code,
                )
            )
            _notify_progress(
                progress_callback,
                StructuredRetryProgress(
                    role=role,
                    phase="rejected",
                    attempt=i + 1,
                    total_attempts=total_attempts,
                    error_class=exc.__class__.__name__,
                    validation_stage=validation_stage,
                    validation_issues=validation_issues,
                    violation_sha256=violation_sha256(rendered_failure),
                    reason_code=reason_code,
                ),
            )
            last_exc = exc
            if i >= max_retries:
                break
            # When retained, keep only the latest failed response beside the
            # immutable base prompt. Accumulating every full JSON attempt grows
            # structured retries quadratically. Large self-contained outputs
            # can opt out and regenerate from the base plus validator feedback.
            # Which kind of rejection this is, decided by the response itself
            # rather than by an exception class name that varies with the
            # parser: if it loads as JSON, the formatting was never the
            # problem and telling it otherwise sends the retry at the wrong
            # thing.
            well_formed_json = False
            if raw and raw.strip():
                try:
                    json.loads(raw)
                except Exception:  # noqa: BLE001 -- any parse failure means "not JSON"
                    well_formed_json = False
                else:
                    well_formed_json = True
            # Local, never a reassignment of the caller's parameters: attempt
            # 2 can be malformed where attempt 1 was not, and an overwritten
            # parameter would keep telling it the formatting was fine.
            using_default_feedback = (
                feedback_preamble is _DEFAULT_FEEDBACK_PREAMBLE
                and feedback_instructions is _DEFAULT_FEEDBACK_INSTRUCTIONS
            )
            attempt_preamble = feedback_preamble
            attempt_instructions = feedback_instructions
            if well_formed_json and using_default_feedback:
                attempt_preamble = _VALIDATION_FEEDBACK_PREAMBLE
                attempt_instructions = _VALIDATION_FEEDBACK_INSTRUCTIONS
            feedback_parts = [
                attempt_preamble,
                f"  {exc.__class__.__name__}: {rendered_failure}",
                "",
                attempt_instructions,
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
                if signature != (exc.__class__.__name__, rendered_failure)
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
                            f"  - {error_class}: "
                            f"{clip_to_whole_lines(message, _EARLIER_FAILURE_MAX_CHARS)}"
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
                    finish_reason=finish_reason,
                    usage_summary=usage_summary,
                    transport_attempts=transport_attempts,
                )
            )
            _notify_progress(
                progress_callback,
                StructuredRetryProgress(
                    role=role,
                    phase="accepted",
                    attempt=i + 1,
                    total_attempts=total_attempts,
                ),
            )
            return value
    # All attempts exhausted.
    failure = StructuredResponseFailure(attempts, role=role)
    if last_exc is not None:
        raise failure from last_exc
    raise failure


__all__ = [
    "StructuredAttempt",
    "StructuredRetryProgress",
    "StructuredResponseFailure",
    "annotate_with_attempt_history",
    "call_llm_with_structured_retry",
    "clip_to_whole_lines",
    "distinct_failures",
    "render_parse_failure",
    "summarise_attempt_history",
    "safe_structured_attempt_metadata",
    "safe_provider_error_category",
]
