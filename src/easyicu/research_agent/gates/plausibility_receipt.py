"""What a ``retain_and_flag`` receipt is, and whether the run produced one.

The static gate in ``plausibility_obligation`` can only prove that a script is
*shaped* to record the out-of-range counts.  It cannot prove the script ran, or
that the number reached the artifact a reader opens.  An earlier draft accepted
any call whose name looked like a serializer, which proved only that the script
had *touched* something that writes: ``json.dump(audit, sys.stdout)``, a debug
file under ``/tmp``, and ``DataFrame(...).to_json()`` with no destination at all
each satisfied it while leaving nothing behind.

So the obligation is settled in two places, and this module owns both halves of
the contract that joins them:

* the **destination** the static gate will accept -- the canonical step summary
  the host itself reads, or a path the script registers in ``output_files``;
* the **receipt** the sealed summary must actually carry once the step has run.

Publishing one spelling is deliberate.  A gate that demands a shape nobody was
told to write blocks every script, so the same constants that the gates read are
rendered into the Coder's binding instructions -- the contract and its enforcer
cannot drift apart because they are the same object.

Case neutrality: the variable names come from ``context``; nothing here knows
which study, benchmark, column or bound is in play.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping, Optional, Sequence, Set

from ..schema import AnalysisStep, ResearchContext, ValidationFinding

#: The sealed host contract key a script reads to obtain the bounds.  Its
#: presence is what proves the script is exercising the typed policy rather
#: than doing arithmetic that happens to involve a comparison.
POLICY_CONTRACT_KEY = "analysis_plausibility_range"

#: The breadcrumb ``repairs/plausibility.py`` leaves after it removes a
#: rejection.  Reading it from the source text rather than the tree is
#: deliberate -- it is a comment, so it cannot be seen any other way, and the
#: asymmetry is safe in this direction: a marker only ever *adds* an
#: obligation, so a forged or stale one costs a repair and never buys a pass.
REPAIR_RECEIPT_MARKER = "_easyicu_flag_only_plausibility_range_retained_v1"

#: The one artifact the host itself opens after a step, and therefore the only
#: destination a pre-execution check can accept without guessing.
CANONICAL_STEP_SUMMARY_FILENAME = "step_summary.json"

#: Where the receipt lives inside that artifact.
RECEIPT_SUMMARY_KEY = "plausibility_audit"

#: The mapping in a step summary through which a script registers any other
#: file it wrote.  A path that appears there is a declared output; a path that
#: does not is a scratch file.
OUTPUT_REGISTRATION_KEY = "output_files"

RECEIPT_POLICY_FIELD = "policy"
RECEIPT_POLICY_VALUE = "retain_and_flag"
RECEIPT_VARIABLE_FIELD = "variable"
RECEIPT_BELOW_FIELD = "below_minimum_n"
RECEIPT_ABOVE_FIELD = "above_maximum_n"
RECEIPT_TOTAL_FIELD = "out_of_range_n"

_VALIDATOR = "mechanical_code_preflight"

#: The compact form carried in the Coder's binding instructions.  It and the
#: sentence below are rendered from the same constants, so the instruction and
#: the gate cannot disagree about a field name -- which is the only part of the
#: wording that has to match.
RECEIPT_CONTRACT_CLAUSE = (
    f"Record the counts in `step_summary[{RECEIPT_SUMMARY_KEY!r}]` as an object "
    "keyed by the exact resolved column, each value "
    f"`{{{RECEIPT_POLICY_FIELD!r}: {RECEIPT_POLICY_VALUE!r}, "
    f"{RECEIPT_BELOW_FIELD!r}: <int>, {RECEIPT_ABOVE_FIELD!r}: <int>, "
    f"{RECEIPT_TOTAL_FIELD!r}: <int>}}`, written on every path including when "
    "every count is 0. Printing it or writing it to an unregistered scratch "
    "path does not satisfy the policy."
)

#: The full contract quoted back in a finding, where the extra words are the
#: repair instruction.
RECEIPT_CONTRACT_SENTENCE = (
    f"Record the result in `step_summary[{RECEIPT_SUMMARY_KEY!r}]` as a JSON "
    "object keyed by the exact resolved column, each value "
    f"`{{{RECEIPT_POLICY_FIELD!r}: {RECEIPT_POLICY_VALUE!r}, "
    f"{RECEIPT_BELOW_FIELD!r}: <int>, {RECEIPT_ABOVE_FIELD!r}: <int>, "
    f"{RECEIPT_TOTAL_FIELD!r}: <int>}}` with "
    f"{RECEIPT_TOTAL_FIELD} equal to {RECEIPT_BELOW_FIELD} + "
    f"{RECEIPT_ABOVE_FIELD}. Write it on every path, including when every "
    "count is 0: a count of zero is a result, and its absence cannot be told "
    "apart from never having looked. Printing it, or writing it to a scratch "
    "path this step does not register in "
    f"`step_summary[{OUTPUT_REGISTRATION_KEY!r}]`, does not satisfy this."
)


def _string_constants(tree: ast.AST) -> Set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def mentions_a_plausibility_range(tree: ast.AST) -> bool:
    """Whether the script names a plausibility range at all.

    Generous on purpose, and only in the direction of asking the question.
    Scripts reach the same bounds through more than one projection of the
    contract, and a step that reads one of them and records nothing should be
    told so whichever spelling it used.  Precision belongs to the checks that
    decide the answer.
    """

    return any(
        value == POLICY_CONTRACT_KEY or value.endswith("plausibility_range")
        for value in _string_constants(tree)
    )


def ranged_variable_names(context: ResearchContext) -> list[str]:
    """The declared variables that carry a plausibility range.

    ``variables`` is a required field on every real context; the ``getattr`` is
    for the composition tests that pass a bare stub, and a context with no
    variables genuinely declares no range.
    """

    return sorted(
        str(descriptor.name)
        for descriptor in getattr(context, "variables", ())
        if descriptor.valid_range is not None
    )


def step_is_under_the_flag_only_obligation(
    *,
    script_text: str,
    tree: Optional[ast.AST],
    context: ResearchContext,
) -> Optional[str]:
    """Why this step owes a receipt, or ``None`` if it does not.

    The trigger is the host's own typed policy plus the script's use of it, so
    it survives the LLM auditor going quiet -- which is the whole point.  A step
    that never reads the sealed range is out of scope here **and is not thereby
    proven compliant**; widening the trigger to every step that merely touches a
    ranged variable would put a domain audit on every step in every study, which
    is a scientific scope decision for the run's owner rather than something a
    gate should take by implication.
    """

    if not ranged_variable_names(context):
        return None
    if REPAIR_RECEIPT_MARKER in str(script_text or ""):
        return "deterministic_repair_receipt"
    if tree is not None and mentions_a_plausibility_range(tree):
        return "declared_range_read"
    return None


def _declares_the_policy(value: Any) -> bool:
    """Whether a receipt's ``policy`` field names this policy.

    Both spellings come from the host.  A script may write the action as a
    string, or echo back the whole ``plausibility_policy`` object it was handed
    in the contract -- and a real generated script does exactly that, which is
    if anything the more faithful of the two.  Reading only the string would
    have rejected a receipt that was right, and in a fail-closed gate a legal
    spelling the check cannot read is a wrong block, not a missed one.
    """

    if isinstance(value, Mapping):
        return str(value.get("out_of_range_action") or "").strip() == (
            RECEIPT_POLICY_VALUE
        )
    return str(value or "").strip() == RECEIPT_POLICY_VALUE


def _nonnegative_int(value: Any) -> Optional[int]:
    """The value as a count, or ``None`` if it is not one.

    ``bool`` is rejected on purpose: ``True`` is an ``int`` in Python, and a
    receipt reporting ``above_maximum_n: True`` has recorded that something
    happened without recording how much.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _receipt_records(payload: Any) -> Optional[list[tuple[Any, Any]]]:
    """The ``(variable, record)`` pairs in either accepted container.

    The canonical spelling is a mapping keyed by column.  A list of records
    each naming its own variable carries exactly the same fields and is just as
    readable, so it is accepted too -- the field contract is what is checked,
    and refusing a second obvious container would only buy repairs.
    """

    if isinstance(payload, Mapping):
        return [(key, value) for key, value in payload.items()]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        records: list[tuple[Any, Any]] = []
        for record in payload:
            if not isinstance(record, Mapping):
                return None
            records.append((record.get(RECEIPT_VARIABLE_FIELD), record))
        return records
    return None


def plausibility_audit_receipt_findings(
    *,
    step_summary: Mapping[str, Any],
    context: ResearchContext,
    step: AnalysisStep,
    script_text: str,
) -> list[ValidationFinding]:
    """Check that the executed step actually left the receipt behind.

    The static gate proves the count was computed from a bound read out of the
    contract; this one proves it reached the artifact.  Neither is sufficient
    alone -- a receipt on its own is only a number someone typed, and a shape on
    its own is only an intention -- so both run.
    """

    tree: Optional[ast.AST]
    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        tree = None
    trigger = step_is_under_the_flag_only_obligation(
        script_text=script_text,
        tree=tree,
        context=context,
    )
    if trigger is None:
        return []

    declared = ranged_variable_names(context)
    step_id = str(step.step_id)
    detail_base = {
        "step_id": step_id,
        "issue_code": "flag_only_plausibility_receipt",
        "policy": RECEIPT_POLICY_VALUE,
        "policy_authority": "typed_research_context_plausibility_policy",
        "trigger": trigger,
        "receipt_key": RECEIPT_SUMMARY_KEY,
        "ranged_variables": declared,
    }

    def _blocked(reason: str, message: str, **extra: Any) -> list[ValidationFinding]:
        return [
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=f"{message} {RECEIPT_CONTRACT_SENTENCE}",
                detail={**detail_base, "reason": reason, **extra},
            )
        ]

    payload = (
        step_summary.get(RECEIPT_SUMMARY_KEY)
        if isinstance(step_summary, Mapping)
        else None
    )
    if payload is None:
        return _blocked(
            "plausibility_audit_receipt_absent",
            (
                f"Step {step_id} reads a declared plausibility range, but the "
                f"step summary it produced carries no {RECEIPT_SUMMARY_KEY!r} "
                "receipt. Code that is shaped to record the out-of-range counts "
                "has not recorded them until the run leaves them in an artifact "
                "a reader can open."
            ),
        )

    records = _receipt_records(payload)
    if not records:
        return _blocked(
            "plausibility_audit_receipt_empty",
            (
                f"Step {step_id} wrote a {RECEIPT_SUMMARY_KEY!r} receipt that "
                "names no variable, so nothing in it says a range was checked."
            ),
            observed_type=type(payload).__name__,
        )

    findings: list[ValidationFinding] = []
    for variable, record in records:
        name = str(variable or "").strip()
        located = {"variable": name or None}
        if not name:
            findings.extend(
                _blocked(
                    "plausibility_audit_variable_missing",
                    (
                        f"Step {step_id} wrote a {RECEIPT_SUMMARY_KEY!r} entry "
                        "that does not name the variable it audited."
                    ),
                    **located,
                )
            )
            continue
        if name not in declared:
            findings.extend(
                _blocked(
                    "plausibility_audit_variable_not_declared",
                    (
                        f"Step {step_id} audited {name!r}, which is not one of "
                        "the variables the host declared a plausibility range "
                        "for. Key the receipt by the exact resolved column."
                    ),
                    **located,
                )
            )
            continue
        if not isinstance(record, Mapping):
            findings.extend(
                _blocked(
                    "plausibility_audit_record_untyped",
                    (
                        f"Step {step_id} recorded {name!r} as a bare value "
                        "rather than the typed record the policy owes."
                    ),
                    **located,
                    observed_type=type(record).__name__,
                )
            )
            continue
        policy = record.get(RECEIPT_POLICY_FIELD)
        if not _declares_the_policy(policy):
            findings.extend(
                _blocked(
                    "plausibility_audit_policy_mismatch",
                    (
                        f"Step {step_id} recorded {name!r} under policy "
                        f"{policy!r}. The host's declared action for this range "
                        f"is {RECEIPT_POLICY_VALUE!r}."
                    ),
                    **located,
                    observed_policy=policy,
                )
            )
            continue
        counts = {
            field: _nonnegative_int(record.get(field))
            for field in (
                RECEIPT_BELOW_FIELD,
                RECEIPT_ABOVE_FIELD,
                RECEIPT_TOTAL_FIELD,
            )
        }
        missing = sorted(field for field, value in counts.items() if value is None)
        if missing:
            findings.extend(
                _blocked(
                    "plausibility_audit_count_missing",
                    (
                        f"Step {step_id} recorded {name!r} without a usable "
                        f"{', '.join(missing)}. Every count is a non-negative "
                        "integer and 0 is a valid one -- omitting it is not the "
                        "same as reporting none."
                    ),
                    **located,
                    missing_fields=missing,
                )
            )
            continue
        below = counts[RECEIPT_BELOW_FIELD]
        above = counts[RECEIPT_ABOVE_FIELD]
        total = counts[RECEIPT_TOTAL_FIELD]
        assert below is not None and above is not None and total is not None
        if total != below + above:
            findings.extend(
                _blocked(
                    "plausibility_audit_count_inconsistent",
                    (
                        f"Step {step_id} recorded {name!r} with "
                        f"{RECEIPT_TOTAL_FIELD}={total}, which is not "
                        f"{RECEIPT_BELOW_FIELD}={below} plus "
                        f"{RECEIPT_ABOVE_FIELD}={above}. A value cannot be both "
                        "below the minimum and above the maximum, so the two "
                        "sides partition the total."
                    ),
                    **located,
                    below_minimum_n=below,
                    above_maximum_n=above,
                    out_of_range_n=total,
                )
            )
    return findings


__all__ = [
    "CANONICAL_STEP_SUMMARY_FILENAME",
    "OUTPUT_REGISTRATION_KEY",
    "POLICY_CONTRACT_KEY",
    "RECEIPT_ABOVE_FIELD",
    "RECEIPT_BELOW_FIELD",
    "RECEIPT_CONTRACT_CLAUSE",
    "RECEIPT_CONTRACT_SENTENCE",
    "RECEIPT_POLICY_FIELD",
    "RECEIPT_POLICY_VALUE",
    "RECEIPT_SUMMARY_KEY",
    "RECEIPT_TOTAL_FIELD",
    "RECEIPT_VARIABLE_FIELD",
    "REPAIR_RECEIPT_MARKER",
    "mentions_a_plausibility_range",
    "plausibility_audit_receipt_findings",
    "ranged_variable_names",
    "step_is_under_the_flag_only_obligation",
]
