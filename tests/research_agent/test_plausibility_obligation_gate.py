"""The host-owned `retain_and_flag` obligation gate.

The shapes below are the ones real generated scripts write, kept in the same
per-column-helper form the r25 Step 06 script uses: the series arrives as a
parameter, so the variable's name never appears near the comparison and only
the bound identifies the check. A gate written against a tidier invented shape
passes its own fixtures and abstains on the only script that matters.
"""

from __future__ import annotations

import ast

import pytest

from easyicu.research_agent.gates.concept import deterministic_code_gate_findings
from easyicu.research_agent.gates.plausibility_obligation import (
    REPAIR_RECEIPT_MARKER,
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

HEADER = """
import json

manifest = json.loads(open("resolved.json").read())


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
"""

# The real shape: bounds read from the sealed contract inside a generic helper.
HELPER_HEAD = """

def validate_allowed_numeric(values, column, manifest):
    contract = manifest["raw_input_contracts"]["contracts"].get(column)
    bounds = contract.get("analysis_plausibility_range")
    if bounds is not None:
        lower = bounds.get("minimum")
        upper = bounds.get("maximum")
        numeric = values.dropna().astype(float)
"""

REJECTS = (
    HEADER
    + HELPER_HEAD
    + """        if lower is not None and (numeric < float(lower)).any():
            raise RuntimeError(f"{column} is below the plausibility minimum")
        if upper is not None and (numeric > float(upper)).any():
            raise RuntimeError(f"{column} is above the plausibility maximum")


write_json("step_summary.json", {"rows": 1})
"""
)

# Exactly what `patch_flag_only_plausibility_range_rejection` leaves behind:
# the guard tests survive, only their bodies are neutered.
RECORDS_NOTHING = (
    HEADER
    + HELPER_HEAD
    + """        if lower is not None and (numeric < float(lower)).any():
            pass  # """
    + REPAIR_RECEIPT_MARKER
    + """
        if upper is not None and (numeric > float(upper)).any():
            pass  # """
    + REPAIR_RECEIPT_MARKER
    + """


write_json("step_summary.json", {"rows": 1})
"""
)

LOCAL_ONLY = (
    HEADER
    + HELPER_HEAD
    + """        below_n = int((numeric < float(lower)).sum()) if lower is not None else 0
        above_n = int((numeric > float(upper)).sum()) if upper is not None else 0


write_json("step_summary.json", {"rows": 1})
"""
)

DECLARED = (
    HEADER
    + """
plausibility_audit = {}
"""
    + HELPER_HEAD
    + """        plausibility_audit[column] = {
            "below_n": int((numeric < float(lower)).sum()) if lower is not None else 0,
            "above_n": int((numeric > float(upper)).sum()) if upper is not None else 0,
        }


write_json("step_summary.json", {"plausibility": plausibility_audit})
"""
)

CONDITIONAL = (
    HEADER
    + """
plausibility_audit = {}
"""
    + HELPER_HEAD
    + """        if lower is not None and (numeric < float(lower)).any():
            plausibility_audit[column] = {
                "below_n": int((numeric < float(lower)).sum()),
            }


write_json("step_summary.json", {"plausibility": plausibility_audit})
"""
)

FILTERS = (
    HEADER
    + HELPER_HEAD
    + """        outside = (numeric < float(lower)) | (numeric > float(upper))
        kept = numeric[~outside]


write_json("step_summary.json", {"rows": 1})
"""
)


def _context(*, ranged: bool = True) -> ResearchContext:
    return ResearchContext(
        research_question="Assess a continuous ICU marker.",
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_stays=3, n_patients=3
        ),
        variables=[
            ConceptDescriptor(
                name="marker",
                dtype="float64",
                valid_range=[0.0, 10.0] if ranged else None,
            )
        ],
    )


def _step() -> AnalysisStep:
    return AnalysisStep(step_id="06_primary", intent="associate", method="logistic")


def _reasons(code: str, *, ranged: bool = True) -> set[str]:
    findings = flag_only_plausibility_obligation_findings(
        ast.parse(code),
        script_text=code,
        context=_context(ranged=ranged),
        step=_step(),
    )
    assert all(finding.severity == "error" for finding in findings)
    return {str((finding.detail or {}).get("reason")) for finding in findings}


def test_a_script_that_records_the_count_in_a_written_output_passes():
    """The one shape that satisfies both halves."""

    assert _reasons(DECLARED) == set()


def test_rejecting_an_out_of_range_value_is_blocked():
    assert "flag_only_plausibility_range_rejected" in _reasons(REJECTS)


def test_filtering_out_of_range_rows_is_blocked():
    assert "flag_only_plausibility_range_filtered" in _reasons(FILTERS)


def test_a_repaired_script_that_records_nothing_is_blocked():
    """The gap this gate exists to close.

    After the deterministic repair the script no longer requests exclusion, so
    the LLM auditor stops reporting and the finding-bound downgrade never runs.
    Every row is retained and nothing is recorded — and before this gate,
    nothing observed that.
    """

    assert _reasons(RECORDS_NOTHING) == {"out_of_range_record_absent"}


def test_a_count_that_never_leaves_the_process_is_not_a_record():
    assert _reasons(LOCAL_ONLY) == {"out_of_range_record_not_in_declared_output"}


def test_a_receipt_written_only_when_the_count_is_positive_is_blocked():
    """Zero is a result; its absence is indistinguishable from not looking."""

    assert _reasons(CONDITIONAL) == {"out_of_range_receipt_conditional_on_count"}


def test_a_script_that_names_no_bound_it_read_is_blocked_not_excused():
    """`not_attributable` costs a repair rather than buying a pass."""

    hard_coded = (
        HEADER
        + """
audit = {}


def validate(values, column, manifest):
    contract = manifest["raw_input_contracts"]["contracts"].get(column)
    if contract.get("analysis_plausibility_range") is not None:
        numeric = values.astype(float)
        audit[column] = int(((numeric < 0.0) | (numeric > 120.0)).sum())


write_json("step_summary.json", {"audit": audit})
"""
    )
    assert _reasons(hard_coded) == {"plausibility_check_not_attributable"}


@pytest.mark.parametrize(
    "sink",
    [
        # Reported against the shipped gate: each of these returned no finding
        # because the check asked only whether *some* serializer had been
        # called, never where the bytes went.
        pytest.param(
            'with open("/tmp/not_declared.json", "w") as scratch:\n'
            "    json.dump(plausibility_audit, scratch)",
            id="scratch_file",
        ),
        pytest.param("json.dump(plausibility_audit, sys.stdout)", id="stdout"),
        pytest.param(
            "pd.DataFrame([plausibility_audit]).to_json()", id="no_destination"
        ),
        # Same class, found while closing the three above.
        pytest.param(
            "sys.stdout.write(json.dumps(plausibility_audit))", id="stdout_write"
        ),
        pytest.param("print(plausibility_audit)", id="print"),
        pytest.param(
            'logging.info("wrote step_summary.json: %s", plausibility_audit)',
            id="log_naming_the_summary",
        ),
    ],
)
def test_a_write_with_no_declared_destination_is_not_a_record(sink):
    """Touching a serializer is not writing to a declared output.

    A scratch path under `/tmp`, a console stream, and a `to_json()` with
    nowhere to go all serialize; none of them leaves an artifact a reader can
    open. The last case is the mirror image: a log line that merely *names* the
    summary file must not count either, which is why the destination check is
    ANDed with the write and does not replace it.
    """

    printed = (
        HEADER
        + """
import logging
import sys

import pandas as pd

plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0


"""
        + sink
        + "\n"
    )
    assert _reasons(printed) == {"out_of_range_record_not_in_declared_output"}


@pytest.mark.parametrize(
    "sink",
    [
        pytest.param(
            '(OUT / "step_summary.json").write_text(json.dumps(summary))',
            id="write_text",
        ),
        pytest.param(
            'with (OUT / "step_summary.json").open("w") as handle:\n'
            "    json.dump(summary, handle)",
            id="open_handle",
        ),
        pytest.param(
            'summary_path = OUT / "step_summary.json"\n'
            'with summary_path.open("w", encoding="utf-8") as handle:\n'
            "    json.dump(summary, handle)",
            id="path_through_a_local",
        ),
        pytest.param(
            "def persist(path, payload):\n"
            '    with open(path, "w") as handle:\n'
            "        json.dump(payload, handle)\n"
            '\n\npersist(OUT / "step_summary.json", summary)',
            id="helper_the_gate_never_heard_of",
        ),
    ],
)
def test_every_way_the_corpus_reaches_the_canonical_summary_is_accepted(sink):
    """Each of these writes the summary in a real generated script.

    The last one matters most: `persist` is in no list anywhere. Helpers are
    recognised by what their body does, which is what lets the writer-name set
    stop growing an entry every time a script invents a name for one.
    """

    code = (
        HEADER
        + """
from pathlib import Path

OUT = Path("outputs")
plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = {
            "below_n": int((numeric < float(lower)).sum()) if lower is not None else 0,
            "above_n": int((numeric > float(upper)).sum()) if upper is not None else 0,
        }


summary = {"plausibility_audit": plausibility_audit}
"""
        + sink
        + "\n"
    )
    assert _reasons(code) == set()


def test_a_path_the_step_registers_as_its_own_output_is_declared():
    """`output_files` is the host's registration surface, so a path filed there
    is declared by the step itself -- unlike the sibling scratch file beside it,
    which is exactly the difference the gate has to see."""

    code = (
        HEADER
        + """
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0


pd.DataFrame([plausibility_audit]).to_csv(OUT / "range_audit.csv")
write_json("step_summary.json", {"output_files": {"table:range": "range_audit.csv"}})
"""
    )
    assert _reasons(code) == set()

    unregistered = code.replace(
        '"table:range": "range_audit.csv"', '"table:other": "other.csv"'
    )
    assert _reasons(unregistered) == {"out_of_range_record_not_in_declared_output"}


def test_no_declared_range_means_no_obligation():
    """A run whose context declares no range never emitted this policy."""

    assert _reasons(REJECTS, ranged=False) == set()


def test_a_step_that_never_touches_a_range_is_out_of_scope():
    """The trigger is the typed policy, not every step in a ranged study."""

    unrelated = (
        HEADER
        + """
rows = [1, 2, 3]
kept = [row for row in rows if row > 1]
write_json("step_summary.json", {"kept": len(kept)})
"""
    )
    assert _reasons(unrelated) == set()


def test_a_cohort_eligibility_threshold_is_not_a_plausibility_check():
    """Measured against the real corpus, this was the gate's worst failure.

    `age >= threshold` is an ordering comparison on a ranged column, so an
    earlier draft that anchored on the column read a cohort inclusion rule as
    a plausibility test — and then followed the retained/excluded masks through
    the whole cohort construction, reporting ordinary contract assertions as
    plausibility rejections.  The bound identifies the check; the column
    cannot.
    """

    eligibility = (
        HEADER
        + """
age_contract = manifest["raw_input_contracts"]["contracts"]["marker"]
plausibility_range = age_contract.get("analysis_plausibility_range")
minimum = float(plausibility_range["minimum"])
maximum = float(plausibility_range["maximum"])
coerced = frame["marker"].astype(float)
outside_n = int(((coerced < minimum) | (coerced > maximum)).sum())

threshold = float(receipt["value"])
retained_mask = coerced >= threshold
excluded_mask = ~retained_mask
if len(receipt_rows) != 1:
    raise ValueError("Expected exactly one host-resolved inclusion predicate")
if bool((retained_mask & excluded_mask).any()):
    raise AssertionError("Retained and excluded masks overlap")

write_json("step_summary.json", {"outside_range_n": outside_n})
"""
    )
    assert _reasons(eligibility) == set()


@pytest.mark.parametrize(
    "bound_source",
    [
        "minimum, maximum = plausibility_range",
        'minimum, maximum = (plausibility_range["minimum"], plausibility_range["maximum"])',
        'minimum, maximum = (float(plausibility_range[0]), float(plausibility_range[1])) if isinstance(plausibility_range, list) else (plausibility_range["minimum"], plausibility_range["maximum"])',
    ],
)
def test_every_spelling_of_the_bound_the_corpus_uses_is_recognized(bound_source):
    """Each of these appears in a real generated script.

    A bound the gate fails to read is not a missing block but a wrong one: the
    step gets told nothing could be attributed when it did the right thing.
    """

    code = (
        HEADER
        + f"""
plausibility_range = contract.get("analysis_plausibility_range")
{bound_source}
numeric = frame["marker"].astype(float)
audit = {{"outside_n": int(((numeric < minimum) | (numeric > maximum)).sum())}}
write_json("step_summary.json", audit)
"""
    )
    assert _reasons(code) == set()


def test_the_gate_runs_inside_the_shared_deterministic_code_gate():
    """It has to reach the pipeline, not only its own unit test."""

    findings = deterministic_code_gate_findings(
        context=_context(),
        step=_step(),
        script_text=RECORDS_NOTHING,
    )
    obligations = [
        finding
        for finding in findings
        if str((finding.detail or {}).get("issue_code") or "")
        == "flag_only_plausibility_obligation"
    ]
    assert obligations, "the obligation gate must run in the shared code gate"
    assert all(finding.severity == "error" for finding in obligations)
    # It is a deterministic validator, so a quarantine cannot be retired by an
    # LLM audit falling silent.
    from easyicu.research_agent.gates.concept import (
        DETERMINISTIC_CODE_GATE_VALIDATORS,
    )

    assert obligations[0].validator in DETERMINISTIC_CODE_GATE_VALIDATORS
