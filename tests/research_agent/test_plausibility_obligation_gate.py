"""The host-owned `retain_and_flag` obligation gate.

The shapes below are the ones real generated scripts write, kept in the same
per-column-helper form the r25 Step 06 script uses: the series arrives as a
parameter, so the variable's name never appears near the comparison and only
the bound identifies the check. A gate written against a tidier invented shape
passes its own fixtures and abstains on the only script that matters.
"""

from __future__ import annotations

import ast
import pathlib

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

# The host hands the step its output directory in the environment, and the
# corpus reads it exactly this way. The directory is half the destination: a
# check that compared only the filename accepted `/tmp/step_summary.json` for
# the artifact the host opens.
HEADER = """
import json
import os
from pathlib import Path

manifest = json.loads(open("resolved.json").read())
STEP_OUT_DIR = Path(os.environ["STEP_OUT_DIR"])
SUMMARY_PATH = STEP_OUT_DIR / "step_summary.json"


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


write_json(SUMMARY_PATH, {"rows": 1})
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


write_json(SUMMARY_PATH, {"rows": 1})
"""
)

LOCAL_ONLY = (
    HEADER
    + HELPER_HEAD
    + """        below_n = int((numeric < float(lower)).sum()) if lower is not None else 0
        above_n = int((numeric > float(upper)).sum()) if upper is not None else 0


write_json(SUMMARY_PATH, {"rows": 1})
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


write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
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


write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
)

FILTERS = (
    HEADER
    + HELPER_HEAD
    + """        outside = (numeric < float(lower)) | (numeric > float(upper))
        kept = numeric[~outside]


write_json(SUMMARY_PATH, {"rows": 1})
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


write_json(SUMMARY_PATH, {"audit": audit})
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

OUT = Path(os.environ["STEP_OUT_DIR"])
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


def test_a_registered_companion_file_does_not_substitute_for_the_receipt():
    """Registering a companion declares an output; it does not deliver this one.

    The post-execution half reads exactly one place, so a count that lands
    anywhere else is a count the host never sees -- however properly the file
    holding it was declared. An earlier draft accepted any registered path, and
    the two halves could then disagree about the same step.
    """

    code = (
        HEADER
        + """
import pandas as pd

plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0


pd.DataFrame([plausibility_audit]).to_csv(STEP_OUT_DIR / "range_audit.csv")
write_json(SUMMARY_PATH, {"output_files": {"table:range": "range_audit.csv"}})
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}

    # The same script, with the receipt also filed where the host reads it.
    delivered = code.replace(
        '{"output_files": {"table:range": "range_audit.csv"}}',
        '{"output_files": {"table:range": "range_audit.csv"},'
        ' "plausibility_audit": plausibility_audit}',
    )
    assert _reasons(delivered) == set()


@pytest.mark.parametrize(
    "summary_key",
    [
        pytest.param("plausibility_audit", id="the_key_the_host_reads"),
        pytest.param("range_audit", id="a_key_of_its_own_choosing"),
    ],
)
def test_the_receipt_must_be_filed_under_the_key_the_host_reads(summary_key):
    """Reaching the file is not the same as reaching the receipt.

    The host opens one key. A summary that carries the counts under a name of
    the script's own choosing has written them somewhere nothing looks, and the
    post-execution half would then block a step the static one had passed.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0


write_json(SUMMARY_PATH, {"%s": plausibility_audit})
"""
        % summary_key
    )
    expected = (
        set()
        if summary_key == "plausibility_audit"
        else {"out_of_range_record_not_in_declared_output"}
    )
    assert _reasons(code) == expected


def test_nesting_the_receipt_below_the_top_level_is_not_delivery():
    """`{"quality": receipt}` moves it one level down, out of the host's sight."""

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0


receipt = {"plausibility_audit": plausibility_audit}
write_json(SUMMARY_PATH, {"quality": receipt})
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}


def test_the_real_counts_in_a_scratch_file_with_the_canonical_name_is_blocked():
    """The reported false green, reproduced whole.

    The script computes the counts honestly and writes them to
    `/tmp/step_summary.json`, then writes the summary the host actually opens
    with hard-coded zeros. Every serializer name matches, and so does the last
    path component -- which is exactly why a destination has to be a directory
    as well as a filename.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = {
            "policy": "retain_and_flag",
            "below_minimum_n": int((numeric < float(lower)).sum()),
            "above_maximum_n": int((numeric > float(upper)).sum()),
            "out_of_range_n": int(
                ((numeric < float(lower)) | (numeric > float(upper))).sum()
            ),
        }


write_json("/tmp/step_summary.json", {"plausibility_audit": plausibility_audit})
write_json(
    SUMMARY_PATH,
    {
        "plausibility_audit": {
            "marker": {
                "policy": "retain_and_flag",
                "below_minimum_n": 0,
                "above_maximum_n": 0,
                "out_of_range_n": 0,
            }
        }
    },
)
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}


def test_the_gate_reads_the_output_directory_the_host_actually_sets():
    """The env aliases are the host's own, not a guess, so they must not drift.

    The set is duplicated rather than imported so a read-only gate does not
    depend on the execution layer; this is what keeps the copy honest.
    """

    from easyicu.research_agent.execution import runner
    from easyicu.research_agent.gates.plausibility_receipt import (
        HOST_OUTPUT_DIR_ENV_KEYS,
    )

    assert HOST_OUTPUT_DIR_ENV_KEYS <= runner.HOST_OWNED_RUNNER_ENV_KEYS
    source = pathlib.Path(runner.__file__).read_text()
    assigned = {
        key
        for key in runner.HOST_OWNED_RUNNER_ENV_KEYS
        if f'"{key}": container_output_dir' in source
    }
    assert assigned, "the runner no longer assigns the output directory by name"
    assert assigned <= HOST_OUTPUT_DIR_ENV_KEYS


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
write_json(SUMMARY_PATH, {"kept": len(kept)})
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

write_json(SUMMARY_PATH, {"plausibility_audit": {"marker": outside_n}})
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
write_json(SUMMARY_PATH, {{"plausibility_audit": {{"marker": audit}}}})
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
