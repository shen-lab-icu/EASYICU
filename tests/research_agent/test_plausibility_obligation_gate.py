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

from easyicu.research_agent.authority.plausibility import (
    FlagOnlyPlausibilityScope,
)
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

HELPER_CALL = """


validate_allowed_numeric(frame["marker"], "marker", manifest)
"""

REJECTS = (
    HEADER
    + HELPER_HEAD
    + """        if lower is not None and (numeric < float(lower)).any():
            raise RuntimeError(f"{column} is below the plausibility minimum")
        if upper is not None and (numeric > float(upper)).any():
            raise RuntimeError(f"{column} is above the plausibility maximum")

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
write_json(SUMMARY_PATH, {"rows": 1})
"""
)

LOCAL_ONLY = (
    HEADER
    + HELPER_HEAD
    + """        below_n = int((numeric < float(lower)).sum()) if lower is not None else 0
        above_n = int((numeric > float(upper)).sum()) if upper is not None else 0

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
)

FILTERS = (
    HEADER
    + HELPER_HEAD
    + """        outside = (numeric < float(lower)) | (numeric > float(upper))
        kept = numeric[~outside]

validate_allowed_numeric(frame["marker"], "marker", manifest)
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


def _scope(*, ranged: bool = True) -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id=_step().step_id,
        expected_columns=("marker",) if ranged else (),
        source_contracts_sha256="0" * 64,
        authority_kind="test_resolved_raw_input_contracts",
    )


def _reasons(code: str, *, ranged: bool = True) -> set[str]:
    findings = flag_only_plausibility_obligation_findings(
        ast.parse(code),
        script_text=code,
        step=_step(),
        scope=_scope(ranged=ranged),
    )
    assert all(finding.severity == "error" for finding in findings)
    return {str((finding.detail or {}).get("reason")) for finding in findings}


def _reasons_for_columns(code: str, *columns: str) -> set[str]:
    scope = FlagOnlyPlausibilityScope(
        step_id=_step().step_id,
        expected_columns=tuple(sorted(columns)),
        source_contracts_sha256="0" * 64,
        authority_kind="test_resolved_raw_input_contracts",
    )
    findings = flag_only_plausibility_obligation_findings(
        ast.parse(code),
        script_text=code,
        step=_step(),
        scope=scope,
    )
    return {str((finding.detail or {}).get("reason")) for finding in findings}


def test_a_script_that_records_the_count_in_a_written_output_passes():
    """The one shape that satisfies both halves."""

    assert _reasons(DECLARED) == set()


def test_empty_mapping_fallback_keeps_the_sealed_range_attributable() -> None:
    """``range = contract.get(...) or {}`` is the real E1 repair shape."""

    with_empty_fallback = DECLARED.replace(
        'bounds = contract.get("analysis_plausibility_range")',
        'bounds = contract.get("analysis_plausibility_range") or {}',
    )

    assert _reasons(with_empty_fallback) == set()


def test_populated_range_fallback_cannot_replace_the_sealed_bounds() -> None:
    with_source_literal_fallback = DECLARED.replace(
        'bounds = contract.get("analysis_plausibility_range")',
        (
            'bounds = contract.get("analysis_plausibility_range") '
            'or {"minimum": 0, "maximum": 10}'
        ),
    )

    assert _reasons(with_source_literal_fallback) == {
        "plausibility_check_not_attributable"
    }


def test_an_uncalled_helper_cannot_certify_literal_receipts() -> None:
    """A perfect-looking dead function is not evidence that a check ran."""

    dead_helper = DECLARED.replace(
        'validate_allowed_numeric(frame["marker"], "marker", manifest)\n',
        "",
    )

    assert _reasons(dead_helper) == {"plausibility_check_not_attributable"}


def test_one_called_column_cannot_certify_another_literal_receipt() -> None:
    forged_second = DECLARED.replace(
        'write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})',
        """
plausibility_audit["second_marker"] = {
    "policy": "retain_and_flag",
    "below_minimum_n": 0,
    "above_maximum_n": 0,
    "out_of_range_n": 0,
}
write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
""",
    )

    assert _reasons_for_columns(forged_second, "marker", "second_marker") == {
        "plausibility_scope_column_not_attributable"
    }


def test_each_called_scope_column_can_share_one_generic_validator() -> None:
    both = DECLARED.replace(
        'validate_allowed_numeric(frame["marker"], "marker", manifest)',
        'validate_allowed_numeric(frame["marker"], "marker", manifest)\n'
        'validate_allowed_numeric(frame["second_marker"], "second_marker", manifest)',
    )

    assert _reasons_for_columns(both, "marker", "second_marker") == set()


def test_named_explicit_column_loop_can_share_one_generic_validator() -> None:
    """The real E1 shape binds the closed host-expected list before looping."""

    looped = DECLARED.replace(
        'validate_allowed_numeric(frame["marker"], "marker", manifest)',
        'PLAUSIBILITY_COLUMNS = ["marker", "second_marker"]\n'
        "for column in PLAUSIBILITY_COLUMNS:\n"
        "    validate_allowed_numeric(frame[column], column, manifest)",
    )

    assert _reasons_for_columns(looped, "marker", "second_marker") == set()


def test_named_column_loop_cannot_certify_an_omitted_scope_column() -> None:
    looped = DECLARED.replace(
        'validate_allowed_numeric(frame["marker"], "marker", manifest)',
        'PLAUSIBILITY_COLUMNS = ["marker"]\n'
        "for column in PLAUSIBILITY_COLUMNS:\n"
        "    validate_allowed_numeric(frame[column], column, manifest)",
    )

    assert _reasons_for_columns(looped, "marker", "second_marker") == {
        "plausibility_scope_column_not_attributable"
    }


def test_module_loop_over_sealed_contracts_covers_each_runtime_column() -> None:
    direct_loop = (
        HEADER
        + """
plausibility_audit = {}
contracts = manifest["raw_input_contracts"]["contracts"]
for column, contract in contracts.items():
    bounds = contract.get("analysis_plausibility_range")
    if bounds is None:
        continue
    lower = bounds.get("minimum")
    upper = bounds.get("maximum")
    numeric = frame[column].dropna().astype(float)
    below_n = int((numeric < lower).sum()) if lower is not None else 0
    above_n = int((numeric > upper).sum()) if upper is not None else 0
    plausibility_audit[column] = {
        "policy": "retain_and_flag",
        "below_minimum_n": below_n,
        "above_maximum_n": above_n,
        "out_of_range_n": below_n + above_n,
    }

write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )

    assert _reasons_for_columns(direct_loop, "marker", "second_marker") == set()


def test_contract_loop_cannot_certify_comparisons_on_unrelated_data() -> None:
    unrelated = (
        HEADER
        + """
plausibility_audit = {}
contracts = manifest["raw_input_contracts"]["contracts"]
for column, contract in contracts.items():
    bounds = contract.get("analysis_plausibility_range")
    lower = bounds.get("minimum")
    numeric = frame["unrelated"].dropna().astype(float)
    below_n = int((numeric < lower).sum()) if lower is not None else 0
    plausibility_audit[column] = {
        "policy": "retain_and_flag",
        "below_minimum_n": below_n,
        "above_maximum_n": 0,
        "out_of_range_n": below_n,
    }

write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )

    assert _reasons_for_columns(unrelated, "marker", "second_marker") == {
        "plausibility_scope_column_not_attributable"
    }


def test_rejecting_an_out_of_range_value_is_blocked():
    assert "flag_only_plausibility_range_rejected" in _reasons(REJECTS)


def test_filtering_out_of_range_rows_is_blocked():
    assert "flag_only_plausibility_range_filtered" in _reasons(FILTERS)


@pytest.mark.parametrize(
    "operation, transform",
    [
        (
            "drop",
            """
        outside = (numeric < float(lower)) | (numeric > float(upper))
        transformed = numeric.drop(index=numeric.index[outside])
""",
        ),
        (
            "query",
            """
        table = numeric.to_frame(name=column)
        expression = f"`{column}` >= {float(lower)} and `{column}` <= {float(upper)}"
        transformed = table.query(expression)
""",
        ),
        (
            "clip",
            """
        transformed = numeric.clip(
            lower=float(lower),
            upper=float(upper),
        )
""",
        ),
        (
            "where",
            """
        outside = (numeric < float(lower)) | (numeric > float(upper))
        transformed = numeric.where(~outside)
""",
        ),
    ],
)
def test_pandas_range_transform_spellings_are_blocked(
    operation: str,
    transform: str,
) -> None:
    code = DECLARED.replace(
        "        plausibility_audit[column] = {",
        transform + "        plausibility_audit[column] = {",
    )

    findings = flag_only_plausibility_obligation_findings(
        ast.parse(code),
        script_text=code,
        step=_step(),
        scope=_scope(),
    )
    transformed = [
        finding
        for finding in findings
        if (finding.detail or {}).get("reason")
        == "flag_only_plausibility_range_transformed"
    ]

    assert len(transformed) == 1
    assert transformed[0].detail["operation"] == operation


@pytest.mark.parametrize(
    "unrelated",
    [
        '        projection = frame.drop(columns=["unused"], errors="ignore")\n',
        '        eligible = frame.query("eligible == 1")\n',
        "        display = frame[\"display\"].clip(lower=0, upper=1)\n",
        "        labelled = frame[\"marker\"].where(frame[\"eligible\"] == 1)\n",
    ],
)
def test_unrelated_pandas_transforms_are_not_plausibility_rejections(
    unrelated: str,
) -> None:
    code = DECLARED.replace(
        "        plausibility_audit[column] = {",
        unrelated + "        plausibility_audit[column] = {",
    )

    assert _reasons(code) == set()


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
        + HELPER_CALL
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
        + HELPER_CALL
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
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

validate_allowed_numeric(frame["marker"], "marker", manifest)
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


def test_a_helper_handed_the_directory_and_the_summary_is_followed_into():
    """Copied from a real generated script, which writes exactly this way.

    `out_dir` and `summary` are *parameters*, so a recogniser that only follows
    assignments sees a write to an unknown directory of an unknown payload and
    blocks a compliant script. Every call site binds both to the right thing,
    which is what makes the parameter readable.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}


def write_summary(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
"""
        + HELPER_HEAD
        + """        plausibility_audit[column] = int(
            (numeric < float(lower)).sum()
        ) if lower is not None else 0

validate_allowed_numeric(frame["marker"], "marker", manifest)
write_summary({"plausibility_audit": plausibility_audit}, STEP_OUT_DIR)
"""
    )
    assert _reasons(code) == set()

    # One call site that hands it somewhere else leaves the parameter unproven,
    # so the write is no longer attributable to the host's own directory.
    ambiguous = code.replace(
        'write_summary({"plausibility_audit": plausibility_audit}, STEP_OUT_DIR)',
        'write_summary({"plausibility_audit": plausibility_audit}, STEP_OUT_DIR)\n'
        'write_summary({"rows": 1}, Path("/tmp"))',
    )
    assert _reasons(ambiguous) == {"out_of_range_record_not_in_declared_output"}


RECORD_THE_COUNTS = """        plausibility_audit[column] = {
            "policy": "retain_and_flag",
            "below_minimum_n": int((numeric < float(lower)).sum()),
            "above_maximum_n": int((numeric > float(upper)).sum()),
            "out_of_range_n": int(
                ((numeric < float(lower)) | (numeric > float(upper))).sum()
            ),
        }
"""

# A receipt the artifact can carry that no post-hoc reader can falsify: it is
# internally consistent, and every count in it is a typed zero.
ZEROS = """
ZEROS = {
    "marker": {
        "policy": "retain_and_flag",
        "below_minimum_n": 0,
        "above_maximum_n": 0,
        "out_of_range_n": 0,
    }
}
"""


def test_rebinding_the_output_directory_after_reading_it_is_blocked():
    """A name is trusted where it is written, not once and for all.

    The script reads the host's directory into `out_dir`, then rebinds it to a
    scratch path and sends the real counts there, leaving the artifact the host
    opens with hard-coded zeros. Every name matches the compliant spelling; the
    first binding is the only honest thing about it. A flat name set had no way
    to notice the second one.
    """

    code = (
        HEADER
        + ZEROS
        + """
plausibility_audit = {}
out_dir = STEP_OUT_DIR
out_dir = Path("/tmp")
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + """

write_json(out_dir / "step_summary.json", {"plausibility_audit": plausibility_audit})
write_json(SUMMARY_PATH, {"plausibility_audit": ZEROS})
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}


def test_a_name_trusted_in_one_function_does_not_lend_itself_to_another():
    """Two functions, one name, two different files.

    `out_dir` is a parameter of the official writer -- every call site hands it
    the host's directory, so it is genuinely trusted there -- and an ordinary
    local in the scratch writer. `payload` is likewise a real receipt in one
    and a literal in the other. Neither name means the same thing in both
    places, and one flat set per script cannot say so.
    """

    code = (
        HEADER
        + ZEROS
        + """
plausibility_audit = {}


def stage_the_official(out_dir, payload):
    write_json(out_dir / "step_summary.json", payload)


def stage_to_scratch(payload):
    out_dir = Path("/tmp")
    write_json(out_dir / "step_summary.json", payload)
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + """

stage_to_scratch({"plausibility_audit": plausibility_audit})
stage_the_official(STEP_OUT_DIR, {"plausibility_audit": ZEROS})
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}


def test_replacing_the_record_with_a_literal_before_writing_it_is_blocked():
    """The same defect one step over, on the payload instead of the path.

    The counts are computed into `plausibility_audit`, then the name is rebound
    to a hard-coded mapping and *that* is what reaches the artifact. Following
    where a value *can* come from is the right question for finding the
    computation and the wrong one for certifying a delivery.
    """

    code = (
        HEADER
        + ZEROS
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + """

plausibility_audit = ZEROS
write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )
    assert _reasons(code) == {"out_of_range_record_not_in_declared_output"}


def test_seeding_an_empty_accumulator_is_not_replacing_it():
    """The shape almost every compliant script writes must survive the rule.

    `plausibility_audit = {}` before `plausibility_audit[column] = ...` is a
    second whole-name binding that carries nothing, and a rule that only asked
    "does every binding carry" would refuse the one spelling the corpus
    actually uses.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + """

write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )
    assert _reasons(code) == set()


# The count the fatal fallback guards on has to be a real one, read out of the
# declared range -- a bare `out_of_range_n = 0` is not a plausibility test and
# the gate rightly ignores it, which would make every case below vacuous.
COUNT_AT_MODULE_LEVEL = """

marker_contract = manifest["raw_input_contracts"]["contracts"]["marker"]
marker_range = marker_contract["analysis_plausibility_range"]
marker_minimum = marker_range["minimum"]
policy = marker_contract["plausibility_policy"]["out_of_range_action"]
out_of_range_n = int((frame["marker"] < float(marker_minimum)).sum())
"""


@pytest.mark.parametrize(
    "guard",
    [
        pytest.param(
            """
if policy == "retain_and_flag":
    pass
elif out_of_range_n != 0:
    raise RuntimeError(f"Unsupported or fatal plausibility policy: {policy}")
""",
            id="elif_after_the_declared_action",
        ),
        pytest.param(
            """
if out_of_range_n > 0 and policy != "retain_and_flag":
    raise RuntimeError(f"Unsupported or fatal plausibility policy: {policy}")
""",
            id="inline_and_not_the_declared_action",
        ),
        pytest.param(
            """
if policy != "retain_and_flag":
    if out_of_range_n:
        raise RuntimeError("Unsupported fatal plausibility policy")
""",
            id="nested_under_a_different_policy",
        ),
    ],
)
def test_a_fatal_fallback_for_some_other_policy_is_not_a_rejection(guard):
    """Both real canary drafts wrote one of these, and both were blocked.

    A script handed `retain_and_flag` still guards its own fatal fallback in
    case it is ever handed something else. Under the declared policy that
    `raise` cannot run, so reading it as a rejection is a wrong block -- and it
    cost a real canary its entire run at step 01. Being defensive about a
    policy the host did not declare is not a violation of the one it did.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + COUNT_AT_MODULE_LEVEL
        + guard
        + """

write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )
    assert _reasons(code) == set()


def test_an_unguarded_fatal_stop_is_still_a_rejection():
    """The control for the exemption: no policy guard, still blocked.

    Without it the exemption could be doing nothing and the tests above would
    pass vacuously -- which is exactly what a first draft of them did, because
    the count they guarded on was a bare literal the gate rightly ignored.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + COUNT_AT_MODULE_LEVEL
        + """
if out_of_range_n != 0:
    raise RuntimeError("out-of-range values present")


write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )
    assert "flag_only_plausibility_range_rejected" in _reasons(code)


@pytest.mark.parametrize(
    "fake_policy",
    [
        'fake_policy = "fail_closed"\n',
        'policy = "fail_closed"\nfake_policy = policy\n',
    ],
)
def test_a_lookalike_or_rebound_policy_cannot_hide_a_fatal_stop(fake_policy):
    """Only the action read from the sealed contract can exempt a fallback."""

    code = (
        HEADER
        + """
plausibility_audit = {}
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + COUNT_AT_MODULE_LEVEL
        + fake_policy
        + """
if out_of_range_n > 0 and fake_policy != "retain_and_flag":
    raise RuntimeError("out-of-range values present")

write_json(SUMMARY_PATH, {"plausibility_audit": plausibility_audit})
"""
    )
    assert "flag_only_plausibility_range_rejected" in _reasons(code)


def test_one_handle_name_reused_across_two_with_blocks_is_read_per_block():
    """Copied from the script the first real canary blocked.

    The step opens a host input with `with open(...) as handle` near the top
    and the summary with `with open(summary_path, "w") as handle` at the
    bottom. Both bind `handle` in module scope, so a rule that demands every
    binding of a name be the summary makes a compliant write invisible -- a
    `with ... as` name means whatever its own block opened, and the two never
    coexist. This shape is why the canary was worth running: the unit fixtures
    were green while the only real script the gate had ever seen was being
    wrongly blocked.
    """

    code = (
        HEADER
        + """
plausibility_audit = {}

with open("resolved.json", "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
"""
        + HELPER_HEAD
        + RECORD_THE_COUNTS
        + HELPER_CALL
        + """

summary_path = STEP_OUT_DIR / "step_summary.json"
step_summary = {"plausibility_audit": plausibility_audit}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle, indent=2)
"""
    )
    assert _reasons(code) == set()

    # The exemption is per block, not per name: a handle on a scratch file
    # still cannot stand in for the artifact the host opens.
    scratch = code.replace(
        'summary_path = STEP_OUT_DIR / "step_summary.json"',
        'summary_path = Path("/tmp") / "step_summary.json"',
    )
    assert _reasons(scratch) == {"out_of_range_record_not_in_declared_output"}


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
    """An empty exact step scope never emitted this policy."""

    assert _reasons(REJECTS, ranged=False) == set()


def test_a_scoped_step_cannot_opt_out_by_omitting_the_range_read():
    """Generated source cannot erase a host-owned non-empty obligation."""

    unrelated = (
        HEADER
        + """
rows = [1, 2, 3]
kept = [row for row in rows if row > 1]
write_json(SUMMARY_PATH, {"kept": len(kept)})
"""
    )
    assert _reasons(unrelated) == {"plausibility_check_not_attributable"}


def test_a_global_ranged_variable_does_not_widen_an_empty_step_scope():
    """The study-wide ResearchContext is descriptive, not step authority."""

    unrelated = (
        HEADER
        + """
rows = [1, 2, 3]
write_json(SUMMARY_PATH, {"rows": len(rows)})
"""
    )
    assert _context(ranged=True).variables[0].valid_range is not None
    assert _reasons(unrelated, ranged=False) == set()


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
        plausibility_scope=_scope(),
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


# ---------------------------------------------------------------------------
# The comparison written straight into the loop body, inside a function.
#
# canary6 (2026-07-30) lost M3 step 04 here.  The step wrote the audit itself
# rather than calling a per-column validator, and every column was reported
# `covered_columns=[]`.  Sweeping the 473 recorded generated scripts before and
# after the fix: 16 of the 133 that carry a plausibility comparison gain
# coverage, and none loses any -- the crediting can only widen.


_INLINE_LOOP_IN_MAIN = HEADER + """

PLAUSIBILITY_SCOPE = ["marker", "second_marker"]


def main():
    contracts = manifest["raw_input_contracts"]["contracts"]
    audit = {}
    for column in PLAUSIBILITY_SCOPE:
        bounds = contracts[column]["analysis_plausibility_range"]
        lower = bounds.get("minimum")
        upper = bounds.get("maximum")
        numeric = frame[column].dropna().astype(float)
        below_n = int((numeric < float(lower)).sum()) if lower is not None else 0
        above_n = int((numeric > float(upper)).sum()) if upper is not None else 0
        audit[column] = {
            "policy": "retain_and_flag",
            "below_minimum_n": below_n,
            "above_maximum_n": above_n,
            "out_of_range_n": below_n + above_n,
        }
    write_json(SUMMARY_PATH, {"plausibility_audit": audit})


main()
"""


def test_an_inline_loop_inside_a_function_covers_the_columns_it_names() -> None:
    """The real M3 shape: no per-column helper, the loop is inside ``main``.

    Being inside a function used to send this down the helper-call branch,
    which looks for the loop around the *call* -- and ``main()`` is called once,
    at module level, not per column. The loop that does the work is inside the
    owner, and nothing looked there.
    """

    assert _reasons_for_columns(
        _INLINE_LOOP_IN_MAIN, "marker", "second_marker"
    ) == set()


def test_the_inline_loop_cannot_certify_a_column_its_list_omits() -> None:
    """Crediting is per literal, so an omitted column is still reported."""

    assert _reasons_for_columns(
        _INLINE_LOOP_IN_MAIN, "marker", "second_marker", "third_marker"
    ) == {"plausibility_scope_column_not_attributable"}


def test_an_inline_loop_at_module_level_covers_its_columns_too() -> None:
    """Same crediting whether or not the work sits in a function."""

    module_level = _INLINE_LOOP_IN_MAIN.replace(
        "def main():\n", "if True:\n"
    ).replace("\n\nmain()\n", "\n")

    assert _reasons_for_columns(
        module_level, "marker", "second_marker"
    ) == set()


def test_a_comparison_that_names_no_column_is_still_not_attributable() -> None:
    """The fix credits named columns; it does not credit an unnamed check."""

    unnamed = HEADER + """


def main():
    bounds = manifest["raw_input_contracts"]["contracts"]["marker"][
        "analysis_plausibility_range"
    ]
    lower = bounds.get("minimum")
    numeric = frame[some_column].dropna().astype(float)
    below_n = int((numeric < float(lower)).sum())
    write_json(SUMMARY_PATH, {"below": below_n})


main()
"""

    assert _reasons_for_columns(unnamed, "second_marker") == {
        "plausibility_scope_column_not_attributable"
    }


def test_a_loop_whose_key_is_unrelated_to_the_compared_data_credits_nothing() -> None:
    """The guard that keeps the loop crediting honest.

    Iterating a list of column names near a comparison proves nothing unless
    the loop variable is what selects the data being compared. Without this,
    any script that happens to loop over its column names anywhere would
    certify every one of them.
    """

    unrelated = HEADER + """

OTHER_COLUMNS = ["marker", "second_marker"]


def main():
    bounds = manifest["raw_input_contracts"]["contracts"]["marker"][
        "analysis_plausibility_range"
    ]
    lower = bounds.get("minimum")
    numeric = frame[fixed_column].dropna().astype(float)
    for label in OTHER_COLUMNS:
        below_n = int((numeric < float(lower)).sum())
        write_json(SUMMARY_PATH, {label: below_n})


main()
"""

    assert _reasons_for_columns(unrelated, "second_marker") == {
        "plausibility_scope_column_not_attributable"
    }
