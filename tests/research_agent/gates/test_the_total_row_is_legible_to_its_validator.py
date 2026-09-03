"""The host's own audit table said "total" in a spelling its own validator cannot read.

``deterministic_missingness`` builds the exposure-completeness table by putting
the whole cohort first and the exposure levels after it, and labels that first
row ``exposure_category='__all__'``.  ``audits/aggregate_row`` refuses any
emitted table that contains a row equal to the sum of the others unless the
table declares a ROLE COLUMN -- and ``exposure_category`` is not one.

So a step whose script exited 0 and wrote exactly the product its plan promised
was failed on a contract violation, and the host then handed its OWN
deterministic script to the Coder to repair.

MEASURED on h1_ventilation_survival, 2026-08-03 (``..._7c6bac6_verify07``),
step ``03_measurement_process_audit``::

    returncode: 0
    stdout: {"n_total": 92398, "n_concepts_audited": 1, "n_structural_no_source": 0}
    step_summary: status=ok, output_files={"table:measurement_process_audit": ...}
    status: repair_failed
    contract_findings: emitted_table_aggregate_row -- "contains a row (row 0)
      whose value equals the sum of every other row in 4 independent count
      columns ... That is a total row, and nothing in the table says so"

    executed_code_sha256 == concept_approved_code_sha256   (host-written)
    deterministic_standard_analysis = declared_missingness_audit_products
    step_provider_call_categories = 3x contract_repair_patch

MEASURED over the recorded corpus (1,374 emitted tables of 3-60 rows): 143
contain a total row.  111 already declare a role column and pass.  Of the 32
that do not, exactly TWO carry a reserved sentinel on the flagged row -- both
``exposure_component_completeness_audit.csv``, both from this producer.  The
other 30 carry no marker of any kind and the validator is right about them.

That split is why the fix is here and not in the validator: widening
``AGGREGATE_ROW_ROLE_COLUMNS`` to accept a sentinel VALUE would weaken the
check for the 30 tables it correctly catches, to serve 2 the producer can
simply label properly.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from easyicu.research_agent.audits.aggregate_row import (  # noqa: E402
    AGGREGATE_ROW_ROLE_COLUMNS,
    LEVEL_ROW_ROLE,
    OVERALL_ROW_ROLE,
)
from easyicu.research_agent.execution.runners.deterministic_missingness import (  # noqa: E402
    _ALL_STRATA_LABEL,
    _LEVEL_ROW_ROLE,
    _OVERALL_ROW_ROLE,
)


def test_the_total_row_is_legible_to_its_validator() -> None:
    """The producer's spelling and the validator's vocabulary must be one word.

    These constants are declared in three places -- here in the missingness
    executor, in ``audits/aggregate_row``, and in
    ``exposure_outcome_distribution_executor`` -- because this module's body is
    rendered inline into the container script and importing the validator
    package there would drag the schema layer across that boundary.  The copy
    is deliberate; drifting is not, and this is what stops it.
    """

    assert _OVERALL_ROW_ROLE == OVERALL_ROW_ROLE
    assert _LEVEL_ROW_ROLE == LEVEL_ROW_ROLE

    from easyicu.research_agent.execution.runners import (
        exposure_outcome_distribution_executor as sibling,
    )

    assert sibling._OVERALL_ROLE == OVERALL_ROW_ROLE
    assert sibling._LEVEL_ROLE == LEVEL_ROW_ROLE


def test_row_role_is_a_column_the_validator_accepts() -> None:
    """The whole point: the emitted column name must be in the accepted set."""

    assert "row_role" in AGGREGATE_ROW_ROLE_COLUMNS


def test_the_producer_still_writes_its_own_stratum_label() -> None:
    """``__all__`` is kept, not replaced.

    Consumers that already read ``exposure_category`` keep working; the role
    column is an ADDITION that makes the same fact legible to a reader who does
    not know this producer.
    """

    assert _ALL_STRATA_LABEL == "__all__"


def _flagged_total_rows(frame: pd.DataFrame) -> list[int]:
    """Re-derive the validator's own finding, so this is not a restatement."""

    from easyicu.research_agent.audits.aggregate_row import (
        MIN_AGREEING_COUNT_COLUMNS,
        _count_columns,
    )

    agreeing: dict[int, int] = {}
    for values in _count_columns(frame).values():
        total = values.sum()
        for index in range(len(frame)):
            value = values.iloc[index]
            if value > 0 and abs((total - value) - value) < 1e-9:
                agreeing[index] = agreeing.get(index, 0) + 1
    return [i for i, n in agreeing.items() if n >= MIN_AGREEING_COUNT_COLUMNS]


def test_the_real_table_that_failed_would_now_declare_its_total_row() -> None:
    """The exact recorded table, re-labelled the way the producer now labels it.

    Rebuilds from the sealed artifact rather than a fixture, so this stops being
    meaningful only if the recorded failure stops existing.
    """

    # Not every recorded copy reproduces it: the table only grows a total row
    # when the plan declared an exposure to stratify by, so pick the artifact
    # that actually carries the defect rather than the newest one.
    frame = None
    for path in sorted(
        pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs").rglob(
            "steps/*/outputs/exposure_component_completeness_audit.csv"
        )
    ):
        candidate = pd.read_csv(path)
        if _flagged_total_rows(candidate) == [0]:
            frame = candidate
            break
    if frame is None:
        pytest.skip("no recorded exposure-completeness table carries a total row")

    flagged = _flagged_total_rows(frame)
    assert flagged == [0], flagged
    assert not [
        column
        for column in frame.columns
        if str(column).strip().lower() in AGGREGATE_ROW_ROLE_COLUMNS
    ], "the recorded table already declares a role -- the defect is gone"

    # Apply exactly what the producer now writes.
    frame["row_role"] = [
        _OVERALL_ROW_ROLE if str(value) == _ALL_STRATA_LABEL else _LEVEL_ROW_ROLE
        for value in frame["exposure_category"]
    ]

    role_columns = [
        column
        for column in frame.columns
        if str(column).strip().lower() in AGGREGATE_ROW_ROLE_COLUMNS
    ]
    assert role_columns == ["row_role"]
    # The flagged row is the one now declared as the total, not a level.
    assert frame["row_role"].iloc[flagged[0]] == _OVERALL_ROW_ROLE
    assert set(frame["row_role"]) == {_OVERALL_ROW_ROLE, _LEVEL_ROW_ROLE}


def test_the_generated_script_labels_the_row_it_writes() -> None:
    """Drives ``missingness_measurement_audit_code``, the function production calls.

    Every other test here checks the rule or the artifact. This one asks the
    generator that actually emits the script, because deleting the label at the
    WRITE SITE leaves all of them green -- the same way a fix earlier in this
    series first passed its own mutation check while driving only a helper.
    """

    from easyicu.research_agent.execution.runners.deterministic_missingness import (
        missingness_measurement_audit_code,
    )

    source = missingness_measurement_audit_code()

    # The completeness row must carry a role key, keyed off the stratum label,
    # and every one of the three strings must be a LITERAL: this block is a
    # template rendered into a container script that defines no host names.
    # The first draft of this fix wrote the module constants into the template
    # instead, which would have raised NameError inside the runner -- caught
    # here, before it reached a run.
    assert (
        f'"row_role": (\n'
        f'                    "{OVERALL_ROW_ROLE}"'
        f' if label == "{_ALL_STRATA_LABEL}"'
        f' else "{LEVEL_ROW_ROLE}"\n'
        f"                ),"
    ) in source, "the generated row must label itself with literal role values"

    # Nothing host-side may leak into the container script.
    for host_name in ("_OVERALL_ROW_ROLE", "_LEVEL_ROW_ROLE", "_ALL_STRATA_LABEL"):
        assert host_name not in source, host_name

    # The generated script must remain syntactically valid Python.
    compile(source, "<missingness_audit>", "exec")


def test_the_validator_accepts_the_relabelled_table(tmp_path: pathlib.Path) -> None:
    """Drives the validator itself, not a re-implementation of its rule."""

    from easyicu.research_agent.audits.aggregate_row import (
        unlabelled_aggregate_row_findings,
    )

    frame = pd.DataFrame(
        {
            "exposure_category": [_ALL_STRATA_LABEL, "0.0", "1.0"],
            "n_stratum": [92398, 62491, 29907],
            "measured_n": [92347, 62440, 29907],
            "value_missing_n": [51, 51, 0],
        }
    )
    unlabelled = tmp_path / "exposure_component_completeness_audit.csv"
    frame.to_csv(unlabelled, index=False)
    before = unlabelled_aggregate_row_findings(
        step_id="03_measurement_process_audit", out_dir=tmp_path
    )
    assert before, "the unlabelled table must still be refused"
    assert before[0].validator == "emitted_table_aggregate_row"

    frame["row_role"] = [_OVERALL_ROW_ROLE, _LEVEL_ROW_ROLE, _LEVEL_ROW_ROLE]
    frame.to_csv(unlabelled, index=False)
    after = unlabelled_aggregate_row_findings(
        step_id="03_measurement_process_audit", out_dir=tmp_path
    )
    assert after == [], [f.message for f in after]
