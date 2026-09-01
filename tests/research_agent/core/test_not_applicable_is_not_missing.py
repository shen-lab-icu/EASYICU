"""A cohort splits into observed, missing, and *not applicable*.

Live E1 blocker, 2026-07-29, ``run_20260729T062855_175303``, step
``06_missingness_event_timing_audit``. The envelope normalizer raised one
error, which surfaced as a ``normalization_error`` shadow mismatch and failed
the step ``contract_failed``::

    code       : inconsistent_registered_missingness_partition
    field_path : row[5]
    product_id : table:missingness_measurement_audit

Row 5 is ``death_time``::

    n_total = 1000   n_nonmissing = 102   missing_n = 0
    eligible_n = 102   not_applicable_n = 898
    missingness_kind = conditional_event_time

The check compared ``n_total`` against ``n_nonmissing + missing_n`` only, which
asserts the value is semantically applicable to every subject. For a
conditional quantity it is not: 102 patients died, so only they *have* a death
time. Reporting the other 898 as missing would claim 89.8 % of death times were
absent when those patients simply did not die -- the opposite of what the audit
is for.

``MissingnessProfile`` already types the third part: ``not_applicable_n`` is
"rows where absence is expected under the typed semantics", default 0. The
check just never read it. With the full partition the row reconciles exactly,
102 + 0 + 898 = 1000, and a row that declares no not-applicable population is
unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.result_envelope import (
    normalize_step_result_shadow,
)

# ``missing_pct`` is load-bearing: without it ``_table_semantic_roles`` never
# assigns the ``missingness`` role and the partition check does not run at
# all -- which made an earlier draft of the positive cases pass vacuously.
_HEADER = (
    "concept,variable,label,value_column,n_total,n_nonmissing,missing_n,"
    "missing_pct,eligible_n,not_applicable_n,missingness_kind\n"
)

_PARTITION_CODE = "inconsistent_registered_missingness_partition"


def _normalize(tmp_path: Path, *rows: str):
    table = tmp_path / "missingness_measurement_audit.csv"
    table.write_text(_HEADER + "".join(rows), encoding="utf-8")
    summary = {
        "status": "ok",
        "output_files": {"table:missingness_measurement_audit": table.name},
    }
    (tmp_path / "step_summary.json").write_text(json.dumps(summary))
    envelope = normalize_step_result_shadow(
        step_id="06_missingness_event_timing_audit",
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )
    return [
        issue
        for issue in envelope.normalization_issues
        if issue.code == _PARTITION_CODE
    ]


def test_a_conditional_event_time_row_is_not_an_inconsistency(tmp_path: Path) -> None:
    """The exact row that failed the real step."""

    issues = _normalize(
        tmp_path,
        "death_time,death_time,death time,death_time,1000,102,0,0.0,102,898,"
        "conditional_event_time\n",
    )

    assert issues == []


def test_a_row_with_no_not_applicable_population_is_unchanged(
    tmp_path: Path,
) -> None:
    """The ordinary measurement row keeps the plain two-way partition."""

    issues = _normalize(
        tmp_path,
        "sofa2_liver,sofa2_liver,sofa2 liver,sofa2_liver_max,1000,452,548,54.8,1000,0,"
        "measurement_missing\n",
    )

    assert issues == []


def test_a_genuinely_inconsistent_row_still_fails(tmp_path: Path) -> None:
    """The check must still catch what it exists to catch.

    Nothing accounts for the 400 subjects this row loses, so it stays an error
    -- the fix widens the partition, it does not retire it.
    """

    issues = _normalize(
        tmp_path,
        "sofa2_liver,sofa2_liver,sofa2 liver,sofa2_liver_max,1000,452,148,14.8,1000,0,"
        "measurement_missing\n",
    )

    assert len(issues) == 1


def test_not_applicable_cannot_absorb_an_arbitrary_shortfall(
    tmp_path: Path,
) -> None:
    """The declared not-applicable count has to be the one that reconciles.

    Otherwise the fix would be a licence to declare any number and pass.
    """

    issues = _normalize(
        tmp_path,
        "death_time,death_time,death time,death_time,1000,102,0,0.0,102,500,"
        "conditional_event_time\n",
    )

    assert len(issues) == 1


@pytest.mark.parametrize("declared", ["898", ""])
def test_the_error_is_decided_by_the_declared_count_not_the_kind_label(
    tmp_path: Path,
    declared: str,
) -> None:
    """Bound to the number, not to the word ``conditional_event_time``.

    A row that omits the count gets the old two-way partition and is correctly
    flagged; the same row declaring 898 reconciles. The label plays no part, so
    a new missingness kind needs no new branch here.
    """

    issues = _normalize(
        tmp_path,
        f"death_time,death_time,death time,death_time,1000,102,0,0.0,102,{declared},"
        "conditional_event_time\n",
    )

    assert len(issues) == (0 if declared else 1)
