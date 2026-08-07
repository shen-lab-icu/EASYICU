"""A cohort fact that every task carried and none was told.

The export writes ``<concept>_time`` as hours elapsed from the cohort anchor.
MEASURED on the real MIMIC-IV cohort: 21 such columns, and **20 of them have
minimum exactly 0.00**. The convention is not in doubt. One column breaks it --
``death_time`` reaches -23, in 28 of 94,458 rows -- placing a death before the
ICU admission that defines the row.

All NINE recorded tasks carry the same 28 rows, and the cohort auditor said
nothing: it reports how much of each column is MISSING, never whether what is
present is possible. Only a time-to-event analysis ever compares those columns,
so h1 found it 20 minutes into its primary step, wrote its own guard, raised
``ValueError: Inconsistent in-hospital event timing``, and died -- with four
provider calls unspent, having discovered a property of the cohort that was
fixed before it was ever handed one.

The rule here is STRUCTURAL and case-neutral: it reads the export's own
convention off the export, and names no concept, database, or study. It is a
``warning`` that drops nothing, exactly like the two cohort-hygiene checks
beside it: 0.03 % of rows is a data-quality fact, not a reason to refuse an
analysis, and which rows to censor is the analyst's call.
"""

from __future__ import annotations

import inspect
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.audits import validators as validators_module
from easyicu.research_agent.audits.validators import cohort_hygiene_findings

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
_SUBKIND = "elapsed_time_precedes_anchor"


def _context():
    return SimpleNamespace(target_outcome="outcome")


def _findings(frame: pd.DataFrame):
    return [
        finding
        for finding in cohort_hygiene_findings(frame, _context())
        if finding.detail.get("subkind") == _SUBKIND
    ]


# ---------------------------------------------------------------------------
# It fires exactly when an elapsed time precedes its anchor
# ---------------------------------------------------------------------------


def test_a_negative_elapsed_time_is_reported():
    frame = pd.DataFrame(
        {"stay_id": [1, 2, 3], "event_time": [10.0, -23.0, 4.0]}
    )

    findings = _findings(frame)

    assert len(findings) == 1
    assert findings[0].detail["column"] == "event_time"
    assert findings[0].detail["negative_n"] == 1
    assert findings[0].detail["minimum"] == -23.0
    # It never fails the run.
    assert findings[0].severity == "warning"
    assert findings[0].detail["impartial"] is True


def test_a_column_that_honours_the_convention_says_nothing():
    frame = pd.DataFrame({"stay_id": [1, 2], "event_time": [0.0, 12.5]})

    assert _findings(frame) == []


def test_a_non_time_column_is_not_audited_for_sign():
    """Plenty of legitimate columns are negative. Only elapsed times are not."""

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "base_excess": [-4.0, -12.0],
            "risk_difference": [-0.2, -0.1],
            "event_time": [0.0, 3.0],
        }
    )

    assert _findings(frame) == []


def test_every_offending_column_is_named_separately():
    frame = pd.DataFrame(
        {
            "first_time": [-1.0, 2.0],
            "last_time": [3.0, -4.0],
            "other_time": [0.0, 1.0],
        }
    )

    findings = _findings(frame)

    assert sorted(f.detail["column"] for f in findings) == ["first_time", "last_time"]


def test_missing_values_are_not_counted_as_negative():
    frame = pd.DataFrame({"event_time": [None, float("nan"), 5.0]})

    assert _findings(frame) == []


def test_a_column_of_only_missing_values_is_skipped():
    frame = pd.DataFrame({"event_time": [None, None]})

    assert _findings(frame) == []


def test_a_non_numeric_time_column_does_not_raise():
    frame = pd.DataFrame({"event_time": ["2026-01-01", "not a number"]})

    assert _findings(frame) == []


# ---------------------------------------------------------------------------
# Case neutrality
# ---------------------------------------------------------------------------


def test_the_rule_names_no_concept_database_or_study():
    """The defect was found on death_time; the rule must not mention it.

    Tokenised, not substring-matched. A first version of this test failed on
    its own subject: it flagged ``vent`` inside ``e-VENT``, which is precisely
    the substring-vs-name bug repaired earlier in this branch (``ols(`` matched
    ``matched_contr-OLS-.append``). A case-neutrality check that cannot tell a
    name from a fragment is not a check.
    """

    import io
    import tokenize

    source = inspect.getsource(cohort_hygiene_findings)
    body = source.split("# (C)", 1)[1] if "# (C)" in source else source

    names: set[str] = set()
    for token in tokenize.generate_tokens(io.StringIO(body).readline):
        if token.type == tokenize.NAME:
            names.add(token.string.lower())
        elif token.type == tokenize.STRING:
            # String literals are the other way a case leaks in.
            names.update(part.lower() for part in token.string.split())

    for banned in (
        "death",
        "death_time",
        "sep3",
        "sepsis",
        "mimic",
        "miiv",
        "los_hosp",
        "sofa",
        "sofa2",
        "lact",
        "vent",
        "mech_vent",
        "kdigo",
    ):
        assert banned not in names, (banned, sorted(names))


def test_it_reads_the_convention_off_any_export():
    """A different naming scheme with the same convention still works."""

    frame = pd.DataFrame({"t_from_anchor_time": [-0.5, 1.0]})

    assert len(_findings(frame)) == 1


# ---------------------------------------------------------------------------
# The real cohort, and the convention the rule relies on
# ---------------------------------------------------------------------------


def _a_real_cohort() -> pd.DataFrame | None:
    for path in sorted(_CORPUS.glob("batch_*/h1_*/aware/run_*/cohort.parquet")):
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - an unreadable export is not the subject
            continue
    return None


def test_the_convention_holds_on_the_real_export():
    """20 of 21 elapsed-time columns start at exactly 0. That is the premise."""

    frame = _a_real_cohort()
    if frame is None:
        pytest.skip("no recorded cohort is mounted")

    time_columns = [c for c in frame.columns if str(c).lower().endswith("_time")]
    if len(time_columns) < 5:
        pytest.skip("this export carries too few elapsed-time columns")

    minima = {
        column: float(pd.to_numeric(frame[column], errors="coerce").min())
        for column in time_columns
    }
    at_zero = [column for column, value in minima.items() if value == 0.0]
    negative = [column for column, value in minima.items() if value < 0.0]

    assert len(at_zero) >= len(time_columns) - 2, minima
    assert len(negative) <= 1, negative


def test_the_real_cohort_is_reported_and_not_blocked():
    frame = _a_real_cohort()
    if frame is None:
        pytest.skip("no recorded cohort is mounted")

    findings = _findings(frame)
    if not findings:
        pytest.skip("this recorded cohort carries no impossible elapsed time")

    assert {f.severity for f in findings} == {"warning"}
    assert all(f.detail["negative_n"] > 0 for f in findings)
    # The whole cohort survives: nothing here drops a row.
    assert all(f.detail["n_rows"] == len(frame) for f in findings)


def test_the_module_still_reports_the_two_checks_it_had():
    """The new check joins (A) and (B); it must not displace them."""

    frame = _a_real_cohort()
    if frame is None:
        pytest.skip("no recorded cohort is mounted")

    subkinds = {
        finding.detail.get("subkind")
        for finding in cohort_hygiene_findings(frame, _context())
    }
    assert "short_stay_exposure" in subkinds
    assert _SUBKIND in subkinds
