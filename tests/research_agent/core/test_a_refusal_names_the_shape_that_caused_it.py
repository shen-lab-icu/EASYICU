"""``no_verifiable_values`` listed the columns and withheld the reason.

When a figure's source-data table carries SEVERAL rows per upstream row -- one
per panel, per statistic, per level -- a single source column holds values from
several different upstream columns.  No single upstream vector can then match
it, so every value column arrives at the refusal unverified and the reader is
handed a list of columns with no explanation covering all of them.

m1's ``09_missingness_audit_figure``, 2026-08-04: 6 source rows over 3 upstream
rows, panel A carrying the upstream ``missing_n`` and panel B its ``measured_n``
under one ``count`` column.  The refusal named
``['count', 'denominator', 'percentage', 'statistic']`` and stopped there.  A
repair reading that reaches for renaming, which the Coder prompt already says
"is not a repair" -- and the step failed with four provider calls unspent, so
what it lacked was not budget.

MEASURED over the recorded corpus: 12 of 361 source-data tables carry duplicate
keys, and of the 8 whose step status is known, 6 failed.  Two passed, so the
shape is NOT fatal on its own and this reports it without judging it.  The
verdict is unchanged; only the explanation is added.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _upstream(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "missingness_measurement_audit.csv"
    pd.DataFrame(
        {
            "variable": ["bili", "sofa2_liver", "death"],
            "missing_n": [52707, 52707, 0],
            "measured_n": [41751, 41751, 94458],
            "n_total": [94458, 94458, 94458],
        }
    ).to_csv(path, index=False)
    return path


def _melted_source(tmp_path: pathlib.Path) -> pd.DataFrame:
    """m1's shape: one row per panel, so ``count`` alternates upstream columns."""

    return pd.DataFrame(
        {
            "panel": ["A", "B", "A", "B", "A", "B"],
            "statistic": ["missing", "measured"] * 3,
            "count": [52707, 41751, 52707, 41751, 0, 94458],
            "denominator": [94458] * 6,
            "source_row_index": [0, 0, 1, 1, 2, 2],
            "source_table": ["missingness_measurement_audit.csv"] * 6,
        }
    )


def _compare(tmp_path: pathlib.Path, source: pd.DataFrame):
    path = tmp_path / "audit_source_data.csv"
    source.to_csv(path, index=False)
    return FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=path,
        upstream_path=_upstream(tmp_path),
    )


def test_the_refusal_says_how_many_source_rows_share_an_upstream_row(tmp_path):
    result = _compare(tmp_path, _melted_source(tmp_path))

    assert result["ok"] is False
    assert result["reason"] == "no_verifiable_values"
    assert result["source_rows_per_upstream_row"] == 2.0
    assert result["n_source_rows"] == 6
    assert result["n_distinct_source_keys"] == 3
    assert "6 rows over 3 upstream rows" in result["message"]


def test_the_verdict_itself_is_unchanged(tmp_path):
    """A diagnostic that changes what passes is not a diagnostic."""

    result = _compare(tmp_path, _melted_source(tmp_path))

    assert result["ok"] is False
    assert result["reason"] == "no_verifiable_values"
    assert sorted(result["unverified_source_value_columns"]) == [
        "count",
        "statistic",
    ]


def test_a_one_to_one_table_is_not_labelled_with_a_shape_it_does_not_have(tmp_path):
    """The field must appear only where it explains something."""

    source = pd.DataFrame(
        {
            "variable": ["bili", "sofa2_liver", "death"],
            "missing_n": [52707, 52707, 0],
            "source_row_index": [0, 1, 2],
            "source_table": ["missingness_measurement_audit.csv"] * 3,
        }
    )
    result = _compare(tmp_path, source)

    assert "source_rows_per_upstream_row" not in result
    # And this one verifies, which is what a traceable subset looks like.
    assert result["ok"] is True, result


def test_a_failing_one_to_one_table_is_still_not_given_the_shape(tmp_path):
    """The case that actually exercises the condition.

    The passing test above proves nothing about it: a verified table returns
    through a different branch, which carries no such field either way. A
    mutation that reported the shape on every table survived until this
    existed.
    """

    source = pd.DataFrame(
        {
            "variable": ["bili", "sofa2_liver", "death"],
            "count": [1, 2, 3],  # matches no upstream vector
            "source_row_index": [0, 1, 2],
            "source_table": ["missingness_measurement_audit.csv"] * 3,
        }
    )
    result = _compare(tmp_path, source)

    assert result["ok"] is False
    assert result["reason"] == "no_verifiable_values"
    assert "source_rows_per_upstream_row" not in result, result
    assert "upstream rows" not in result["message"], result["message"]


def test_the_diagnostic_cannot_fail_the_audit_on_its_own(tmp_path):
    """It is wrapped, because an explanation must never become the verdict."""

    import inspect

    source = FigureSourceDataValidator._compare_source_to_upstream
    body = inspect.getsource(source)
    marker = body.index("rows_per_upstream_row: dict[str, object] = {}")
    guarded = body[marker : marker + 1400]
    assert "try:" in guarded and "except Exception" in guarded


def test_the_recorded_m1_artifacts_reproduce_it():
    """Replays the run that motivated this, from its own outputs."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    runs = sorted(
        _CORPUS.glob("batch_*verify15/m1_hepatobiliary_missingness/aware/run_*/")
    )
    if not runs:
        pytest.skip("the verify15 m1 run is not on this disk")
    run = runs[-1]
    source_path = (
        run / "steps/09_missingness_audit_figure/outputs"
        "/missingness_audit_source_data.csv"
    )
    upstream_path = (
        run / "steps/03_missingness_and_measurement_audit/outputs"
        "/missingness_measurement_audit.csv"
    )
    if not source_path.exists() or not upstream_path.exists():
        pytest.skip("the recorded figure artifacts are not on this disk")

    result = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=pd.read_csv(source_path),
        source_path=source_path,
        upstream_path=upstream_path,
    )

    assert result["ok"] is False
    assert result["reason"] == "no_verifiable_values"
    assert result["source_rows_per_upstream_row"] == 2.0
    assert result["n_source_rows"] == 6


def test_the_shape_is_reported_not_condemned():
    """Two recorded melted tables passed, so this may not become a rule.

    A duplicate key is evidence about why no column matched, not a verdict of
    its own -- there is no branch that refuses a table for being melted.
    """

    import inspect

    body = inspect.getsource(FigureSourceDataValidator._compare_source_to_upstream)
    assert '"reason": "duplicate_source_keys"' not in body
    assert "source_rows_per_upstream_row" in body
