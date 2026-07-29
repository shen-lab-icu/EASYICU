"""A figure-lineage join key must identify a row, not merely vary across rows.

Measured on a real run (2026-07-29): the exposure-outcome distribution figure
was rejected with "source-data values disagree with
exposure_outcome_distribution.csv" while its source-data CSV was a
*byte-identical copy* of that upstream table.

The key selector accepted ``row_role`` because it was "mostly distinct"
(2 unique values across 3 rows, over the 0.5 threshold). But the table holds
two ``exposure_level`` rows, so the join was many-to-many: exposure level 0 in
the figure was compared against level 1 upstream, and every numeric column was
then reported as disagreeing. A correct figure was accused of fabricating the
numbers it had copied exactly.

The shape below is the real one, values included.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator


def _distribution_table() -> pd.DataFrame:
    """The real three-row shape: two exposure levels plus an overall row."""

    return pd.DataFrame(
        [
            {
                "row_role": "exposure_level",
                "exposure_level": 0.0,
                "n_rows": 660,
                "exposure_denominator": 1000,
                "exposure_pct": 66.0,
                "outcome_events": 57,
                "outcome_denominator": 660,
                "outcome_rate_pct": 8.636364,
                "ci_low_pct": 6.725538,
                "ci_high_pct": 11.025908,
            },
            {
                "row_role": "exposure_level",
                "exposure_level": 1.0,
                "n_rows": 340,
                "exposure_denominator": 1000,
                "exposure_pct": 34.0,
                "outcome_events": 45,
                "outcome_denominator": 340,
                "outcome_rate_pct": 13.235294,
                "ci_low_pct": 10.040713,
                "ci_high_pct": 17.251359,
            },
            {
                "row_role": "overall",
                "exposure_level": None,
                "n_rows": 1000,
                "exposure_denominator": 1000,
                "exposure_pct": 100.0,
                "outcome_events": 102,
                "outcome_denominator": 1000,
                "outcome_rate_pct": 10.2,
                "ci_low_pct": 8.473914,
                "ci_high_pct": 12.230696,
            },
        ]
    )


def _compare(source: pd.DataFrame, upstream: pd.DataFrame, tmp_path: Path) -> dict:
    upstream_path = tmp_path / "exposure_outcome_distribution.csv"
    upstream.to_csv(upstream_path, index=False)
    return FigureSourceDataValidator._compare_source_to_upstream(
        source_df=source,
        source_path=tmp_path / "figure_input_source_data.csv",
        upstream_path=upstream_path,
    )


def test_an_exact_copy_is_not_reported_as_disagreeing(tmp_path: Path) -> None:
    """The measured false rejection."""

    table = _distribution_table()

    result = _compare(table.copy(), table, tmp_path)

    assert result["ok"], (
        "a byte-identical copy of the upstream table was rejected: "
        f"{result.get('reason')} / {result.get('message')}"
    )


def test_the_reported_key_names_what_separates_the_rows(tmp_path: Path) -> None:
    """The key is also the explanation, so it must read like a key.

    Several columns happen to be distinct per row (a confidence bound, for
    one). Widening on the level indicator is what a reader can act on.
    """

    table = _distribution_table()

    result = _compare(table.copy(), table, tmp_path)

    assert result["key_column"] == "row_role+exposure_level"


def test_a_forged_value_is_still_rejected(tmp_path: Path) -> None:
    """Widening the key must not blunt the check it exists to enable."""

    upstream = _distribution_table()
    source = _distribution_table()
    source.loc[source["exposure_level"] == 1.0, "outcome_events"] = 999

    result = _compare(source, upstream, tmp_path)

    assert not result["ok"]
    assert result["reason"] == "source_values_disagree"


def test_swapping_the_two_levels_is_rejected(tmp_path: Path) -> None:
    """The exact mix-up the broken join hallucinated must be caught for real.

    Under the old many-to-many join this comparison was already being made
    against the wrong row, so a figure that genuinely swapped the levels and
    one that copied them faithfully produced the same complaint. Only one of
    those is a defect.
    """

    upstream = _distribution_table()
    source = _distribution_table()
    first = source.loc[source["exposure_level"] == 0.0, "outcome_events"].iloc[0]
    second = source.loc[source["exposure_level"] == 1.0, "outcome_events"].iloc[0]
    source.loc[source["exposure_level"] == 0.0, "outcome_events"] = second
    source.loc[source["exposure_level"] == 1.0, "outcome_events"] = first

    result = _compare(source, upstream, tmp_path)

    assert not result["ok"]
    assert result["reason"] == "source_values_disagree"


def test_a_key_that_cannot_be_made_unique_says_so(tmp_path: Path) -> None:
    """Refusing beats comparing cross-matched rows and blaming the figure.

    Two rows identical in every shared column cannot be told apart, so no
    key exists. The old code would have joined them many-to-many and
    reported whatever fell out.
    """

    table = pd.DataFrame(
        [
            {"row_role": "exposure_level", "value": 1.0},
            {"row_role": "exposure_level", "value": 1.0},
            {"row_role": "overall", "value": 2.0},
        ]
    )

    result = _compare(table.copy(), table, tmp_path)

    assert not result["ok"]
    assert result["reason"] == "ambiguous_join_key"
    assert "many-to-many" in result["message"]
    assert result["duplicate_key_values"]["source"] == ["exposure_level"]


@pytest.mark.parametrize("duplicated_rows", [2, 3])
def test_more_duplicate_rows_do_not_change_the_verdict(
    tmp_path: Path, duplicated_rows: int
) -> None:
    """Uniqueness, not a distinct-ratio threshold, decides.

    With more repeated rows the old 0.5 "mostly distinct" ratio eventually
    rejects the key for the wrong reason. The verdict should depend on
    whether a row can be identified, not on how many rows share a label.
    """

    rows = [
        {"row_role": "exposure_level", "exposure_level": float(index), "value": index}
        for index in range(duplicated_rows)
    ]
    rows.append({"row_role": "overall", "exposure_level": None, "value": 99})
    table = pd.DataFrame(rows)

    result = _compare(table.copy(), table, tmp_path)

    assert result["ok"], result.get("reason")
