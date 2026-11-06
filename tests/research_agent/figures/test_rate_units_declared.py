"""Rate columns are percent or proportion by their name, never by their size.

These are lead-r13 magnitude/format sweep #16 sites: figure and review readers
that decided percent vs proportion from the column maximum. A table of rare
event rates stored as percent (every value <= 1, e.g. 0.8 %) was shown or
reviewed as 80 %. The name now declares the unit (``pct``/``percent``
suffix = percent, every other accepted rate name = proportion). A proportion
column holding a value above 1 contradicts its name and raises
``UndeclaredUnitError``. Synthetic tables only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.figures.strata import normalise_strata_frame
from easyicu.research_agent.reporting.publication_bundles import _declared_rate_scale
from easyicu.research_agent.review import methodological_rigor
from easyicu.utils.declared_units import UndeclaredUnitError


def test_strata_percent_column_of_small_rates_is_divided_by_its_name():
    out = normalise_strata_frame(pd.DataFrame({"score": [1, 2], "mortality_pct": [0.4, 0.8], "n": [50, 60]}))
    assert out["rate"].tolist() == pytest.approx([0.004, 0.008])


def test_strata_proportion_column_stays_and_a_percent_value_in_it_is_refused():
    out = normalise_strata_frame(pd.DataFrame({"score": [1, 2], "mortality_rate": [0.1, 0.3], "n": [50, 60]}))
    assert out["rate"].tolist() == pytest.approx([0.1, 0.3])
    with pytest.raises(UndeclaredUnitError) as caught:
        normalise_strata_frame(pd.DataFrame({"score": [1, 2], "mortality_rate": [10.0, 30.0], "n": [50, 60]}))
    assert caught.value.reason == "rate_unit_contradiction"


def test_publication_rate_scale_follows_the_column_name():
    assert _declared_rate_scale("mortality_rate", [0.004, 0.2], site="t") == 100.0
    assert _declared_rate_scale("mortality_pct", [0.4, 20.0], site="t") == 1.0
    with pytest.raises(UndeclaredUnitError):
        _declared_rate_scale("outcome_rate", [12.0, 30.0], site="t")


@pytest.mark.parametrize(
    "column,values,expected",
    [("missing_pct", [0.8, 0.5], 0.008), ("missing_pct", [46.0], 0.46), ("missing_fraction", [0.46], 0.46)],
)
def test_missing_fraction_reads_the_declared_unit(tmp_path: Path, monkeypatch, column, values, expected):
    pd.DataFrame({column: values}).to_csv(tmp_path / "missingness.csv", index=False)
    monkeypatch.setattr(
        methodological_rigor, "_record_by_current_name",
        lambda evidence, name, evidence_records=None: (
            SimpleNamespace(relative_path="missingness.csv") if name == "missingness" else None
        ),
    )
    evidence = SimpleNamespace(root=tmp_path)
    assert methodological_rigor._missing_fraction(evidence) == pytest.approx(expected)


def test_missing_fraction_column_above_one_is_refused(tmp_path: Path, monkeypatch):
    pd.DataFrame({"missing_frac": [46.0]}).to_csv(tmp_path / "missingness.csv", index=False)
    monkeypatch.setattr(
        methodological_rigor, "_record_by_current_name",
        lambda evidence, name, evidence_records=None: SimpleNamespace(relative_path="missingness.csv"),
    )
    with pytest.raises(UndeclaredUnitError):
        methodological_rigor._missing_fraction(SimpleNamespace(root=tmp_path))
