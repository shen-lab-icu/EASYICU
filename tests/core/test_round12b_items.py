"""Round-12b acceptance items carried into lead-r13."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callback_apply import _drop_declared_row_exclusions

ROOT = Path(__file__).resolve().parents[2]
DICT = json.loads((ROOT / "src/easyicu/data/concept-dict.json").read_text())
CHANGELOG = " ".join((ROOT / "docs/extraction_changelog.md").read_text().split())


def _abx_hicl_sources(database):
    return [s for s in DICT["abx"]["sources"][database] if s.get("sub_var") == "drughiclseqno"]


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_abx_hicl_branch_admits_only_orders_without_a_drugname(database):
    (source,) = _abx_hicl_sources(database)
    assert source["require_empty_var"] == "drugname"
    assert "drugname" in source["extra_vars"]
    frame = pd.DataFrame({
        "drughiclseqno": [10093, 10093, 10093],
        "drugname": ["SODIUM CHLORIDE 0.9 % IV 250 ML BAG", "", None],
        "dosage": ["1", "1", "1"],
        "routeadmin": ["IV", "IV", "IV"],
    })
    kept = _drop_declared_row_exclusions(frame, SimpleNamespace(params=source), "abx")
    assert kept["drugname"].isna().sum() + kept["drugname"].eq("").sum() == len(kept) == 2


def test_d_dimer_out_of_bound_counts_state_both_denominators():
    comment = next(s for s in DICT["d_dimer"]["sources"]["aumc"] if "callback" in s)["_comment"]
    for text in (comment, CHANGELOG):
        assert "932" in text and "782" in text
        assert "20 stays" in text or "20 admissions" in text
        assert "25/929" not in text and "25 of 929" not in text


def test_aumc_assumption_wording_names_the_death_date_check():
    assert "no valid `dateofdeath` after the existing death-date checks" in CHANGELOG
