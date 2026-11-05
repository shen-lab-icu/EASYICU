"""Round-12 basis records: bounds that differ from v6 and deliberate ricu deviations.

The auditor accepted these as audited corrections that predate the R2 bound
freeze (baseline cd2bae05d); each must state its basis where a reader of the
dictionary will see it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DICTIONARY = json.loads((ROOT / "src/easyicu/data/concept-dict.json").read_text())
CHANGELOG = (ROOT / "docs/extraction_changelog.md").read_text()

# concept: (min, max) now; v6 differs for every row.
BOUNDS = {
    "dbp": (1, 200), "map": (1, 250), "sbp": (1, 300), "plateau_pres": (1, 100),
    "tidal_vol": (1, 2000), "epi_rate": (0, 5), "norepi_rate": (0, 5), "ins": (0, 50),
    "icp": (-10, 100), "plt": (1, 1200), "hbco": (0, 100),
}


@pytest.mark.parametrize("concept", sorted(BOUNDS))
def test_bounds_that_differ_from_v6_state_their_basis(concept):
    entry = DICTIONARY[concept]
    assert (entry.get("min"), entry.get("max")) == BOUNDS[concept]
    comment = entry.get("_comment", "")
    assert "bound basis" in comment or "lower bound is 1" in comment, concept
    assert f"| `{concept}` |" in CHANGELOG


def test_pafi_records_its_deviation_from_ricu_and_the_direction():
    comment = DICTIONARY["pafi"]["_comment"]
    assert "deviates from ricu" in comment and "[min PaO2, max FiO2]" in comment
    assert "SOFA respiratory component is equal or lower" in comment
    assert "P/F FiO2 aggregation deviates from ricu" in CHANGELOG
