"""The dictionary marks SOFA-2 concepts SICdb cannot compute as unavailable.

``scores.sofa2_aggregate`` records that SICdb has no CNS source, so its SOFA-2
total stays empty. The catalog's declared list is a union over inputs (an
airway flag gave SICdb a GCS "mapping"), so SICdb used to appear as mapped for
SOFA-2 CNS, the SOFA-2 total and SOFA-2 based sepsis.
"""

from __future__ import annotations

import pytest

from easyicu.scores.sofa2_aggregate import SOFA2_STRUCTURALLY_UNAVAILABLE_COMPONENTS_BY_DATABASE
from easyicu.webserver import catalog, catalog_metadata


@pytest.fixture(autouse=True)
def _isolated_definitions():
    catalog_metadata.catalog_definitions.cache_clear()
    yield
    catalog_metadata.catalog_definitions.cache_clear()


def test_the_owner_still_marks_sicdb_cns_unavailable():
    assert SOFA2_STRUCTURALLY_UNAVAILABLE_COMPONENTS_BY_DATABASE == {"sic": frozenset({"sofa2_cns"})}


def test_sofa2_cns_total_and_sepsis_are_unavailable_in_sicdb_and_nothing_else():
    unavailable = catalog._structurally_unavailable(catalog_metadata.catalog_definitions())
    assert unavailable == {"sofa2_cns": ["sic"], "sofa2": ["sic"], "sep3_sofa2": ["sic"]}


def test_coverage_moves_sicdb_from_declared_to_structurally_unavailable():
    definitions = catalog_metadata.catalog_definitions()
    metadata, _ = catalog._coverage_metadata(definitions, definitions)
    for key in ("sofa2_cns", "sofa2", "sep3_sofa2"):
        entry = metadata[key]
        assert "sic" not in entry["declared_databases"], key
        assert entry["structurally_unavailable"] == ["sic"], key
        assert entry["databases"] == len(entry["declared_databases"]), key
    for key in ("sofa2_coag", "sofa2_resp", "sofa"):
        assert "sic" in metadata[key]["declared_databases"], key
        assert "structurally_unavailable" not in metadata[key], key
