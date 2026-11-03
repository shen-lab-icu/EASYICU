"""MIMIC-IV eMAR drug exposure counts given doses only.

2026-11-03 extraction audit 2 (P2): cort and apixaban were empty for MIMIC-IV
although emar records their administrations.
"""

from __future__ import annotations

import json
import re
from importlib import resources

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback


def _dictionary() -> dict:
    return json.loads(
        resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("concept", ["cort", "apixaban"])
def test_emar_source_keeps_administered_events_only(concept):
    source = ConceptSource.from_mapping(_dictionary()[concept]["sources"]["miiv"][0])
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "charttime": pd.to_datetime(["2150-01-01 08:00"] * 4),
            concept: ["x"] * 4,
            "event_txt": ["Administered", "Not Given", "Delayed Administered", "Hold Dose"],
        }
    )

    out = _apply_callback(frame, source, concept_name=concept)

    assert out["event_txt"].tolist() == ["Administered", "Delayed Administered"]
    assert out[concept].tolist() == [True, True]


def test_systemic_corticosteroid_names_exclude_local_preparations():
    sources = _dictionary()["cort"]["sources"]
    for database in ("miiv", "mimic", "eicu"):
        pattern = re.compile(sources[database][0]["regex"], re.IGNORECASE)
        for systemic in ("Prednisone", "Hydrocortisone Na Succ.", "methylprednisolone sodium succ",
                         "Dexamethasone", "Fludrocortisone Acetate"):
            assert pattern.search(systemic), (database, systemic)
        for local in ("Hydrocortisone Cream 1%", "Prednisolone Acetate 1% Ophth. Susp.",
                      "Methylprednisolone Acetate", "Hydrocortisone Enema",
                      "Dexamethasone Ophthalmic Susp 0.1%", "Hydrocortisone Study Drug"):
            assert not pattern.search(local), (database, local)
    eicu = re.compile(sources["eicu"][0]["regex"], re.IGNORECASE)
    assert eicu.search("SOLU-MEDROL") and eicu.search("1 ML VIAL : DEXAMETHASONE SODIUM PHOSPHATE")
