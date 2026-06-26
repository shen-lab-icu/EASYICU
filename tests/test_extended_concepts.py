"""Validation for the extended-feature concepts merged into the main
``concept-dict.json`` (severity scores + extra labs) and the reusable
``na_below`` sentinel transform that backs the native-score concepts.

Real-data extraction smoke checks are behind ``needs_real_data``
(run with ``--run-real`` and ``EASYICU_DATA_PATH`` set).
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.resources import load_dictionary

VALID_SOURCE_KEYS = {
    "mimic", "mimic_demo", "miiv", "miiv_demo",
    "eicu", "eicu_demo", "aumc", "hirid", "sic", "sic_demo",
}

SEVERITY = ["apache_iv", "apache_iv_pred_hosp_mort", "saps3"]
EXTRA_LABS = ["ammonia", "amylase", "d_dimer", "ferritin", "ldh",
              "lipase", "osmolality"]


def test_extended_concepts_present_in_main_dict():
    d = load_dictionary()
    for name in SEVERITY + EXTRA_LABS:
        assert name in d, f"{name} not merged into concept-dict.json"


def test_extended_concept_sources_reference_known_databases():
    d = load_dictionary()
    for name in SEVERITY + EXTRA_LABS:
        for src_key in d[name].sources:
            assert src_key in VALID_SOURCE_KEYS, (
                f"concept '{name}' references unknown source '{src_key}'"
            )


def test_severity_concepts_typed():
    d = load_dictionary()
    for name in SEVERITY:
        assert d[name].category == "severity"
        assert d[name].target == "id_tbl"


def test_extra_labs_cover_five_databases():
    d = load_dictionary()
    # MIMIC-III/IV + eICU verified itemids/labnames; SIC + AmsterdamUMC mapped.
    for lab in EXTRA_LABS:
        srcs = set(d[lab].sources)
        assert {"miiv", "mimic"} <= srcs
    # d_dimer is intentionally MIMIC-only (FEU vs DDU unit mismatch elsewhere)
    assert "sic" in d["ferritin"].sources
    assert "aumc" in d["ldh"].sources
    assert "sic" not in d["d_dimer"].sources


def test_eicu_lab_casing_matches_source_strings():
    d = load_dictionary()
    assert d["ldh"].sources["eicu"][0].ids == ["LDH"]
    assert d["ferritin"].sources["eicu"][0].ids == ["Ferritin"]


def test_na_below_maps_sentinels_to_nan():
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.concept.schema import ConceptSource

    src = ConceptSource(value_var="apache_iv", callback="transform_fun(na_below(0))")
    frame = pd.DataFrame({"id": [1, 2, 3, 4], "apache_iv": [-1.0, 0.0, 51.0, 211.0]})
    out = _apply_callback(frame.copy(), src, "apache_iv")
    vals = out["apache_iv"]
    assert pd.isna(vals.iloc[0])      # -1 sentinel -> NaN
    assert vals.iloc[1] == 0.0
    assert vals.iloc[2] == 51.0
    assert vals.iloc[3] == 211.0


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database,concept,lo,hi", [
    ("eicu", "apache_iv", 0, 286),
    ("eicu", "apache_iv_pred_hosp_mort", 0, 1),
    ("sic", "saps3", 0, 217),
])
def test_severity_extracts_within_bounds(database, concept, lo, hi):
    from easyicu import load_concepts

    df = load_concepts([concept], database=database)
    assert concept in df.columns
    vals = pd.to_numeric(df[concept], errors="coerce").dropna()
    assert len(vals) > 0
    assert vals.min() >= lo
    assert vals.max() <= hi
