"""The cohort feature catalog says which columns are bookkeeping.

The hospital walk (10-22) found Cohort Statistics offering 345 features,
among them SOFA-2 observed/available flags, ascertainment receipts and AKI
reference metadata. Each feature now carries the catalog's display audience;
the picker offers clinical ones. Selection, extraction and the comparison are
unchanged. Fixtures are synthetic.
"""

from __future__ import annotations

from easyicu.webserver import cohort_review


def test_feature_rows_carry_the_catalog_audience():
    desc = {"files": [
        {"module": "vitals", "rows": 10, "columns": ["stay_id", "charttime", "hr", "map"]},
        {"module": "sofa2_score", "rows": 10, "columns": [
            "stay_id", "charttime", "sofa2", "sofa2_cardio", "sofa2_cardio_observed",
            "sofa2_cardio_available", "sofa2_cns_ascertainment",
        ]},
        {"module": "renal", "rows": 10, "columns": ["stay_id", "aki_reference_status", "aki_stage"]},
    ]}
    catalog = cohort_review._feature_catalog(desc, [])
    audience = {
        feature["column"]: feature["audience"]
        for module in catalog["modules"] for feature in module["features"]
    }
    clinical = {"hr", "map", "sofa2", "sofa2_cardio", "aki_stage"}
    internal = {"sofa2_cardio_observed", "sofa2_cardio_available", "sofa2_cns_ascertainment", "aki_reference_status"}
    assert {c for c in clinical if c in audience} == clinical
    assert {c: audience[c] for c in clinical} == dict.fromkeys(clinical, "clinical")
    assert {c: audience[c] for c in internal} == dict.fromkeys(internal, "internal")
    # Still part of the catalog: only the picker hides them.
    assert catalog["total_features"] == len(audience)
    payload = cohort_review._feature_catalog_payload(catalog, [])
    assert all("audience" in f for m in payload["modules"] for f in m["features"])
