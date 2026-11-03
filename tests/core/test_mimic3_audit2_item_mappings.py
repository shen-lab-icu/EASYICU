"""MIMIC-III/MIMIC-IV item mappings from the 2026-11-03 extraction audit (batch 2).

Each mapping was checked against MIMIC d_items labels and raw units:
CareVue (inputevents_cv) holds the same products/drugs as the MetaVision
items these concepts mapped; NICU length 4188/4187 have their valueuom
swapped; MetaVision/MIMIC-IV pre-ICU intake items follow the CareVue
ICU-intake rule.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

CONCEPTS = json.loads(
    resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
)


def _sources(concept: str, database: str, table: str | None = None) -> list[dict]:
    found = CONCEPTS[concept]["sources"][database]
    return [s for s in found if table is None or s.get("table") == table]


def _ids(sources: list[dict]) -> set[int]:
    out: set[int] = set()
    for source in sources:
        ids = source.get("ids", [])
        out.update(ids if isinstance(ids, list) else [ids])
    return out


CAREVUE = {
    "packed_rbc": ({30001, 30004, 30179, 30104}, set()),
    "ffp": ({30005, 30180, 30103}, {40801, 44031}),  # plasmapheresis, Plasma-Lyte
    "platelets": ({30006, 30105}, set()),
    "albumin_iv": ({30008, 30009, 30181}, {44952}),  # OR Albumin (MV excludes OR colloid)
    # 30299 Potassium Acetate: MetaVision excludes 222139 Potassium ACEtate too.
    "potassium_iv": ({30026, 30297, 30029}, {30299}),
    "bicarbonate": ({30030, 30338}, set()),
    "magnesium_iv": ({30027}, set()),
    "dexmedetomidine": ({30167}, set()),
    "octreotide": ({30133}, set()),
    "diltiazem": ({30115}, set()),
    "esmolol": ({30117}, set()),
    "nicardipine": ({30178}, set()),
}


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
@pytest.mark.parametrize("concept", sorted(CAREVUE))
def test_carevue_holds_the_same_product_as_the_metavision_items(concept, database):
    present, absent = CAREVUE[concept]
    carevue = _sources(concept, database, "inputevents_cv")
    assert len(carevue) == 1
    assert carevue[0]["callback"] == "transform_fun(set_val(TRUE))"
    assert _ids(carevue) == present
    assert not _ids(_sources(concept, database)) & absent
    # The MetaVision source is kept alongside.
    assert _sources(concept, database, "inputevents_mv")


@pytest.mark.parametrize("database", ["mimic", "mimic_demo", "miiv"])
def test_platelets_include_or_and_pacu_platelets_like_rbc_and_ffp(database):
    table = "inputevents" if database == "miiv" else "inputevents_mv"
    assert {225170, 226369, 227071} <= _ids(_sources("platelets", database, table))


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_nicu_length_in_cm_is_not_inch_converted_and_its_inch_twin_is_dropped(database):
    sources = _sources("height", database)
    converted = [s for s in sources if "binary_op(`*`, 2.54)" in (s.get("callback") or "")]
    plain = [s for s in sources if not s.get("callback")]
    assert 4187 not in _ids(sources)  # 'Length Calc Inches': inches labelled cm
    assert 4188 not in _ids(converted)  # 'Length in cm': cm labelled Inches
    assert _ids(plain) == {4188}
    assert {920, 1394, 3485, 3486, 226707} <= _ids(converted)


PRE_ICU_INTAKE = {
    226361, 226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371,
    226372, 226375, 226376, 226377, 227070, 227071, 227072,
}


@pytest.mark.parametrize(
    ("database", "table"),
    [("mimic", "inputevents_mv"), ("mimic_demo", "inputevents_mv"), ("miiv", "inputevents")],
)
def test_metavision_and_miiv_intake_excludes_pre_icu_or_pacu_items(database, table):
    (source,) = _sources("total_input_ml", database, table)
    ids = set(source["ids"])
    assert not ids & PRE_ICU_INTAKE
    # ICU fluids, blood products and feeds remain.
    assert {225158, 225828, 225168, 220970, 225170, 226452} <= ids
