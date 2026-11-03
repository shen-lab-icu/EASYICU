"""Diuretic family concepts and the loop-diuretic furosemide equivalent (audit #19)."""
import pandas as pd
import pytest

from easyicu.concept.diuretics import (
    AUDIT_ATTR,
    FUROSEMIDE_EQUIVALENT_FACTORS,
    DiureticSourceContractError,
    furosemide_equivalent,
)
from easyicu.resources import load_dictionary

MAIN = ["aumc", "eicu", "hirid", "miiv", "mimic", "sic"]
FLAGS = ["loop_diuretic", "thiazide_diuretic", "potassium_sparing_diuretic", "acetazolamide"]


def test_equivalence_follows_brater_1998():
    # IV loading doses 40 mg furosemide = 1 mg bumetanide = 20 mg torsemide;
    # oral furosemide needs twice the IV dose, the others do not.
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("furosemide", "iv")] * 40 == 40
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("bumetanide", "iv")] * 1 == 40
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("torasemide", "iv")] * 20 == 40
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("furosemide", "oral")] * 80 == 40
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("bumetanide", "oral")] == 40
    assert FUROSEMIDE_EQUIVALENT_FACTORS[("torasemide", "oral")] == 2


def test_amounts_convert_units_routes_and_drop_unknown_units():
    frame = pd.DataFrame({
        "givendose": [1.0, 0.04, 10.0, 5.0, 3.0],
        "doseunit": ["mg", "g", "mg", "ml", "mg"],
        "route": ["iv", "iv", "po", "iv", "iv"],
        "status": ["ok", "ok", "ok", "ok", "Rewritten"],
    })
    out = furosemide_equivalent(
        frame, concept_name="fe", value_var="givendose", unit_var="doseunit",
        params={"fe_drug": "furosemide", "fe_route_var": "route", "fe_oral_values": ["po"],
                "status_var": "status", "excluded_statuses": ["Rewritten"]},
    )
    assert out["fe"].tolist() == [1.0, 40.0, 5.0]
    assert out.attrs[AUDIT_ATTR]["rows_unit_not_mass"] == 1
    bumetanide = furosemide_equivalent(
        frame.iloc[:1], concept_name="fe", value_var="givendose", unit_var="doseunit",
        params={"fe_drug": "bumetanide", "fe_route": "iv"},
    )
    assert bumetanide["fe"].tolist() == [40.0]
    with pytest.raises(DiureticSourceContractError):
        furosemide_equivalent(frame, concept_name="fe", value_var="givendose",
                              unit_var="doseunit", params={"fe_drug": "ethacrynic acid", "fe_route": "iv"})


def test_diuretic_concepts_are_registered_per_database():
    dictionary = load_dictionary()
    fe = dictionary.get("loop_diuretic_furosemide_eq")
    assert fe.units == ["mg"] and fe.aggregate == "sum" and fe.minimum == 0
    # Per-record plausibility bound (HiRID holds a 102424 mg oral torasemide entry).
    assert fe.maximum == 2000
    for database, sources in fe.sources.items():
        for source in sources:
            assert source.callback == "furosemide_equivalent"
            assert source.params["fe_drug"] in {"furosemide", "bumetanide", "torasemide"}
    # eICU records diuretic boluses as orders without administered amounts.
    assert "eicu" not in fe.sources
    for name in FLAGS:
        definition = dictionary.get(name)
        assert definition.class_name == "lgl_cncpt" and definition.aggregate == "max"
    for database in MAIN:
        assert database in dictionary.get("loop_diuretic").sources


def test_audit_19_items_are_mapped():
    dictionary = load_dictionary()

    def ids(name, database, table=None, sub_var=None):
        return {i for s in dictionary.get(name).sources.get(database, [])
                if (table is None or s.table == table) and (sub_var is None or s.sub_var == sub_var)
                for i in (s.ids or [])}

    assert {1000520, 1000521, 1000522, 1000986} <= ids("loop_diuretic", "hirid")
    assert 1450 in ids("loop_diuretic", "sic")
    assert 6882 in ids("loop_diuretic", "aumc")
    assert 229639 in ids("loop_diuretic", "miiv")
    assert 3660 in ids("furosemide", "eicu", "medication", "drughiclseqno")
    assert 3664 in ids("loop_diuretic", "eicu", "medication", "drughiclseqno")
    # MIMIC-IV eMAR is not used (its loader ignores the stay filter); the
    # class flags read prescriptions, as aspirin does.
    for name in ["furosemide", *FLAGS]:
        assert all(s.table != "emar" for s in dictionary.get(name).sources["miiv"])
    for name in FLAGS:
        assert any(s.table == "prescriptions" for s in dictionary.get(name).sources["miiv"])
    assert 1000428 in ids("thiazide_diuretic", "hirid")
    assert {7011, 20177} <= ids("potassium_sparing_diuretic", "aumc")
    assert 6804 in ids("acetazolamide", "aumc")
