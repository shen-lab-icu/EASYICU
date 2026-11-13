"""S732 B2+B3: diuretic and nephrotoxin exposures in the opt-in diuretic_nephrotoxin module."""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept import ConceptResolver, ConceptSource, _apply_callback
from easyicu.concept.callbacks import CALLBACK_REGISTRY, ConceptCallbackContext
from easyicu.concept.catalog import (
    CONCEPT_DICTIONARY,
    CONCEPT_GROUPS_INTERNAL,
    DIURETIC_NEPHROTOXIN_CONCEPTS,
    HIDDEN_DICTIONARY_CONCEPTS,
)
from easyicu.concept.nephrotoxin_coexposure import (
    CoexposureDerivationError,
    coexposure,
    vpt_coexposure_callback,
)
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable

H = pd.Timedelta(hours=1)


def test_module_is_opt_in_and_its_concepts_are_hidden():
    assert EXTRACT_MODULES["diuretic_nephrotoxin"] == list(DIURETIC_NEPHROTOXIN_CONCEPTS)
    assert "diuretic_nephrotoxin" not in EXTRACT_MODULE_ORDER
    grouped = {c for members in CONCEPT_GROUPS_INTERNAL.values() for c in members}
    for concept in (*DIURETIC_NEPHROTOXIN_CONCEPTS, "vancomycin_iv"):
        assert concept in HIDDEN_DICTIONARY_CONCEPTS
        assert concept not in grouped
        assert concept not in CONCEPT_DICTIONARY


@pytest.mark.parametrize("concept, databases", [
    ("furosemide_active", ("miiv", "aumc", "hirid", "sic", "eicu")),
    ("torasemide", ("hirid", "sic")),
    ("piperacillin_tazobactam", ("miiv", "hirid", "sic")),
    ("nsaid", ("miiv", "aumc", "hirid", "sic")),
])
def test_exposures_are_boolean_hourly_max_with_the_census_databases(concept, databases):
    definition = load_dictionary()[concept]
    assert definition.aggregate == "max"
    assert set(definition.sources) == set(databases)


def test_furosemide_active_spreads_only_where_a_stop_is_recorded():
    raw = load_dictionary()
    active, sealed = raw["furosemide_active"].sources, raw["furosemide"].sources
    assert [s.dur_var for s in active["aumc"]] == ["stop"]
    assert [s.dur_var for s in active["miiv"]] == ["endtime"]
    assert all(not s.dur_var for s in sealed["aumc"])


def test_vancomycin_iv_drops_only_hirid_oral_capsules():
    raw = load_dictionary()
    assert list(raw["vancomycin_iv"].sources["hirid"][0].ids) == [189]
    assert 331 in set(raw["vancomycin"].sources["hirid"][0].ids)
    assert raw["vancomycin_piperacillin_tazobactam"].callback == "vpt_coexposure"
    assert CALLBACK_REGISTRY["vpt_coexposure"] is vpt_coexposure_callback


def test_miiv_piperacillin_without_tazobactam_is_not_mapped():
    ids = set(load_dictionary()["piperacillin_tazobactam"].sources["miiv"][0].ids)
    assert ids == {225893}


def test_unreviewed_emar_is_excluded():
    for name in ("furosemide_active", "torasemide", "piperacillin_tazobactam", "nsaid", "vancomycin_iv"):
        assert all(source.table != "emar" for sources in load_dictionary()[name].sources.values() for source in sources)


def _sic(cases):
    return SimpleNamespace(config=SimpleNamespace(name="sic"), load_table=lambda *a, **kw: cases)


def test_sic_active_hours_cover_infusion_intervals_and_point_doses():
    origin = 7_200.0
    cases = pd.DataFrame({"CaseID": [5], "ICUOffset": [origin], "TimeOfStay": [origin + 72 * 3_600]})
    source = ConceptSource.from_mapping({
        "ids": [1417], "table": "medication", "sub_var": "DrugID", "val_var": "Amount",
        "index_var": "Offset", "callback": "sicdb_active_exposure", "extra_vars": ["OffsetDrugEnd"],
    })
    frame = pd.DataFrame({
        "CaseID": [5, 5, 5, 5],
        # infusion over ICU hours 0.5-3.0; a point dose at 5.25 h; zero and missing amounts.
        "Offset": [origin + 1_800, origin + 18_900, origin + 36_000, origin + 39_600],
        "OffsetDrugEnd": [origin + 10_800, origin + 18_900, origin + 43_200, origin + 43_200],
        "furosemide_active": [0.25, 0.04, 0.0, None],
    })
    result = _apply_callback(frame, source, concept_name="furosemide_active", data_source=_sic(cases))
    aligned = ConceptResolver.__new__(ConceptResolver)._align_time_to_admission(
        result, _sic(cases), ["CaseID"], "Offset")
    hours = sorted(aligned.Offset.floordiv(1).astype(int).unique().tolist())
    assert hours == [0, 1, 2, 5]
    assert aligned.furosemide_active.eq(True).all()


def _events(rows):
    return pd.DataFrame(rows, columns=["stay_id", "_h"]).astype({"_h": float})


def test_coexposure_needs_both_agents_within_24_hours():
    vanco = _events([(1, 0), (1, 30), (2, 0)])
    piptazo = _events([(1, 23), (1, 60), (2, 24)])
    rows = coexposure(vanco, piptazo, ["stay_id"])
    # Stay 1: pip-tazo at 23 h follows vanco at 0 h (23 h < 24); vanco at 30 h follows
    # pip-tazo at 23 h; pip-tazo at 60 h is 30 h after the last vanco. Stay 2: exactly 24 h apart.
    assert rows.values.tolist() == [[1, 23.0], [1, 30.0]]


def _table(rows, value):
    frame = pd.DataFrame(rows, columns=["stay_id", "charttime", value])
    frame["charttime"] = pd.to_timedelta(frame["charttime"], unit="h")
    return ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column=value)


def _ctx():
    return ConceptCallbackContext(concept_name="vancomycin_piperacillin_tazobactam", target=None,
                                  interval=H, resolver=None, data_source=None, patient_ids=None)


def test_coexposure_callback_counts_only_true_administrations():
    tables = {
        "vancomycin_iv": _table([(1, 2, True), (1, 5, False)], "vancomycin_iv"),
        "piperacillin_tazobactam": _table([(1, 6, True), (1, 40, True)], "piperacillin_tazobactam"),
    }
    out = vpt_coexposure_callback(tables, _ctx()).data
    assert out.charttime.tolist() == [6 * H]
    assert out.vancomycin_piperacillin_tazobactam.tolist() == [True]


def test_coexposure_accepts_relative_hours_and_refuses_absolute_times():
    frame = pd.DataFrame({"stay_id": [1, 1], "charttime": [3.0, 50.0], "vancomycin_iv": [True, True]})
    hours = ICUTable(frame, id_columns=["stay_id"], index_column="charttime", value_column="vancomycin_iv")
    other = pd.DataFrame({"stay_id": [1], "charttime": [10.0], "piperacillin_tazobactam": [True]})
    other = ICUTable(other, id_columns=["stay_id"], index_column="charttime", value_column="piperacillin_tazobactam")
    out = vpt_coexposure_callback({"vancomycin_iv": hours, "piperacillin_tazobactam": other}, _ctx()).data
    assert out.charttime.tolist() == [10.0]
    absolute = frame.assign(charttime=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    absolute = ICUTable(absolute, id_columns=["stay_id"], index_column="charttime", value_column="vancomycin_iv")
    with pytest.raises(CoexposureDerivationError, match="coexposure_time_unit_unavailable"):
        vpt_coexposure_callback({"vancomycin_iv": absolute, "piperacillin_tazobactam": other}, _ctx())

