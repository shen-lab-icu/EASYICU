"""eICU vasopressor orders fill in exposure where no infusion is charted.

2026-11-03 extraction audit round 2 (V1): 11,454 of 24,591 eICU stays with a
norepinephrine order have no norepinephrine row in infusiondrug.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.eicu_vasopressor_orders import (
    ORDER_ONLY_STAYS_ATTR,
    order_exposure_durations,
)
from easyicu.resources import load_dictionary


def test_order_span_counts_only_for_stays_without_an_infusion_row():
    orders = pd.DataFrame({
        "patientunitstayid": [1, 2, 2, 3],
        "drugstartoffset": [60, 30, 600, 0],
        "drugstopoffset": [600, 150, 0, 120],
    })

    out = order_exposure_durations(orders, concept_name="norepi_dur", infused_stays={3})

    # Stay 2's second order has no later stop: a zero-hour exposure at start.
    assert out.values.tolist() == [[1, 60.0, 9.0], [2, 30.0, 2.0], [2, 600.0, 0.0]]
    assert out.attrs[ORDER_ONLY_STAYS_ATTR] == 2


def test_missing_order_columns_fail_closed():
    with pytest.raises(ValueError, match="eicu_order_exposure_columns_missing"):
        order_exposure_durations(pd.DataFrame({"patientunitstayid": [1]}),
                                 concept_name="norepi_dur", infused_stays=())


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_norepi_dur_reads_orders_by_name_and_hicl(database):
    sources = [s for s in load_dictionary()["norepi_dur"].sources[database]
               if s.callback == "eicu_order_exposure_duration"]
    assert {s.sub_var for s in sources} == {"drugname", "drughiclseqno"}
    for source in sources:
        assert source.params["infused_regex"] == "(?i)^(?:norepi|levophed)"
        assert source.params["excluded_statuses"] == ["true"]


# Round-12 ruling: order-derived exposure must be distinguishable from charted
# infusions, so an analysis can keep infusion records only (sensitivity).
@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
def test_norepi_order_dur_reads_exactly_the_norepi_dur_order_sources(database):
    dictionary = load_dictionary()
    companion = dictionary["norepi_order_dur"].sources
    assert set(companion) == {"eicu", "eicu_demo"}
    duration = [s for s in dictionary["norepi_dur"].sources[database]
                if s.callback == "eicu_order_exposure_duration"]

    def key(source):
        return (source.sub_var, str(source.params.get("regex") or source.ids), source.callback,
                source.params["infused_regex"], tuple(source.params["excluded_statuses"]))

    assert sorted(map(key, companion[database])) == sorted(map(key, duration))


def test_norepi_order_dur_is_published_with_the_vasopressors():
    from easyicu.api.extraction import EXTRACT_MODULES

    assert "norepi_order_dur" in EXTRACT_MODULES["vasopressors"]


@pytest.mark.parametrize("charted_infusion", [False, True])
def test_norepi_order_dur_repeats_only_order_derived_spans_on_their_rows(tmp_path, charted_infusion):
    from easyicu.concept import ConceptDictionary, ConceptResolver
    from tests.support.eicu_source import DICTIONARY, _eicu_source

    medication = pd.DataFrame({
        "medicationid": [1], "drugname": ["Norepinephrine 4 mg/250 mL"], "drughiclseqno": [2051],
        "drugstartoffset": [90], "drugstopoffset": [390], "drugordercancelled": ["No"],
        "dosage": ["4 mg"], "routeadmin": ["IV"], "frequency": [""], "drugivadmixture": ["Yes"],
        "loadingdose": [""], "prn": ["No"], "gtc": [0], "drugorderoffset": [80],
    })
    infusion = pd.DataFrame({
        "infusiondrugid": [1], "infusionoffset": [120],
        "drugname": ["Norepinephrine (mcg/min)" if charted_infusion else "Propofol (mcg/kg/min)"],
        "drugrate": ["5"], "infusionrate": [""], "drugamount": [""], "volumeoffluid": [""],
        "patientweight": ["80"],
    })
    source = _eicu_source(tmp_path, "eicu", medication=medication, infusiondrug=infusion)
    # Published together, as in the vasopressors module: same rows, same clock.
    merged = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["norepi_dur", "norepi_order_dur"], source, merge=True, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
        concept_workers=1, verbose=False,
    )
    merged = merged.data if hasattr(merged, "data") else merged
    companion = merged.loc[merged["norepi_order_dur"].notna()] if "norepi_order_dur" in merged else merged.iloc[:0]
    if charted_infusion:
        assert companion.empty
    else:
        assert companion[["norepi_dur", "norepi_order_dur"]].values.tolist() == [[5.0, 5.0]]
