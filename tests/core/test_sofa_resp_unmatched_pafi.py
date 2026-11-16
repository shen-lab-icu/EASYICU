"""sofa_resp scores only hours that have a P/F ratio or ventilation.

ricu's pafi() drops a PaO2 or FiO2 that has no partner, so its sofa_resp has
no row there. EasyICU's internal pafi table keeps such rows with P/F missing
(missing_fio2 provenance); scoring them 0 published respiratory hours that no
P/F supports (paper decomposition 2026-11-16, B1).
"""

import pandas as pd

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sofa_resp
from easyicu.table import ICUTable


def _ctx():
    return ConceptCallbackContext(concept_name="sofa_resp", target=None, interval=pd.Timedelta(hours=1),
                                  resolver=object(), data_source=None, patient_ids=None)


def _run(pafi_rows, vent_rows):
    pafi = ICUTable(pd.DataFrame(pafi_rows, columns=["stay_id", "charttime", "pafi"]),
                    id_columns=["stay_id"], index_column="charttime", value_column="pafi")
    vent = ICUTable(pd.DataFrame(vent_rows, columns=["stay_id", "charttime", "vent_ind"]),
                    id_columns=["stay_id"], index_column="charttime", value_column="vent_ind")
    out = _callback_sofa_resp({"pafi": pafi, "vent_ind": vent}, _ctx()).data
    return {(int(r.stay_id), float(r.charttime)): int(r.sofa_resp) for r in out.itertuples()}


def test_an_unmatched_gas_hour_gets_no_respiratory_score():
    got = _run([(1, 0.0, 150.0), (1, 1.0, None), (1, 2.0, 350.0)], [(1, 0.0, True)])
    assert got == {(1, 0.0): 3, (1, 2.0): 1}


def test_a_ventilated_hour_without_pf_still_scores_zero_as_ricu_does():
    got = _run([(1, 3.0, None)], [(1, 3.0, True)])
    assert got == {(1, 3.0): 0}


def test_scores_with_a_ratio_are_unchanged():
    got = _run([(2, 0.0, 90.0), (2, 1.0, 180.0), (2, 2.0, 450.0)], [])
    assert got == {(2, 0.0): 2, (2, 1.0): 2, (2, 2.0): 0}
