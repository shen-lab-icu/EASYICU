"""Round-7 extraction audit (2026-11-05): regressions and residues on 91858ef31."""
import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.resources import load_dictionary


def test_high_frequency_presampling_keeps_the_requested_minimum(tmp_path):
    # R7-F002: >1,000 same-hour HiRID PaO2 rows were pre-sampled with a median,
    # so P/F used 120 instead of the requested hourly minimum 80.
    from tests.support.blood_gas_sources import _source
    from tests.support.hirid_source import hirid_source_root

    times = np.linspace(0.01, 0.99, 1002)
    rows = [{"patientid": 1, "datetime": float(t), "variableid": 20000200, "value": v}
            for t, v in zip(times, np.tile([80.0, 160.0], 501))]
    rows.append({"patientid": 1, "datetime": 0.5, "variableid": 2010, "value": 50.0})
    root = hirid_source_root(tmp_path, "observations", rows)
    loaded = ConceptResolver(load_dictionary()).load_concepts(
        ["pafi"], _source(root, "hirid"), merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientid": [1]}, concept_workers=1, verbose=False,
    )
    assert loaded["pafi"].data.pafi.tolist() == pytest.approx([160.0])


@pytest.mark.parametrize("database", ["mimic", "miiv"])
@pytest.mark.parametrize("names", [["urine"], ["urine", "urine24", "uo_6h", "sbp", "dbp", "hr"]])
def test_all_null_urine_hour_stays_missing_and_true_zero_stays_zero(tmp_path, database, names):
    # R7-F001: an hour whose urine rows all lack a value was published as 0 mL.
    from tests.core.test_extraction_audit_round5b_r6v import _load, _urine_source

    source = _urine_source(tmp_path, database, [1.25, 1.5, 2.25, 3.25], [np.nan, np.nan, 0., 100.])
    table = _load(source, names)["urine"]
    hourly = table.data.set_index(table.index_column).urine
    assert 1. not in hourly.dropna().index
    assert hourly.loc[2.] == 0. and hourly.loc[3.] == 100.


def _counter_total(hours, values, rate="output"):
    from easyicu.concept.hirid_fluid_counters import MAX_OUTPUT_RATE_ML_PER_H, counter_increments

    day = pd.Timestamp("2150-01-01 12:00")
    frame = pd.DataFrame({"patientid": 1, "datetime": day + pd.to_timedelta(hours, unit="h"), "value": values})
    return counter_increments(frame, id_col="patientid", time_col="datetime", value_col="value",
                              max_rate_ml_per_h=MAX_OUTPUT_RATE_ML_PER_H if rate == "output" else rate).volume.sum()


@pytest.mark.parametrize("hours, values, total", [
    # Zero level after the 12:00 restart, an 8.3 L reading 14 min later, then back near zero.
    ([0.0, 0.23, 0.38, 4.0, 23.9], [0., 8305.7, 11.7, 1009.2, 5859.], 5859.),
    # The next reading falls below the pre-spike level but is no restart (>= half of it).
    ([0.0, 9.9, 11.0, 11.5, 13.0, 23.0], [0., 2943.4, 11834.6, 1522.6, 1676.4, 4400.4], 4400.4),
    # A +10 L reading over a 22 L baseline cannot double its neighbors.
    ([0.0, 6.0, 6.13, 6.25, 7.0], [0., 22182.2, 32182.2, 22220., 22381.2], 22381.2),
])
def test_r7_f029_output_counter_reverted_fast_rises_are_spikes(hours, values, total):
    assert _counter_total(hours, values) == pytest.approx(total)
    # Input counters carry no physiological rate bound: the rise still counts there.
    assert _counter_total(hours, values, rate=None) > total


@pytest.mark.parametrize("hours, values, total", [
    # A sustained fast rise is kept.
    ([0.0, 1.0, 1.1, 2.0], [0., 500., 6500., 6600.], 6600.),
    # A genuine restart after a real large increment is kept (next < half the level).
    ([0.0, 2.0, 3.0, 4.0], [0., 3000., 9000., 200.], 9200.),
    # A slow downward correction keeps the running maximum.
    ([0.0, 2.0, 4.0, 6.0], [0., 100., 150., 104.], 150.),
])
def test_r7_f029_output_counter_keeps_sustained_rises_restarts_and_slow_corrections(hours, values, total):
    assert _counter_total(hours, values) == pytest.approx(total)
