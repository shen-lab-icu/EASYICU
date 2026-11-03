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
