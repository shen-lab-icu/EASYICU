"""The standalone wide single-column SQL refuses a unit-suspect read too (Codex delta5 D5V-WIDE-SINGLE).

``load_concepts`` batches wide-table concepts only for the default median;
an explicit ``min``/``max``/``mean`` on a single-source wide concept (eICU
``pap_sys`` in ``vitalperiodic``) goes to ``_load_single_concept``'s own
wide single-column SQL. That SQL keeps the declared bounds in WHERE, so a
read whose values all contradict the bounds came back empty and the
concept was published as absent, while every other bounded path refuses
it with ``UndeclaredUnitError(bounds_unit_suspect)``. Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.aggregation_route import record_duckdb_routes
from easyicu.utils.declared_units import UndeclaredUnitError
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_N = 150  # stays, one value each: one read holds _N values


def _eicu(root, *, pasystolic=500.0):
    ids = np.arange(1, _N + 1)
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": ids, "patienthealthsystemstayid": ids,
            "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        # pap_sys bounds [0, 120] mmHg: 500 is another unit.
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": ids, "patientunitstayid": ids, "observationoffset": 150,
            "pasystolic": pasystolic, "padiastolic": 15.0, "pamean": 30.0, "etco2": 0.0,
        }),
    )


@pytest.mark.parametrize("aggregate", ["min", "max", "mean"])
def test_an_explicit_aggregate_on_the_wide_single_column_path_refuses(tmp_path, aggregate):
    source = _eicu(tmp_path / "eicu")
    with record_duckdb_routes() as routes:
        with pytest.raises(UndeclaredUnitError) as caught:
            _load(source, ["pap_sys"], aggregate=aggregate)
    assert caught.value.reason == "bounds_unit_suspect"
    assert caught.value.quantity == "pap_sys"
    assert dict(routes) == {"wide_single_column": 1}


@pytest.mark.parametrize("aggregate", ["min", "max", "mean"])
def test_in_bounds_values_still_publish_on_the_wide_single_column_path(tmp_path, aggregate):
    source = _eicu(tmp_path / "eicu", pasystolic=25.0)
    with record_duckdb_routes() as routes:
        out = _load(source, ["pap_sys"], aggregate=aggregate)
    assert dict(routes) == {"wide_single_column": 1}
    assert int(out["pap_sys"].data["pap_sys"].notna().sum()) == _N

