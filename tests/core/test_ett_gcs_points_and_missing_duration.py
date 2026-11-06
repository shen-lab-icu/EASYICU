"""ett_gcs point tables and windows without a duration follow ricu 0.6.3.

ricu ``gcs`` expands the airway table only when it is a ``win_tbl``
(``if (is_win_tbl(sed)) sed <- expand(sed, aggregate = "any")``); a point
table (SICdb and AUMC publish hourly airway points without a duration) is
merged at its own hours, so a pre-admission point never reaches hour zero
(delta4 D4G-POINT). The expected GCS maps are ricu's own output for the same
tables (Rscript, ricu 0.6.3: ``ricu:::gcs`` on ``as_ts_tbl`` components and
an ``as_ts_tbl`` ``ett_gcs``). Synthetic rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.window_expansion import (
    expand_windows_inclusive_any,
    publish_inclusive_any_window,
)
from easyicu.table import ICUTable
from tests.support.gcs_ricu import ID, TIME, airway_points, component, run_gcs

pytestmark = [pytest.mark.ricu_parity, pytest.mark.clinical_conformance]


def _components():
    return (
        component("egcs", [0, 1, 2], [2, 2, 2]),
        component("mgcs", [0, 1, 2], [4, 4, 4]),
        component("vgcs", [0, 1, 2], [1, 1, 1]),
    )


def test_a_point_table_keeps_each_point_in_its_own_hour():
    frame = pd.DataFrame({
        ID: 1, TIME: [-1.5, 1.0, 2.3, 2.6], "ett_gcs": [True, False, True, False],
    })
    cells = expand_windows_inclusive_any(
        frame, idx_col=TIME, id_cols=[ID], value_col="ett_gcs", dur_hours=None,
    )
    assert dict(zip(cells[TIME], cells["ett_gcs"])) == {-2.0: True, 1.0: False, 2.0: True}
    assert cells["ett_gcs"].dtype == bool


def test_the_publisher_never_moves_a_pre_admission_point_to_hour_zero():
    table = airway_points([-2, 1], [True, False])
    published = publish_inclusive_any_window(table, "ett_gcs")
    cells = published.data
    assert dict(zip(cells[TIME], cells["ett_gcs"])) == {-2.0: True, 1.0: False}
    assert isinstance(published, ICUTable)


def test_gcs_with_point_airway_evidence_matches_ricu():
    # ricu 0.6.3: points TRUE at -2 and 2, FALSE at 1 -> {0: 7, 1: 7, 2: 15}.
    got = run_gcs(*_components(), airway_points([-2, 1, 2], [True, False, True]))
    assert got == {0.0: 7.0, 1.0: 7.0, 2.0: 15.0}
