"""Root-authored independent combinations; only synthetic writer is reused."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from easyicu.io.arterial_blood_gas import extract_miiv_arterial_blood_gas


def test_root_independent_specimen_pair_and_availability_combinations(tmp_path):
    path = Path(__file__).with_name("test_arterial_blood_gas.py")
    spec = importlib.util.spec_from_file_location("abg_root_synthetic_fixture", path)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    labs = [
        f.lab(50821, 80, native=1),
        f.lab(52033, "ART.", stored=30, native=2),
        f.lab(50816, 0.4, stored=27, native=3),
        f.lab(50821, 120, specimen=901, time=29, stored=30, native=4),
        f.lab(52033, "ART.", specimen=901, time=29, stored=None, native=5),
        f.lab(50816, 0.5, specimen=901, time=29, stored=30, native=6),
        f.lab(50821, 90, specimen=902, time=27, stored=28, native=7),
        f.lab(52033, "ART.", specimen=902, time=27, stored=28, native=8),
        f.lab(52033, "VEN.", specimen=902, time=27, stored=28, native=9),
    ]
    charts = [
        f.chart(0.5, time=24, stored=24.5),
        f.chart(0.6, time=24, stored=26),
        f.chart(0.8, time=25, stored=20),
        f.chart(101.0, time=28, stored=29),
    ]
    # Anchor900 at25 sees25 instead of24; shift that last-valid25 to25.5 to test both ties.
    charts[2]["charttime"] = f.ORIGIN + pd.Timedelta(hours=25.5)
    r = extract_miiv_arterial_blood_gas(
        f.source(tmp_path, labs=labs, charts=charts), allowed_stay_ids=[20]
    )
    assert set(r.events.stay_id) == {20} and len(r.events) == 13 and len(r.po2) == 3
    p = r.pairs.merge(r.po2[["po2_event_key", "specimen_id"]], on="po2_event_key")
    p900 = p[p.specimen_id == 900]
    assert len(p900) == 3
    assert np.allclose(sorted(p900.pafi_mmhg), sorted([200, 160, 100 * 80 / 60]))
    assert p900.available_at.eq(f.ORIGIN + pd.Timedelta(hours=30)).all()
    p901 = p[p.specimen_id == 901]
    assert sorted(p901.pafi_mmhg) == [150.0, 240.0]
    assert p901.available_at.isna().all() and p901.arterial_certified.all()
    assert "STORE_BEFORE_CHART" in ";".join(p901.availability_status)
    p902 = p[p.specimen_id == 902]
    assert p902.pafi_mmhg.isna().all() and not p902.arterial_certified.any()
    assert r.events.loc[r.events.itemid.eq(223835), "converted_value"].eq(101).any()
