"""Independent composite replay: invented data only, three source channels/timing."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fixtures", Path(__file__).with_name("test_scoped_measurements.py")
)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def test_root_independent_source_and_dual_clock_combinations(tmp_path):
    labs = [
        f.lab_event(50912, 1.2, time=24, stored=30, native=1),
        f.lab_event(52024, 1.8, time=25, stored=26, native=2),
        f.lab_event(52546, 1.6, time=26, stored=None, native=3),
        f.lab_event(50912, 9.9, time=5, stored=6, native=4),
    ]
    charts = [
        f.chart_event(220615, 1.2, time=24, stored=31),
        f.chart_event(229761, 1.8, time=25, stored=25.5),
        f.chart_event(220052, 60, time=26, stored=25),
        f.chart_event(220181, 80, time=26, stored=27),
        f.chart_event(225312, 75, time=26, stored=27),
        f.chart_event(220074, 8, time=26, stored=None),
    ]
    r = f.extract(tmp_path, allowed=[20, 30], labs=labs, charts=charts)
    assert len(r.events) == 9 and set(r.clock_context.stay_id) == {20, 30}
    assert len(r.events.loc[r.events.concept.eq("crea")]) == 5
    assert r.events.event_key.nunique() == 9
    assert r.events.retained_for_analysis.all()
    assert set(r.events.loc[r.events.concept.eq("map"), "converted_value"]) == {
        60,
        75,
        80,
    }
    assert (
        r.events.loc[r.events.source_item_id.eq(220052), "clock_order_status"].item()
        == "store_before_chart"
    )
    assert (
        r.events.loc[r.events.source_item_id.eq(52546), "storetime_status"].item()
        == "missing"
    )
    assert (
        r.events.loc[r.events.source_item_id.eq(220074), "storetime_status"].item()
        == "missing"
    )
    assert 9.9 not in set(r.events.converted_value)
    # A consumer at ICU h=7 sees the newer whole-blood assay, not late old chemistry.
    visible = r.events.loc[
        r.events.concept.eq("crea")
        & r.events.measurement_hours.lt(7)
        & r.events.store_hours.lt(7)
    ]
    assert set(visible.source_item_id) == {52024, 229761}
    assert visible.converted_value.eq(1.8).all()
    # At h=11 the older late lab becomes visible but must not become latest measured.
    visible = r.events.loc[
        r.events.concept.eq("crea")
        & r.events.measurement_hours.lt(11)
        & r.events.store_hours.lt(11)
    ]
    assert (
        visible.loc[
            visible.measurement_hours.eq(visible.measurement_hours.max()),
            "converted_value",
        ]
        .eq(1.8)
        .all()
    )
    # No source-API collapse of MAP disagreement or chart/lab copies.
    assert (
        len(r.events.loc[r.events.measurement_hours.eq(6) & r.events.concept.eq("map")])
        == 3
    )
