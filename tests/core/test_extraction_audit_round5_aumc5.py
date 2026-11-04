"""Round-5 AUMC regressions, using tiny synthetic source tables only."""

import json
from pathlib import Path

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from tests.support.blood_gas_sources import _source, _write

DATA = Path(easyicu.__file__).parent / "data"
MINUTE = 60_000


def _dictionary(sofa=False):
    return json.loads((DATA / ("sofa2-dict.json" if sofa else "concept-dict.json")).read_text())


def _db(tmp_path, *, origin=17.25, numeric=(), drugs=(), processes=(), lists=(), texts=(), layout="file"):
    _write(pd.DataFrame({
        "admissionid": [1], "patientid": [11], "admittedat": [origin * MINUTE],
        "dischargedat": [(origin + 48 * 60) * MINUTE], "weightgroup": ["80-89"],
    }), tmp_path / "admissions.parquet")
    for table, rows, cols in (
        ("numericitems", numeric, ["measuredat", "itemid", "value", "unit"]),
        ("drugitems", drugs, ["start", "stop", "itemid", "dose", "doseunit", "doserateunit", "iscontinuous", "action", "ordercategory"]),
        ("processitems", processes, ["start", "stop", "itemid", "item"]),
        ("listitems", lists, ["measuredat", "itemid", "value"]),
        ("freetextitems", texts, ["measuredat", "itemid", "value"]),
    ):
        frame = pd.DataFrame(rows, columns=cols)
        frame["admissionid"] = 1
        for col in ("start", "stop", "measuredat"):
            if col in frame:
                frame[col] = (frame[col] + origin) * MINUTE
        if "measuredat" in frame:
            frame["registeredat"] = frame["measuredat"]
        if table == "numericitems":
            frame["tag"] = None
        if table == "drugitems":
            frame["doserateperkg"] = True
            frame["orderid"] = range(len(frame))
            frame["duration"] = (frame["stop"] - frame["start"]) / MINUTE
        path = tmp_path / f"{table}.parquet"
        if layout == "bucket":
            path = tmp_path / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        _write(frame, path)
    return _source(tmp_path, "aumc")


def _load(db, name, *, sofa=False, interval=pd.Timedelta(hours=1)):
    dictionary = ConceptDictionary.from_json(DATA / "concept-dict.json")
    if sofa:
        dictionary.update(ConceptDictionary.from_json(DATA / "sofa2-dict.json"))
    result = ConceptResolver(dictionary).load_concepts(
        [name], db, merge=False, r_compatible=False, interval=interval,
        patient_ids={"admissionid": [1]}, concept_workers=1, verbose=False,
    )[name]
    frame = result.data
    return frame.dropna(subset=[name]).set_index(result.index_column)[name].to_dict()


def _drug(start, stop, item, dose=1.0, unit="mcg", rate="min", continuous=True, action="Toedienen", category="IV"):
    return (start, stop, item, dose, unit, rate, continuous, action, category)


@pytest.mark.parametrize("name,item,unit,rate", [
    ("norepi_rate", 7229, "mcg", "min"), ("epi_rate", 6818, "mcg", "min"),
    ("dopa_rate", 7179, "mcg", "min"), ("dobu_rate", 7178, "mcg", "min"),
    ("phn_rate", 19929, "mcg", "min"), ("propofol_rate", 7480, "mcg", "min"),
    ("fentanyl_rate", 7219, "mcg", "uur"), ("midazolam_rate", 7194, "mg", "uur"),
])
def test_r5c_aumc_003_rate_hours_overlap_the_exact_relative_interval(tmp_path, name, item, unit, rate):
    db = _db(tmp_path, drugs=[_drug(0.25, 120, item, unit=unit, rate=rate)])
    # Both touched ICU hours survive; the exact stop at 2h excludes hour 2.
    if name in {"fentanyl_rate", "midazolam_rate"}:
        frame = pd.read_parquet(tmp_path / "drugitems.parquet")
        frame["doserateperkg"] = False
        frame.to_parquet(tmp_path / "drugitems.parquet", index=False)
    assert _load(db, name) == {0.0: 1.0, 1.0: 1.0}
    if name == "norepi_rate":
        assert _load(db, "norepi60") == {0.0: 1.0, 1.0: 1.0}
        assert _load(db, "vaso_ind") == {0.0: True, 1.0: True}
        assert list(_load(db, "norepi_dur", interval=None).values()) == pytest.approx([119.75 / 60])


def test_r5c_aumc_004_each_bag_and_stay_conserve_d10_volume(tmp_path):
    # Per-minute rate and D40 concentration must be converted BEFORE allocation.
    bags = [_drug(50, 80, 8940, dose=2, unit="ml", rate="min", continuous=False),
            _drug(110, 190, 7255, dose=30, unit="ml", rate="uur", continuous=False)]
    expected = [{0.0: 80, 1.0: 160}, {1.0: 10, 2.0: 60, 3.0: 10}]
    for i, bag in enumerate(bags):
        db = _db(tmp_path / str(i), drugs=[bag])
        got = _load(db, "dex")
        assert got == pytest.approx(expected[i])
        assert sum(got.values()) == pytest.approx([240, 80][i])
    db = _db(tmp_path / "all", drugs=bags + [_drug(10, 20, 8940, dose=999, unit="ml", continuous=False, action="Flush")])
    assert _load(db, "dex") == pytest.approx({0: 80, 1: 170, 2: 60, 3: 10})
    assert sum(_load(db, "dex").values()) == pytest.approx(320)


def test_r5c_aumc_005_set_tidal_volume_preserves_ml_and_rejects_ambiguous_values(tmp_path):
    rows = [(5, 8851, .5, "ml"), (65, 8851, 500, "ml"),
            (125, 8851, 2, "ml"), (185, 8851, 20, "ml"), (245, 12291, 480, "ml")]
    db = _db(tmp_path, numeric=rows)
    assert _load(db, "tidal_vol_set") == {0: 500, 1: 500, 4: 480}


def test_r5c_aumc_006_midazolam_is_positive_iv_administration(tmp_path):
    db = _db(tmp_path, drugs=[_drug(5, 10, 7194), _drug(65, 70, 7194, category="Niet iv Tabletten"),
                             _drug(125, 130, 7194, dose=0)])
    assert _load(db, "midazolam") == {0: True}


@pytest.mark.parametrize("sofa", [False, True])
def test_r5c_aumc_007_negative_circuit_pressure_is_evidence_but_zero_is_unknown(tmp_path, sofa):
    db = _db(tmp_path, numeric=[(5, 7666, -80, "mmHg"), (65, 7666, 0, "mmHg"),
                               (125, 8805, -10, "ml"), (185, 8805, 20, "ml"),
                               (245, 7666, float("inf"), "mmHg"), (305, 7666, None, "mmHg")])
    assert _load(db, "rrt", sofa=sofa) == {0: True, 3: True}
    assert _dictionary(sofa)["rrt"]["aggregate"] == "max"


@pytest.mark.parametrize("name,processes,drugs,lists,expected", [
    ("mech_vent", [(5, 20, 9328, "Beademen"), (40, 121, 9328, "Beademen"),
                   (185, 240, 10740, "Beademen non-invasief")], [], [],
     {0: "invasive", 1: "invasive", 2: "invasive", 3: "noninvasive"}),
    ("ett_gcs", [(40, 121, 12634, "Tube")], [], [(5, 6735, "Geïntubeerd")], {0: True, 1: True, 2: True}),
    ("other_vaso", [], [_drug(5, 20, 19929), _drug(40, 121, 19929), _drug(185, 240, 19929, dose=0)], [], {0: True, 1: True, 2: True}),
])
def test_r5c_aumc_008_windows_keep_longer_same_hour_and_true_endpoints(tmp_path, name, processes, drugs, lists, expected):
    db = _db(tmp_path, processes=processes, drugs=drugs, lists=lists)
    assert _load(db, name, sofa=name == "other_vaso") == expected
    if name == "mech_vent":
        assert _load(db, "vent_ind") == {0: True, 1: True, 2: True}
        assert _load(db, "vent_ind", interval=None) == {0: True, 1: True, 2: True}


def test_r5c_aumc_009_calcium_actual_ph_does_not_pool_corrected_measurements(tmp_path):
    db = _db(tmp_path, numeric=[(5, 9560, 1.1, "mmol/L"), (5, 9561, 1.4, "mmol/L"),
                               (65, 8915, 1.3, "mmol/L")])
    assert _load(db, "cai") == {0: 1.1}
    assert _load(db, "cai_ph74") == {0: 1.4, 1: 1.3}


def test_r5c_aumc_011_unconfirmed_assays_stay_unmapped():
    # Review disposition, not a base-failing regression: no repository item
    # evidence establishes the FEU/DDU basis of historical 10393/12163.
    assert not _dictionary()["d_dimer"]["sources"].get("aumc")


def test_r5c_aumc_012_scvo2_requires_unanimous_central_site_without_extra_specimen(tmp_path):
    numeric = [(5 + 60 * i, 12311 if i % 2 == 0 else 8903, .72 if i % 2 == 0 else 72, "Geen") for i in range(6)]
    lists = [(5, 7432, "Centrale lijn"), (65, 7432, "Centrale lijn"),
             (125, 7432, "Centrale lijn"), (125, 7432, "PAP"), (185, 7432, "PAP"),
             (245, 7432, "Centrale lijn"), (305, 7432, "Perifeer veneus")]
    db = _db(tmp_path, numeric=numeric, lists=lists, texts=[(245, 11646, "Ven.")])
    assert _load(db, "scvo2") == {0: 72, 1: 72}


def test_r5c_aumc_013_chart_fio2_has_only_bedside_channels(tmp_path):
    db = _db(tmp_path, numeric=[(5, 12279, 40, "Geen"), (65, 6699, 35, "Geen"),
                               (125, 20656, 100, "Geen"), (185, 8845, 5, "L/min")])
    assert _load(db, "fio2_chart") == {0: 40, 1: 35}
    for source in _dictionary()["fio2_chart"]["sources"]["aumc"]:
        assert source["chart_channel"] == "set_or_analyzed"


@pytest.mark.parametrize("sofa", [False, True])
def test_r5c_aumc_014_ecmo_process_covers_only_the_half_open_interval(tmp_path, sofa):
    db = _db(tmp_path, processes=[(59.75, 120, 20664, "ECMO")])
    assert _load(db, "ecmo", sofa=sofa) == {0: True, 1: True}


def test_r5c_aumc_016_peep_settings_cannot_override_standby(tmp_path):
    db = _db(tmp_path, numeric=[(5, 12284, 8, "cmH2O"), (65, 12284, 0, "cmH2O"),
                               (125, 12284, 8, "cmH2O"), (185, 12284, 8, "cmH2O")],
             lists=[(5, 12290, "Stand By"), (125, 12290, "PC"),
                    (185, 12290, "Stand By"), (185, 6685, "IPPV")])
    assert _load(db, "adv_resp", sofa=True) == {2: True, 3: True}


@pytest.mark.parametrize("name,primary,secondary,cuff", [("sbp", 6641, 8841, 6678), ("dbp", 6643, 8842, 6680)])
def test_r5c_aumc_017_secondary_arterial_line_is_below_primary_above_cuff(tmp_path, name, primary, secondary, cuff):
    rows = [(5, primary, 100, "mmHg"), (5, secondary, 80, "mmHg"), (5, cuff, 90, "mmHg"),
            (65, secondary, 80, "mmHg"), (65, cuff, 90, "mmHg"),
            (125, secondary, 0, "mmHg"), (125, cuff, 90, "mmHg")]
    assert _load(_db(tmp_path, numeric=rows), name) == {0: 100, 1: 80, 2: 90}


def test_r5c_aumc_019_total_peep_uses_12364_only_as_fallback(tmp_path):
    rows = [(5, 12301, 12, "cmH2O"), (5, 12364, 14, "cmH2O"),
            (65, 12364, 11, "cmH2O"), (125, 8879, 5, "cmH2O")]
    assert _load(_db(tmp_path, numeric=rows), "peep_total") == {0: 12, 1: 11}


def test_r5c_aumc_003_subminute_episode_keeps_both_touched_hours(tmp_path):
    # Both absolute endpoints fall in the same minute. Flooring before
    # subtracting the fractional admission origin destroys this interval.
    db = _db(tmp_path, origin=6_000_000.125, drugs=[_drug(59.9, 60.1, 7229)])
    assert _load(db, "norepi_rate") == {0: 1., 1: 1.}
    assert _load(db, "vaso_ind") == {0: True, 1: True}


def test_r5c_aumc_003_sixty_minute_filter_excludes_a_later_short_episode(tmp_path):
    # An exactly representable two-hour episode exercises equality at stop;
    # fractional durations can round a few nanoseconds below that boundary.
    db = _db(tmp_path, drugs=[_drug(0, 120, 7229), _drug(126, 136, 7229, dose=2)])
    assert _load(db, "norepi_rate") == {0: 1., 1: 1., 2: 2.}
    assert _load(db, "norepi60") == {0: 1., 1: 1.}


def test_r5c_aumc_003_sixty_minute_filter_keeps_fractional_episode_end(tmp_path):
    db = _db(tmp_path, drugs=[_drug(29.9, 120.2, 7229)])
    assert _load(db, "norepi60") == {0: 1., 1: 1., 2: 1.}


def test_r5c_aumc_004_continuous_rate_and_labelled_bag_total_are_distinct():
    from easyicu.concept.aumc_sources import dextrose

    frame = pd.DataFrame({
        "admissionid": [1, 2, 3, 4, 5, 6], "itemid": [8940, 8940, 7255, 7255, 8940, 7255],
        "start": [50., 50., 0., 0., 0., 0.], "stop": [80., 80., 60., 60., 60., 60.],
        "dex": [120., 120., 1., 1., 120., float("inf")], "doseunit": ["ml", "ml", "mg", "ml", "ml", "ml"],
        "doserateunit": ["uur", "uur", "uur", None, "uur", "uur"], "iscontinuous": [False, True, True, True, False, True],
        "administered": [100., 100., 100., 100., 0., 100.], "administeredunit": ["ml"] * 6,
        "action": ["Toedienen"] * 6,
    })
    # Intermittent: use the labelled 100 mL bag, even if the pump rate is
    # inconsistent. Continuous: 120 mL/h for 30 min is 60 mL. D40 is x4.
    result = dextrose(frame, "dex", None).groupby("admissionid").dex.sum()
    assert result.to_dict() == pytest.approx({1: 400, 2: 240})


def test_r5c_aumc_005_tidal_unit_rejections_are_counted():
    from easyicu.concept.aumc_sources import tidal_volume

    out = tidal_volume(pd.DataFrame({"tidal_vol_set": [.5, 500., 2., 20., None]}), "tidal_vol_set")
    assert out.attrs["aumc_tidal_unit_rejected"] == 2


def test_r5c_aumc_016_device_conflict_is_counted_before_hour_aggregation():
    from easyicu.concept.aumc_sources import respiratory_modes

    frame = pd.DataFrame({"admissionid": [1, 1, 1, 1], "measuredat": [5., 5., 5., 65.],
                          "itemid": [12290, 12290, 6685, 12290], "adv_resp": ["Stand By", "PC", "IPPV", "PC"]})
    out = respiratory_modes(frame, "adv_resp")
    assert out.itemid.tolist() == [6685, 12290]
    assert out.attrs["aumc_active_standby_conflicts"] == 1


def test_r5c_aumc_016_distinct_evita_devices_keep_independent_activity():
    from easyicu.concept.aumc_sources import respiratory_modes

    # The dictionary labels 6685 Evita 4 and 9534 Evita 1. A standby
    # record on one device channel cannot erase activity on the other.
    frame = pd.DataFrame({"admissionid": [1, 1], "measuredat": [5., 5.],
                          "itemid": [6685, 9534], "adv_resp": ["Stand By", "IPPV"]})
    out = respiratory_modes(frame, "adv_resp")
    assert out.itemid.tolist() == [9534]
    assert out.attrs["aumc_active_standby_conflicts"] == 1


def test_r5c_aumc_016_explicit_process_is_independent_active_support(tmp_path, caplog):
    db = _db(tmp_path, numeric=[(5, 12284, 8, "cmH2O")],
             lists=[(5, 12290, "Stand By")], processes=[(5, 60, 9328, "Beademen")])
    with caplog.at_level("INFO", logger="easyicu.concept.aumc_sources"):
        assert _load(db, "adv_resp", sofa=True) == {0: True}
    assert "AUMC active/standby mode conflicts at original times: 1" in caplog.text


def test_r5c_aumc_008_positive_window_wins_over_point_counterevidence(tmp_path):
    db = _db(tmp_path, lists=[(5, 6735, "Geïntubeerd"), (65, 6735, "Verward")],
             processes=[(5, 121, 12634, "Tube")])
    resolver = ConceptResolver(ConceptDictionary.from_json(DATA / "concept-dict.json"))
    table = resolver.load_concepts(["ett_gcs"], db, merge=False, r_compatible=False,
                                  interval=pd.Timedelta(hours=1), patient_ids={"admissionid": [1]},
                                  concept_workers=1, verbose=False)["ett_gcs"]
    assert not table.data.duplicated(["admissionid", table.index_column]).any()
    assert table.data.ett_gcs.eq(True).all()
    assert set(table.data[table.index_column]) == {0., 1., 2.}


def test_r5c_aumc_008_airway_assessment_is_a_point_at_the_hour_boundary(tmp_path):
    db = _db(tmp_path, lists=[(59.9, 6735, "Geïntubeerd")])
    assert _load(db, "ett_gcs") == {0: True}


@pytest.mark.parametrize("name,item", [("rocuronium", 6960), ("warfarin", 6802)])
def test_r5c_aumc_027_zero_dose_orders_are_not_exposure(tmp_path, name, item):
    db = _db(tmp_path, drugs=[_drug(5, 10, item), _drug(65, 70, item, dose=0),
                             _drug(125, 130, item, dose=-1)])
    assert _load(db, name) == {0: True}


@pytest.mark.parametrize("name,item,sofa", [("midazolam", 7194, False), ("other_vaso", 19929, True)])
def test_r5c_aumc_006_008_nonfinite_doses_are_not_exposure(tmp_path, name, item, sofa):
    db = _db(tmp_path, drugs=[_drug(5, 10, item), _drug(65, 70, item, dose=float("inf"))])
    assert _load(db, name, sofa=sofa) == {0: True}


def _review_sources(tmp_path, layout):
    return _db(
        tmp_path, layout=layout,
        drugs=[_drug(.25, 120, 7229), _drug(50, 80, 8940, dose=2, unit="ml", continuous=False),
               _drug(5, 10, 7194), _drug(65, 70, 7194, category="Niet iv Tabletten"),
               _drug(5, 20, 19929), _drug(40, 121, 19929)],
        processes=[(5, 20, 9328, "Beademen"), (40, 121, 9328, "Beademen"),
                   (40, 121, 12634, "Tube"), (59.9, 120, 20664, "ECMO")],
        numeric=[(5, 8851, 500., "ml"), (5, 7666, -80., "mmHg"), (65, 7666, 0., "mmHg"),
                 (5, 9560, 1.1, "mmol/L"), (5, 9561, 1.4, "mmol/L"),
                 (125, 12311, .72, "Geen"), (5, 12279, 40., "Geen"),
                 (5, 6641, 100., "mmHg"), (5, 8841, 80., "mmHg"), (65, 8841, 80., "mmHg"),
                 (5, 6643, 60., "mmHg"), (5, 8842, 50., "mmHg"), (65, 8842, 50., "mmHg"),
                 (5, 12301, 12., "cmH2O"), (5, 12364, 14., "cmH2O"), (65, 12364, 11., "cmH2O"),
                 (185, 12284, 8., "cmH2O")],
        lists=[(5, 6735, "Geïntubeerd"), (65, 6735, "Verward"),
               (125, 7432, "Centrale lijn"), (185, 12290, "Stand By")],
    )


REVIEW_EXPECTED = {
    "norepi_rate": {0: 1., 1: 1.}, "norepi60": {0: 1., 1: 1.}, "vaso_ind": {0: True, 1: True},
    "dex": {0: 80., 1: 160.}, "tidal_vol_set": {0: 500.}, "midazolam": {0: True},
    "rrt": {0: True}, "mech_vent": {0: "invasive", 1: "invasive", 2: "invasive"},
    "vent_ind": {0: True, 1: True, 2: True},
    "ett_gcs": {0: True, 1: True, 2: True}, "other_vaso": {0: True, 1: True, 2: True},
    "cai": {0: 1.1}, "cai_ph74": {0: 1.4}, "scvo2": {2: 72.}, "fio2_chart": {0: 40.},
    "ecmo": {0: True, 1: True}, "adv_resp": {0: True, 1: True, 2: True},
    "sbp": {0: 100., 1: 80.}, "dbp": {0: 60., 1: 50.}, "peep_total": {0: 12., 1: 11.},
}


def _assert_values(actual, expected, name):
    if any(isinstance(v, str) for v in expected.values()):
        assert actual == expected, name
    else:
        # DuckDB/native publication stores numeric measurements as float32.
        assert actual == pytest.approx(expected), name


@pytest.mark.parametrize("layout", ["file", "bucket"])
@pytest.mark.parametrize("merged", [False, True])
def test_r5c_aumc_review_sources_match_in_batch_and_bucket_paths(tmp_path, monkeypatch, layout, merged):
    import easyicu.datasource as ds

    db = _review_sources(tmp_path, layout)
    calls = []
    original = ds.load_bucketed_table_multi_aggregated

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(set(args[2]))
        return result

    monkeypatch.setattr(ds, "load_bucketed_table_multi_aggregated", observed)
    dictionary = ConceptDictionary.from_json(DATA / "concept-dict.json")
    dictionary.update(ConceptDictionary.from_json(DATA / "sofa2-dict.json"))
    result = ConceptResolver(dictionary).load_concepts(
        list(REVIEW_EXPECTED), db, merge=merged, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"admissionid": [1]}, concept_workers=1, verbose=False,
    )
    for name, expected in REVIEW_EXPECTED.items():
        frame = result if merged else result[name].data
        time = next(c for c in ["charttime", "start", "measuredat", "measuredat_minutes"] if c in frame) if merged else result[name].index_column
        present = frame.dropna(subset=[name])
        assert not present.duplicated(["admissionid", time]).any(), name
        _assert_values(present.set_index(time)[name].to_dict(), expected, name)
        if not merged:
            _assert_values(_load(db, name, sofa=True), expected, name)
    if layout == "bucket":
        assert any({"cai", "cai_ph74", "fio2_chart"} <= names for names in calls)


def test_r5c_aumc_review_native_publication_preserves_the_audited_values(tmp_path, monkeypatch):
    from easyicu.api import extraction

    db = _review_sources(tmp_path / "source", "bucket")
    # Exercise the real module publication path with only the audited inputs;
    # the adjacent all-module test checks the complete installed module list.
    names = [name for name in REVIEW_EXPECTED if name != "cai_ph74"]
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "vitals", names)
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    result = extraction.extract_database(
        "aumc", data_path=str(db.base_path), output_dir=str(tmp_path / "out"),
        modules=["vitals", "outcome"], patient_ids={"admissionid": [1]}, native_export_v2=True, verbose=False,
    )
    assert not result["modules"]["vitals"].get("errors")
    published = pd.read_parquet(tmp_path / "out/vitals.parquet")
    assert "stay_id" in published and "admissionid" not in published
    for name in names:
        _assert_values(published.dropna(subset=[name]).set_index("charttime")[name].to_dict(), REVIEW_EXPECTED[name], name)


@pytest.mark.parametrize("name,item,values,unit", [
    ("fio2_chart", 12279, (40., 60.), "Geen"),
    ("cai", 9560, (1.1, 1.4), "mmol/L"),
    ("cai_ph74", 9561, (1.2, 1.5), "mmol/L"),
])
def test_r5c_aumc_009_013_dense_points_keep_the_relative_hour(tmp_path, name, item, values, unit):
    # Just over the existing 1000-row optimization threshold; not a benchmark.
    rows = [(59.9, item, values[0], unit)] * 501 + [(60.1, item, values[1], unit)] * 501
    db = _db(tmp_path, numeric=rows)
    assert _load(db, name) == pytest.approx({0: values[0], 1: values[1]})


@pytest.mark.parametrize("sofa,names", [
    (False, ["rrt", "ecmo", "ett_gcs"]), (True, ["rrt", "ecmo", "adv_resp", "other_vaso"]),
])
def test_r5c_aumc_007_008_014_016_logical_pools_declare_positive_evidence_aggregation(sofa, names):
    dictionary = _dictionary(sofa)
    assert all(dictionary[name]["aggregate"] == "max" for name in names)


def test_subconcept_cached_on_parent_grid_does_not_answer_a_coarser_request(tmp_path):
    # SOFA-2 vaso_ind loads other_vaso on its one-minute grid; the requested
    # other_vaso must still come back on the requested hourly grid.
    db = _review_sources(tmp_path, "file")
    dictionary = ConceptDictionary.from_json(DATA / "concept-dict.json")
    dictionary.update(ConceptDictionary.from_json(DATA / "sofa2-dict.json"))
    for order in (["vaso_ind", "other_vaso"], ["other_vaso", "vaso_ind"]):
        result = ConceptResolver(dictionary).load_concepts(
            order, db, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
            patient_ids={"admissionid": [1]}, concept_workers=1, verbose=False,
        )
        assert {name: table.interval for name, table in result.items()} == {
            "vaso_ind": pd.Timedelta(hours=1), "other_vaso": pd.Timedelta(hours=1),
        }, order
