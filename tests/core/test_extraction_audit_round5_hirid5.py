"""HiRID round-5 regressions; every source row here is synthetic."""

from tests.support.bucket_storage import write_fixture_parquet

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tests.support.hirid_source import hirid_source_root as _source_root, hirid_load as _load


def _dictionary(filename="concept-dict.json"):
    import easyicu

    return json.loads((Path(easyicu.__file__).parent / "data" / filename).read_text())


def _table(frame, name, time="givenat"):
    from easyicu.table import ICUTable

    return ICUTable(frame, id_columns=["patientid"], index_column=time, value_column=name)


def _context(name, resolver=None):
    from easyicu.concept.callbacks import ConceptCallbackContext

    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1), resolver=resolver,
        data_source=SimpleNamespace(config=SimpleNamespace(name="hirid")), patient_ids=None,
    )


@pytest.mark.parametrize("name,unit,target", [
    ("adh_rate", "U", None),
    ("fentanyl_rate", "mg", "mcg/hour"),
    ("midazolam_rate", "µg", "mg/hour"),
])
def test_r5c_hirid_002_non_weight_rates_split_previous_valid_interval(name, unit, target):
    from easyicu.utils.callback_utils import hirid_rate, hirid_rate_mass

    frame = pd.DataFrame({
        "patientid": 1, "infusionid": 7, "doseunit": unit,
        "givenat": [0.75, 1.25, 2.0, 10.0], "givendose": [0.0, 12.0, 6.0, 3.0],
        "recordstatus": 0,
    })
    kwargs = dict(concept_name=name, val_col="givendose", unit_col="doseunit",
                  grp_var="infusionid", index_col="givenat")
    out = hirid_rate_mass(frame, target_unit=target, **kwargs) if target else hirid_rate(frame, **kwargs)
    scale = {"adh_rate": 1 / 60, "fentanyl_rate": 1000, "midazolam_rate": .001}[name]
    # Zero start markers bound the first interval, and an exact end at 2 h
    # gives no dose to hour 2. Gaps over 6 h stay at their record boundary.
    got = dict(zip(out.givenat, out[name] / scale))
    assert got == pytest.approx({0: 6.0, 1: 12.0, 10: 3.0})
    assert sum(got.values()) == pytest.approx(21.0)


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r5c_hirid_002_public_batch_rates_conserve_dose(tmp_path, layout):
    names = ["adh_rate", "fentanyl_rate", "midazolam_rate"]
    rows = []
    for infusion, name in enumerate(names):
        source = _dictionary()[name]["sources"]["hirid"][0]
        item = source["ids"][0]
        unit = {"adh_rate": "U", "fentanyl_rate": "mg", "midazolam_rate": "µg"}[name]
        rows.extend({"patientid": 1, "givenat": t, "pharmaid": item, "givendose": dose,
                     "doseunit": unit, "route": "cv-inf", "infusionid": infusion, "recordstatus": 0}
                    for t, dose in [(.75, 0.), (1.25, .012), (2., .006), (10., .003)])
    root = _source_root(tmp_path, "pharma", rows, layout)
    out = _load(root, names)
    for name in names:
        scale = {"adh_rate": 1 / 60, "fentanyl_rate": 1000, "midazolam_rate": .001}[name]
        got = dict(zip(out[name].givenat, out[name][name] / scale))
        assert got == pytest.approx({0.: .006, 1.: .012, 10.: .003})
        pd.testing.assert_frame_equal(out[name], _load(root, [name])[name], check_like=True)


@pytest.mark.parametrize("prefix", ["norepi", "epi", "dobu"])
def test_r5c_hirid_003_exact_duration_and_original_half_open_occupancy(prefix):
    from easyicu.concept.callbacks import _callback_vaso60, _callback_vaso_ind
    from easyicu.utils.callback_utils import hirid_duration

    name = f"{prefix}_dur"
    frame = pd.DataFrame({
        "patientid": [1, 1, 2, 2, 3, 3], "infusionid": [7, 7, 8, 8, 9, 9],
        "givenat": [0.75, 2.25, 4.9, 5.1, 7.0, 8.0],
        "givendose": [0., 6., 0., 2., 0., 3.],
        "recordstatus": 0,
    })
    duration = hirid_duration(frame, concept_name=name, val_col="givendose",
                              index_col="givenat", grp_var="infusionid")
    assert duration.givenat.tolist() == pytest.approx([.75, 4.9, 7.0])
    assert duration[name].tolist() == pytest.approx([1.5, .2, 1.0])
    durations = {name: _table(duration, name)}
    occupied = _callback_vaso_ind(durations, _context("vaso_ind")).data
    assert set(zip(occupied.patientid, occupied.givenat)) == {
        (1, 0.), (1, 1.), (1, 2.), (2, 4.), (2, 5.), (3, 7.),
    }
    rate_name = f"{prefix}_rate"
    rates = pd.DataFrame({"patientid": [1, 1, 1, 2, 2, 3, 3],
                          "givenat": [0., 1., 2., 4., 5., 7., 8.], rate_name: .1})
    out = _callback_vaso60({**durations, rate_name: _table(rates, rate_name)},
                           _context(f"{prefix}60")).data
    assert set(zip(out.patientid, out.givenat)) == {(1, 0.), (1, 1.), (1, 2.), (3, 7.)}


@pytest.mark.parametrize("blocked_level", ["noninvasive", None])
def test_r5c_hirid_004_windows_expand_and_explicit_states_break_bridges(tmp_path, blocked_level):
    from easyicu.concept.callbacks import _callback_vent_ind
    from easyicu.table import WinTbl

    mech = WinTbl(
        pd.DataFrame({"patientid": [1, 1], "datetime": [.5, 3.5],
                      "dur_var": [3., 4.], "mech_vent": ["invasive", blocked_level]}),
        id_vars=["patientid"], index_var="datetime", dur_var="dur_var", dur_unit="hours",
    )
    evidence = {
        n: _table(pd.DataFrame({"patientid": [1, 1], "datetime": [5., 8.], n: [8., 8.]}),
                  n, "datetime") for n in ("peep", "etco2", "tidal_vol")
    }
    evidence["vent_mode"] = _table(pd.DataFrame(columns=["patientid", "datetime", "vent_mode"]),
                                    "vent_mode", "datetime")
    resolver = SimpleNamespace(load_concepts=lambda names, *a, **kw: {n: evidence[n] for n in names})
    out = _callback_vent_ind({"mech_vent": mech}, _context("vent_ind", resolver)).data
    # Hour 3 contains both invasive and NIV/no-airway intervals: retain its
    # actual invasive overlap, but neither bridge through NIV nor seed at 8.
    assert out.datetime.tolist() == [0., 1., 2., 3.]
    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": .5, "variableid": 15001552, "value": 1.},
        {"patientid": 1, "datetime": 3.5, "variableid": 15001552,
         "value": 3. if blocked_level else 6.},
        {"patientid": 1, "datetime": 5., "variableid": 2600, "value": 8.},
        {"patientid": 1, "datetime": 5., "variableid": 2200, "value": 35.},
    ])
    # The public loader must preserve exact starts and the missing-airway
    # window as counterevidence, rather than dropping it during projection.
    public = _load(root, ["vent_ind"])["vent_ind"]
    assert public.charttime.tolist() == [0., 1., 2., 3.]


def test_r5c_hirid_004_standby_mode_stops_an_inferred_bridge():
    from easyicu.concept.callbacks import _hirid_bridge_ventilator_evidence

    airway = _table(pd.DataFrame({"patientid": [1], "datetime": [0.], "vent_ind": [True]}),
                     "vent_ind", "datetime")
    loaded = {n: _table(pd.DataFrame({"patientid": [1], "datetime": [5.], n: [8.]}), n, "datetime")
              for n in ("peep", "etco2", "tidal_vol")}
    loaded["vent_mode"] = _table(pd.DataFrame({"patientid": [1, 1], "datetime": [2.5, 6.],
                                               "vent_mode": ["standby", "pressure"]}), "vent_mode", "datetime")
    resolver = SimpleNamespace(load_concepts=lambda names, *a, **kw: {n: loaded[n] for n in names})
    out = _hirid_bridge_ventilator_evidence(airway, _context("vent_ind", resolver), "patientid", "datetime")
    assert out.data.datetime.tolist() == [0.]


def test_r5c_hirid_005_advanced_support_covers_reliable_airway_intervals(tmp_path):
    assert _dictionary("sofa2-dict.json")["adv_resp"]["aggregate"] == "max"
    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": .5, "variableid": 15001552, "value": 1.},
        {"patientid": 1, "datetime": 3.5, "variableid": 15001552, "value": 3.},
        {"patientid": 1, "datetime": 5.5, "variableid": 15001552, "value": 6.},
        {"patientid": 2, "datetime": 1., "variableid": 15001552, "value": 6.},
    ])
    out = _load(root, ["adv_resp"])["adv_resp"]
    supported = out.loc[out.adv_resp.eq(True)]
    assert set(supported.loc[supported.patientid.eq(1), "datetime"]) == {0., 1., 2., 3., 4., 5.}
    assert not out.loc[out.patientid.eq(2), "adv_resp"].eq(False).any()


def test_r5c_hirid_005_support_after_unknown_in_same_hour_is_retained(tmp_path):
    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": .1, "variableid": 15001552, "value": 6.},
        {"patientid": 1, "datetime": .5, "variableid": 15001552, "value": 1.},
        {"patientid": 1, "datetime": 1.5, "variableid": 15001552, "value": 6.},
    ])
    out = _load(root, ["adv_resp", "vent_ind"])
    assert set(out["adv_resp"].loc[out["adv_resp"].adv_resp.eq(True), "datetime"]) == {0., 1.}
    assert set(out["vent_ind"].charttime) == {0., 1.}


@pytest.mark.parametrize("prefix", ["norepi", "epi", "dobu"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r5c_hirid_003_public_load_preserves_episode_start(tmp_path, prefix, layout):
    # Read the actual mapping so this tests temporal plumbing, not an unrelated ID.
    ids = _dictionary()[f"{prefix}_dur"]["sources"]["hirid"][0]["ids"]
    item = ids[0] if isinstance(ids, list) else ids
    rows = [{"patientid": 1, "givenat": t, "pharmaid": item, "givendose": dose,
             "doseunit": "µg", "route": "iv-inf", "infusionid": 7, "recordstatus": 0}
            for t, dose in [(.75, 0.), (2.25, 6.)]]
    root = _source_root(tmp_path, "pharma", rows, layout)
    out = _load(root, [f"{prefix}_dur", "vaso_ind", f"{prefix}60"])
    duration = out[f"{prefix}_dur"]
    assert duration.givenat.tolist() == [.75]
    assert duration[f"{prefix}_dur"].tolist() == [1.5]
    indicator = out["vaso_ind"]
    clock = next(c for c in ("charttime", "givenat", "datetime") if c in indicator)
    assert set(indicator[clock]) == {0., 1., 2.}
    sixty = out[f"{prefix}60"]
    clock = next(c for c in ("charttime", "givenat", "datetime") if c in sixty)
    assert set(sixty[clock]) == {0., 1., 2.}


def test_r5c_hirid_006_isolated_counter_spikes_do_not_raise_peak_or_shift_time():
    from easyicu.concept.hirid_fluid_counters import counter_increments, hourly_counter_volume

    origin = pd.Timestamp("2020-01-01 12:00")
    frame = pd.DataFrame({"patientid": 1, "datetime": origin + pd.to_timedelta([.5, 1., 1.5, 2.], unit="h"),
                          "value": [1000., 10000., 1100., 1200.]})
    kwargs = dict(id_col="patientid", time_col="datetime", value_col="value")
    out = counter_increments(frame, **kwargs)
    assert out.volume.sum() == pytest.approx(1200.)
    clean = frame.drop(index=1)
    actual = hourly_counter_volume(frame, origins=pd.Series({1: origin}), **kwargs)
    expected = hourly_counter_volume(clean, origins=pd.Series({1: origin}), **kwargs)
    pd.testing.assert_frame_equal(actual, expected)
    for values, total in [([1000., 10000., 11000., 12000.], 12000.),
                           ([1000., 1200., 100., 200.], 1400.)]:
        assert counter_increments(frame.assign(value=values), **kwargs).volume.sum() == total


def _drug_recall(tmp_path, name, filename, ids, route):
    declared = _dictionary(filename)[name]["sources"]["hirid"]
    mapped = {i for s in declared for i in (s["ids"] if isinstance(s["ids"], list) else [s["ids"]])}
    assert set(ids) <= mapped
    rows = [{"patientid": 1, "givenat": float(h), "pharmaid": item, "givendose": dose,
             "doseunit": "mg", "route": route, "infusionid": item, "recordstatus": 0}
            for h, (item, dose) in enumerate((i, d) for i in ids for d in (0., 1e-9, 2.))]
    if name in {"abx", "enoxaparin", "magnesium_iv"}:
        rows.append({**rows[-1], "givenat": float(len(rows)),
                     "route": "iv-inf" if name == "enoxaparin" else "po"})
    root = _source_root(tmp_path, "pharma", rows)
    out = _load(root, [name])[name]
    assert set(out.loc[out[name].eq(True), "givenat"]) == {float(h) for h in range(2, len(rows), 3)}
    if name == "other_vaso":
        from easyicu.scores.sofa2 import sofa2_cardio

        assert _dictionary(filename)[name]["aggregate"] == "max"
        scores = sofa2_cardio(pd.Series(75., index=out.index), other_vaso=out[name])
        assert scores.loc[out[name].eq(True)].eq(2).all()
        component = _load(root, ["sofa2_cardio"])["sofa2_cardio"]
        clock = next(c for c in ("charttime", "givenat", "datetime") if c in component)
        by_hour = component.loc[component.patientid.eq(1)].set_index(clock).sofa2_cardio
        assert by_hour.loc[out.loc[out[name].eq(True), "givenat"]].eq(2).all()
    if name == "insulin":
        rate_ids = _dictionary()["ins"]["sources"]["hirid"][0]["ids"]
        assert not set(ids) & set(rate_ids)


def test_r5c_hirid_007_abx_recall_requires_positive_doses(tmp_path):
    _drug_recall(tmp_path, 'abx', 'concept-dict.json', [1000233], 'iv-inf')


def test_r5c_hirid_008_enoxaparin_recall_requires_positive_doses(tmp_path):
    _drug_recall(tmp_path, 'enoxaparin', 'concept-dict.json', [1000451, 1000831, 1000832, 1000833, 1000834], 'sc-inj')


def test_r5c_hirid_009_insulin_recall_requires_positive_doses(tmp_path):
    _drug_recall(tmp_path, 'insulin', 'concept-dict.json', [1000379, 1000381, 1000963], 'sc-inj')


def test_r5c_hirid_010_magnesium_iv_recall_requires_positive_doses(tmp_path):
    _drug_recall(tmp_path, 'magnesium_iv', 'concept-dict.json', [1000070], 'cv-inf')


def test_r5c_hirid_011_other_vaso_recall_requires_positive_doses(tmp_path):
    _drug_recall(tmp_path, 'other_vaso', 'sofa2-dict.json', [1000441, 1001061], 'cv-inf')


def test_r5c_hirid_012_enteral_diltiazem_stays_out_of_the_iv_concept(tmp_path):
    # Round-11 ruling: diltiazem is IV exposure in every database, and HiRID
    # 1001071/1000331 are tablets, so R5C-HIRID-012 is withdrawn for them.
    assert "hirid" not in _dictionary()["diltiazem"]["sources"]
    rows = [{"patientid": 1, "givenat": 1., "pharmaid": item, "givendose": 60., "doseunit": "mg",
             "route": "po", "infusionid": item, "recordstatus": 0} for item in (1001071, 1000331)]
    out = _load(_source_root(tmp_path, "pharma", rows), ["diltiazem"])["diltiazem"]
    assert "diltiazem" not in out or not out["diltiazem"].eq(True).any()


def test_r5c_hirid_013_ckmb_raw_is_separate_from_mass_pool(tmp_path):
    dictionary = _dictionary()
    assert not dictionary["ckmb"]["sources"].get("hirid")
    raw = dictionary["ckmb_hirid_raw"]
    assert raw["unit"] == "U/L"
    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": 1., "variableid": 24000220, "value": 12.3},
    ])
    out = _load(root, ["ckmb_hirid_raw"])["ckmb_hirid_raw"]
    assert out.ckmb_hirid_raw.tolist() == pytest.approx([12.3])


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r5c_hirid_019_status_bits_are_removed_before_dose_and_episode_aggregation(tmp_path, layout):
    from easyicu.utils.callback_utils import hirid_duration, hirid_rate_kg

    frame = pd.DataFrame({
        "patientid": 1, "infusionid": 7, "doseunit": "µg", "weight": 50.,
        "givenat": [.25, .5, 1.25, 2.25, 3.25], "givendose": [0., 99., 12., 99., 99.],
        "recordstatus": [0, 2 | 8, 8, 32 | 8, 2 | 32],
    })
    rate = hirid_rate_kg(frame, concept_name="norepi_rate", val_col="givendose", unit_col="doseunit",
                         index_col="givenat", grp_var="infusionid")
    assert dict(zip(rate.givenat, rate.norepi_rate * 60 * 50)) == pytest.approx({0.: 9., 1.: 3.})
    dur = hirid_duration(frame, concept_name="norepi_dur", val_col="givendose",
                         index_col="givenat", grp_var="infusionid")
    assert dur.givenat.tolist() == [.25]
    assert dur.norepi_dur.tolist() == [1.]
    rows = [{"patientid": 1, "givenat": float(h), "pharmaid": 1000724, "givendose": 2.,
             "doseunit": "units", "route": "iv-inf", "infusionid": 7, "recordstatus": status}
            for h, status in enumerate([0, 2, 32, 34, 8, 1 | 8])]
    out = _load(_source_root(tmp_path, "pharma", rows, layout), ["ins"])["ins"]
    assert set(out.givenat) == {0., 4., 5.}
    assert out.ins.sum() == 6.


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r5c_hirid_019_raw_source_filters_status_before_sql_aggregation(tmp_path, layout):
    # A raw amount definition is a supported public dictionary contract.
    # Status eligibility must also hold without a complex value callback
    # incidentally forcing the loader to bypass SQL aggregation.
    dictionary = _dictionary()
    dictionary["ins"]["aggregate"] = "sum"
    source = dictionary["ins"]["sources"]["hirid"][0]
    source.pop("callback")
    source["val_var"] = "givendose"
    mapping = tmp_path / "raw-ins.json"
    mapping.write_text(json.dumps({"ins": dictionary["ins"]}))
    rows = [{"patientid": 1, "givenat": h / 10, "pharmaid": 1000724, "givendose": 2.,
             "doseunit": "units", "route": "iv-inf", "infusionid": 7, "recordstatus": status}
            for h, status in enumerate([0, 2, 32, 34, 8, 1 | 8])]
    out = _load(_source_root(tmp_path, "pharma", rows, layout), ["ins"], dict_path=mapping)["ins"]
    assert dict(zip(out.givenat, out.ins)) == {0.: 6.}


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r5c_hirid_023_dex_iv_filter_retains_grouped_infusion_callback(tmp_path, layout):
    from easyicu.concept import ConceptSource, _apply_callback

    rows = [{"patientid": 1, "givenat": t, "pharmaid": 1000690, "givendose": dose,
             "doseunit": "ml", "route": route, "infusionid": infusion, "recordstatus": 0}
            for infusion, route, amount in [(7, "iv-inf", 12.), (8, "cv-inf", 24.), (9, "po", 900.)]
            for t, dose in [(.25, 0.), (1.25, amount), (2.25, 0.)]]
    root = _source_root(tmp_path, "pharma", rows, layout)
    out = _load(root, ["dex"])["dex"]
    # Both IV infusion groups must survive the route filter. The enteral
    # group must never enter the amount-to-rate or concentration conversion.
    assert out.dex.notna().any()
    path = root / ("pharma/1.parquet" if layout == "flat" else "pharma_bucket/bucket_id=0/1.parquet")
    raw = pd.read_parquet(path)
    write_fixture_parquet(raw.loc[raw.route.ne("po")], path, index=False)
    iv_only = _load(root, ["dex"])["dex"]
    pd.testing.assert_frame_equal(out, iv_only)
    source = _dictionary()["dex"]["sources"]["hirid"][0]
    assert "route" in source.get("extra_vars", [])
    grouped = _apply_callback(pd.DataFrame(rows).rename(columns={"givendose": "dex"}),
                               ConceptSource.from_mapping(source), "dex",
                               data_source=_context("dex").data_source)
    assert dict(zip(grouped.infusionid, grouped.dex)) == pytest.approx({7: 4., 8: 8.})


def test_r5c_hirid_023_native_medications_ignore_only_non_iv_dex(tmp_path):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database

    rows = [{"patientid": 1, "givenat": t, "pharmaid": 1000690, "givendose": dose,
             "doseunit": "ml", "route": route, "infusionid": infusion, "recordstatus": 0}
            for infusion, route, amount in [(7, "iv-inf", 12.), (8, "cv-inf", 24.), (9, "po", 900.)]
            for t, dose in [(.25, 0.), (1.25, amount), (2.25, 0.)]]
    root = _source_root(tmp_path, "pharma", rows, "bucket", preserve_other=True)

    def medications(label):
        output = tmp_path / label
        try:
            result = extract_database("hirid", data_path=str(root), output_dir=str(output),
                                      modules=["outcome", "medications"], native_export_v2=True, verbose=False)
            assert not result["modules"]["medications"].get("errors")
            return pd.read_parquet(output / "medications.parquet").sort_values(
                ["stay_id", "charttime"]).reset_index(drop=True)
        finally:
            clear_global_loader()

    mixed = medications("mixed")
    path = root / "pharma_bucket/bucket_id=0/1.parquet"
    raw = pd.read_parquet(path)
    write_fixture_parquet(raw.loc[~(raw.pharmaid.eq(1000690) & raw.route.eq("po"))], path, index=False)
    iv_only = medications("iv-only")
    assert mixed.dex.notna().any()
    assert {"abx", "insulin", "enoxaparin"}.issubset(mixed.columns)
    assert mixed[["abx", "insulin", "enoxaparin"]].notna().any().all()
    pd.testing.assert_frame_equal(mixed, iv_only)
