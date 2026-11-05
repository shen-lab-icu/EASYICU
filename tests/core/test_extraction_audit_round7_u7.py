"""Round-7 urine receipts and breakpoints, using invented records only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from easyicu.concept import ConceptResolver, callbacks
from easyicu.databases.time_origin import native_export_time_origin
from easyicu.resources import load_dictionary
from easyicu.scores.aki_profiles import apply_reference_aki, build_renal_aki_bundle
from easyicu.scores.kdigo_aki import _calc_aki_stage_uo, kdigo_stages, kdigo_uo
from easyicu.scores.urine_output import retain_availability_lag, source_availability
from easyicu.scores.urine_windows import assess_urine_windows
from easyicu.table import ICUTable
from tests.support.urine_clock_source import urine_clock_source


DATABASES = ("miiv", "mimic", "eicu", "sic", "aumc", "hirid")
WINDOWS = (6, 12, 24)
HOUR = pd.Timedelta(hours=1)


def _table(frame, name, id_column="stay_id"):
    return ICUTable(frame, id_columns=[id_column], index_column="charttime", value_column=name)


def _context(database, name):
    return callbacks.ConceptCallbackContext(
        name, None, HOUR, None, SimpleNamespace(config=SimpleNamespace(name=database)), None,
    )


@pytest.mark.parametrize("axis", ["datetime", "mixed_object"])
def test_r7_f005_seconds_survive_alignment_and_contributing_receipts(axis):
    origin = pd.Timestamp("2200-01-01 00:00:17")
    # Seconds on either side of hour boundaries, two channels in hour 20,
    # nonmonotone entry times, and source rows deliberately out of chart order.
    hours = np.r_[np.arange(25.) + 59 / 3600, 20 + 1 / 3600]
    entries = hours + 2 / 3600
    entries[20] = 50 + 37 / 3600
    raw = pd.DataFrame({
        "icustay_id": 1, "charttime": origin + pd.to_timedelta(hours, unit="h"),
        "storetime": origin + pd.to_timedelta(entries, unit="h"), "urine": 25.,
    }).sample(frac=1, random_state=7).reset_index(drop=True)
    expected_chart = (raw.charttime - origin) / HOUR
    expected_entry = (raw.storetime - origin) / HOUR
    retain_availability_lag(raw, "mimic", "charttime")
    if axis == "mixed_object":
        raw["charttime"] = raw.charttime.astype(object)
        raw.loc[::2, "charttime"] = expected_chart.loc[::2]
    admissions = pd.DataFrame({
        "icustay_id": [1], "intime": [origin], "outtime": [origin + 100 * HOUR], "los": [100 / 24],
    })
    source = SimpleNamespace(config=SimpleNamespace(name="mimic", id_configs={}))
    resolver = ConceptResolver(load_dictionary())
    resolver._icustays_cache = admissions
    aligned = resolver._align_time_to_admission(raw, source, ["icustay_id"], "charttime")
    # Absolute one-second bound; a tolerance proportional to LOS is unsafe.
    assert np.max(np.abs(aligned.charttime.to_numpy() - expected_chart.to_numpy())) * 3600 <= 1
    if axis == "datetime":
        assert np.max(np.abs(aligned.storetime.to_numpy() - expected_entry.to_numpy())) * 3600 <= 1
    receipts = source_availability(aligned, ["icustay_id"], "charttime", HOUR)
    aligned["charttime"] = np.floor(aligned.charttime)
    urine = aligned.groupby(["icustay_id", "charttime"], as_index=False).urine.sum().merge(
        receipts, on=["icustay_id", "charttime"], validate="one_to_one",
    )
    urine = _table(urine, "urine", "icustay_id")
    weight = pd.DataFrame({"icustay_id": [1], "weight": [70.]})
    tables = {"urine": urine, "weight": ICUTable(weight, id_columns=["icustay_id"], value_column="weight")}
    latest = entries.max()
    for window in WINDOWS:
        name = f"uo_{window}h"
        row = callbacks._callback_uo_window(tables.copy(), _context("mimic", name), window, name).data.iloc[-1]
        for suffix in ("_available_at", "_assessment_available_at"):
            available = row[f"{name}{suffix}"]
            assert abs(available - latest) * 3600 <= 1
            assert not available <= latest - 2 / 3600
    daily = callbacks._callback_urine24(tables.copy(), _context("mimic", "urine24")).data.iloc[-1]
    assert abs(daily.urine24_available_at - latest) * 3600 <= 1
    assert daily.urine24_uses_future
    bundle = build_renal_aki_bundle(
        "mimic", urine_df=urine.data, weight_df=weight,
        id_col="icustay_id", time_col="charttime", time_unit="hours", interval=HOUR,
    ).iloc[-1]
    assert abs(bundle.aki_stage_uo_reference_available_at - latest) * 3600 <= 1
    assert not bundle.aki_stage_uo_reference_available_at <= latest - 2 / 3600


@pytest.mark.parametrize("database", DATABASES)
def test_r7_f006_native_assessment_columns_replay_the_same_owner(database, tmp_path, monkeypatch):
    import tempfile

    from easyicu.api import concepts as api
    from easyicu.api.extraction import extract_database
    from easyicu.concept.metadata_sidecar import parse_column_metadata_sidecar

    root, output = tmp_path / "source", tmp_path / "out"
    urine_clock_source(root, database)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        monkeypatch.delenv(name, raising=False)
    names = [f"uo_{window}h" for window in WINDOWS]
    try:
        loaded = api.load_concepts(
            ["urine", "weight", *names], database=database, data_path=root,
            merge=False, verbose=False, concept_workers=1, parallel_workers=1,
        )
        urine = loaded["urine"]
        owner = assess_urine_windows(
            urine.data, loaded["weight"].data, id_columns=urine.id_columns,
            time_column=urine.index_column, interval=HOUR, source_is_rate=database == "hirid",
        )
        result = extract_database(
            database, data_path=str(root), output_dir=str(output), modules=["outcome", "renal"],
            native_export_v2=True, verbose=False,
        )
        assert not {m: r["errors"] for m, r in result["modules"].items() if r.get("errors")}
        published = pd.read_parquet(output / "renal.parquet")
        schema = pq.read_schema(output / "renal.parquet")
        manifest = json.loads((output / "_manifest.json").read_text())
        sidecar = parse_column_metadata_sidecar((output / manifest["column_metadata"]["file"]).read_bytes())
        metadata = next(file.columns for file in sidecar.files if file.relative_path == "renal.parquet")
        keys = ["stay_id", "charttime"]
        expected = owner.rename(columns={urine.id_columns[0]: "stay_id", urine.index_column: "charttime"})
        expected = expected.set_index(keys).sort_index()
        actual = published.set_index(keys).sort_index().reindex(expected.index)
        for window in WINDOWS:
            name = f"uo_{window}h"
            for suffix, owner_suffix in (
                ("assessment_rate", "assessment_rate"),
                ("assessment_covered_h", "covered_h"),
                ("assessment_available_at", "assessment_available_at"),
            ):
                column = f"{name}_{suffix}"
                assert column in schema.names
                assert column in metadata
                binding = metadata[column]
                assert binding.metadata.source_concept == name
                assert binding.derivation_window.start_hours == -window
                assert binding.derivation_window.end_hours == 0
                if suffix == "assessment_available_at":
                    assert binding.metadata.time_unit == "h"
                    assert binding.metadata.time_origin == native_export_time_origin(database)
                else:
                    assert binding.metadata.canonical_unit == ("h" if suffix == "assessment_covered_h" else "mL/kg/h")
                    assert binding.metadata.extraction_bounds.minimum == 0
                    assert binding.metadata.extraction_bounds.maximum == (window if suffix == "assessment_covered_h" else None)
                wanted = expected[f"{name}_{owner_suffix}"]
                np.testing.assert_array_equal(actual[column].isna(), wanted.isna())
                np.testing.assert_allclose(actual[column], wanted, rtol=1e-6, atol=1e-6, equal_nan=True)
            # A descriptive partial window remains different from a complete
            # assessment; preserving only the former is not an alias contract.
            partial = actual[f"{name}_assessment_covered_h"].between(0, window, inclusive="neither")
            assert partial.any()
            assert actual.loc[partial, f"{name}_assessment_rate"].isna().all()
    finally:
        api.clear_global_loader()


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("case", ["nan", "negative", "mixed_nan", "mixed_negative", "gap", "zero"])
@pytest.mark.parametrize("rate_source", [False, True])
def test_r7_f007_invalid_records_remain_collection_breakpoints(database, case, rate_source):
    times = [0., 4., 17., 21.] if case == "gap" else [0., 4., 8., 12.]
    amounts = [0., 28., 0., 28.]
    if case == "zero":
        amounts = [0.] * 4
    elif case in {"nan", "mixed_nan"}:
        amounts[2] = np.nan
    elif case in {"negative", "mixed_negative"}:
        amounts[2] = -1.
    if rate_source:
        amounts = np.asarray(amounts) / 4
    urine = pd.DataFrame({"stay_id": 1, "charttime": times, "urine": amounts})
    if case.startswith("mixed_"):
        urine = pd.concat([urine, urine.iloc[[2]].assign(urine=28.)], ignore_index=True)
    urine = urine.sample(frac=1, random_state=7)
    urine["_urine_available_at"] = urine.charttime + 37 / 3600
    weight = pd.DataFrame({"stay_id": [1], "weight": [70.]})
    expected = assess_urine_windows(
        urine, weight, id_columns=["stay_id"], time_column="charttime", interval=HOUR,
        source_is_rate=rate_source,
    ).set_index("charttime")
    actual = kdigo_uo(
        urine, weight, id_col="stay_id", time_col="charttime", time_unit="hours",
        interval=HOUR, source_is_rate=rate_source,
    ).set_index("charttime")
    pd.testing.assert_index_equal(actual.index, expected.index)
    for window in WINDOWS:
        np.testing.assert_allclose(actual[f"uo_rt_{window}hr_covered_h"], expected[f"uo_{window}h_covered_h"])
        np.testing.assert_allclose(
            actual[f"uo_rt_{window}hr"], expected[f"uo_{window}h_assessment_rate"], equal_nan=True,
        )
    stages = _calc_aki_stage_uo(*(expected[f"uo_{window}h_assessment_rate"] for window in WINDOWS))
    pd.testing.assert_series_equal(actual.aki_stage_uo, stages, check_names=False)
    kwargs = dict(
        urine_df=urine, weight_df=weight, id_col="stay_id", time_col="charttime",
        urine_source_is_rate=rate_source, time_unit="hours", interval=HOUR,
    )
    # pandas 3 keeps the AKI owners' time key as object; compare values, not index dtype.
    reference = apply_reference_aki(**kwargs).set_index("charttime")
    quality = kdigo_stages(**kwargs).set_index("charttime")
    pd.testing.assert_series_equal(reference.aki_stage_uo_reference, stages, check_names=False, check_index_type=False)
    pd.testing.assert_series_equal(quality.aki_stage_uo, stages, check_names=False, check_index_type=False)
    # Raw duplicate channels test the harmonized reference and quality owners.
    # Source-native profiles have separate, frozen channel rules (R8).
    if not case.startswith("mixed_"):
        bundle = build_renal_aki_bundle(database, **kwargs).set_index("charttime")
        pd.testing.assert_series_equal(bundle.aki_stage_uo_reference, stages, check_names=False, check_index_type=False)
    if case == "zero":
        assert actual.iloc[-1].uo_rt_6hr == 0
        assert actual.iloc[-1].aki_stage_uo == 3
    else:
        assert actual.iloc[-1].uo_rt_6hr_covered_h == 4
        assert pd.isna(actual.iloc[-1].uo_rt_6hr)
        assert pd.isna(actual.iloc[-1].aki_stage_uo)
        assert pd.isna(quality.iloc[-1].aki_stage_uo_available_at)


def test_r7_f024_retrospective_values_are_gated_by_actual_availability():
    urine = _table(pd.DataFrame({
        "stay_id": 1, "charttime": np.arange(25.), "urine": 100.,
        "_urine_available_at": np.r_[np.arange(24.), 30 + 59 / 3600],
    }), "urine")
    row = callbacks._callback_urine24({"urine": urine}, _context("miiv", "urine24")).data.iloc[-1]
    assert row.urine24 == 2400
    assert row.urine24_uses_future
    assert row.urine24_available_at == 30 + 59 / 3600
    assert not row.urine24_available_at <= 24
    assert row.urine24_available_at <= 30 + 59 / 3600


def test_r7_f025_float32_clock_warning_uses_absolute_seconds():
    source = np.array([1000 + 59 / 3600, 2048 + 37 / 3600, 4096 + 11 / 3600, 7000 + 17 / 3600])
    stored = source.astype("float32").astype("float64")
    early_seconds = (source - stored) * 3600
    assert (early_seconds > .036).any()
    assert not (early_seconds > 1).any()
    np.testing.assert_array_equal(stored.astype("float32"), source.astype("float32"))


@pytest.mark.parametrize("database", ["eicu", "sic"])
def test_r7_f026_gap_partial_values_use_only_accepted_historical_intervals(database):
    urine = _table(pd.DataFrame({
        "stay_id": 1, "charttime": [*np.arange(25.), 44.], "urine": [*[100.] * 25, 99999.],
    }), "urine")
    daily = callbacks._callback_urine24({"urine": urine}, _context(database, "urine24")).data.set_index("charttime")
    gap = daily.loc[25.:]
    assert gap.urine24.isna().all()
    partial = gap.loc[gap.urine24_est.notna()]
    assert not partial.empty
    assert partial.urine24_covered_h.between(18, 24, inclusive="left").all()
    np.testing.assert_allclose(partial.urine24_est, 2400.)
    observed = gap.loc[gap.urine24_observed_ml.notna()]
    np.testing.assert_allclose(observed.urine24_observed_ml, observed.urine24_covered_h * 100)
    assert gap.loc[37.:, "urine24_est"].isna().all()
    assert gap.loc[44., "urine24_covered_h"] == 0
