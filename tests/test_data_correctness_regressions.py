from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu import api
from easyicu.datasource import load_bucketed_table_aggregated
from easyicu.io.data_converter import ConversionStatus, DataConverter
from easyicu.table import ICUTable


def test_align_to_icu_admission_fails_instead_of_returning_input_unchanged():
    frame = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2026-01-01T00:00:00")]}
    )

    with pytest.raises(NotImplementedError, match="no longer returns unaligned data"):
        api.align_to_icu_admission(frame, verbose=False)

    with pytest.raises(NotImplementedError, match="load_concepts"):
        api.align_to_icu_admission({"hr": frame}, verbose=False)


def test_explicit_empty_cohort_returns_all_requested_special_outputs(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_global_loader",
        lambda **kwargs: pytest.fail("empty cohort must not construct a loader"),
    )

    result = api.load_concepts(
        ["aki", "mort_28d"], patient_ids=[], merge=False, verbose=False
    )

    assert set(result) == {"aki", "mort_28d"}
    assert all(frame.empty for frame in result.values())

    via_alias = api.load_concepts(["hr"], stay_id=[], merge=False, verbose=False)
    assert via_alias["hr"].empty


def test_special_loaders_receive_resolved_source_and_patient_filter(
    monkeypatch, tmp_path
):
    resolved_path = tmp_path / "resolved"
    resolved_path.mkdir()
    loader = SimpleNamespace(database="miiv", data_path=resolved_path)
    monkeypatch.setattr(api, "_get_global_loader", lambda **kwargs: loader)

    calls = {}

    def record(name, frame):
        def _loader(database, data_path=None, **kwargs):
            calls[name] = (database, data_path, kwargs.get("patient_ids"))
            return frame

        return _loader

    import easyicu.scores.circ_failure as circ
    import easyicu.scores.comorbidity as comorbidity
    import easyicu.scores.kdigo_aki as kdigo
    import easyicu.scores.microbiology as microbiology
    import easyicu.scores.outcomes as outcomes

    monkeypatch.setattr(
        kdigo,
        "load_kdigo_aki",
        record("aki", pd.DataFrame({"stay_id": [7], "aki": [1]})),
    )
    monkeypatch.setattr(
        circ,
        "load_circ_failure",
        record("circ", pd.DataFrame({"stay_id": [7], "circ_failure": [1]})),
    )
    monkeypatch.setattr(
        comorbidity,
        "load_comorbidity",
        record("charlson", pd.DataFrame({"stay_id": [7], "charlson_index": [2]})),
    )
    monkeypatch.setattr(
        outcomes,
        "load_outcomes",
        record("outcome", pd.DataFrame({"stay_id": [7], "mort_28d": [False]})),
    )
    monkeypatch.setattr(
        microbiology,
        "load_microbiology",
        record("micro", pd.DataFrame({"stay_id": [7], "culture_positive": [True]})),
    )

    result = api.load_concepts(
        ["aki", "circ_failure", "charlson", "mort_28d", "culture_positive"],
        patient_ids={"stay_id": [7]},
        merge=False,
        concept_workers=1,
        parallel_workers=1,
        verbose=False,
    )

    assert set(result) == {
        "aki",
        "circ_failure",
        "charlson",
        "mort_28d",
        "culture_positive",
    }
    assert set(calls) == {"aki", "circ", "charlson", "outcome", "micro"}
    assert all(call == ("miiv", str(resolved_path), [7]) for call in calls.values())


def test_batched_load_routes_special_concepts_through_subprocess_with_full_list(
    monkeypatch, tmp_path
):
    """Regression for the 2026-07 batching data-loss bug.

    When a request mixes special concepts (KDIGO/CIRC/COMORB/OUTCOME/MICRO) with a
    base concept AND batching triggers, the batched branch used to return before the
    special re-attach block, silently dropping the whole special group. The fix forces
    the subprocess batch path and passes the FULL concept list (specials included) so
    each per-batch child re-runs the dedicated special loaders. This test pins that
    invariant without forking: it captures what concept list reaches subprocess_batch_load.
    """
    import easyicu.runtime.memory_manager as mm

    loader = SimpleNamespace(database="miiv", data_path=tmp_path)
    monkeypatch.setattr(api, "_get_global_loader", lambda **kwargs: loader)

    captured = {}

    def fake_subprocess_batch_load(concepts, **kwargs):
        captured["concepts"] = list(concepts)
        return {c: pd.DataFrame({c: [1]}) for c in concepts}

    monkeypatch.setattr(mm, "subprocess_batch_load", fake_subprocess_batch_load)

    # 200 patients with batch_size 50 -> 4 chunks -> batching triggers.
    result = api.load_concepts(
        ["aki", "hr"],
        patient_ids={"stay_id": list(range(200))},
        batch_size=50,
        merge=False,
        verbose=False,
    )

    assert "concepts" in captured, "batching must route through subprocess_batch_load"
    # The special concept 'aki' must survive into the batched subprocess call.
    assert "aki" in captured["concepts"], (
        "special concept 'aki' was stripped from the batched path — the KDIGO/CIRC/"
        "COMORB/OUTCOME/MICRO drop regression is back"
    )
    assert "hr" in captured["concepts"]
    assert "aki" in result


def test_concept_cache_isolates_cohort_source_and_data_fingerprint(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    source_file = data_path / "stays.parquet"
    source_file.write_bytes(b"v1")
    calls = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"stay_id": kwargs["patient_ids"]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)
    monkeypatch.setattr(
        api,
        "align_to_icu_admission",
        lambda frame, **kwargs: frame.assign(aligned=True),
    )

    first = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[2],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    cached = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )
    source_file.write_bytes(b"version-two")
    os.utime(source_file, None)
    api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        n_patients=3,
        align_time=True,
        verbose=False,
    )

    assert len(calls) == 3
    assert calls[0]["patient_ids"] == [1]
    assert calls[0]["n_patients"] == 3
    assert first["aligned"].all() and cached["aligned"].all()


def test_concept_cache_does_not_truncate_digest_to_collision_prone_prefix(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    calls = []

    def fake_cache_key(concepts, source, **kwargs):
        cohort = kwargs["patient_ids"][0]
        suffix = "1" * 24 if cohort == 1 else "2" * 24
        return "deadbeef" + suffix + "0" * 32

    def fake_load_concepts(**kwargs):
        calls.append(kwargs["patient_ids"])
        return pd.DataFrame({"stay_id": kwargs["patient_ids"]})

    monkeypatch.setattr(api, "_get_cache_key", fake_cache_key)
    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)

    first = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[1],
        verbose=False,
    )
    second = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        patient_ids=[2],
        verbose=False,
    )

    assert calls == [[1], [2]]
    assert first["stay_id"].tolist() == [1]
    assert second["stay_id"].tolist() == [2]


def test_transformed_bounds_are_applied_before_hourly_aggregation(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1],
            "labresultoffset": [0, 10],
            "labname": ["x", "x"],
            "value": [80.0, 10_000.0],
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        _resolve_bucket_directory=lambda name: None,
        _resolve_flat_parquet_directory=lambda name: tmp_path,
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=0,
        value_max=100,
        value_transform='TRY_CAST("value" AS DOUBLE)',
    )

    assert result["value"].tolist() == [80.0]
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 2,
        "bounds_bounded_transformed_non_null": 1,
        "bounds_bounded_aggregate_non_null": 1,
        "bounds_unit_suspect": False,
        "bounds_unbounded_retry": False,
    }


def test_transformed_bounds_retry_unbounded_when_all_values_are_unit_suspect(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 100,
            "labresultoffset": [0] * 100,
            "labname": ["x"] * 100,
            "value": [98.6] * 100,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        _resolve_bucket_directory=lambda name: None,
        _resolve_flat_parquet_directory=lambda name: tmp_path,
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=32,
        value_max=42,
        value_transform='TRY_CAST("value" AS DOUBLE)',
    )

    assert result["value"].tolist() == pytest.approx([98.6])
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 100,
        "bounds_bounded_transformed_non_null": 0,
        "bounds_bounded_aggregate_non_null": 0,
        "bounds_unit_suspect": True,
        "bounds_unbounded_retry": True,
    }


def test_untransformed_bounds_retry_unbounded_when_all_values_are_unit_suspect(
    tmp_path,
):
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 100,
            "labresultoffset": [0] * 100,
            "labname": ["x"] * 100,
            "value": [98.6] * 100,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        _resolve_bucket_directory=lambda name: None,
        _resolve_flat_parquet_directory=lambda name: tmp_path,
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=32,
        value_max=42,
    )

    assert result["value"].tolist() == pytest.approx([98.6])
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 100,
        "bounds_bounded_transformed_non_null": 0,
        "bounds_bounded_aggregate_non_null": 0,
        "bounds_unit_suspect": True,
        "bounds_unbounded_retry": True,
    }


def test_untransformed_bounds_keep_empty_non_unit_suspect_batch(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1] * 99,
            "labresultoffset": [0] * 99,
            "labname": ["x"] * 99,
            "value": [98.6] * 99,
        }
    ).to_parquet(tmp_path / "labs.parquet", index=False)

    defaults = SimpleNamespace(
        index_var="labresultoffset", sub_var="labname", unit_var=None
    )
    config = SimpleNamespace(
        name="eicu", get_table=lambda name: SimpleNamespace(defaults=defaults)
    )
    source = SimpleNamespace(
        config=config,
        base_path=tmp_path,
        _resolve_bucket_directory=lambda name: None,
        _resolve_flat_parquet_directory=lambda name: tmp_path,
        _get_parquet_columns_for_files=lambda files: {
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "value",
        },
    )

    result = load_bucketed_table_aggregated(
        source,
        "labs",
        "value",
        ["x"],
        patient_ids=[1],
        value_min=32,
        value_max=42,
    )

    assert result.empty
    assert result.attrs["easyicu_bounds_loader"] == {
        "bounds_raw_transformed_non_null": 99,
        "bounds_bounded_transformed_non_null": 0,
        "bounds_bounded_aggregate_non_null": 0,
        "bounds_unit_suspect": False,
        "bounds_unbounded_retry": False,
    }


def test_sofa2_overlay_bounds_are_included(monkeypatch):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", None)
    assert api._load_concept_bounds_map()["motor_response"] == (1.0, 6.0)


def test_converter_rejects_partial_shards_after_interrupted_status(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(shard_dir / "1.parquet")
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.CONVERTING,
        "shards": 2,
    }
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "converting" in reason


def test_converter_rejects_valid_looking_orphan_shard_without_completion_status(
    tmp_path,
):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(shard_dir / "1.parquet")
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status.clear()
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "without completed conversion status" in reason


def test_converter_rejects_valid_looking_single_file_without_completion_status(
    tmp_path,
):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status.clear()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "without completed conversion status" in reason


@pytest.mark.parametrize(
    "status",
    [
        ConversionStatus.PENDING,
        ConversionStatus.CONVERTING,
        ConversionStatus.FAILED,
        "in_progress",
    ],
)
def test_converter_rejects_single_file_with_non_completed_status(tmp_path, status):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": status,
        "row_count": 1,
        "shards": 0,
    }

    needed, _ = converter._is_conversion_needed(csv_path)

    assert needed is True


def test_converter_rejects_completed_single_file_with_row_count_mismatch(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        tmp_path / "events.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.COMPLETED,
        "row_count": 2,
        "shards": 0,
    }

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "row-count mismatch" in reason

    converter._status[csv_path.name]["row_count"] = 1
    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is False
    assert reason == "already converted and verified"


def test_converter_rejects_completed_shards_with_row_count_mismatch(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("id,value\n1,2\n2,3\n", encoding="utf-8")
    shard_dir = tmp_path / "events"
    shard_dir.mkdir()
    pd.DataFrame({"id": [1], "value": [2]}).to_parquet(
        shard_dir / "1.parquet", index=False
    )
    pd.DataFrame({"id": [2], "value": [3]}).to_parquet(
        shard_dir / "2.parquet", index=False
    )
    converter = DataConverter(tmp_path, database="miiv", verbose=False)
    converter._status[csv_path.name] = {
        "status": ConversionStatus.COMPLETED,
        "row_count": 3,
        "shards": 2,
    }
    converter._invalidate_dir_caches()

    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is True
    assert "row-count mismatch" in reason

    converter._status[csv_path.name]["row_count"] = 2
    needed, reason = converter._is_conversion_needed(csv_path)

    assert needed is False
    assert reason == "sharded (2 files)"


def test_converter_manifest_keeps_existing_tables_and_counts_bad_rows(tmp_path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("id,value\n1,2\n", encoding="utf-8")
    bad.write_text("id,value\n1,2\n3,4,5\n6,7\n", encoding="utf-8")
    converter = DataConverter(
        tmp_path, database="miiv", parallel_workers=1, verbose=False
    )
    first = converter.convert_all(force=True)
    assert first["bad.csv"]["bad_rows_skipped"] == 1

    good.write_text("id,value\n1,2\n2,3\n", encoding="utf-8")
    os.utime(good, None)
    converter.convert_all()
    manifest = json.loads((tmp_path / "conversion_manifest.json").read_text())
    tables = {row["file"]: row for row in manifest["tables"]}

    assert set(tables) == {"good.csv", "bad.csv"}
    assert tables["bad.csv"]["bad_rows_skipped"] == 1


def test_converter_distinguishes_mimic_generations_and_rejects_ambiguity(tmp_path):
    mimic3 = tmp_path / "mimic-iii"
    mimic4 = tmp_path / "mimic-iv"
    ambiguous = tmp_path / "mimic-dataset"
    for path in (mimic3, mimic4, ambiguous):
        path.mkdir()

    assert DataConverter(mimic3, verbose=False).database == "mimic"
    assert DataConverter(mimic4, verbose=False).database == "miiv"
    with pytest.raises(ValueError, match="Ambiguous MIMIC"):
        DataConverter(ambiguous, verbose=False)


def test_outcomes_preserve_missing_free_days_and_sort_readmissions(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    icu = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "stay_id": [2, 1],
            "intime": ["2020-01-02", "2020-01-01"],
            "los": [None, 2.0],
        }
    )
    patients = pd.DataFrame({"subject_id": [1], "dod": [None]})
    monkeypatch.setattr(
        outcomes,
        "_raw_table",
        lambda database, data_path, table: icu if table == "icustays" else patients,
    )

    result = outcomes.load_outcomes("miiv")
    by_stay = result.set_index("stay_id")

    assert by_stay.loc[1, "icu_free_days_28"] == 26.0
    assert pd.isna(by_stay.loc[2, "icu_free_days_28"])
    assert bool(by_stay.loc[1, "icu_readmission"]) is False
    assert bool(by_stay.loc[2, "icu_readmission"]) is True


def test_missing_eicu_vent_days_are_not_treated_as_28_free_days(monkeypatch):
    import easyicu.scores.outcomes as outcomes

    apache = pd.DataFrame(
        {
            "apacheversion": ["IVa", "IVa", "IVa"],
            "patientunitstayid": [1, 2, 3],
            "actualventdays": [-1, None, 4],
            "actualhospitalmortality": ["ALIVE", "EXPIRED", "ALIVE"],
        }
    )
    monkeypatch.setattr(outcomes, "_raw_table", lambda *args: apache)

    result = outcomes.load_outcomes("eicu").set_index("patientunitstayid")

    assert pd.isna(result.loc[1, "vent_free_days_28"])
    assert result.loc[2, "vent_free_days_28"] == 0
    assert result.loc[3, "vent_free_days_28"] == 24


def test_eicu_microbiology_uses_all_stays_as_denominator(monkeypatch):
    import easyicu.scores.microbiology as microbiology

    patient = pd.DataFrame({"patientunitstayid": [1, 2]})
    microlab = pd.DataFrame(
        {
            "patientunitstayid": [1],
            "organism": ["E. coli"],
            "culturesite": ["Blood"],
        }
    )
    monkeypatch.setattr(microbiology, "_build_datasource", lambda *args: object())
    monkeypatch.setattr(
        microbiology,
        "_table_df",
        lambda ds, table: patient if table == "patient" else microlab,
    )

    result = microbiology.load_microbiology("eicu").set_index("patientunitstayid")

    assert bool(result.loc[1, "culture_positive"]) is True
    assert bool(result.loc[2, "culture_positive"]) is False
    assert bool(result.loc[2, "bld_culture_positive"]) is False


def test_miiv_microbiology_pushes_stay_subset_to_hospital_table(monkeypatch):
    import easyicu.scores.microbiology as microbiology

    captured = []
    stays = pd.DataFrame({"stay_id": [10, 20], "hadm_id": [100, 200]})
    microbiology_events = pd.DataFrame(
        {
            "hadm_id": [100, 200],
            "org_name": ["E. coli", ""],
            "spec_type_desc": ["Blood", "Urine"],
        }
    )

    monkeypatch.setattr(microbiology, "_build_datasource", lambda *args: object())

    def fake_table(_ds, table, columns=None, filters=None):
        if table == "icustays":
            return stays
        captured.extend(filters or [])
        assert table == "microbiologyevents"
        return microbiology_events[microbiology_events["hadm_id"].isin([100])]

    monkeypatch.setattr(microbiology, "_table_df", fake_table)

    result = microbiology.load_microbiology("miiv", patient_ids=[10])

    assert result["stay_id"].tolist() == [10]
    assert bool(result["culture_positive"].iloc[0]) is True
    assert len(captured) == 1
    assert captured[0].column == "hadm_id"
    assert captured[0].value == [100]


def test_icu_table_to_wide_sets_column_axis_name():
    table = ICUTable(
        pd.DataFrame({"stay_id": [1, 1], "hour": [0, 1], "value": [2.0, 3.0]}),
        id_columns=["stay_id"],
        index_column="hour",
        value_column="value",
    )

    wide = table.to_wide("hr")

    assert wide.columns.name == "hr"
    assert wide.loc[1].tolist() == [2.0, 3.0]
