"""First-load Parquet work stays in SQL; complete payloads match pandas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.webserver import cohort_review
from tests.support.cohort_export import write_cohort_export
from tests.support.cohort_reference import install_reference


def _run(monkeypatch, path, desc, selected=None):
    monkeypatch.setattr(cohort_review, "_SUMMARY_CACHE", {})
    monkeypatch.setattr(
        cohort_review,
        "_resolve_registered_source",
        lambda body: ({"id": "synthetic", "path": str(path)}, desc),
    )
    return cohort_review.cohort_review_summary(
        {} if selected is None else {"selected_features": selected}
    )


def test_summary_never_materializes_parquet_modules(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    original = pd.read_parquet

    def bounded(path, *args, **kwargs):
        assert Path(path).name == "demographics.parquet", (
            "time-indexed modules must aggregate in DuckDB"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", bounded)
    payload = _run(monkeypatch, tmp_path, desc)
    assert payload["summary"]["cohort_size"] == 24
    assert payload["sofa_reclassification"]["status"] == "ready"
    assert all(
        row["kind"] != "blocked" for row in payload["selected_feature_distributions"]
    )
    assert all(row["covered_entities"] == 24 for row in payload["coverage"])


@pytest.mark.parametrize(
    "case",
    [
        "regular",
        "empty",
        "all_null",
        "duplicates",
        "csv",
        "missing",
        "string_ids",
        "float_ids",
    ],
)
@pytest.mark.parametrize("selection", ["default", "all"])
def test_complete_payload_matches_original_readers(
    tmp_path, monkeypatch, case, selection
):
    desc = write_cohort_export(tmp_path, case=case)
    selected = (
        None
        if selection == "default"
        else [
            f"{entry['module']}:{column}"
            for entry in desc["files"]
            for column in entry["concept_ids"]
        ]
    )
    with monkeypatch.context() as reference:
        install_reference(reference)
        expected = _run(reference, tmp_path, desc, selected)
    actual = _run(monkeypatch, tmp_path, desc, selected)
    assert actual == expected
    assert json.dumps(actual, ensure_ascii=False, allow_nan=False) == json.dumps(
        expected, ensure_ascii=False, allow_nan=False
    )


def _replace_module(root, desc, module, frame):
    item = next(entry for entry in desc["files"] if entry["module"] == module)
    frame.to_parquet(root / item["file"], index=False)
    item["columns"] = list(frame.columns)
    item["rows"] = len(frame)
    item["concept_ids"] = [
        column for column in frame if column not in {"stay_id", "charttime"}
    ]


def _assert_parity(monkeypatch, root, desc, selected=None):
    with monkeypatch.context() as reference:
        install_reference(reference)
        expected = _run(reference, root, desc, selected)
    actual = _run(monkeypatch, root, desc, selected)
    assert actual == expected
    assert json.dumps(actual, ensure_ascii=False, allow_nan=False) == json.dumps(
        expected, ensure_ascii=False, allow_nan=False
    )
    return actual


@pytest.mark.parametrize("entity_column", ["patientunitstayid", "icustay_id"])
def test_source_identifier_aliases_keep_complete_payload(
    tmp_path, monkeypatch, entity_column
):
    desc = write_cohort_export(tmp_path, entity_column=entity_column)
    _assert_parity(monkeypatch, tmp_path, desc)


@pytest.mark.parametrize("dtype", ["float32", "float64", "string", "Int64"])
def test_numeric_conversions_and_rounding_match_pandas(tmp_path, monkeypatch, dtype):
    desc = write_cohort_export(tmp_path)
    values = (
        [2, 5, None, 0, 9, 7, 3, 6]
        if dtype == "Int64"
        else [2.1245, 5.0055, None, 0.0005, 9.1255, -7.0005, 3.3335, 6.0065]
    )
    frame = pd.DataFrame(
        {"stay_id": [1, 1, 2, 2, 3, 3, 4, 4], "lact": pd.Series(values, dtype=dtype)}
    )
    if dtype == "string":
        frame.loc[2, "lact"] = "1_0"  # TRY_CAST and pandas use different grammars.
    _replace_module(tmp_path, desc, "blood_gas", frame)
    _assert_parity(monkeypatch, tmp_path, desc, ["blood_gas:lact"])


def test_boolean_unknowns_presence_and_survival_reductions(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "death": ["\u00a0yes\u3000", "no", None, "?", "0.0", "false", "1.0", "TRUE"],
            "icu_death": pd.Series(
                [False, True, None, None, False, False, True, True], dtype="boolean"
            ),
            "mort_28d": [True, True, None, False, False, False, True, True],
            "los_icu": [1.1245, 9.1255, 4.0, 8.0, None, 3.0, 1.0, 2.0],
            "los_hosp": [4.0, 12.0, 5.0, 15.0, 1.0, 8.0, None, 9.0],
            "followup_days_28d": [5.0, 40.0, 28.0, 28.0, 2.0, 27.0, 1.0, 8.0],
        }
    )
    _replace_module(tmp_path, desc, "outcome", frame)
    payload = _assert_parity(
        monkeypatch,
        tmp_path,
        desc,
        ["outcome:death", "outcome:icu_death", "outcome:los_icu"],
    )
    assert payload["summary"]["mortality"]["unknown_count"] == 21
    assert (
        payload["survival_analysis"]["outcomes"][2]["event_summary"][
            "excluded_inconsistent_entities"
        ]
        == 2
    )


def test_kind_inference_uses_first_5000_non_null_in_file_order(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2] * 3001,
            "lact": [None] * 1000 + [0.0, 1.0] * 2500 + [9.0, 10.0],
        }
    )
    _replace_module(tmp_path, desc, "blood_gas", frame)
    with monkeypatch.context() as reference:
        install_reference(reference)
        expected = _run(reference, tmp_path, desc, ["blood_gas:lact"])
    original = cohort_review._infer_feature_kind

    def bounded_sample(frame, column):
        assert len(frame[column].dropna()) <= 5000
        return original(frame, column)

    monkeypatch.setattr(cohort_review, "_infer_feature_kind", bounded_sample)
    payload = _run(monkeypatch, tmp_path, desc, ["blood_gas:lact"])
    assert json.dumps(payload, ensure_ascii=False, allow_nan=False) == json.dumps(
        expected, ensure_ascii=False, allow_nan=False
    )
    assert payload["selected_feature_distributions"][0]["kind"] == "binary"


def test_empty_feature_selection_and_cap_are_unchanged(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    _assert_parity(monkeypatch, tmp_path, desc, [])
    features = [f"vitals:synthetic_{index}" for index in range(55)]
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            **{feature.split(":")[1]: [3.0, 9.0, 5.0] for feature in features},
        }
    )
    _replace_module(tmp_path, desc, "vitals", frame)
    payload = _assert_parity(monkeypatch, tmp_path, desc, features)
    assert len(payload["selected_feature_distributions"]) == 48


def test_generator_is_a_verified_native_v2_package(tmp_path):
    from easyicu.research_agent.intake.export_package import open_export_package

    desc = write_cohort_export(tmp_path, batch_rows=11)
    package = open_export_package(tmp_path)
    assert package.database == "eicu"
    assert package.column_metadata_sha256
    assert len(package.files) == 19
    assert desc["summary"]["total_rows"] == sum(
        entry["rows"] for entry in desc["files"]
    )


@pytest.mark.parametrize("nullable", [False, True])
def test_large_identifiers_keep_coverage_normalization(tmp_path, monkeypatch, nullable):
    desc = write_cohort_export(tmp_path)
    ids = [2**53 + 1, 2**53 + 2, None] if nullable else [2**53 + 1, 2**53 + 2]
    # Nullable extension integers differ from the default Arrow-to-pandas float
    # representation. Both must preserve the original normalization boundary.
    series = pd.Series(ids, dtype="Int64" if nullable else "int64")
    for item in desc["files"]:
        module_ids = series.copy()
        if item["module"] == "vitals":
            # These distinct physical keys share the original float-normalized
            # spelling. Coverage must compare canonical keys across files.
            module_ids.iloc[0] = 2**53
        frame = pd.DataFrame(
            {"stay_id": module_ids, item["concept_ids"][0]: range(len(ids))}
        )
        _replace_module(tmp_path, desc, item["module"], frame)
    _assert_parity(monkeypatch, tmp_path, desc, ["blood_gas:lact"])


def test_feature_coverage_reuses_only_the_same_cohort(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path, case="missing")
    original = cohort_review._covered_entities
    selected_modules = {
        "blood_gas",
        "vitals",
        "respiratory",
        "chemistry",
        "hematology",
        "renal",
        "vasopressors",
        "ventilator",
    }

    def no_duplicate_scan(path, item, ids):
        assert item["module"] not in selected_modules, (
            "selected-feature stay groups should supply coverage"
        )
        return original(path, item, ids)

    monkeypatch.setattr(cohort_review, "_covered_entities", no_duplicate_scan)
    payload = _run(monkeypatch, tmp_path, desc)
    coverage = {row["module"]: row for row in payload["coverage"]}
    assert coverage["vitals"]["covered_entities"] == 18
    assert coverage["vitals"]["coverage_pct"] == 75.0
    monkeypatch.setattr(cohort_review, "_covered_entities", original)
    result = cohort_review._coverage_payload(
        tmp_path,
        desc,
        precomputed=({"another-cohort"}, {str(tmp_path / "vitals.parquet"): 999}),
    )
    assert (
        next(row for row in result if row["module"] == "vitals")["covered_entities"]
        == 18
    )


def test_nan_observations_are_missing_before_stay_medians(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    desc = write_cohort_export(tmp_path)
    _replace_module(
        tmp_path,
        desc,
        "blood_gas",
        pd.DataFrame({"stay_id": [1, 1, 2, 2], "lact": [0.0] * 4}),
    )
    pq.write_table(
        pa.table(
            {
                "stay_id": [1, 1, 2, 2],
                "lact": pa.array(
                    [float("nan"), 3.0, 9.0, float("nan")], from_pandas=False
                ),
            }
        ),
        tmp_path / "blood_gas.parquet",
    )
    _assert_parity(monkeypatch, tmp_path, desc, ["blood_gas:lact"])


def test_float32_stay_medians_keep_pandas_precision(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "lact": pd.Series([1.11, 2.22, 4.44, 7.77], dtype="float32"),
        }
    )
    _replace_module(tmp_path, desc, "blood_gas", frame)
    catalog = cohort_review._feature_catalog(desc, [])
    args = (tmp_path, desc, {str(i) for i in range(1, 25)}, catalog, ["blood_gas:lact"])
    with monkeypatch.context() as reference:
        install_reference(reference)
        expected = cohort_review._selected_feature_profiles(*args)
    assert cohort_review._selected_feature_profiles(*args) == expected


def test_table_one_rounds_to_three_decimals_before_group_median(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    _replace_module(
        tmp_path,
        desc,
        "blood_gas",
        pd.DataFrame({"stay_id": [1, 2], "lact": [2.6754, 2.6754]}),
    )
    payload = _assert_parity(monkeypatch, tmp_path, desc, ["blood_gas:lact"])
    age_groups = next(
        row for row in payload["groups"]["supported"] if row["id"] == "age"
    )
    row = next(
        row
        for row in age_groups["profile"]["rows"]
        if row.get("feature_id") == "blood_gas:lact"
    )
    assert row["values"][0] == 2.67


@pytest.mark.parametrize("first", [0.0, -0.0], ids=["positive", "negative"])
def test_signed_zero_keeps_the_original_json_spelling(tmp_path, monkeypatch, first):
    desc = write_cohort_export(tmp_path)
    _replace_module(
        tmp_path,
        desc,
        "blood_gas",
        pd.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6],
                "lact": [first, -0.0, -0.0, 0.0, -0.0, 2.0],
            }
        ),
    )
    _assert_parity(monkeypatch, tmp_path, desc, ["blood_gas:lact"])


def test_missing_demographics_never_materializes_the_denominator_module(
    tmp_path, monkeypatch
):
    desc = write_cohort_export(tmp_path)
    desc["files"] = [
        entry for entry in desc["files"] if entry["module"] != "demographics"
    ]
    with monkeypatch.context() as reference:
        install_reference(reference)
        expected = _run(reference, tmp_path, desc)

    def no_rows(*args, **kwargs):
        raise AssertionError("fallback stay keys must also aggregate in DuckDB")

    monkeypatch.setattr(pd, "read_parquet", no_rows)
    actual = _run(monkeypatch, tmp_path, desc)
    assert actual == expected


def test_all_null_feature_needs_no_observation_sample(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    desc = write_cohort_export(tmp_path, case="all_null")
    original = pq.ParquetFile

    class GuardedParquet:
        def __init__(self, path, *args, **kwargs):
            self.file = original(path, *args, **kwargs)
            self.path = path

        def __getattr__(self, name):
            return getattr(self.file, name)

        def iter_batches(self, *args, **kwargs):
            assert Path(self.path).name != "blood_gas.parquet", (
                "all-null statistics should avoid a row sample"
            )
            return self.file.iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", GuardedParquet)
    payload = _run(monkeypatch, tmp_path, desc, ["blood_gas:lact"])
    assert payload["selected_feature_distributions"][0]["kind"] != "blocked"
    assert payload["selected_feature_distributions"][0]["observed"] == 0
