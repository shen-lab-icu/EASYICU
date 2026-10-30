"""Mortality unknowns and survival group timing are not interchangeable with negatives."""

import json
import pandas as pd
import pytest

from easyicu.webserver import cohort_review
from tests.support.cohort_export import write_cohort_export
from tests.webserver.test_cohort_parquet import _replace_module, _run


def test_mortality_unknowns_excluded_in_parquet_and_summary(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path)
    _replace_module(tmp_path, desc, "outcome", pd.DataFrame({
        "stay_id": [1, 2, 3], "death": pd.Series([True, False, None], dtype="boolean"),
        "los_hosp": [3.0, 4.0, 5.0],
    }))
    payload = _run(monkeypatch, tmp_path, desc, ["outcome:death"])
    mortality = payload["summary"]["mortality"]
    assert mortality["deceased_count"] == 1
    assert mortality["survived_count"] == 1
    assert mortality["known_count"] == 2
    assert mortality["unknown_count"] == 22
    assert mortality["pct"] == 50.0
    survival = next(g for g in payload["groups"]["supported"] if g["id"] == "survival")
    assert [row["count"] for row in survival["groups"]] == [1, 1, 22]
    feature = next(f for f in payload["selected_feature_distributions"] if f["id"] == "outcome:death")
    assert [row["count"] for row in feature["categories"]] == [1, 1, 22]


def test_sparse_non_death_binary_retains_its_existing_denominator():
    features = [{"id": "medications:abx", "module": "medications", "column": "abx", "label": "abx", "kind": "binary", "mapping": {"1": True}}]
    rows = cohort_review._selected_feature_rows([["1", "2"]], features)
    assert rows[0]["values"] == [50.0]


def test_unknown_sex_is_excluded_from_summary_and_group_percentages():
    summary = cohort_review._sex_summary(["F", "M", None, "unknown"])
    assert summary["female_count"] == 1
    assert summary["known_count"] == 2
    assert summary["unknown_count"] == 2
    assert summary["female_pct"] == 50.0
    assert cohort_review._female_pct(["1", "2", "3"], {"1": "F", "2": "M"}) == 50.0
    assert cohort_review._sex_bucket("not female") is None
    assert cohort_review._sex_bucket(1) is None


def test_unverified_future_sepsis_and_worst_sofa_are_never_km_groups():
    groups = cohort_review._survival_group_options(
        entity_ids=["1", "2"], age_by_entity={"1": 40, "2": 80},
        sex_by_entity={"1": "F", "2": "M"},
        sofa_by_entity={"1": 1, "2": 9}, sepsis_by_entity={"1": False, "2": True},
    )
    by_id = {row["id"]: row for row in groups}
    assert by_id["sepsis"]["status"] == "blocked"
    assert "time origin" in by_id["sepsis"]["reason"]
    assert by_id["sofa2"]["status"] == "blocked"
    assert "not a verified baseline" in by_id["sofa2"]["reason"]
    assert by_id["age"]["status"] == by_id["sex"]["status"] == "ready"


def test_custom_groups_bind_cache_and_reject_future_km(tmp_path, monkeypatch):
    desc = write_cohort_export(tmp_path, stays=4)
    _replace_module(tmp_path, desc, "demographics", pd.DataFrame({
        "stay_id": [1, 2, 3, 4], "age": [30, 50, 80, None],
        "sex": ["F", "M", "F", None], "adm": ["Emergency", "Elective", "Other", None],
    }))
    _replace_module(tmp_path, desc, "blood_gas", pd.DataFrame({
        "stay_id": [1, 1, 2, 2, 3], "lact": [1.0, 3.0, 4.0, 6.0, 8.0],
    }))
    monkeypatch.setattr(cohort_review, "_SUMMARY_CACHE", {})
    monkeypatch.setattr(cohort_review, "_resolve_registered_source", lambda body: ({"path": str(tmp_path)}, desc))
    initial = cohort_review.cohort_review_summary({})
    assert initial["custom_group_options"]["version"] == "1"
    assert initial["custom_group_applied"] is None
    def run(rule):
        return cohort_review.cohort_review_summary({"custom_group": rule})

    age = run({"feature_id": "demographics:age", "kind": "numeric_threshold", "threshold": 50})
    assert age["custom_group_applied"] == {"feature_id": "demographics:age", "kind": "numeric_threshold", "threshold": 50.0}
    custom = next(g for g in age["groups"]["supported"] if g["id"] == "custom")
    assert [g["count"] for g in custom["groups"]] == [2, 1, 1, 0]
    assert next(g for g in age["survival_analysis"]["group_options"] if g["id"] == "custom")["status"] == "ready"
    changed = run({"feature_id": "demographics:age", "kind": "numeric_threshold", "threshold": 30})
    assert changed["custom_group_applied"]["threshold"] == 30.0
    assert [g["count"] for g in next(g for g in changed["groups"]["supported"] if g["id"] == "custom")["groups"]] == [1, 2, 1, 0]
    dynamic = run({"feature_id": "blood_gas:lact", "kind": "numeric_threshold", "threshold": 4})
    custom = next(g for g in dynamic["groups"]["supported"] if g["id"] == "custom")
    assert custom["basis"] == "whole_stay_entity_median"
    assert [g["count"] for g in custom["groups"]] == [1, 2, 1, 0]
    assert next(g for g in dynamic["survival_analysis"]["group_options"] if g["id"] == "custom")["status"] == "blocked"
    categorical = run({"feature_id": "demographics:adm", "kind": "categorical_values", "values": ["Emergency", "Elective"]})
    assert categorical["custom_group_applied"] == {"feature_id": "demographics:adm", "kind": "categorical_values", "values": ["Emergency", "Elective"]}
    custom = next(g for g in categorical["groups"]["supported"] if g["id"] == "custom")
    assert [g["count"] for g in custom["groups"]] == [1, 1, 1, 1]
    assert next(o for o in categorical["custom_group_options"]["categories"] if o["feature_id"] == "demographics:adm")["values"] == ["Elective", "Emergency", "Other"]
    assert next(g for g in categorical["survival_analysis"]["group_options"] if g["id"] == "custom")["status"] == "blocked"
    assert "stay_id" not in json.dumps(categorical)
    empty = run({"feature_id": "demographics:age", "kind": "numeric_threshold", "threshold": 100})
    assert next(g for g in empty["groups"]["supported"] if g["id"] == "custom")["status"] == "blocked"
    assert next(g for g in empty["survival_analysis"]["group_options"] if g["id"] == "custom")["status"] == "blocked"
    with pytest.raises(cohort_review.CohortReviewError):
        run({"feature_id": "demographics:age", "kind": "numeric_threshold", "threshold": float("nan")})
    with pytest.raises(cohort_review.CohortReviewError):
        run({"feature_id": "blood_gas:lact", "kind": "categorical_values", "values": ["1", "2"]})
