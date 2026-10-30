"""Synthetic checks for ICU landmark timing and export authority."""
import json
from dataclasses import replace

import pandas as pd
import pytest

from easyicu.concept.metadata_projection import ColumnProjectionSpec, ConceptColumnRole, project_concept_column_metadata
from easyicu.concept.metadata_sidecar import (EXPORT_PHYSICAL_SCOPE, ColumnMetadataBinding, ColumnMetadataFileBinding,
    ColumnMetadataSidecar, TimeCoordinate, write_content_addressed_sidecar)
from easyicu.concept.schema import ConceptDefinition
from easyicu.webserver import cohort_landmark, cohort_review
from tests.support.typed_export import metadata_binding


def fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ids = list(range(1, 12))
    durations = [60., 2., 1., .5, 2., None, 2., 2., 2., 31., 31.1]
    status = ["Alive", "Expired", "Expired", "Alive", None, "Alive", "Alive", "Alive", "Alive", "Expired", "Expired"]
    pd.DataFrame({"patientunitstayid": ids, "unitdischargestatus": status,
                  "unitdischargeoffset": [x * 1440 if x is not None else None for x in durations]}).to_parquet(raw / "patient.parquet")
    outcomes = pd.DataFrame({"patientunitstayid": ids, "icu_death": pd.Series([False, True, True, False, None, False, False, False, False, True, True], dtype="boolean"), "los_icu": durations})
    outcomes.to_parquet(tmp_path / "outcome.parquet")
    lab = pd.DataFrame({"patientunitstayid": [1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11],
                        "charttime": [0., 23., 5., 5., 0., 0., 0., 24., 0., 0., -1., 1., 1.],
                        "pt": [10., 40., 20., 10., 10., 10., 10., 20., 10., 20., 10., 20., 20.]})
    lab.to_parquet(tmp_path / "hematology.parquet")
    missing_source = ConceptDefinition(name="icu_death", sources={}, description="synthetic ICU death")
    event = ColumnMetadataBinding(metadata=project_concept_column_metadata(missing_source,
        spec=ColumnProjectionSpec(column_name="icu_death", source_concept="icu_death", role=ConceptColumnRole.EVENT_STATUS), source_database="eicu"))
    files = (
        ColumnMetadataFileBinding(relative_path="hematology.parquet", module="hematology", identity_column="patientunitstayid",
            time_coordinates=(TimeCoordinate(column="charttime", origin="icu_admission", unit="h"),),
            columns={"pt": metadata_binding("pt", "pt", ConceptColumnRole.VALUE, database="eicu")}),
        ColumnMetadataFileBinding(relative_path="outcome.parquet", module="outcome", identity_column="patientunitstayid", time_coordinates=(),
            columns={"icu_death": event, "los_icu": metadata_binding("los_icu", "los_icu", ConceptColumnRole.VALUE, database="eicu")}),
    )
    ref = write_content_addressed_sidecar(tmp_path, ColumnMetadataSidecar(source_database="eicu", source_database_class_prefixes=(), scope=EXPORT_PHYSICAL_SCOPE, files=files))
    manifest = {"schema_version": "easyicu_native_export_v2", "export_kind": "web_module_extraction", "database": "eicu",
                "data_path": str(raw), "cohort_contract": {"observation_window_hours": 720}, "column_metadata": ref.to_dict()}
    (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
    desc = {"files": [{"file": "hematology.parquet", "module": "hematology", "rows": len(lab), "columns": list(lab)},
                      {"file": "outcome.parquet", "module": "outcome", "rows": len(outcomes), "columns": list(outcomes)}]}
    return desc, outcomes, lab, [str(x) for x in ids]


def run(tmp_path, desc, outcomes, ids, hours=24, threshold=15):
    return cohort_landmark.build_landmark_group(path=tmp_path, desc=desc, entity_ids=ids, outcome=outcomes.rename(columns={"patientunitstayid":"stay_id"}),
        feature={"module":"hematology", "column":"pt"}, threshold=threshold, landmark_hours=hours)


def test_landmark_strict_boundaries_first_value_risk_set_and_censoring(tmp_path):
    desc, outcome, lab, ids = fixture(tmp_path)
    result = run(tmp_path, desc, outcome, ids)
    assert result["group"]["status"] == "ready", result
    assert [g["count"] for g in result["group"]["groups"]] == [2, 3]
    counts = result["details"]["landmark_counts"]
    assert counts == {"cohort_total": 11, "missing_followup": 1, "missing_event": 1,
                      "death_at_or_before_landmark": 1, "discharged_at_or_before_landmark": 1,
                      "missing_measurement": 2, "included": 5}
    events, times = result["vectors"]
    assert events == {"1": False, "2": True, "8": False, "10": True, "11": False}
    assert times == {"1": 30., "2": 1., "8": 1., "10": 30., "11": 30.}
    custom = {"label":"PT · custom", "basis":"whole_stay_entity_median", "_members":[], "groups":[]}
    payload = cohort_review._survival_analysis_payload(outcome=outcome.rename(columns={"patientunitstayid":"stay_id"}), entity_ids=ids,
        age_by_entity={}, sex_by_entity={}, sofa_by_entity={}, sepsis_by_entity={}, custom_group=custom, landmark=result)
    curve = next(c for c in payload["curves"] if c["group_id"] == "custom")
    assert curve["outcome_id"] == "icu_death"
    assert payload["default_outcome"] == "icu_death" and payload["default_group"] == "custom"
    assert [g["n"] for g in curve["groups"]] == [2, 3]
    assert [g["events"] for g in curve["groups"]] == [0, 2]
    assert [row["values"][0] for row in curve["number_at_risk"]["rows"]] == [2, 3]
    assert not any(c["group_id"] == "custom" and c["outcome_id"] != "icu_death" for c in payload["curves"])


def test_source_disagreement_duplicate_or_missing_is_blocked(tmp_path):
    desc, outcome, _, ids = fixture(tmp_path)
    source = tmp_path / "raw" / "patient.parquet"
    raw = pd.read_parquet(source)
    raw.loc[0, "unitdischargestatus"] = "Expired"
    raw.to_parquet(source)
    assert "differs" in run(tmp_path, desc, outcome, ids)["group"]["reason"]
    raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    raw.to_parquet(source)
    assert "does not match" in run(tmp_path, desc, outcome, ids)["group"]["reason"]


def test_missing_or_tampered_time_metadata_fails_closed(tmp_path):
    desc, outcome, _, ids = fixture(tmp_path)
    manifest_path = tmp_path / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["column_metadata"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert run(tmp_path, desc, outcome, ids)["group"]["status"] == "blocked"
    manifest["database"] = "sic"
    manifest_path.write_text(json.dumps(manifest))
    assert "not verified for this source" in run(tmp_path, desc, outcome, ids)["group"]["reason"]


@pytest.mark.parametrize("coordinate", [None, TimeCoordinate(column="charttime", origin="hospital_admission", unit="h"),
                                         TimeCoordinate(column="charttime", origin="icu_admission", unit="min")])
def test_wrong_or_missing_time_coordinate_blocks(tmp_path, coordinate):
    from easyicu.concept.metadata_sidecar import read_content_addressed_sidecar
    desc, outcome, _, ids = fixture(tmp_path)
    manifest_path = tmp_path / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ref = manifest["column_metadata"]
    sidecar = read_content_addressed_sidecar(tmp_path / ref["file"], expected_sha256=ref["sha256"], expected_size=ref["size"])
    bindings = tuple(replace(b, time_coordinates=(() if coordinate is None else (coordinate,))) if b.module == "hematology" else b for b in sidecar.files)
    manifest["column_metadata"] = write_content_addressed_sidecar(tmp_path, replace(sidecar, files=bindings)).to_dict()
    manifest_path.write_text(json.dumps(manifest))
    assert "verified ICU-relative hours" in run(tmp_path, desc, outcome, ids)["group"]["reason"]


def test_empty_group_and_all_censored_logrank(tmp_path):
    desc, outcome, _, ids = fixture(tmp_path)
    empty = run(tmp_path, desc, outcome, ids, threshold=1000)
    assert empty["group"]["status"] == "blocked"
    assert "two non-empty" in empty["group"]["reason"]
    outcome["icu_death"] = pd.Series([False] * len(outcome), dtype="boolean")
    outcome.to_parquet(tmp_path / "outcome.parquet")
    source_path = tmp_path / "raw" / "patient.parquet"
    source = pd.read_parquet(source_path)
    source["unitdischargestatus"] = "Alive"
    source.to_parquet(source_path)
    result = run(tmp_path, desc, outcome, ids)
    assert result["group"]["status"] == "ready"
    custom = {"label":"PT · custom", "basis":"whole_stay_entity_median", "_members":[], "groups":[]}
    payload = cohort_review._survival_analysis_payload(outcome=outcome.rename(columns={"patientunitstayid":"stay_id"}), entity_ids=ids,
        age_by_entity={}, sex_by_entity={}, sofa_by_entity={}, sepsis_by_entity={}, custom_group=custom, landmark=result)
    curve = next(c for c in payload["curves"] if c["group_id"] == "custom")
    assert curve["logrank"]["status"] != "ready"
    assert sum(g["n"] for g in curve["groups"]) == result["details"]["landmark_counts"]["included"]
    assert sum(v for k, v in result["details"]["landmark_counts"].items() if k != "cohort_total") == len(ids)


def test_file_identity_change_blocks_and_busy_limit_fails_closed(tmp_path, monkeypatch):
    desc, outcome, _, ids = fixture(tmp_path)
    original = cohort_landmark.source_file_identity
    seen = {}
    def changing(path):
        identity = original(path)
        if path.name == "patient.parquet":
            seen[path] = seen.get(path, 0) + 1
            return (*identity, seen[path])
        return identity
    monkeypatch.setattr(cohort_landmark, "source_file_identity", changing)
    assert "changed during landmark" in run(tmp_path, desc, outcome, ids)["group"]["reason"]
    assert cohort_landmark._LANDMARK_SLOT.acquire(blocking=False)
    try:
        assert "Another landmark" in run(tmp_path, desc, outcome, ids)["group"]["reason"]
    finally:
        cohort_landmark._LANDMARK_SLOT.release()


def test_summary_exposes_landmark_receipt_and_aggregate_only(tmp_path, monkeypatch):
    desc, outcome, _, ids = fixture(tmp_path)
    demo = pd.DataFrame({"patientunitstayid": [int(i) for i in ids], "age": [50] * len(ids), "sex": ["F", "M"] * 5 + ["F"]})
    demo.to_parquet(tmp_path / "demographics.parquet")
    desc["files"].append({"file": "demographics.parquet", "module": "demographics", "rows": len(ids), "columns": list(demo)})
    desc["path"] = str(tmp_path)
    desc["summary"] = {"stays": len(ids), "modules": 3, "file_count": 3, "total_rows": 35}
    monkeypatch.setattr(cohort_review, "_resolve_registered_source", lambda body: ({"path": str(tmp_path)}, desc))
    response = cohort_review.cohort_review_summary({"custom_group": {"feature_id": "hematology:pt", "kind": "numeric_threshold", "threshold": 15, "landmark_hours": 24}})
    assert response["custom_group_applied"]["landmark_hours"] == 24
    assert response["custom_group_options"]["landmark_hours"] == [6, 12, 24, 48]
    custom = next(g for g in response["survival_analysis"]["group_options"] if g["id"] == "custom")
    assert custom["status"] == "ready" and custom["measurement_unit"] == "sec"
    assert "patientunitstayid" not in json.dumps(response)
    assert "_members" not in json.dumps(response)
