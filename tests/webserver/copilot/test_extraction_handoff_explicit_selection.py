"""Explicit research variables and source identity govern export reuse."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import dataio
from easyicu.webserver.pi_copilot.extraction_handoff import (
    compile_registered_export_handoff,
    study_binding_for_export,
    submit_study_extraction,
)


def _export(path: Path, raw: Path, database: str, concepts: list[str], *, bounded: bool = True) -> None:
    path.mkdir()
    contract = dataio.normalize_export_cohort_contract({"preset": "all_icu"})
    manifest = {
        "schema_version": "easyicu_native_export_v2",
        "database": database,
        "data_path": str(raw),
        "format": "parquet",
        "cohort_contract": contract,
        "cohort_execution": dataio.export_cohort_execution(contract),
        "files": [{"file": "labs.parquet", "module": "labs", "concept_ids": concepts}],
        "concept_selection": {"mode": "explicit", "modules": {"labs": concepts}},
        "concept_availability": {"structurally_unavailable": []},
        "time_window_authority": {"required": bounded},
    }
    (path / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _study(path: Path, database: str) -> dict:
    return {
        "id": "study-1", "revision": 1,
        "data_source": {"path": str(path), "database": database},
        "modules": ["labs"], "export_format": "parquet",
        "time_window": {"observation_hours": 24, "anchor": "ICU admission"},
        "execution_concepts": {"primary_exposure": "creat", "outcome": "death", "covariates": ["lact"]},
    }


@pytest.mark.parametrize("database", ["miiv", "eicu_demo"])
def test_partial_module_selection_cannot_be_reused(database: str, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    export = tmp_path / "export"
    _export(export, raw, database, ["creat", "death"])
    study = _study(export, database)
    source = {"id": "source-1", "path": str(export), "database": database, "ok": True}
    handoff = compile_registered_export_handoff(study, source)
    assert "registered_export_concepts_missing" in handoff.mismatch_codes
    submitted = []
    for _ in range(2):
        with pytest.raises(dataio.ExportCohortError, match="missing_review_extraction") as failure:
            submit_study_extraction(study=study, registered_source=source, submit=lambda body: submitted.append(body) or {"job_id": "1"})
        assert failure.value.detail["next_action"] == "prepare_and_bind_export_with_required_concepts"
    assert submitted == []
    assert str(raw) not in json.dumps(handoff.public_receipt())


def test_explicit_selection_requires_every_requested_concept_and_time_authority(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    export = tmp_path / "export"
    _export(export, raw, "miiv", ["creat", "death", "lact"], bounded=False)
    study = _study(export, "miiv")
    source = {"path": str(export), "database": "miiv"}
    assert compile_registered_export_handoff(study, source).mismatch_codes == ("registered_export_time_window_unbounded",)
    manifest_path = export / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["time_window_authority"]["required"] = True
    manifest_path.write_text(json.dumps(manifest))
    assert compile_registered_export_handoff(study, source).reusable
    reused = submit_study_extraction(
        study=study, registered_source=source,
        submit=lambda _body: pytest.fail("a complete export must not be submitted again"),
    )
    assert reused.reused is True
    manifest["concept_selection"]["mode"] = "all_in_selected_modules"
    manifest_path.write_text(json.dumps(manifest))
    assert compile_registered_export_handoff(study, source).reusable
    missing_selection = dict(manifest)
    missing_selection.pop("concept_selection")
    manifest_path.write_text(json.dumps(missing_selection))
    assert "registered_export_concepts_unrecorded" in compile_registered_export_handoff(study, source).mismatch_codes
    manifest["files"][0]["concept_ids"].remove("lact")
    manifest_path.write_text(json.dumps(manifest))
    assert "registered_export_concepts_unresolved" in compile_registered_export_handoff(study, source).mismatch_codes
    manifest["files"][0]["concept_ids"].append("lact")
    manifest["files"][0]["concept_status"] = {"lact": {"availability": "produced_all_null"}}
    manifest_path.write_text(json.dumps(manifest))
    assert "registered_export_concepts_unavailable" in compile_registered_export_handoff(study, source).mismatch_codes
    manifest["files"].append({"file": "labs__another.parquet", "module": "labs", "concept_ids": ["lact"], "concept_status": {"lact": {"availability": "available"}}})
    manifest_path.write_text(json.dumps(manifest))
    assert compile_registered_export_handoff(study, source).reusable
    manifest["concept_availability"]["structurally_unavailable"] = [{"concept_id": "lact"}]
    manifest_path.write_text(json.dumps(manifest))
    assert "registered_export_concepts_unavailable" in compile_registered_export_handoff(study, source).mismatch_codes


def test_new_export_requires_same_reviewed_raw_source_and_database(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    other = tmp_path / "other"
    raw.mkdir()
    other.mkdir()
    old, same, different = (tmp_path / name for name in ("old", "same", "different"))
    for path, root in ((old, raw), (same, raw), (different, other)):
        _export(path, root, "miiv", ["creat", "death", "lact"])
    study = _study(old, "miiv")
    same_source = {"path": str(same), "database": "miiv", "ok": True}
    assert study_binding_for_export(study, same_source)["path"] == str(same)
    with pytest.raises(dataio.ExportCohortError, match="registered_export_study_source_mismatch"):
        compile_registered_export_handoff(study, {"path": str(different), "database": "miiv"})
    assert study_binding_for_export(study, {"path": str(different), "database": "miiv", "ok": True}) is None
    study["data_source"]["database"] = "eicu_demo"
    with pytest.raises(dataio.ExportCohortError, match="registered_export_study_database_mismatch"):
        compile_registered_export_handoff(study, same_source)


@pytest.mark.parametrize("database", ["miiv", "eicu_demo"])
@pytest.mark.parametrize("case", ["alias", "all_modules_missing", "all_null", "structural"])
def test_unresolvable_source_does_not_submit_duplicate_extraction(tmp_path: Path, database: str, case: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    export = tmp_path / "export"
    _export(export, raw, database, ["crea", "death", "lact"])
    manifest_path = export / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    study = _study(export, database)
    if case != "alias":
        study["execution_concepts"]["primary_exposure"] = "crea"
    if case == "all_modules_missing":
        manifest["concept_selection"]["mode"] = "all_in_selected_modules"
        study["execution_concepts"]["primary_exposure"] = "unmapped_measurement"
    elif case == "all_null":
        manifest["files"][0]["concept_status"] = {"crea": {"availability": "produced_all_null"}}
    elif case == "structural":
        manifest["concept_availability"]["structurally_unavailable"] = [{"concept_id": "crea"}]
    manifest_path.write_text(json.dumps(manifest))
    source = {"path": str(export), "database": database}
    submissions = []
    expected = "unresolved_review_mapping" if case in {"alias", "all_modules_missing"} else "unavailable_review_source"
    for _ in range(2):
        with pytest.raises(dataio.ExportCohortError, match=expected) as failure:
            submit_study_extraction(study=study, registered_source=source, submit=lambda body: submissions.append(body) or {})
        assert failure.value.detail["next_action"].startswith("review_")
    assert submissions == []


def test_an_unverified_derived_column_cannot_cause_repeated_full_extractions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    export = tmp_path / "export"
    _export(export, raw, "miiv", ["crea", "death", "lact"])
    study = _study(export, "miiv")
    study["execution_concepts"]["primary_exposure"] = "crea"
    study["execution_concepts"]["covariates"] = ["lact_max"]
    source = {"path": str(export), "database": "miiv"}
    submitted = []
    with pytest.raises(dataio.ExportCohortError, match="missing_review_extraction"):
        submit_study_extraction(study=study, registered_source=source, submit=lambda body: submitted.append(body) or {"job_id": "once"})
    assert submitted == []
    # Even an externally produced full-module export cannot claim that the
    # requested derived-column mapping has been proved.
    manifest_path = export / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["concept_selection"]["mode"] = "all_in_selected_modules"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(dataio.ExportCohortError, match="unresolved_review_mapping"):
        submit_study_extraction(study=study, registered_source=source, submit=lambda body: submitted.append(body) or {})
    assert submitted == []
