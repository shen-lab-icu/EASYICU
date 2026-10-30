"""Run provenance travels through existing immutable input/evidence owners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.data_provenance import ExportJob, patch_demographic_export, synthetic_dictionary


def test_run_records_the_snapshot_and_exposes_digest_checked_evidence(tmp_path: Path):
    from easyicu.concept.provenance import capture_data_provenance, ExportDataProvenance
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.research_evidence_preview import build_evidence_preview
    from easyicu.webserver.agent_runs import _public_single_artifact_payload

    snapshot = capture_data_provenance(
        synthetic_dictionary(), concepts=["lab_signal", "coded_condition"], source_database="miiv",
    )
    captured = ExportDataProvenance(snapshots=(snapshot,)).model_dump(mode="json")
    evidence = EvidenceStore(root=tmp_path)
    context_path = tmp_path / "research_context.json"
    context_path.write_text(json.dumps({
        "cohort": {"provenance": {
            "data_provenance": captured, "data_provenance_export_authority_sha256": "a" * 64,
        }},
        "variables": [{"name": "lab_signal_mean", "source_concept": "lab_signal"}],
    }), encoding="utf-8")
    context_record = evidence.register_file(
        source_path=context_path, evidence_id="research_context", kind="log",
        description="Synthetic context",
    )
    record = publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    preview = build_evidence_preview(tmp_path, evidence_id=record.evidence_id, expected_sha256=record.sha256)
    payload = preview["value"]
    assert payload["status"] == "recorded"
    assert payload["research_context_sha256"] == context_record.sha256
    assert payload["export_authority_sha256"] == "a" * 64
    assert payload["data_provenance"] == captured
    assert payload["variables"][0]["concept_id"] == "lab_signal"
    assert _public_single_artifact_payload("data_provenance.json", payload) == payload
    # Repeated projection and resume preserve the bytes; no current catalog read.
    assert publish_run_data_provenance(run_dir=tmp_path, evidence=evidence).sha256 == record.sha256
    assert preview["declared_lineage"][0]["evidence_id"] == "research_context"


def test_legacy_run_has_typed_not_recorded_provenance(tmp_path: Path):
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    evidence = EvidenceStore(root=tmp_path)
    path = tmp_path / "research_context.json"
    path.write_text("{}", encoding="utf-8")
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    payload = json.loads((tmp_path / "data_provenance.json").read_text())
    assert payload["status"] == "not_recorded"
    assert payload["schema_version"] == "easyicu.run-data-provenance/1"
    assert payload["data_provenance"] is None


def test_forged_context_without_verified_run_cohort_is_unrecorded(tmp_path: Path):
    from types import SimpleNamespace
    from easyicu.concept.provenance import capture_data_provenance, ExportDataProvenance
    from easyicu.research_agent.authority.provenance import provenance_from_verified_cohort, publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.data_provenance_projection import load_data_provenance_projection

    snapshot = capture_data_provenance(synthetic_dictionary(), concepts=["lab_signal"], source_database="miiv")
    captured = ExportDataProvenance(snapshots=(snapshot,)).model_dump(mode="json")
    semantic = {"export_authority": {"data_provenance": captured, "authority_sha256": "a" * 64},
                "cohort_file_sha256": "b" * 64, "source_mode": "export",
                "cohort_window_hours": [0, 24], "columns": ["stay_id", "hr_mean"]}
    authority = SimpleNamespace(to_dict=lambda: {"semantic_provenance": semantic},
        cohort_columns=("stay_id", "hr_mean"), producer_implementation_sha256="c" * 64)
    verified = SimpleNamespace(authority=authority, reference=SimpleNamespace(sha256="d" * 64))
    provenance = provenance_from_verified_cohort(verified)
    assert provenance["data_provenance_materialization"] == {
        "source_mode": "export", "cohort_window_hours": [0.0, 24.0],
        "output_columns": ["stay_id", "hr_mean"], "producer_implementation_sha256": "c" * 64,
    }
    path = tmp_path / "research_context.json"
    path.write_text(json.dumps({"cohort": {"provenance": provenance}, "variables": []}))
    evidence = EvidenceStore(root=tmp_path)
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    record = publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    payload = load_data_provenance_projection(tmp_path)
    assert payload["schema_version"] == "easyicu.run-data-provenance/1"
    assert payload["status"] == "not_recorded"
    assert payload["data_provenance"] is None
    assert payload["materialization_sha256"] is None
    assert publish_run_data_provenance(run_dir=tmp_path, evidence=evidence).sha256 == record.sha256
    # An unverified or mismatched source cannot make the new claim.
    authority.cohort_columns = ("stay_id", "other")
    assert "data_provenance_materialization" not in provenance_from_verified_cohort(verified)
    legacy = {**semantic, "data_provenance": captured, "provenance_sha256": "d" * 64}
    assert "data_provenance_materialization" not in provenance_from_verified_cohort(None, legacy)


def test_materialization_summary_rejects_unbound_or_unsafe_payload():
    from easyicu.research_agent.authority.provenance import RunDataProvenanceV2
    summary = {"source_mode": "export", "cohort_window_hours": [0, 24],
               "output_columns": ["stay_id"], "producer_implementation_sha256": "c" * 64}
    with pytest.raises(ValueError):
        RunDataProvenanceV2(run_id="synthetic", status="not_recorded", materialization=summary)
    for change in ({"cohort_window_hours": [24, 0]}, {"output_columns": ["x"] * 257},
                   {"source_mode": "unknown"}, {"producer_implementation_sha256": "invalid"}):
        from easyicu.research_agent.authority.provenance import VerifiedMaterializationSummary
        with pytest.raises(ValueError):
            VerifiedMaterializationSummary.model_validate({**summary, **change})


def test_synthetic_verified_authority_keeps_materialization_when_source_snapshot_absent(tmp_path):
    from tests.support.typed_trajectory import typed_trajectory_bundle
    from easyicu.research_agent.authority.provenance import provenance_from_verified_cohort, publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.data_provenance_projection import load_data_provenance_projection
    from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority

    (tmp_path / "bundle").mkdir()
    paths, source, _trajectory = typed_trajectory_bundle(tmp_path / "bundle")
    verified = stage_materialized_cohort_authority(paths["parquet"], tmp_path / "cohort.parquet",
        producer_implementation_sha256="c" * 64, expected_source_authority=source.reference)
    provenance = provenance_from_verified_cohort(verified)
    assert "data_provenance" not in provenance
    summary = provenance["data_provenance_materialization"]
    assert summary["output_columns"] == list(verified.authority.cohort_columns)
    assert summary["cohort_window_hours"] == list(verified.authority.semantic_provenance["cohort_window_hours"])
    path = tmp_path / "research_context.json"
    path.write_text(json.dumps({"cohort": {"database": "miiv", "provenance": provenance}, "variables": []}))
    evidence = EvidenceStore(root=tmp_path)
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    record = publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    payload = load_data_provenance_projection(tmp_path)
    assert payload["schema_version"] == "easyicu.run-data-provenance/2"
    assert payload["status"] == "not_recorded"  # Raw-source snapshot remains unavailable.
    assert payload["materialization"] == summary
    assert publish_run_data_provenance(run_dir=tmp_path, evidence=evidence).sha256 == record.sha256


@pytest.mark.parametrize("tampering", ["summary", "authority", "cohort", "missing", "database", "source"])
def test_new_materialization_claim_fails_closed_on_unverified_run_input(tmp_path, tampering):
    from tests.support.typed_trajectory import typed_trajectory_bundle
    from easyicu.research_agent.authority.provenance import provenance_from_verified_cohort, publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.intake.materialized_metadata import (
        materialized_provenance_path, stage_materialized_cohort_authority,
    )
    from easyicu.webserver.data_provenance_projection import load_data_provenance_projection

    (tmp_path / "bundle").mkdir()
    paths, source, _trajectory = typed_trajectory_bundle(tmp_path / "bundle")
    cohort = tmp_path / "cohort.parquet"
    verified = stage_materialized_cohort_authority(paths["parquet"], cohort,
        producer_implementation_sha256="c" * 64, expected_source_authority=source.reference)
    provenance = provenance_from_verified_cohort(verified)
    database = "miiv"
    if tampering == "summary":
        provenance["data_provenance_materialization"]["cohort_window_hours"] = [0, 999]
    elif tampering == "authority":
        materialized_provenance_path(cohort).write_text("{}")
    elif tampering == "cohort":
        cohort.write_bytes(b"not a valid cohort")
    elif tampering == "missing":
        cohort.unlink()
    elif tampering == "database":
        database = "sic"
    elif tampering == "source":
        provenance["data_provenance_export_authority_sha256"] = "f" * 64
    path = tmp_path / "research_context.json"
    path.write_text(json.dumps({"cohort": {"database": database, "provenance": provenance}, "variables": []}))
    evidence = EvidenceStore(root=tmp_path)
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    payload = load_data_provenance_projection(tmp_path)
    assert payload["schema_version"] == "easyicu.run-data-provenance/1"
    assert payload["status"] == "not_recorded"
    assert payload["data_provenance"] is None
    assert payload["materialization_sha256"] is None


def test_metadata_uses_the_existing_evidence_preview_budget(tmp_path):
    from easyicu.concept.provenance import capture_data_provenance, ExportDataProvenance
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.data_provenance_projection import load_data_provenance_projection
    from easyicu.webserver.research_evidence_preview import MAX_PREVIEW_BYTES

    snapshot = capture_data_provenance(
        synthetic_dictionary(), concepts=[f"synthetic_missing_{i}" for i in range(1200)],
        source_database="miiv",
    )
    path = tmp_path / "research_context.json"
    path.write_text(json.dumps({"cohort": {"provenance": {
        "data_provenance": ExportDataProvenance(snapshots=(snapshot,)).model_dump(mode="json"),
        "data_provenance_export_authority_sha256": "a" * 64,
    }}}))
    evidence = EvidenceStore(root=tmp_path)
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    record = publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    assert (tmp_path / record.relative_path).stat().st_size <= MAX_PREVIEW_BYTES
    assert load_data_provenance_projection(tmp_path)["status"] == "recorded"


def test_tampered_context_and_missing_input_binding_fail_closed(tmp_path: Path):
    from easyicu.concept.provenance import ExportDataProvenance
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance, RunDataProvenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    evidence = EvidenceStore(root=tmp_path)
    with pytest.raises(ValueError, match="registered research context"):
        publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    path = tmp_path / "research_context.json"
    path.write_text("{}", encoding="utf-8")
    record = evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    (tmp_path / record.relative_path).write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    with pytest.raises(ValueError, match="bindings"):
        RunDataProvenance(run_id="synthetic", status="recorded")
    with pytest.raises(ValueError, match="invented"):
        RunDataProvenance(run_id="synthetic", status="not_recorded", data_provenance=ExportDataProvenance(snapshots=()))


@pytest.mark.parametrize("changed", ["context", "metadata"])
def test_resume_refuses_changed_registered_inputs_or_metadata(tmp_path, monkeypatch, changed):
    import hashlib
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    path = tmp_path / "research_context.json"
    path.write_text("{}")
    evidence = EvidenceStore(root=tmp_path)
    context = evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    record = publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)
    if changed == "context":
        new_bytes = b'{"variables":[]}'
        (tmp_path / context.relative_path).write_bytes(new_bytes)
        updated = context.model_copy(update={"sha256": hashlib.sha256(new_bytes).hexdigest()})
        original_get = evidence.get
        monkeypatch.setattr(evidence, "get", lambda key: updated if key == "research_context" else original_get(key))
    else:
        (tmp_path / record.relative_path).write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        publish_run_data_provenance(run_dir=tmp_path, evidence=evidence)


@pytest.mark.parametrize("unsafe", [
    {"patient_rows": [{"value": 4.2}]},
    {"note": "/home/example/patients.csv"},
    {"note": "stay_id=synthetic-private"},
])
def test_provenance_preview_download_and_bundle_keep_the_existing_privacy_boundary(tmp_path, unsafe):
    from easyicu.webserver import agent_runs

    if "note" in unsafe:
        unsafe = {"run_id": "synthetic", "status": "not_recorded", "variables": [
            {"column": unsafe["note"], "concept_id": None},
        ]}
    (tmp_path / "data_provenance.json").write_text(json.dumps(unsafe), encoding="utf-8")
    results = [
        agent_runs.read_run_artifact(str(tmp_path), "data_provenance.json"),
        agent_runs.read_run_artifact_bytes(str(tmp_path), "data_provenance.json"),
        agent_runs.build_run_bundle(str(tmp_path)),
    ]
    assert all(result["error"] == "artifact_privacy_scan_failed" for result in results)
    assert all("content" not in result and "payload" not in result for result in results)


def test_native_export_records_provenance_even_without_feature_definitions(tmp_path, monkeypatch):
    import easyicu.api as api
    from easyicu.webserver import dataio
    from easyicu.research_agent.intake.export_package import open_export_package, ExportPackageError
    from easyicu.research_agent.cohort.materializer import _export_authority_provenance

    patch_demographic_export(monkeypatch)
    monkeypatch.setattr(api, "get_all_patient_ids", lambda *_, **__: ([1, 2], "stay_id"))
    out = tmp_path / "export"
    result = dataio.make_export_runner(
        data_path=str(tmp_path), database="miiv", modules=["demographics"],
        concepts={"demographics": ["age"]}, out_dir=str(out),
        include_feature_definitions=False,
    )(ExportJob())
    assert result["manifest"]
    with open_export_package(out) as package:
        captured = package.data_provenance.model_dump(mode="json")
        assert captured["snapshots"][0]["concepts"][0]["concept_id"] == "age"
        authority = _export_authority_provenance(package)
        assert authority["data_provenance"] == captured
        original_authority = package.authority_sha256
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.data_provenance_projection import load_data_provenance_projection

    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=out, database="miiv",
        feature_concepts=(), outcome_concepts=(), static_concepts=("age",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stage_materialized_cohort_authority(paths["parquet"], run_dir / "cohort.parquet",
                                        producer_implementation_sha256="c" * 64)
    context = build_research_context(
        research_question="Describe the synthetic age measurements.",
        cohort=run_dir / "cohort.parquet", cohort_name="synthetic", database="miiv",
        id_columns=("stay_id",), outcome_columns=(),
    )
    assert context.cohort.provenance["data_provenance"] == captured
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(), encoding="utf-8")
    evidence = EvidenceStore(root=run_dir)
    evidence.register_file(source_path=context_path, kind="log", evidence_id="research_context", description="Synthetic")
    record = publish_run_data_provenance(run_dir=run_dir, evidence=evidence)
    projected = load_data_provenance_projection(run_dir)
    assert projected["data_provenance"] == captured
    assert projected["export_authority_sha256"] == original_authority
    assert projected["evidence_ref"]["sha256"] == record.sha256
    from types import SimpleNamespace
    from easyicu.webserver.agent_pipeline_runs import _write_projection
    wrapper = tmp_path / "wrapper"
    review = _write_projection(
        wrapper_dir=wrapper, run_dir=run_dir,
        study={"id": "synthetic", "question": "Describe synthetic measurements."},
        provider={"provider": "mock", "model": "synthetic"}, acquisition=SimpleNamespace(),
    )
    assert "data_provenance.json" in {item["name"] for item in review["artifacts"]}
    assert json.loads((wrapper / "data_provenance.json").read_text())["data_provenance"] == captured
    (run_dir / record.relative_path).write_text("{}", encoding="utf-8")
    assert load_data_provenance_projection(run_dir)["status"] == "not_recorded"
    manifest_path = out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_provenance"]["snapshots"][0]["concepts"][0]["definition"] = "Changed later"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ExportPackageError, match="invalid export data provenance"):
        open_export_package(out)
    manifest["data_provenance"]["snapshots"] = []
    manifest_path.write_text(json.dumps(manifest))
    with open_export_package(out) as package:
        assert package.authority_sha256 != original_authority


def test_pipeline_registers_metadata_before_any_planner_or_execution(tmp_path, monkeypatch):
    import pandas as pd
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.providers.mocks import MockLLMClient

    class MetadataRegistered(RuntimeError):
        pass

    register = pipeline_module.register_run_metadata

    def stop_after_metadata(**kwargs):
        register(**kwargs)
        assert (kwargs["run_dir"] / "data_provenance.json").is_file()
        raise MetadataRegistered("stop before planning")

    monkeypatch.setattr(pipeline_module, "register_run_metadata", stop_after_metadata)
    pipeline = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path, llm=MockLLMClient(), runner_kind="subprocess", enable_literature=False,
    )
    with pytest.raises(MetadataRegistered):
        pipeline.run(
            question="Describe synthetic age measurements.",
            cohort=pd.DataFrame({"stay_id": [1, 2], "age": [30.0, 40.0]}),
            database="miiv",
        )


def test_direct_native_materialization_freezes_provenance_without_an_export(tmp_path, monkeypatch):
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.cohort.schema import CohortDefinition, ConceptPredicate, TimeWindow
    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.research_agent.authority.provenance import publish_run_data_provenance
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    patch_demographic_export(monkeypatch)
    monkeypatch.setattr(materializer, "_resolve_source", lambda *_: ("native", tmp_path))
    paths = materializer.materialize_to_parquet(
        tmp_path / "native", data_path=tmp_path, database="miiv",
        feature_concepts=(), outcome_concepts=(), static_concepts=("age",),
        patient_ids=[1, 2],
        cohort_definition=CohortDefinition("synthetic_adults", inclusion=(ConceptPredicate(
            "age", TimeWindow("icu_admission", 0, 24), "first", ">=", 18,
        ),)),
    )
    context = build_research_context(
        research_question="Describe synthetic ages.", cohort=paths["parquet"],
        cohort_name="synthetic", database="miiv", id_columns=("stay_id",), outcome_columns=(),
    )
    captured = context.cohort.provenance["data_provenance"]
    assert captured["snapshots"][0]["concepts"][0]["concept_id"] == "age"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "research_context.json"
    path.write_text(context.model_dump_json())
    evidence = EvidenceStore(root=run_dir)
    evidence.register_file(source_path=path, evidence_id="research_context", kind="log", description="Synthetic")
    publish_run_data_provenance(run_dir=run_dir, evidence=evidence)
    payload = json.loads((run_dir / "data_provenance.json").read_text())
    assert payload["status"] == "recorded"
    assert payload["materialization_sha256"] == context.cohort.provenance["materialized_cohort_provenance_sha256"]
    assert payload["cohort_file_sha256"] is not None
    assert payload["export_authority_sha256"] is None
    assert payload["cohort_selection"]["inclusion"][0]["value"] == 18
    assert payload["cohort_selection"]["inclusion"][0]["time_window"]["anchor"] == "icu_admission"
