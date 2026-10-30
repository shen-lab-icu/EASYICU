"""Synthetic SIC time-axis provenance; no clinical rows or approval."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE, ColumnMetadataSidecar, write_content_addressed_sidecar,
)
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
from easyicu.research_agent.intake.export_package import (
    NATIVE_MANIFEST, NATIVE_MANIFEST_SCHEMA_V2, ExportPackageError,
    open_export_package, require_canonical_time_projection,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedColumnMetadataCollector, MaterializedMetadataError,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.cohort.materializer import materialize_cohort, materialize_to_parquet
from easyicu.research_agent.cohort.schema import (
    CohortDefinition, ConceptPredicate, TimeWindow, materialize_locked_analysis_cohort,
    recover_approved_source_analysis_cohort,
)


def _package(root, *, legacy=False):
    frame = pd.DataFrame({"stay_id": [1, 2, 1], "charttime": [1.0, 2.0, 24.0], "lact": [1.2, 3.4, 9.0]})
    frame.to_parquet(root / "blood_gas.parquet", index=False)
    demographics = pd.DataFrame({"stay_id": [1, 2], "age": [50, 55]})
    demographics.to_parquet(root / "demographics.parquet", index=False)
    demographics_binding = build_export_file_metadata_binding(
        relative_path="demographics.parquet", module="demographics", frame=demographics,
        concept_ids=("age",), database="sic", database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )
    binding = build_export_file_metadata_binding(
        relative_path="blood_gas.parquet", module="blood_gas", frame=frame,
        concept_ids=("lact",), database="sic", database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )
    if legacy:
        binding = replace(binding, time_coordinates=(
            replace(binding.time_coordinates[0], origin="icu_admission"),
        ))
    reference = write_content_addressed_sidecar(root, ColumnMetadataSidecar(
        source_database="sic", source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE, files=(binding, demographics_binding),
    ))
    manifest = {
        "schema_version": NATIVE_MANIFEST_SCHEMA_V2, "database": "sic", "format": "parquet",
        "canonical_physical_schema": {"time_origin": "icu_admission" if legacy else SIC_CARE_UNIT_ORIGIN},
        "time_window_authority": {"required": True, "source": "outcome.los_icu", "bounded_stays": 2},
        "concept_selection": {"mode": "explicit", "modules": {"blood_gas": ["lact"], "demographics": ["age"]}},
        "files": [{"file": "blood_gas.parquet", "module": "blood_gas", "concepts": 1,
                   "concept_ids": ["lact"], "rows": 3,
                   "column_metadata_columns": list(binding.columns)},
                  {"file": "demographics.parquet", "module": "demographics", "concepts": 1,
                   "concept_ids": ["age"], "rows": 2,
                   "column_metadata_columns": list(demographics_binding.columns)}],
        "feature_definitions": {"included": False}, "column_metadata": reference.to_dict(),
    }
    (root / NATIVE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return open_export_package(root)


def test_bound_source_axis_reaches_window_receipt_but_does_not_grant_execution(tmp_path):
    with _package(tmp_path) as package:
        # A scientific consumer still requires the plan's source-bound approval;
        # merely writing a truthful physical coordinate is not approval.
        with pytest.raises(ExportPackageError, match="source-bound care-unit axis"):
            require_canonical_time_projection(package, "lact")
        axis = SourceTimeAxisBinding(
            source_identity_sha256="a" * 64, export_authority_sha256=package.authority_sha256,
            manifest_sha256=package.manifest_sha256, selected_origin=SIC_CARE_UNIT_ORIGIN,
        )
        require_canonical_time_projection(package, "lact", source_time_axis=axis)
        with pytest.raises(ExportPackageError, match="source-bound care-unit axis"):
            require_canonical_time_projection(
                package, "lact", source_time_axis=replace(axis, manifest_sha256="b" * 64),
            )
        collector = MaterializedColumnMetadataCollector(package)
        collector.add_timeseries("lact", output_columns=["lact_max", "lact_first_time"], window=(0, 24))
        maximum = collector.binding_for_output("lact_max")
        first_time = collector.binding_for_output("lact_first_time")
        assert maximum.derivation_window.origin == SIC_CARE_UNIT_ORIGIN
        assert first_time.metadata.time_origin == SIC_CARE_UNIT_ORIGIN
        assert first_time.derivation_window.origin == SIC_CARE_UNIT_ORIGIN


def test_legacy_sic_icu_label_cannot_be_reclassified_or_written_to_a_new_receipt(tmp_path):
    with _package(tmp_path, legacy=True) as package:
        axis = SourceTimeAxisBinding(
            source_identity_sha256="a" * 64, export_authority_sha256=package.authority_sha256,
            manifest_sha256=package.manifest_sha256, selected_origin=SIC_CARE_UNIT_ORIGIN,
        )
        with pytest.raises(ExportPackageError, match="source-bound care-unit axis"):
            require_canonical_time_projection(package, "lact", source_time_axis=axis)
        collector = MaterializedColumnMetadataCollector(package)
        with pytest.raises(MaterializedMetadataError, match="cannot be labelled ICU admission"):
            collector.add_timeseries("lact", output_columns=["lact_max"], window=(0, 24))
        assert collector.binding_for_output("lact_max") is None


def test_bound_synthetic_source_materializes_and_retains_exact_axis(tmp_path, monkeypatch):
    with _package(tmp_path) as package:
        axis = SourceTimeAxisBinding(
            source_identity_sha256="a" * 64, export_authority_sha256=package.authority_sha256,
            manifest_sha256=package.manifest_sha256, selected_origin=SIC_CARE_UNIT_ORIGIN,
        )
    args = dict(
        feature_concepts=["lact"], database="sic", data_path=tmp_path,
        outcome_concepts=(), static_concepts=("age",), cohort_window=(0.0, 24.0),
    )
    with pytest.raises(ExportPackageError, match="source-bound care-unit axis"):
        materialize_cohort(**args)
    with pytest.raises(ExportPackageError, match="source-bound care-unit axis"):
        materialize_cohort(**args, source_time_axis=replace(axis, manifest_sha256="b" * 64))
    frame, receipt = materialize_cohort(**args, source_time_axis=axis)
    assert frame["lact_max"].tolist() == [1.2, 3.4]
    assert receipt["source_time_axis"] == {
        "source_identity_sha256": "a" * 64, "export_authority_sha256": axis.export_authority_sha256,
        "manifest_sha256": axis.manifest_sha256, "selected_origin": SIC_CARE_UNIT_ORIGIN,
    }
    predicate = ConceptPredicate(
        concept_id="lact", time_window=TimeWindow(SIC_CARE_UNIT_ORIGIN, 0, 24),
        aggregation="max", op=">", value=2.0,
    )
    definition = CohortDefinition(name="primary", inclusion=(predicate,))
    selected, selected_receipt = materialize_cohort(**args, cohort_definition=definition, source_time_axis=axis)
    assert selected["stay_id"].tolist() == [2]
    assert selected_receipt["source_time_axis"]["selected_origin"] == SIC_CARE_UNIT_ORIGIN
    wrong = replace(predicate, time_window=TimeWindow("icu_admission", 0, 24))
    with pytest.raises(MaterializedMetadataError, match="explicitly use the selected care-unit origin"):
        materialize_cohort(**args, cohort_definition=CohortDefinition(name="wrong", inclusion=(wrong,)), source_time_axis=axis)
    paths = materialize_to_parquet(
        tmp_path / "output", emit_trajectory=True, trajectory_concepts=["lact"],
        trajectory_window=(0, 24), **args, source_time_axis=axis,
    )
    from easyicu.research_agent.intake.materialized_trajectory import load_verified_materialized_trajectory_authority
    trajectory = load_verified_materialized_trajectory_authority(paths["trajectory"])
    assert trajectory.authority.time_origin == SIC_CARE_UNIT_ORIGIN
    assert trajectory.authority.window.origin == SIC_CARE_UNIT_ORIGIN
    assert trajectory.authority.window.inclusive is False
    assert 24.0 not in pd.read_parquet(paths["trajectory"])["charttime"].tolist()
    from easyicu.research_agent.intake.materialized_trajectory import MaterializedTrajectoryError
    malformed = dict(trajectory.authority.semantic_provenance)
    malformed["source_time_axis"] = {"selected_origin": SIC_CARE_UNIT_ORIGIN,
                                     "export_authority_sha256": axis.export_authority_sha256}
    with pytest.raises(MaterializedTrajectoryError, match="source time axis is invalid"):
        replace(trajectory.authority, semantic_provenance=malformed)
    # A sealed source axis alone is not approval of a new plan or its cohort.
    with pytest.raises(MaterializedMetadataError, match="requires the approved plan"):
        materialize_locked_analysis_cohort(
            run_dir=tmp_path / "output", plan=SimpleNamespace(cohort=definition),
            universe_path=paths["parquet"], context=None,
        )
    # Isolate the materialization owner from the separately tested signed gate.
    # This stub does not confer real research approval.
    from easyicu.databases import research_approval
    monkeypatch.setattr(research_approval, "require_approved_database_capabilities", lambda *_a, **_k: axis)
    descriptor = SimpleNamespace(name="lact_max", source_concept="lact", role="other",
                                 analysis_window=f"{SIC_CARE_UNIT_ORIGIN}[0,24]h")
    context = SimpleNamespace(
        user_preferences=None, variables=[descriptor], cohort=SimpleNamespace(database="sic"),
        model_dump=lambda **_: {"synthetic": "context"},
    )
    plan = SimpleNamespace(cohort=definition, steps=[], model_dump=lambda **_: {"synthetic": "plan"})
    plan_result = SimpleNamespace(plan=plan, context=context)
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path / "output", plan=plan, context=context, universe_path=paths["parquet"],
        approved_plan_result=plan_result, time_origin_attestor=lambda *_: None,
    )
    assert result["status"] == "applied"
    assert pd.read_parquet(result["path"])["stay_id"].tolist() == [2]
    assert recover_approved_source_analysis_cohort(
        run_dir=tmp_path / "output", plan=plan, universe_path=paths["parquet"],
    )["authority_ref"] == result["authority_ref"]


def test_source_analysis_recovery_refuses_partial_transaction(tmp_path):
    assert recover_approved_source_analysis_cohort(
        run_dir=tmp_path, plan=SimpleNamespace(), universe_path=tmp_path / "universe.parquet",
    ) is None
    (tmp_path / "cohort_analysis.parquet").write_bytes(b"incomplete")
    with pytest.raises(MaterializedMetadataError, match="incomplete or invalid"):
        recover_approved_source_analysis_cohort(
            run_dir=tmp_path, plan=SimpleNamespace(), universe_path=tmp_path / "universe.parquet",
        )


def test_signed_synthetic_plan_materializes_and_recovers_bound_cohort(tmp_path):
    from easyicu.databases.research_approval import CapabilityApprovalError, bind_database_capability_review
    from easyicu.databases.research_substitutions import time_origin_selection
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.time_origin_source_authority import RegisteredExportTimeOrigin
    from tests.research_agent.planning.test_time_origin_review_binding import _review
    from tests.support.signed_time_origin import signed_origin_case
    from tests.support.source_substitutions import _approve

    context, plan, runtime, _, _ = signed_origin_case()
    selection = time_origin_selection(context)
    with _package(tmp_path) as package:
        axis = SourceTimeAxisBinding(
            source_identity_sha256=selection.source_identity_sha256,
            export_authority_sha256=package.authority_sha256,
            manifest_sha256=package.manifest_sha256,
            selected_origin=SIC_CARE_UNIT_ORIGIN,
        )
    receipt = RegisteredExportTimeOrigin(
        axis.source_identity_sha256, axis.export_authority_sha256, axis.manifest_sha256, axis.selected_origin,
    )
    def attestor(_context, _selection):
        return selection, receipt

    paths = materialize_to_parquet(
        tmp_path / "output", feature_concepts=["lact"], database="sic", data_path=tmp_path,
        outcome_concepts=(), static_concepts=("age",), cohort_window=(0.0, 24.0),
        source_time_axis=axis,
    )
    base = _review(context, plan).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=context, plan=plan,
                                             time_origin_attestor=attestor, runtime_authority=runtime)
    assert review.approval_allowed
    approved = _approve(tmp_path / "output", EvidenceStore(tmp_path / "output"), context, plan, review)
    stale_receipt = replace(receipt, manifest_sha256="c" * 64)
    with pytest.raises(CapabilityApprovalError, match="time_origin"):
        materialize_locked_analysis_cohort(
            run_dir=tmp_path / "output", plan=plan, context=context, universe_path=paths["parquet"],
            approved_plan_result=approved,
            time_origin_attestor=lambda *_: (selection, stale_receipt), runtime_authority=runtime,
        )
    assert not (tmp_path / "output" / "cohort_analysis.parquet").exists()
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path / "output", plan=plan, context=context, universe_path=paths["parquet"],
        approved_plan_result=approved, time_origin_attestor=attestor, runtime_authority=runtime,
    )
    assert result["status"] == "applied"
    assert recover_approved_source_analysis_cohort(
        run_dir=tmp_path / "output", plan=plan, universe_path=paths["parquet"],
    )["authority_ref"] == result["authority_ref"]
