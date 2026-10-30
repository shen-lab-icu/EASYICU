"""Synthetic tests of deterministic landmark route, not human authorization."""
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.case_duration_landmark import propose_case_duration_landmark_authority
from easyicu.research_agent.execution.runners.care_unit_landmark_cohort_executor import care_unit_landmark_cohort_executor_code
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDefinition
from tests.research_agent.authority.test_case_duration_landmark_authority import _inputs, _PROVENANCE


def _case():
    origin, outcome, axis, primary = _inputs()
    authority = propose_case_duration_landmark_authority(
        origin=origin, outcome=outcome, source_axis=axis, source_provenance=_PROVENANCE, primary_step=primary,
    )
    risk = AnalysisStep(
        step_id="landmark", intent="Apply the approved risk set", method="signed_care_unit_landmark_cohort",
        planned_analysis_role="auxiliary", expected_outputs=["artifact:analysis_cohort"],
        inputs=[authority.proposed_outcome, authority.death_status_column, authority.death_time_column,
                authority.exposure_column], icu_rule_refs=[authority.plan_rule_ref],
    )
    primary = primary.model_copy(update={"inputs": ["artifact:analysis_cohort"], "icu_rule_refs": [authority.plan_rule_ref]})
    plan = AnalysisPlan(research_question="Synthetic case-duration proxy", analysis_type="association_study",
                        cohort=CohortDefinition(name="all"), steps=[risk, primary])
    authority.validate_plan(plan)
    return risk, plan, authority


def test_landmark_route_never_publishes_unsealed_cohort(tmp_path: Path, monkeypatch):
    risk, plan, authority = _case()
    script = care_unit_landmark_cohort_executor_code(risk, plan=plan, authority=authority)
    source, output = tmp_path / "cohort.parquet", tmp_path / "out"
    pd.DataFrame({"stay_id": [1]}).to_parquet(source)
    monkeypatch.setenv("COHORT_PARQUET", str(source))
    monkeypatch.setenv("STEP_OUT_DIR", str(output))
    with pytest.raises(ValueError, match="no sealed authority"):
        exec(compile(script, "<landmark>", "exec"), {})
    assert not (output / "analysis_cohort.parquet").exists()
    with pytest.raises(ValueError, match="signed case-duration authority"):
        care_unit_landmark_cohort_executor_code(risk, plan=plan, authority=None)


def test_pending_candidate_never_grants_case_duration_preparation():
    from types import SimpleNamespace
    from easyicu.webserver.agent_pipeline_runs import ResearchPipelineRunError, _require_no_unsigned_case_duration

    _require_no_unsigned_case_duration(SimpleNamespace(facts={}))
    with pytest.raises(ResearchPipelineRunError, match="complete signed research plan"):
        _require_no_unsigned_case_duration(SimpleNamespace(facts={"database_case_duration_review": {
            "execution_contract_available": True,
        }}))


def test_metadata_attestor_resolves_only_one_matching_registered_source(tmp_path, monkeypatch):
    from easyicu.databases.research_substitutions import time_origin_selection
    from easyicu.webserver import case_duration_materialization as host
    from easyicu.webserver import time_origin_source_authority as source
    from tests.support.signed_case_duration import signed_case_duration_case

    context, *_ = signed_case_duration_case()
    selection = time_origin_selection(context)
    row = {"id": "src_example", "database": "sic", "version": "1.0.6", "path": str(tmp_path), "ok": True}
    monkeypatch.setattr(host.sources, "load_registry", lambda: {"sources": [row]})
    monkeypatch.setattr(source, "prepare_case_duration_metadata_plan", lambda path, **_: (path, "metadata-only"))
    assert host.attest_case_duration_metadata(context, selection) == (str(tmp_path), "metadata-only")
    monkeypatch.setattr(host.sources, "load_registry", lambda: {"sources": [row, row]})
    with pytest.raises(ValueError, match="registration_not_unique"):
        host.attest_case_duration_metadata(context, selection)


def test_signed_synthetic_landmark_materializes_reuses_and_publishes(tmp_path, monkeypatch):
    from easyicu.databases.research_approval import bind_database_capability_review, require_approved_database_capabilities
    from easyicu.databases.research_substitutions import case_duration_selection, time_origin_selection
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN, SourceTimeAxisBinding
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.cohort.schema import materialize_approved_care_unit_landmark
    from easyicu.research_agent.intake.export_package import open_export_package
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.webserver.time_origin_source_authority import CaseDurationSourceProvenance
    from easyicu.webserver import time_origin_source_authority as source_authority
    from easyicu.webserver.dataio import make_export_runner
    from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files
    from easyicu.webserver import case_duration_materialization as host_materializer
    from tests.research_agent.planning.test_time_origin_review_binding import _review
    from tests.support.data_provenance import ExportJob
    from tests.support.signed_case_duration import signed_case_duration_case
    from tests.support.source_substitutions import _approve

    export = tmp_path / "export"
    export.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    raw.mkdir()
    lact = [float(x) for x in range(1, 11)]
    days = [2.0 + 0.3*x + (0.1 if x % 2 else -0.1) for x in range(1, 9)] + [1.0, 2.0]
    pd.DataFrame({"CaseID": list(range(1, 11)), "ICUOffset": [0]*10,
                  "TimeOfStay": [round(day * 86400) for day in days],
                  "OffsetOfDeath": [None]*8 + [43200, None],
                  "HospitalDischargeType": [2026]*8 + [2028, 2028]}).to_parquet(raw / "cases.parquet", index=False)
    pd.DataFrame({"CaseID": list(range(1, 11)) + [1], "Offset": [3600]*10 + [86400],
                  "LaboratoryID": [657]*11, "LaboratoryValue": lact + [999.0]}).to_parquet(
        raw / "laboratory.parquet", index=False)
    stage_reviewed_source_files(source_root=raw, staging_root=stage,
        relative_files=["cases.parquet", "laboratory.parquet"], source_identity_sha256="a" * 64)
    (raw / "cases.parquet").unlink()
    (raw / "laboratory.parquet").unlink()
    make_export_runner(data_path=str(stage), database="sic", modules=["outcome", "blood_gas", "sic_case_landmark"],
        concepts={"outcome": ["los_icu"], "blood_gas": ["lact"], "sic_case_landmark": [
            "care_unit_to_case_closure_days", "death_time_during_case_hours", "death_during_case"]},
        export_format="parquet", out_dir=str(export), include_feature_definitions=False)(ExportJob())
    raw_case_output = pd.read_parquet(export / "sic_case_landmark.parquet")
    assert raw_case_output["death_during_case"].iloc[:9].tolist() == [False]*8 + [True]
    assert pd.isna(raw_case_output["death_during_case"].iloc[9])
    assert raw_case_output["death_time_during_case_hours"].iloc[8] == 12
    context, plan, _, _, _, _, _ = signed_case_duration_case()
    context = context.model_copy(update={"primary_exposure": "lact_max", "variables": [
        v.model_copy(update={"name": "lact_max", "source_concept": "lact"}) if v.name == "lact" else v
        for v in context.variables]})
    origin, outcome = time_origin_selection(context), case_duration_selection(context)
    with open_export_package(export) as package:
        axis = SourceTimeAxisBinding(origin.source_identity_sha256, package.authority_sha256,
                                     package.manifest_sha256, SIC_CARE_UNIT_ORIGIN)
        proof = source_authority.case_duration_source_provenance(package)
    monkeypatch.setattr(host_materializer.sources, "load_registry", lambda: {"sources": [{
        "id": "src_example", "database": "sic", "version": "1.0.6", "path": str(export), "ok": True,
    }]})
    origin_attestor = source_authority.review_time_origin_attestor
    case_attestor = source_authority.attest_case_duration_for_review
    assert origin_attestor(context, origin)[1].to_dict() == case_attestor(context, origin)[0].to_dict()
    old_primary = plan.steps[1]
    old_requirement = old_primary.model_requirements[0]
    term = old_requirement.model_terms[0].model_copy(update={"name": "lact_max"})
    requirement = old_requirement.model_copy(update={"exposure_source": "lact_max", "model_terms": [term]})
    primary = old_primary.model_copy(update={"model_requirements": [requirement],
        "inputs": ["artifact:analysis_cohort", "lact_max", outcome.proposed_outcome]})
    authority = propose_case_duration_landmark_authority(origin=origin, outcome=outcome, source_axis=axis,
                                                        source_provenance=proof, primary_step=primary)
    risk = plan.steps[0].model_copy(update={"icu_rule_refs": [authority.plan_rule_ref],
        "inputs": [outcome.proposed_outcome, "death_during_case", "death_time_during_case_hours", "lact_max"]})
    primary = primary.model_copy(update={"icu_rule_refs": [authority.plan_rule_ref]})
    plan = plan.model_copy(update={"steps": [risk, primary]})
    receipt = case_attestor(context, origin)[0]
    base = _review(context, plan).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=context, plan=plan, time_origin_attestor=origin_attestor,
                                             case_duration_attestor=case_attestor, runtime_authority=authority)
    assert review.approval_allowed
    from types import SimpleNamespace
    from easyicu.databases.research_approval import CapabilityApprovalError
    from easyicu.research_agent.execution.phase import _materialize_approved_case_duration_if_needed
    evidence = EvidenceStore(run_dir)
    catalog = run_dir / "planning_catalog.parquet"
    pd.DataFrame({"stay_id": pd.Series(dtype="int64")}).to_parquet(catalog, index=False)
    calls = []
    services = SimpleNamespace(time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor,
        approved_case_duration_materializer=lambda **kwargs: calls.append(kwargs) or catalog)
    pipeline = SimpleNamespace(_services=services, _scientific_runtime_authorities=SimpleNamespace(current_case=authority))
    with pytest.raises(CapabilityApprovalError):
        _materialize_approved_case_duration_if_needed(pipeline,
            plan_result=SimpleNamespace(context=context, plan=plan, evidence=evidence), cohort_path=catalog, run_dir=run_dir)
    assert calls == []
    approved = _approve(run_dir, evidence, context, plan, review)
    services.approved_case_duration_materializer = host_materializer.materialize_approved_case_duration_universe
    universe_path = _materialize_approved_case_duration_if_needed(
        pipeline, plan_result=approved, cohort_path=catalog, run_dir=run_dir)
    from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
    verified_universe = load_verified_materialized_cohort_authority(universe_path)
    host_materializer.require_case_duration_exposure_materialization(
        verified_universe, authority, source_concept="lact",
    )
    with pytest.raises(ValueError, match="exposure_source_concept_mismatch"):
        host_materializer.require_case_duration_exposure_materialization(
            verified_universe, authority, source_concept="glu",
        )
    with pytest.raises(ValueError, match="exposure_window_unverified"):
        host_materializer.require_case_duration_exposure_materialization(
            verified_universe, authority.model_copy(update={"exposure_column": "death_during_case"}),
        )
    assert _materialize_approved_case_duration_if_needed(
        pipeline, plan_result=approved, cohort_path=catalog, run_dir=run_dir) == universe_path
    # A separately reviewed context can keep the same plan and column name
    # while binding it to a different raw concept. It must not reuse the old
    # materialization, even when source and plan hashes are unchanged.
    changed_context = context.model_copy(update={"variables": [
        v.model_copy(update={"source_concept": "glu"}) if v.name == "lact_max" else v
        for v in context.variables
    ]})
    from easyicu.databases.research_approval import ApprovedCaseDurationBinding
    changed_result = SimpleNamespace(context=changed_context, plan=plan)
    changed_binding = ApprovedCaseDurationBinding(source_axis=axis, authority=authority)
    with monkeypatch.context() as patch:
        def require_new_materialization(*args, **kwargs):
            assert kwargs["stem"] != universe_path.stem
            assert kwargs["feature_concepts"] == ["glu"]
            raise ValueError("new_context_requires_materialization")
        patch.setattr(host_materializer, "materialize_to_parquet", require_new_materialization)
        with pytest.raises(ValueError, match="new_context_requires_materialization"):
            host_materializer.materialize_approved_case_duration_universe(
                plan_result=changed_result, run_dir=run_dir, approved=changed_binding,
            )
    arguments = dict(run_dir=run_dir, universe_path=universe_path, plan_result=approved,
                     time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor,
                     runtime_authority=authority)
    stale = CaseDurationSourceProvenance(proof.snapshot_sha256, "0" * 64, proof.source_config_sha256)
    with pytest.raises(CapabilityApprovalError):
        materialize_approved_care_unit_landmark(**{**arguments, "case_duration_attestor": lambda *_: (receipt, stale)})
    assert not (run_dir / "cohort_landmark.parquet").exists()
    result = materialize_approved_care_unit_landmark(**arguments)
    assert result["counts"] == {"input_rows": 10, "eligible_rows": 8, "unidentifiable_rows": 1,
                                "closed_or_dead_at_landmark_rows": 1}
    assert pd.read_parquet(result["path"])["stay_id"].tolist() == list(range(1, 9))
    assert pd.read_parquet(result["path"])["lact_max"].iloc[0] == 1.0
    assert materialize_approved_care_unit_landmark(**arguments)["reused"] is True
    from easyicu.research_agent.authority.execution_input import ExecutionInputAuthorityState
    from easyicu.research_agent.intake.materialized_metadata import MaterializedMetadataError
    binding = require_approved_database_capabilities(approved, run_dir, cohort_path=universe_path,
        time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor, runtime_authority=authority)
    input_args = dict(universe_path=universe_path, analysis_path=result["path"], trajectory_binding=None,
        run_dir=run_dir, legacy_trajectory_verifier=None, plan=plan, context=context)
    with pytest.raises(MaterializedMetadataError):
        ExecutionInputAuthorityState.bind(**input_args)
    input_state = ExecutionInputAuthorityState.bind(**input_args, approved_case_duration_binding=binding)
    assert input_state.selected_path == result["path"]
    input_state.rebind_cohort(plan=plan, context=context)
    wrong_plan = plan.model_copy(update={"research_question": "Changed after approval"})
    with pytest.raises(MaterializedMetadataError, match="signed plan"):
        ExecutionInputAuthorityState.bind(**{**input_args, "plan": wrong_plan}, approved_case_duration_binding=binding)
    with pytest.raises(MaterializedMetadataError, match="landmark cohort is missing"):
        ExecutionInputAuthorityState.bind(**{**input_args, "analysis_path": run_dir / "absent.parquet"},
                                          approved_case_duration_binding=binding)
    selected = select_standard_executor(risk, plan=plan, current_case_scientific_runtime_authority=authority)
    assert selected is not None
    monkeypatch.setenv("COHORT_PARQUET", str(result["path"]))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "step"))
    exec(compile(selected.code, "<signed-landmark>", "exec"), {})
    assert (tmp_path / "step" / "analysis_cohort.parquet").read_bytes() == result["path"].read_bytes()
    import json
    summary = json.loads((tmp_path / "step" / "step_summary.json").read_text())
    assert summary["output_files"] == {"artifact:analysis_cohort": "analysis_cohort.parquet"}
    from tests.research_agent.core.test_cross_sectional_phenotyping_executor import _binding
    product = tmp_path / "step" / "analysis_cohort.parquet"
    product_frame = pd.read_parquet(product)
    typed = _binding("artifact:analysis_cohort", product_frame, product, primary.step_id)
    typed["declared_kind"] = typed["identity_row"]["declared_kind"] = "artifact"
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps({"step_id": primary.step_id, "inputs": {"artifact:analysis_cohort": typed}}))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "model"))
    model_executor = select_standard_executor(primary, plan=plan, current_case_scientific_runtime_authority=authority)
    assert model_executor is not None
    exec(compile(model_executor.code, "<signed-ols>", "exec"), {})
    import numpy as np
    model_summary = json.loads((tmp_path / "model" / "step_summary.json").read_text())
    design = np.column_stack([np.ones(8), np.asarray(lact[:8])])
    expected = np.linalg.lstsq(design, np.asarray(days[:8]), rcond=None)[0][1]
    assert model_summary["primary_estimate"] == pytest.approx(expected)
    assert model_summary["n_total"] == 8
    assert model_summary["typed_cohort_input"] == "artifact:analysis_cohort"
