"""Exercise the real source -> plan -> registered executor owner chain."""

from dataclasses import replace
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

from easyicu.research_agent.acquisition.foundation import AcquisitionResult
from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HospitalMortalityFollowup,
)
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.acquisition.time_varying_materialization import (
    TimeVaryingMaterializationError,
    materialize_time_varying_acquisition,
)
from easyicu.research_agent.canonical_json import sha256_file
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.time_varying_exposure import (
    TimeVaryingExposureSpecification,
    TIME_VARYING_INPUT_METADATA_KEY,
)
from easyicu.research_agent.contracts.time_varying_validation import (
    time_varying_runtime_bundle_errors,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.execution.runners.time_varying_executor import (
    run_time_varying_association,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.capability_registry import (
    assess_scientific_capability,
    resolve_primary_capability,
)
from easyicu.research_agent.planning.scientific_review import timing_design_closed
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
from easyicu.research_agent.reporting.scientific_maturity import (
    build_scientific_maturity_audit,
)
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)


def _spec():
    return TimeVaryingExposureSpecification.model_validate(
        {
            "exposure_concept": "lact",
            "exposure_summary": "running_max_of_direct_measurements",
            "exposure_window_hours": 24,
            "followup": "hospital_death_or_discharge_from_icu_admission",
            "missingness_policy": "observed_state_indicator",
            "baseline_columns": ["age", "sex"],
            "baseline_categorical_encodings": {
                "sex": {
                    "kind": "binary_indicator",
                    "output_column": "sex_binary",
                    "positive_level": "Male",
                    "negative_level": "Female",
                    "unknown_or_missing_policy": "reject",
                }
            },
            "interpretation": "descriptive_time_updated_association_not_causal",
        }
    )


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(1279)
    n = 300
    stay = np.arange(100001, 100001 + n)
    baseline = pd.DataFrame(
        {
            "stay_id": stay,
            "age": rng.uniform(20, 80, n),
            "sex": np.where(rng.random(n) < 0.5, "Female", "Male"),
        }
    )
    cohort_path = tmp_path / "source_cohort.parquet"
    baseline.to_parquet(cohort_path, index=False)
    provenance_path = tmp_path / "source_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.cohort_materializer/1",
                "cohort_window_hours": [0.0, 24.0],
                "feature_concepts": ["lact"],
                "outcome_concepts": ["death"],
                "static_concepts": ["age", "sex"],
                "n_stays_after_inclusion_exclusion": len(baseline),
                "columns": list(baseline),
                "cohort_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        baseline.reset_index(drop=True), index=False
                    ).values.tobytes()
                ).hexdigest(),
                "cohort_file_sha256": sha256_file(cohort_path),
                "cohort_file_size": cohort_path.stat().st_size,
                "producer_parameters": {},
            }
        )
    )
    measured = stay[np.arange(n) % 3 != 0]
    trajectory = pd.DataFrame(
        {
            "stay_id": measured,
            "charttime": 2.0,
            "concept": "lact",
            "value_num": rng.uniform(0.7, 6, len(measured)),
            "value_str": None,
            "evidence_state": "direct_observed",
            "owner_observed": True,
            "owner_available": True,
        }
    )
    trajectory_path = tmp_path / "source_trajectory.parquet"
    trajectory.to_parquet(trajectory_path, index=False)
    mapping_path = tmp_path / "private_mapping.parquet"
    pd.DataFrame(
        {"stay_id": stay, "patient_key": np.arange(n) // 2 + 91000}
    ).to_parquet(mapping_path, index=False)
    mapping_sha = sha256_file(mapping_path)
    grouping = PatientGroupingBinding(
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/private-map",
            "mapping_sha256": mapping_sha,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    duration = rng.uniform(3, 100, n)
    event = (rng.random(n) < 0.45).astype("int8")
    duration[0], event[0] = 0.5, 1
    followup = HospitalMortalityFollowup(
        frame=pd.DataFrame(
            {
                "stay_id": stay,
                "hospital_death": event,
                "death_time_hours": np.where(event, duration, np.nan),
                "hospital_followup_time_hours": duration,
            }
        ),
        exclusions=pd.DataFrame(columns=["stay_id", "reason_code"]),
        receipt={
            "schema_version": "easyicu.mimic_iv_hospital_mortality_followup/1",
            "database": "synthetic",
            "analysis_unit": "icu_stay",
            "time_origin": "icu_admission",
            "time_unit": "hours",
            "event": {
                "column": "hospital_death",
                "definition": "synthetic binary event",
                "event_time_column": "death_time_hours",
                "event_time_source": "synthetic",
            },
            "censoring": {
                "followup_time_column": "hospital_followup_time_hours",
                "rule": "event else administrative censoring",
                "source": "synthetic",
            },
            "input_stays": n,
            "valid_stays": n,
            "excluded_stays": 0,
            "event_stays": int(event.sum()),
            "censored_stays": int(n - event.sum()),
            "zero_time_event_stays": 0,
            "zero_time_censored_stays": 0,
            "exclusion_counts": {},
            "privacy": {
                "identifier_values_returned": False,
                "raw_rows_returned": False,
                "source_paths_returned": False,
            },
        },
    )
    acquisition = AcquisitionResult(
        universe_path=cohort_path,
        provenance_path=provenance_path,
        trajectory_path=trajectory_path,
        trajectory_provenance_path=provenance_path,
        selection=None,
        coverage=None,
        materialized_concepts=["lact", "age", "sex", "death"],
    )
    return acquisition, followup, grouping


def _materialize(source):
    acquisition, followup, grouping = source
    return materialize_time_varying_acquisition(
        acquisition,
        specification=_spec(),
        hospital_followup=followup,
        patient_grouping=grouping,
        raw_source_receipt={"authority_ref": "synthetic/raw"},
        exposure_column="lact",
    )


def _projection(
    path,
    *,
    cohort=None,
    literature_citation_keys=(),
    direct_comparator_literature_keys=(),
):
    row = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "time_varying_exposure",
            "axis": "timing",
            "strategy": "time_varying",
            "execution_variables": ["lact"],
            "time_varying_execution": _spec().model_dump(mode="json"),
        }
    )
    return compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact", "cohort": cohort or {}},
        sensitivity_specs=[row],
        primary_exposure="lact",
        primary_exposure_source="lact",
        target_outcome="death",
        declared_covariates=["age", "sex"],
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=path,
        scientific_configuration_sha256="a" * 64,
        literature_citation_keys=literature_citation_keys,
        direct_comparator_literature_keys=direct_comparator_literature_keys,
        dependence=PlannedDependenceRequirement(
            group_source="patient_stay_id",
            group_derivation="prefix_before_delimiter",
            delimiter=":s",
        ),
    )


def test_time_varying_projection_binds_sealed_method_and_comparator_sources(source):
    acquired = _materialize(source)
    projection = _projection(
        acquired.universe_path,
        literature_citation_keys=(
            "strobe_2007",
            "record_2015",
            "suissa_immortal_time_2008",
            "grambsch_therneau_ph_1994",
            "chebl_serum_2020_31179840",
        ),
        direct_comparator_literature_keys=("chebl_serum_2020_31179840",),
    )
    authorities = ScientificRuntimeAuthorities.load(
        trajectory=None, current_case=projection.authority
    )
    plan, _ = authorities.development_execution_only_plan(
        research_question="Time-updated lactate and hospital mortality"
    )
    primary = authorities.current_case.governed_step(plan)

    assert primary.literature_citation_keys == [
        "suissa_immortal_time_2008",
        "strobe_2007",
        "record_2015",
        "grambsch_therneau_ph_1994",
        "chebl_serum_2020_31179840",
    ]
    bindings = {
        item.citation_key: item for item in primary.literature_design_bindings
    }
    assert bindings["suissa_immortal_time_2008"].design_elements == [
        "time_zero",
        "exposure",
        "estimand",
    ]
    assert bindings["strobe_2007"].design_elements == ["reporting", "dependence"]
    assert bindings["chebl_serum_2020_31179840"].divergence
    selected = next(
        item for item in plan.design_selection.candidates if item.disposition == "selected"
    )
    assert selected.literature_citation_keys == primary.literature_citation_keys


def test_source_projection_preserves_early_events_unmeasured_and_private_groups(source):
    acquired = _materialize(source)
    cohort, panel = (
        pd.read_parquet(acquired.universe_path),
        pd.read_parquet(acquired.trajectory_path),
    )
    assert len(cohort) == 300
    assert len(panel) > len(cohort)
    assert (
        int(cohort.death.sum())
        == int(panel.hospital_death.sum())
        == int(source[1].frame.hospital_death.sum())
    )
    assert panel.analysis_cluster_index.nunique() == 150
    assert panel.loc[panel.interval_stop_hours.eq(0.5), "hospital_death"].sum() == 1
    assert cohort.lact.isna().sum() == 100
    assert not {"stay_id", "patient_key", "__private_patient_group"}.intersection(
        cohort.columns
    )
    assert not {"stay_id", "patient_key", "__private_patient_group"}.intersection(
        panel.columns
    )
    assert source[0].universe_path != acquired.universe_path
    receipt = json.loads(
        pq.read_metadata(acquired.trajectory_path).metadata[
            TIME_VARYING_INPUT_METADATA_KEY.encode()
        ]
    )
    assert receipt["specification_sha256"] == _spec().sha256
    assert receipt["source_cohort_sha256"] == sha256_file(source[0].universe_path)
    assert receipt["claim_ceiling"] == "analysis_only"
    assert str(source[2].mapping_path) not in json.dumps(receipt)
    assert "91000" not in json.dumps(receipt)


def test_registered_runtime_replaces_static_model_and_keeps_analysis_only(source):
    acquired = _materialize(source)
    projection = _projection(acquired.universe_path)
    authorities = ScientificRuntimeAuthorities.load(
        trajectory=None, current_case=projection.authority
    )
    plan, finding = authorities.development_execution_only_plan(
        research_question="Time-updated lactate and hospital mortality"
    )
    assert finding.detail["analysis_only"] is True
    authorities.validate_plan(plan)
    assert timing_design_closed(plan)
    assert plan.cohort.selection_mode == "all_input_rows"
    assert not plan.cohort.inclusion and not plan.cohort.exclusion
    assert plan.endpoint is not None
    assert plan.endpoint.model_dump(mode="json") == {
        "name": "death",
        "kind": "binary",
        "absence_semantics": "no_absent_rows",
        "levels": [0, 1],
        "event_column": None,
        "time_column": None,
        "time_origin": None,
        "censoring_rule": None,
    }
    # The final host compiler must preserve the exact denominator as well as
    # the execution step when generic plan shaping is reapplied.
    rebound, _ = authorities.bind_plan(plan)
    assert rebound.cohort == plan.cohort
    primary = authorities.current_case.governed_step(plan)
    selection = select_standard_executor(
        primary,
        plan=plan,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert selection.analysis_kind == "signed_time_varying_exposure_cox"
    assert "COHORT_TRAJECTORY_PARQUET" in selection.code
    compile(selection.code, "registered_time_varying.py", "exec")
    verdict = resolve_primary_capability(analysis_type="association", plan=plan)
    assert verdict.capability.scientific_validation == "analysis_only"
    assert all("logit" not in (step.method or "") for step in plan.steps)
    bad = plan.model_copy(deep=True)
    bad.steps[1].method = "adjusted_association_models"
    with pytest.raises(ValueError, match="drifted"):
        authorities.validate_plan(bad)
    bad = plan.model_copy(update={"cohort": None})
    with pytest.raises(ValueError, match="cohort selection"):
        authorities.validate_plan(bad)
    bad = plan.model_copy(update={"endpoint": None})
    with pytest.raises(ValueError, match="hospital endpoint"):
        authorities.validate_plan(bad)


@pytest.mark.parametrize(
    "cohort",
    [{"age_min": 18}, {"exclude_readmissions": True}, {"min_icu_los_hours": 24}],
)
def test_projection_does_not_erase_declared_cohort_filters(source, cohort):
    from easyicu.webserver.scientific_runtime_projection import (
        WebScientificRuntimeProjectionError,
    )

    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        _projection(source[0].universe_path, cohort=cohort)
    assert caught.value.code == "web_time_varying_filtered_cohort_not_bound"


def test_pipeline_persists_time_varying_plan_with_required_cohort_without_provider(
    source, tmp_path
):
    from pathlib import Path

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    acquired = _materialize(source)
    projection = _projection(acquired.universe_path)
    client = ScriptedMockLLMClient([])
    pipeline = ResearchAgentPipeline(
        config=PipelineConfig(
            workdir=tmp_path / "pipeline",
            planner_only=True,
            development_diagnostic=True,
            require_human_plan_review=True,
            required_primary_cohort_selection_mode="all_input_rows",
            current_case_scientific_runtime_authority=projection.authority,
            scientific_runtime_projection_sha256=projection.projection_sha256,
            enable_memory=False,
            enable_replanning=False,
        ),
        services=PipelineServices(llm=client),
    )
    outcome = pipeline.run(
        question="Time-updated lactate and hospital mortality",
        cohort=acquired.universe_path,
        trajectory_path=acquired.trajectory_path,
        database="synthetic",
        target_outcome="death",
        primary_exposure="lact",
        id_columns=["patient_stay_id"],
        user_preferences={
            "covariates": ["age", "sex"],
            "covariate_selection": "exact",
            "covariate_rationales": {"age": "baseline age", "sex": "baseline sex"},
            "covariate_temporal_roles": {
                "age": "baseline_static",
                "sex": "baseline_static",
            },
            "covariate_operationalizations": {"age": "age", "sex": "sex"},
        },
        stop_after_analysis=True,
    )
    assert type(outcome).__name__ == "HumanReviewPending"
    assert not client.calls
    plan = json.loads(Path(outcome.run_dir, "analysis_plan.json").read_text())
    context = json.loads(Path(outcome.run_dir, "research_context.json").read_text())
    assert plan["cohort"]["selection_mode"] == "all_input_rows"
    assert plan["endpoint"] == {
        "name": "death",
        "kind": "binary",
        "absence_semantics": "no_absent_rows",
        "levels": [0, 1],
        "event_column": None,
        "time_column": None,
        "time_origin": None,
        "censoring_rule": None,
    }
    assert len(plan["steps"]) == 2
    assert plan["steps"][1]["method"] == "time_varying_exposure_model"
    identity = context["cohort"]["provenance"]["replacement_row_identity"]
    assert identity["mapped_cohort_rows"] == 300
    assert identity["patient_group_derivation"] == {
        "algorithm": "prefix_before_:s",
        "delimiter": ":s",
    }
    typed_context = ResearchContext.model_validate(context)
    typed_plan = AnalysisPlan.model_validate(plan)
    assessment = assess_scientific_capability(
        analysis_type="association",
        context=typed_context,
        plan=typed_plan,
    )
    assert assessment.scientific_validator_available is True
    assert assessment.claim_ceiling == "analysis_only"
    assert assessment.issue_code is None
    maturity = build_scientific_maturity_audit(
        context=typed_context,
        plan=typed_plan,
        run_dir=Path(outcome.run_dir),
    )
    assert "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE" not in {
        finding.code for finding in maturity.findings
    }
    assert maturity.facts["primary_covariates"] == ["age", "sex"]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="real R survival required")
def test_registered_execution_fits_real_r_and_records_source_and_policy(
    source, tmp_path
):
    acquired = _materialize(source)
    projection = _projection(acquired.universe_path)
    authorities = ScientificRuntimeAuthorities.load(
        trajectory=None, current_case=projection.authority
    )
    plan, _ = authorities.development_execution_only_plan(
        research_question="Time-updated lactate and hospital mortality"
    )
    step = plan.steps[1]
    output_dir = tmp_path / "steps" / step.step_id / "outputs"
    result = run_time_varying_association(
        frame=pd.read_parquet(acquired.universe_path),
        trajectory_path=acquired.trajectory_path,
        authority=projection.authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=output_dir,
    )
    assert result["status"] == "ok"
    assert result["n_total"] == 300
    assert result["cluster_count"] == 150
    receipt = result["scientific_runtime_receipt"]
    assert receipt["publication_ready"] is False
    assert receipt["fit"]["diagnostics"] == {"converged": True, "warnings": []}
    assert receipt["counting_process_input_sha256"] == sha256_file(
        acquired.trajectory_path
    )
    assert len(result["output_files"]) == 3
    result["input_bindings"] = [
        {
            "input_key": "table:analysis_cohort",
            "loaded": True,
            "row_count": 300,
            "sha256": receipt["construction"]["analysis_cohort_sha256"],
        }
    ]
    records = [
        {
            "step_id": step.step_id,
            "status": "ok",
            "deterministic_standard_analysis": "signed_time_varying_exposure_cox",
            "step_summary": result,
        }
    ]
    assert time_varying_runtime_bundle_errors(
        plan=plan, records=records, run_dir=tmp_path
    ) == []

    estimates = output_dir / "time_varying_cox_estimates.csv"
    estimates.write_text(
        estimates.read_text(encoding="utf-8").replace(
            "exposure_running_max_when_observed,", "unexpected_term,", 1
        ),
        encoding="utf-8",
    )
    assert "estimate table" in " ".join(
        time_varying_runtime_bundle_errors(
            plan=plan, records=records, run_dir=tmp_path
        )
    )


@pytest.mark.parametrize(
    "change", ["cohort_rows", "outcome", "baseline", "exposure", "metadata"]
)
def test_runtime_rejects_drift_before_fitting(source, tmp_path, change):
    acquired = _materialize(source)
    projection = _projection(acquired.universe_path)
    frame = pd.read_parquet(acquired.universe_path)
    if change == "cohort_rows":
        frame = frame.iloc[1:]
    elif change == "metadata":
        pd.read_parquet(acquired.trajectory_path).to_parquet(
            acquired.trajectory_path, index=False
        )
    elif change == "outcome":
        frame.loc[1, "death"] = 1 - int(frame.loc[1, "death"])
    else:
        column = {"baseline": "age", "exposure": "lact"}[change]
        frame.loc[1, column] = 500.0
    out = tmp_path / "bad_outputs"
    with pytest.raises(ValueError):
        run_time_varying_association(
            frame=frame,
            trajectory_path=acquired.trajectory_path,
            authority=projection.authority,
            runtime_projection_sha256=projection.projection_sha256,
            out_dir=out,
        )
    assert not out.exists()


def test_reject_native_authority_downgrade_and_existing_output(source):
    acquisition, followup, grouping = source
    native = replace(
        acquisition,
        cohort_authority_path=acquisition.universe_path,
        cohort_authority_ref=object(),
    )
    with pytest.raises(TimeVaryingMaterializationError) as caught:
        _materialize((native, followup, grouping))
    assert caught.value.code == "time_varying_native_lineage_extension_required"
    _materialize(source)
    with pytest.raises(TimeVaryingMaterializationError) as caught:
        _materialize(source)
    assert caught.value.code == "time_varying_artifact_exists"


def test_landmark_materialization_uses_hospital_hours_not_icu_duration(source):
    from easyicu.research_agent.acquisition.hospital_followup_materialization import (
        materialize_hospital_followup_acquisition,
    )
    from easyicu.research_agent.intake.legacy_materialization import (
        load_verified_legacy_materialization_provenance,
    )

    acquisition, followup, _ = source
    prepared = materialize_hospital_followup_acquisition(
        acquisition, followup=followup, raw_source_receipt={"authority_ref": "test"}
    )
    frame = pd.read_parquet(prepared.universe_path)
    assert prepared.provenance_path.name == "hospital_followup_cohort_provenance.json"
    verified = load_verified_legacy_materialization_provenance(
        prepared.universe_path,
        cohort=frame,
    )
    assert verified is not None
    assert verified["n_stays_after_inclusion_exclusion"] == len(frame)
    assert verified["cohort_file_sha256"] == sha256_file(prepared.universe_path)
    assert len(frame) == 300
    assert frame.loc[0, "death_time_hours"] == 0.5
    np.testing.assert_allclose(
        frame.hospital_followup_time_hours, followup.frame.hospital_followup_time_hours
    )
    row = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "landmark_24h",
            "axis": "timing",
            "strategy": "landmark",
            "landmark_hours": 24,
            "execution_variables": ["death_time_hours", "hospital_followup_time_hours"],
            "event_time_variable": "death_time_hours",
            "observation_duration_variable": "hospital_followup_time_hours",
            "observation_duration_unit": "hours",
        }
    )
    assert row.source_materialization_variables == ()


def test_landmark_followup_keeps_verified_patient_grouping_visible_to_planning(
    source,
):
    from easyicu.research_agent.acquisition.hospital_followup_materialization import (
        materialize_hospital_followup_acquisition,
    )
    from easyicu.research_agent.cohort.materializer import _hash_df
    from easyicu.research_agent.planning.dependence_authority import (
        context_dependence_authority,
    )
    from easyicu.research_agent.research_context.builder import build_research_context

    acquisition, followup, grouping = source
    frame = pd.read_parquet(acquisition.universe_path)
    mapping = pd.read_parquet(grouping.mapping_path).set_index("stay_id")
    frame["patient_stay_id"] = [
        f"p{int(mapping.loc[stay, 'patient_key'])}:s{int(stay)}"
        for stay in frame["stay_id"]
    ]
    frame = frame.drop(columns=["stay_id"])
    frame.to_parquet(acquisition.universe_path, index=False)
    provenance = json.loads(acquisition.provenance_path.read_text())
    provenance.update(
        {
            "columns": list(frame),
            "cohort_sha256": _hash_df(frame.reset_index(drop=True)),
            "cohort_file_sha256": sha256_file(acquisition.universe_path),
            "cohort_file_size": acquisition.universe_path.stat().st_size,
            "replacement_row_identity": {
                "mapping_file_sha256": grouping.mapping_sha256,
                "output_identity_column": "patient_stay_id",
                "mapped_cohort_rows": len(frame),
                "patient_group_derivation": {
                    "algorithm": "prefix_before_:s",
                    "delimiter": ":s",
                },
                "authority_coordinates": dict(grouping.authority_coordinates),
            },
        }
    )
    acquisition.provenance_path.write_text(json.dumps(provenance))

    prepared = materialize_hospital_followup_acquisition(
        acquisition,
        followup=followup,
        raw_source_receipt={"authority_ref": "test"},
    )
    context = build_research_context(
        research_question="Is a first-day exposure associated with hospital death?",
        cohort=prepared.universe_path,
        cohort_name="patient-grouped landmark cohort",
        database="miiv",
        target_outcome="death",
        id_columns=["patient_stay_id"],
        user_preferences={
            "data_constraints": json.dumps(
                {
                    "analysis_design": {
                        "analysis_unit": "icu_stay",
                        "cluster_unit": "patient",
                        "variance_estimator": "cluster_robust",
                    }
                }
            )
        },
    )

    replacement = context.cohort.provenance["replacement_row_identity"]
    assert replacement["mapped_cohort_rows"] == len(pd.read_parquet(prepared.universe_path))
    assert context.cohort.n_patients == 150
    dependence = context_dependence_authority(context)
    assert dependence is not None
    assert dependence.group_source == "patient_stay_id"
    assert dependence.group_derivation == "prefix_before_delimiter"


def test_landmark_followup_accounts_for_excluded_and_uncovered_stays(source):
    from easyicu.research_agent.acquisition.hospital_followup_materialization import (
        materialize_hospital_followup_acquisition,
    )

    acquisition, followup, _ = source
    clipped = HospitalMortalityFollowup(
        frame=followup.frame.iloc[1:].copy(), exclusions=followup.exclusions, receipt={}
    )
    with pytest.raises(ValueError, match="coverage_incomplete"):
        materialize_hospital_followup_acquisition(
            acquisition, followup=clipped, raw_source_receipt={}
        )
    clipped = HospitalMortalityFollowup(
        frame=clipped.frame,
        exclusions=pd.DataFrame(
            {
                "stay_id": [followup.frame.iloc[0].stay_id],
                "reason_code": ["invalid_chronology"],
            }
        ),
        receipt={},
    )
    prepared = materialize_hospital_followup_acquisition(
        acquisition, followup=clipped, raw_source_receipt={}
    )
    assert len(pd.read_parquet(prepared.universe_path)) == 299
    receipt = json.loads(prepared.provenance_path.read_text())[
        "hospital_followup_materialization"
    ]
    assert receipt["excluded_stays"] == 1
