"""Explicit family builders describe existing owners without changing routing."""

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.current_case_scientific_runtime import build_current_case_scientific_runtime_authority
from easyicu.research_agent.authority.time_varying_runtime import TimeVaryingRuntimeAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning import family_spec
from easyicu.research_agent.planning.family_spec import FamilyPlanSpec, FamilySpecError, FamilySpecRequest
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from tests.support.target_trial import target_trial_authority_body


BUILDERS = {
    "trial": "build_target_trial_skeleton",
    "time_varying": "build_time_varying_survival_skeleton",
    "dynamic": "build_dynamic_landmark_skeleton",
}


def _request(kind):
    fields = dict(
        research_question="Evaluate the prespecified longitudinal study in synthetic ICU stays.",
        cohort_name="synthetic_study", cohort_selection_mode="all_input_rows", cluster_unit="patient",
        primary_exposure="", exposure_kind="none", exposure_levels=[], reference_level_index=0,
        primary_contrast_level_index=0, exposure_is_ordered=False,
        outcome_levels=["0", "1"], event_level_index=1, adjustment_selection="exact",
    )
    if kind == "trial":
        owner = build_current_case_scientific_runtime_authority(target_trial_authority_body())
        fields.update(
            family_id="target_trial_emulation", analysis_type="causal_inference", sealed_target_trial=owner,
            identity_column=owner.unit_id_column, outcome=owner.event_column,
            variable_roster=list(owner.required_columns), exact_roster=[item.column for item in owner.covariates],
        )
    elif kind == "time_varying":
        body = dict(
            schema_version="easyicu.time_varying_runtime_authority/1", authority_kind="time_varying_exposure_association",
            protocol_content_sha256="a" * 64, sensitivity_spec_id="time_varying_exposure",
            specification=dict(
                exposure_concept="lab_value", exposure_summary="running_max_of_direct_measurements",
                exposure_window_hours=24, followup="hospital_death_or_discharge_from_icu_admission",
                missingness_policy="observed_state_indicator", baseline_columns=["age"], baseline_categorical_encodings={},
                interpretation="descriptive_time_updated_association_not_causal",
            ),
            exposure_column="lab_value", outcome_column="death", identity_column="patient_stay_id",
            primary_cohort_selection_mode="all_input_rows", development_execution_only_allowed=True,
            plan_method="time_varying_exposure_model", plan_intent="Estimate the prespecified time-updated association.",
            plan_outputs=["table:time_varying_cox_estimates", "table:time_varying_input_audit", "log:time_varying_runtime_receipt"],
        )
        owner = TimeVaryingRuntimeAuthority.model_validate_json(json.dumps({**body, "execution_contract_sha256": canonical_sha256(body)}))
        fields.update(
            family_id="time_varying_exposure_survival", analysis_type="association_study", sealed_time_varying=owner,
            identity_column=owner.identity_column, outcome=owner.outcome_column,
            primary_exposure=owner.exposure_column, exposure_kind="continuous",
            variable_roster=list(owner.required_columns), exact_roster=list(owner.specification.baseline_columns),
        )
    else:
        coordinates = family_spec.DynamicLandmarkCoordinates(
            trajectory_bound=True, trajectory_time_unit="hours", trajectory_time_origin="icu_admission",
            patient_column="patient_id", event_time_column="event_hours", followup_end_column="followup_hours",
            feature_concepts=["pulse", "lab_value"], landmark_hours=[6, 12], lookback_hours=6,
            horizon_hours=[12, 24], aggregations=["last", "mean"],
        )
        fields.update(
            family_id="dynamic_landmark_prediction", analysis_type="dynamic_prediction", dynamic_landmark=coordinates,
            identity_column="stay_id", outcome="death", variable_roster=["stay_id", "death", *coordinates.source_columns],
        )
    fields["required_reader_label_keys"] = list(fields["variable_roster"])
    fields["level_label_keys"] = [f"{fields['outcome']}=0", f"{fields['outcome']}=1"]
    return FamilySpecRequest(**fields)


def _spec(request):
    return FamilyPlanSpec(
        family_id=request.family_id, request_sha256=request.request_sha256,
        reader_display_labels={key: "Study " + key.replace("_", " ") for key in [*request.required_reader_label_keys, *request.level_label_keys]},
        roster_decision_note="Use the host coordinates and their fixed analysis-only boundary.",
    )


def _context(request):
    patient = request.dynamic_landmark.patient_column if request.dynamic_landmark else (
        request.sealed_target_trial.patient_group_column if request.sealed_target_trial else request.identity_column
    )
    return ResearchContext(
        research_question=request.research_question,
        cohort=CohortDescriptor(
            cohort_name=request.cohort_name, database="miiv", n_stays=0, id_columns=[request.identity_column],
            outcome_columns=[request.outcome],
        ),
        variables=[ConceptDescriptor(
            name=name, description="Study " + name.replace("_", " "), source_concept=name,
            dtype="int64" if name == request.outcome else "float64",
            role="id" if name in {request.identity_column, patient} else "outcome" if name == request.outcome else "demographic",
            observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]} if name == request.outcome else None,
        ) for name in request.variable_roster],
        target_outcome=request.outcome, primary_exposure=request.primary_exposure or None,
        user_preferences=UserPreferences(inferred_analysis_family=request.analysis_type),
    )


def _build(kind):
    request = _request(kind)
    return request, getattr(family_spec, BUILDERS[kind])(request, _spec(request))


@pytest.mark.parametrize(
    ("family_constant", "family_id", "builder_name"),
    [
        ("TARGET_TRIAL_FAMILY_ID", "target_trial_emulation", "build_target_trial_skeleton"),
        ("TIME_VARYING_SURVIVAL_FAMILY_ID", "time_varying_exposure_survival", "build_time_varying_survival_skeleton"),
        ("DYNAMIC_LANDMARK_FAMILY_ID", "dynamic_landmark_prediction", "build_dynamic_landmark_skeleton"),
    ],
)
def test_families_publish_explicit_builders(family_constant, family_id, builder_name):
    assert getattr(family_spec, family_constant, None) == family_id
    assert callable(getattr(family_spec, builder_name, None))


@pytest.mark.parametrize("kind", BUILDERS)
def test_synthetic_requests_compile_through_the_existing_product_and_edge_validators(kind):
    request, built = _build(kind)
    skeleton = assemble_progressive_skeleton(
        outline=built.outline, foundation=built.foundation.foundation,
        steps=[item.step for item in built.materializations],
    )
    plan, receipt = compile_progressive_plan(skeleton=skeleton, context=_context(request))
    assert len(plan.steps) == len(skeleton.steps) == len(receipt.compiled_steps)
    assert sum(step.planned_analysis_role == "primary" for step in plan.steps) == 1
    assert plan.design_selection.claim_ceiling == "analysis_only"
    assert plan.steps[-1].expected_outputs == ["report:report"]
    producers = {output: step.step_id for step in plan.steps for output in step.expected_outputs}
    assert len(producers) == sum(len(step.expected_outputs) for step in plan.steps)
    for step in skeleton.steps:
        for reference in step.product_inputs:
            assert producers[reference.product_id] == reference.producer_step_id
            assert reference.producer_step_id in step.depends_on


@pytest.mark.parametrize("kind", BUILDERS)
def test_builders_fail_closed_on_missing_coordinates_even_after_model_copy(kind):
    request = _request(kind)
    field = {"trial": "sealed_target_trial", "time_varying": "sealed_time_varying", "dynamic": "dynamic_landmark"}[kind]
    request = request.model_copy(update={field: None})
    with pytest.raises(FamilySpecError) as error:
        getattr(family_spec, BUILDERS[kind])(request, _spec(request))
    assert error.value.reason_code == "family_spec_longitudinal_coordinates"


@pytest.mark.parametrize("kind", BUILDERS)
def test_builders_bind_the_exact_request_digest_and_family(kind):
    request = _request(kind)
    builder = getattr(family_spec, BUILDERS[kind])
    with pytest.raises(FamilySpecError, match="family_spec_request_digest_mismatch"):
        builder(request, _spec(request).model_copy(update={"request_sha256": "0" * 64}))
    other = _request("dynamic" if kind != "dynamic" else "trial")
    with pytest.raises(FamilySpecError, match="family_spec_template_mismatch"):
        builder(other, _spec(other))
    with pytest.raises(FamilySpecError, match="family_spec_family_mismatch"):
        builder(request, _spec(request).model_copy(update={"family_id": other.family_id}))


@pytest.mark.parametrize("kind", BUILDERS)
@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"variable_roster": ["unrelated"]}, "family_spec_longitudinal_source_missing"),
        ({"cluster_unit": None}, "family_spec_longitudinal_owner_mismatch"),
        ({"outcome_levels": ["1", "0"]}, "family_spec_longitudinal_endpoint"),
        ({"cohort_selection_mode": "predicate_filtered"}, "family_spec_longitudinal_population"),
        ({"subgroup_columns": ["age"]}, "family_spec_longitudinal_coordinate_unsupported"),
        ({"exact_roster": ["unrelated"]}, "family_spec_longitudinal_roster_mismatch"),
        ({"exposure_levels": ["0", "1"]}, "family_spec_longitudinal_exposure"),
        ({"execution_tier": "first_result"}, "family_spec_longitudinal_coordinate_unsupported"),
    ],
)
def test_unserved_requests_have_a_typed_refusal(kind, change, code):
    request = _request(kind).model_copy(update=change)
    with pytest.raises(FamilySpecError) as error:
        getattr(family_spec, BUILDERS[kind])(request, _spec(request))
    assert error.value.reason_code == code


@pytest.mark.parametrize("kind", BUILDERS)
def test_the_planner_cannot_add_features_or_adjustment_to_host_coordinates(kind):
    request = _request(kind)
    with pytest.raises(FamilySpecError, match="family_spec_feature_variables_not_applicable"):
        getattr(family_spec, BUILDERS[kind])(request, _spec(request).model_copy(update={"feature_variables": ["age", "pulse"]}))


def test_target_trial_products_and_figure_inputs_are_the_signed_owners():
    request, built = _build("trial")
    steps = {item.step.step_id: item.step for item in built.materializations}
    owner = request.sealed_target_trial
    assert [item.product_id for item in steps["target_trial_suite"].outputs] == list(owner.analysis_plan_outputs)
    assert [item.product_id for item in steps["visualization"].product_inputs] == list(owner.figure_input_products)
    assert [item.product_id for item in steps["visualization"].outputs] == [owner.figure_product]
    assert steps["target_trial_suite"].custom_method == owner.plan_method
    assert steps["target_trial_suite"].raw_inputs == list(owner.required_columns)
    review = " ".join(built.outline.design_selection.selected.reviewable_plan)
    for stage in ("protocol", "eligibility", "grace period", "time zero", "confounder", "weight", "positivity", "ESS", "balance", "RD/RR", "truncation", "unweighted", "E-value"):
        assert stage in review


def test_time_varying_does_not_invent_diagnostic_or_figure_owners():
    request, built = _build("time_varying")
    steps = {item.step.step_id: item.step for item in built.materializations}
    primary = steps["time_varying_cox"]
    assert [item.product_id for item in primary.outputs] == list(request.sealed_time_varying.plan_outputs)
    assert primary.custom_method == request.sealed_time_varying.plan_method
    assert steps["baseline_context"].table_one_group_by == request.outcome
    assert not any(item.step.module_id == "visualization" for item in built.materializations)
    assert "patient-cluster-robust" in primary.objective
    assert "PH validation" in steps["report"].objective
    request = request.model_copy(update={"requires_visualization": True})
    with pytest.raises(FamilySpecError, match="family_spec_longitudinal_product_unowned"):
        family_spec.build_time_varying_survival_skeleton(request, _spec(request))


def test_dynamic_landmark_graph_preserves_time_grouping_and_kernel_limits():
    _request_value, built = _build("dynamic")
    steps = {item.step.step_id: item.step for item in built.materializations}
    assert steps["landmark_features"].custom_method == "build_landmark_feature_matrix"
    assert "[landmark-lookback, landmark)" in steps["landmark_features"].objective
    assert steps["horizon_labels"].custom_method == "attach_landmark_outcomes"
    assert "missing labels, never a non-event" in steps["horizon_labels"].objective
    assert "GroupShuffleSplit" in steps["patient_split"].objective
    assert "groups=patient_id" in steps["patient_split"].objective
    assert steps["model_fit"].depends_on == ["patient_split"]
    assert "Fit preprocessing inside training" in steps["model_fit"].objective
    assert steps["landmark_performance"].custom_method == "evaluate_landmark_probabilities"
    assert "no confidence intervals" in steps["landmark_performance"].objective
    assert steps["model_fit"].scientific_action_id == "prediction.dynamic_prediction"
    assert not any(step.scientific_action_id == "prediction.discrimination_calibration" for step in steps.values())
    assert not {"internal_validation", "decision_curve", "subgroup_performance"}.intersection(steps)


@pytest.mark.parametrize("changes", [
    {"landmark_hours": [12, 6]}, {"landmark_hours": [6, 6]}, {"landmark_hours": [6]},
    {"landmark_hours": [float("nan"), 6]}, {"landmark_hours": [-1, 6]},
    {"horizon_hours": [0]}, {"horizon_hours": [float("inf")]}, {"lookback_hours": 0},
    {"split_unit": "row"}, {"feature_concepts": ["event_hours"]},
    {"feature_concepts": ["pulse", "pulse"]}, {"aggregations": ["last", "last"]}, {"aggregations": ["count"]},
    {"trajectory_bound": False}, {"trajectory_time_unit": "days"}, {"trajectory_time_origin": "hospital_admission"},
    {"validation_fraction": 1}, {"seed": -1}, {"model_pipeline": "random_forest"}, {"calibration_bins": 1},
])
def test_dynamic_coordinates_refuse_unsupported_timing_features_or_split(changes):
    payload = _request("dynamic").dynamic_landmark.model_dump()
    with pytest.raises(ValidationError):
        family_spec.DynamicLandmarkCoordinates.model_validate({**payload, **changes})


@pytest.mark.parametrize("kind", ["trial", "time_varying"])
def test_signed_drafts_bind_to_the_existing_executor_without_new_runtime_routing(kind):
    request, built = _build(kind)
    skeleton = assemble_progressive_skeleton(
        outline=built.outline, foundation=built.foundation.foundation,
        steps=[item.step for item in built.materializations],
    )
    plan, _receipt = compile_progressive_plan(skeleton=skeleton, context=_context(request))
    owner = request.sealed_target_trial or request.sealed_time_varying
    bound = owner.bind_plan(plan)
    owner.validate_plan(bound)
    primary = owner.governed_step(bound)
    assert primary.method == owner.plan_method
    assert primary.expected_outputs == list(owner.analysis_plan_outputs if kind == "trial" else owner.plan_outputs)


@pytest.mark.parametrize("kind", ["trial", "time_varying"])
def test_copied_authorities_must_still_pass_their_digest_validation(kind):
    request = _request(kind)
    field = "sealed_target_trial" if kind == "trial" else "sealed_time_varying"
    owner = getattr(request, field).model_copy(update={"plan_outputs": ("table:invented_result",)})
    request = request.model_copy(update={field: owner})
    with pytest.raises(FamilySpecError, match="family_spec_longitudinal_coordinates"):
        getattr(family_spec, BUILDERS[kind])(request, _spec(request))


def test_longitudinal_coordinates_cannot_be_attached_to_another_family():
    request = _request("trial")
    payload = request.model_dump(mode="json")
    payload["family_id"] = "baseline_adjusted_association"
    with pytest.raises(ValidationError, match="family_spec_longitudinal_coordinates"):
        FamilySpecRequest.model_validate_json(json.dumps(payload))


def test_new_families_do_not_activate_the_context_router():
    for kind in ("trial", "dynamic"):
        request = _request(kind)
        disclosure = request.sealed_target_trial.planning_contract_context() if kind == "trial" else ""
        assert family_spec.family_template_id_for_context(
            _context(request), analysis_types=[request.analysis_type], planning_contract_context=disclosure,
        ) is None


@pytest.mark.parametrize("kind", BUILDERS)
def test_an_outline_binder_cannot_change_the_host_graph(kind):
    request = _request(kind)
    with pytest.raises(FamilySpecError, match="family_spec_template_step_mismatch"):
        getattr(family_spec, BUILDERS[kind])(
            request, _spec(request), bind_outline=lambda outline: outline.model_copy(update={"steps": outline.steps[:-1]}),
        )


def test_an_outline_binder_cannot_change_a_materialized_step():
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

    request = _request("dynamic")

    def change_objective(outline):
        steps = list(outline.steps)
        steps[0] = steps[0].model_copy(update={"objective": "Silently change the approved source population."})
        return outline.model_copy(update={"steps": steps})

    with pytest.raises(ProgressivePlanCompileError, match="progressive_step_materialization_mismatch"):
        family_spec.build_dynamic_landmark_skeleton(request, _spec(request), bind_outline=change_objective)


def test_event_followup_coordinates_are_exposed_without_features_as_outcomes():
    trial = _request("trial")
    dynamic = _request("dynamic")
    assert trial.covered_outcome_columns == (trial.outcome, trial.sealed_target_trial.followup_time_column)
    assert dynamic.covered_outcome_columns == (dynamic.outcome, "event_hours", "followup_hours")


@pytest.mark.parametrize("kind", BUILDERS)
def test_unavailable_audit_columns_and_oversized_designs_fail_closed(kind):
    original = _request(kind)
    for changes, code in (
        ({"measurement_audit_columns": ["unbound_measurement"]}, "family_spec_longitudinal_source_missing"),
        ({"variable_roster": [*original.variable_roster, *(f"extra_{index}" for index in range(25))]}, "family_spec_roster_exceeds_design"),
    ):
        request = original.model_copy(update=changes)
        with pytest.raises(FamilySpecError, match=code):
            getattr(family_spec, BUILDERS[kind])(request, _spec(request))


def test_dynamic_source_roles_cannot_alias_identity_or_outcome():
    request = _request("dynamic")
    for field, value in (("patient_column", "stay_id"), ("event_time_column", request.outcome)):
        coordinates = request.dynamic_landmark.model_copy(update={field: value})
        invalid = request.model_copy(update={"dynamic_landmark": coordinates})
        with pytest.raises(FamilySpecError, match="family_spec_longitudinal_endpoint"):
            family_spec.build_dynamic_landmark_skeleton(invalid, _spec(invalid))


def test_existing_request_digests_do_not_gain_null_longitudinal_fields():
    payload = _request("dynamic").model_dump(mode="json")
    payload.pop("dynamic_landmark")
    payload.update(family_id="baseline_adjusted_association", analysis_type="association_study", primary_exposure="pulse", exposure_kind="continuous")
    request = FamilySpecRequest.model_validate(payload)
    assert not {"dynamic_landmark", "sealed_target_trial", "sealed_time_varying"}.intersection(request.model_dump(mode="json"))
    for key in ("accepted_feature_groups", "accepted_baseline_rows"):
        payload.pop(key)
    assert request.request_sha256 == canonical_sha256(payload)


@pytest.mark.parametrize("kind", BUILDERS)
def test_the_compiler_still_rejects_unowned_product_references(kind):
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

    request, built = _build(kind)
    steps = [item.step.model_copy(deep=True) for item in built.materializations]
    report = steps[-1]
    report.product_inputs[0] = report.product_inputs[0].model_copy(update={"product_id": "table:no_owner"})
    skeleton = assemble_progressive_skeleton(outline=built.outline, foundation=built.foundation.foundation, steps=steps)
    with pytest.raises(ProgressivePlanCompileError):
        compile_progressive_plan(skeleton=skeleton, context=_context(request))
