"""Metadata plans and prepared re-plans use the same sealed patient grouping."""

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.patient_grouping import (
    PatientGroupingBinding, derive_patient_grouping,
)
from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.cohort.schema import (
    CohortSchemaError, validate_plan_typed_bindings_against_context,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.gates.plan_declared_inputs import declared_raw_input_plan_findings
from easyicu.research_agent.intake.materialized_metadata import (
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
    stage_exposure_grouped_cohort_authority, stage_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration.exposure_grouping_phase import _declare_on_metadata
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.dependence_authority import context_dependence_authority
from easyicu.research_agent.planning.exposure_group_compile import CompiledGrouping, CompiledGroupings
from easyicu.research_agent.planning.exposure_group_spec import read_stated_exposure_groupings
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import resolved_raw_input_contracts
from tests.support.baseline_association import baseline_context, baseline_plan_attempt
from tests.support.typed_export import typed_export


def _grouping(levels):
    groups = [{"id": "g1", "label": "Lower", "rule": {"summary": "value", "op": "<", "value": 40}}]
    if levels == 3:
        groups.append({"id": "g3", "label": "Upper", "rule": {
            "summary": "value", "op": ">=", "value": 60,
        }})
    groups.append({"id": "g2", "label": "Middle" if levels == 3 else "Upper", "rule": "otherwise"})
    stated = read_stated_exposure_groupings({"groupings": [{
        "id": "x1", "concept": "age", "window": None, "scale": "ordinal",
        "groups": groups, "unmeasured": {"handling": "exclude"},
        "reference": "g1", "contrast": f"g{levels}",
        "quote": "Compare synthetic baseline bands", "source": "question",
    }]})
    return CompiledGroupings((CompiledGrouping(
        stated.groupings[0], "applied", None, "Synthetic baseline bands", {"value": "age"},
    ),))


def _metadata(tmp_path, database, first_stay, levels):
    stays = [11, 12, 21, 22, 31, 32]
    # The official source schemas differ; preparation maps both to its own
    # opaque composite row key, rather than requiring the private patient id.
    stay_column, patient_column = (
        ("patientunitstayid", "uniquepid") if database == "eicu" else ("stay_id", "subject_id")
    )
    patients = ["synthetic-a", "synthetic-a", "synthetic-b", "synthetic-b", "synthetic-c", "synthetic-c"]
    if database == "miiv":
        patients = [101, 101, 202, 202, 303, 303]
    bridge = derive_patient_grouping(
        pd.DataFrame({stay_column: stays, patient_column: patients}),
        stay_column=stay_column, patient_column=patient_column, identity_table_name="synthetic_identity",
    )
    mapping = tmp_path / "grouping.parquet"
    bridge.frame.to_parquet(mapping, index=False)
    mapping_sha = hashlib.sha256(mapping.read_bytes()).hexdigest()
    binding = PatientGroupingBinding(
        mapping_path=mapping, mapping_sha256=mapping_sha,
        mapping_stay_column="stay_id", mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "synthetic/grouping", "mapping_sha256": mapping_sha,
            "grouping_derivation": "prefix_before_:s", "provider_visible_values": False,
        },
    )
    restriction = {}
    planned_restriction = {}
    if first_stay:
        coordinate = tmp_path / "first_stay.parquet"
        pd.DataFrame({"stay_id": stays, "first_icu_stay": [True, False] * 3}).to_parquet(coordinate, index=False)
        digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
        restriction = {"first_icu_stay_path": coordinate, "first_icu_stay_sha256": digest,
                       "first_icu_stay_authority_coordinates": {"coordinate_sha256": digest}}
        planned_restriction = {"first_icu_stay_restriction": {
            "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA, "coordinate_sha256": digest,
            "provider_visible_values": False,
        }}
    frame = pd.DataFrame({
        "patient_stay_id": pd.Series(dtype="string"), "age": pd.Series(dtype="float64"),
        "lact_first": pd.Series(dtype="float64"), "death": pd.Series(dtype="bool"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
        "patient_grouping_status": "bound", **planned_restriction,
        "replacement_row_identity": {
            "output_identity_column": binding.output_identity_column,
            "mapping_file_sha256": binding.mapping_sha256, "mapped_cohort_rows": 0,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
            "authority_coordinates": dict(binding.authority_coordinates),
        },
    }
    catalog = tmp_path / "catalog.parquet"
    frame.to_parquet(catalog, index=False)
    compiled = _grouping(levels)
    grouping_sha = hashlib.sha256(json.dumps(compiled.record(), sort_keys=True).encode()).hexdigest()
    _declare_on_metadata(catalog, compiled.applied, groupings_record_sha256=grouping_sha)
    preferences = baseline_context(covariates=("lact_first",)).user_preferences
    preferences.data_constraints = json.dumps({"analysis_design": {
        "analysis_family": "association_study", "analysis_unit": "icu_stay",
        "variance_estimator": "model_based" if first_stay else "cluster_robust",
        **({} if first_stay else {"cluster_unit": "patient"}),
    }})
    kwargs = dict(
        research_question="Compare synthetic baseline bands and hospital death, adjusting for a baseline measurement.",
        database=database, cohort_name="synthetic", target_outcome="death",
        primary_exposure=compiled.applied[0].variable, user_preferences=preferences,
    )
    context = build_research_context(cohort=catalog, **kwargs)
    return context, kwargs, binding, restriction, compiled, grouping_sha, stays


def _prepare(tmp_path, database, binding, restriction, compiled, grouping_sha, stays):
    source = typed_export(
        tmp_path / "source", database=database,
        labs=pd.DataFrame({
            "stay_id": stays, "charttime": [0.0] * 6, "age": [25, 27, 50, 52, 75, 77],
            "lact": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6], "mech_vent": [False, True] * 3,
        }),
        outcomes=pd.DataFrame({"stay_id": stays, "death": [False, True] * 3}),
    )
    paths = materialize_to_parquet(
        tmp_path / "prepared", database=database, data_path=source,
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
        **binding.materializer_kwargs(), **restriction,
    )
    staged = tmp_path / "replan" / "cohort.parquet"
    stage_materialized_cohort_authority(paths["parquet"], staged, producer_implementation_sha256="a" * 64)
    stage_exposure_grouped_cohort_authority(
        staged, producer_implementation_sha256="b" * 64,
        groupings_record_sha256=grouping_sha, groupings=[{
            "variable": item.variable, "parameters": item.derivation(),
            "labels": item.record()["labels"], "compared": item.record()["compared"],
        } for item in compiled.applied],
    )
    return staged


def _runtime(context, levels):
    return ScientificRuntimeAuthorities(trajectory=None, current_case=build_baseline_authority(
        protocol_content_sha256="c" * 64, exposure_column=context.primary_exposure,
        outcome_column="death", exposure_encoding="treatment_contrast",
        exposure_levels=tuple(str(i) for i in range(1, levels + 1)),
        exposure_reference_level="1", primary_contrast_level=str(levels),
        required_adjustment_columns=("lact_first",), dependence=context_dependence_authority(context),
    ))


@pytest.mark.parametrize("database", ["eicu", "miiv"])
@pytest.mark.parametrize("first_stay", [False, True])
@pytest.mark.parametrize("levels", [2, 3])
def test_metadata_plan_preparation_and_prepared_replan_bind_the_same_grouping(
    tmp_path, database, first_stay, levels,
):
    metadata, kwargs, binding, restriction, compiled, grouping_sha, stays = _metadata(tmp_path, database, first_stay, levels)
    runtime = _runtime(metadata, levels)
    candidate, _ = baseline_plan_attempt(metadata, scientific_authorities=runtime)
    staged = _prepare(tmp_path, database, binding, restriction, compiled, grouping_sha, stays)
    prepared = build_research_context(cohort=staged, **kwargs)
    assert prepared.materialized_inputs.cohort.identity_column == "patient_stay_id"
    assert prepared.cohort.n_stays == (3 if first_stay else 6)
    assert prepared.cohort.n_patients == 3
    assert context_dependence_authority(prepared) == runtime.current_case.dependence
    replanned, llm = baseline_plan_attempt(prepared, scientific_authorities=runtime)
    assert len(llm.calls) == 1
    primary = replanned.output.steps[3]
    expected = candidate.output.steps[3].model_requirements[0].dependence
    assert primary.model_requirements[0].dependence == expected
    assert (expected is None) == first_stay
    assert not declared_raw_input_plan_findings(plan=replanned.output, context=prepared)
    if expected:
        assert expected.group_source in primary.inputs
        assert expected.group_source in prepared.materialized_inputs.cohort.cohort_columns
        assert expected.group_source not in prepared.materialized_inputs.cohort.column_bindings
        contracts = resolved_raw_input_contracts(prepared, primary.inputs)["contracts"]
        assert contracts[expected.group_source]["authority_kind"] == "materialized_cohort_identity"
    validate_plan_typed_bindings_against_context(plan=replanned.output, context=prepared)


@pytest.mark.parametrize("fault", [
    "no_contract", "wrong_derivation", "no_host_design", "independent_population",
    "different_identity", "unmaterialized_source", "other_step",
    "outcome", "exposure", "covariate", "model_term", "effect_modifier",
])
def test_grouping_permission_is_exact_and_never_makes_identity_a_model_variable(tmp_path, fault):
    first_stay = fault == "independent_population"
    metadata, kwargs, binding, restriction, compiled, grouping_sha, stays = _metadata(tmp_path, "miiv", first_stay, 2)
    draft = baseline_plan_attempt(metadata)[0]
    staged = _prepare(tmp_path, "miiv", binding, restriction, compiled, grouping_sha, stays)
    context = build_research_context(cohort=staged, **kwargs)
    # Compile against the prepared roster: metadata-stage navigation inputs
    # must not mask the rule this refusal fixture is meant to exercise.
    plan, _ = compile_progressive_plan(skeleton=draft.facts.skeleton, context=context)
    primary = plan.steps[3]
    requirement = primary.model_requirements[0]
    identity = context.materialized_inputs.cohort.identity_column
    clinical_control = plan.model_copy(deep=True)
    clinical_control.steps[3].inputs = [name for name in primary.inputs if name != identity]
    validate_plan_typed_bindings_against_context(plan=clinical_control, context=context)
    primary.inputs = list(dict.fromkeys([*primary.inputs, identity]))
    if fault == "no_contract":
        requirement.dependence = None
    elif fault == "wrong_derivation":
        requirement.dependence = requirement.dependence.model_copy(update={"delimiter": ":unverified"})
    elif fault == "no_host_design":
        context.user_preferences.data_constraints = "{}"
    elif fault == "independent_population":
        requirement.dependence = PlannedDependenceRequirement(
            group_source=identity, group_derivation="prefix_before_delimiter", delimiter=":s",
        )
    elif fault == "different_identity":
        context.cohort.id_columns.append("lact_first")
        context.cohort.provenance["replacement_row_identity"]["output_identity_column"] = "lact_first"
        requirement.dependence = context_dependence_authority(context)
    elif fault == "unmaterialized_source":
        requirement.dependence = requirement.dependence.model_copy(update={"group_source": "absent_patient_key"})
        primary.inputs = [name for name in primary.inputs if name != identity] + ["absent_patient_key"]
    elif fault == "other_step":
        plan.steps[1].inputs.append(identity)
    elif fault == "outcome":
        requirement.outcome = identity
    elif fault == "exposure":
        requirement.exposure_source = identity
    elif fault == "covariate":
        requirement.covariates.append(identity)
    else:
        field = "model_terms" if fault == "model_term" else "effect_modifiers"
        getattr(requirement, field).append(ModelTermSpec(
            name=identity, role="covariate", coding="continuous", transform="identity",
        ))
    with pytest.raises(CohortSchemaError, match="raw name"):
        validate_plan_typed_bindings_against_context(plan=plan, context=context)
