"""Synthetic contexts, scripted family planning and baseline runtime bindings."""

import json

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent, FAMILY_SPEC_STRATEGY
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.schema import (
    CohortDescriptor, ConceptDescriptor, EndpointSpec, ResearchContext, UserPreferences,
)


def baseline_context(exposure="sex", *, outcome="death", covariates=(), subgroup=None):
    context = ResearchContext(
        research_question=f"Association of {exposure} with {outcome}, adjusted for {', '.join(covariates)}.",
        cohort=CohortDescriptor(
            cohort_name="synthetic_admissions", database="synthetic", n_stays=0,
            id_columns=["stay_id"], outcome_columns=[outcome], requested_outcome_columns=[outcome],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
            ConceptDescriptor(name="age", role="demographic", dtype="float64", source_concept="age", unit="years"),
            ConceptDescriptor(name="sex", role="demographic", dtype="int64", source_concept="sex",
                              observed_domain={"levels": [0, 1], "n_unique": 2, "is_binary": True}),
            ConceptDescriptor(name="admission_type", role="demographic", dtype="string",
                              observed_domain={"levels": ["elective", "emergency", "urgent"], "n_unique": 3}),
            ConceptDescriptor(name="chronic_flag", role="demographic", dtype="int64",
                              observed_domain={"levels": [0, 1], "n_unique": 2, "is_binary": True}),
            ConceptDescriptor(name=outcome, role="outcome", dtype="bool",
                              # "unit_death" is a synthetic column with no registered outcome
                              # authority; "icu_death" is a derived outcome that a synthetic
                              # source cannot supply.
                              description="Death during the ICU stay" if outcome == "unit_death" else "Death before hospital discharge",
                              observed_domain={"levels": [False, True], "n_unique": 2, "is_binary": True}),
        ],
        primary_exposure=exposure, target_outcome=outcome,
        endpoint=EndpointSpec(name=outcome, kind="binary", absence_semantics="no_absent_rows", levels=[False, True]),
        user_preferences=UserPreferences(
            inferred_analysis_family="association_study", covariate_selection="exact",
            covariates=list(covariates),
            covariate_rationales={name: "The study specifies this baseline adjustment variable." for name in covariates},
            covariate_temporal_roles={name: "baseline_static" for name in covariates},
            data_constraints=json.dumps({"stated_subgroups": [{"variable": subgroup, "concept": subgroup,
                                          "words": "within the stated strata", "source_status": "available"}] if subgroup else []}),
        ),
    )
    if exposure == "age_group":
        from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_TRANSFORM_ID, EXPOSURE_GROUP_LABELS_KEY

        context.variables.append(ConceptDescriptor(
            name="age_group", role="other", dtype="int64", derived_from_concepts=["age"],
            unit_normalization=EXPOSURE_GROUP_TRANSFORM_ID,
            observed_domain={"levels": [1, 2], "n_unique": 2, "is_binary": True},
        ))
        context = context.model_copy(update={
            "research_question": "Age ≥65 versus <65 and hospital death, adjusted for sex.",
            "cohort": context.cohort.model_copy(update={"provenance": {EXPOSURE_GROUP_LABELS_KEY: [{
                "variable": "age_group", "concept": "age", "scale": "ordinal",
                "parameters": {
                    "concept": "age", "window": None, "scale": "ordinal",
                    "groups": [
                        {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 65.0}},
                        {"id": "g2", "rule": "otherwise"},
                    ], "unmeasured": "exclude", "source_columns": {"value": "age"},
                },
                "levels": [{"code": 1, "group": "g1", "label": "Under 65"},
                           {"code": 2, "group": "g2", "label": "65 or older"}],
            }]}}),
        })
    return context


def baseline_plan_attempt(context, *, planning_contract_context="", scientific_authorities=None):
    request = build_family_spec_request(
        context, analysis_types=("association_study",), variable_roster=[v.name for v in context.variables],
        allowed_literature_citation_keys=(), required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_contract_context,
    )
    payload = {
        "schema_version": "easyicu.family_plan_spec/1", "family_id": request.family_id,
        "request_sha256": request.request_sha256, "adjustment_set": [],
        "reader_display_labels": [{"key": key, "value": "Clinical " + key.replace("_", " ")} for key in request.required_reader_label_keys],
        "comparator_applications": [], "roster_decision_note": "Keep the study's exact stated adjustment roster.",
    }
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    result = ProgressivePlannerAgent(llm, scientific_authorities=scientific_authorities).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY, allowed_literature_citation_keys=(),
        required_primary_cohort_selection_mode="all_input_rows",
        planning_contract_context=planning_contract_context,
    )
    return result, llm


def baseline_runtime(context, plan):
    from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
    from easyicu.research_agent.contracts.model_terms import ModelTermSpec
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
    from easyicu.research_agent.planning.adjustment_authority import context_stated_variables

    requirement = plan.steps[3].model_requirements[0]
    authority = build_baseline_authority(
        protocol_content_sha256="f" * 64, exposure_column=context.primary_exposure, outcome_column=context.target_outcome,
        exposure_levels=tuple(requirement.exposure_levels or ()),
        exposure_reference_level=requirement.exposure_reference_level,
        primary_contrast_level=requirement.primary_contrast_level,
        exposure_encoding="restricted_cubic_spline" if context.primary_exposure == "age" else "treatment_contrast",
        required_adjustment_columns=tuple(requirement.covariates), dependence=requirement.dependence,
        subgroup_terms=tuple(ModelTermSpec(name=item.concept, role="covariate", coding="binary", levels=["0", "1"],
                                          reference_level="0", transform="treatment_contrast")
                             for item in context_stated_variables(context, "stated_subgroups")),
    )
    return ScientificRuntimeAuthorities(trajectory=None, current_case=authority)


def grouped_case(tmp_path, source="bmi", *, covariates=("sex",), clustered=False, middle_band=True):
    import hashlib
    import pandas as pd
    from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
    from easyicu.research_agent.orchestration.exposure_grouping_phase import _declare_on_metadata
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
    from easyicu.research_agent.planning.dependence_authority import context_dependence_authority
    from easyicu.research_agent.planning.exposure_group_compile import CompiledGrouping, CompiledGroupings
    from easyicu.research_agent.planning.exposure_group_spec import read_stated_exposure_groupings
    from easyicu.research_agent.schema import VariableRole

    context = baseline_context("age", covariates=covariates)
    context.variables[1] = context.variables[1].model_copy(update={
        "name": source, "source_concept": source, "unit": None,
        "role": VariableRole.LAB if source == "assay" else VariableRole.DEMOGRAPHIC,
        "analysis_window": "icu_admission[0,0]h" if source == "assay" else None,
    })
    context = context.model_copy(update={"primary_exposure": source})
    constraints = {"stated_adjustments": [
        {"variable": name, "concept": name, "words": f"adjust for {name}", "source_status": "available"}
        for name in covariates
    ]}
    if clustered:
        constraints["analysis_design"] = {"analysis_family": "association_study", "analysis_unit": "icu_stay",
                                           "cluster_unit": "patient", "variance_estimator": "cluster_robust"}
        constraints["verified_patient_grouping"] = {
            "authority_ref": "synthetic/grouping", "coordinate": "stay_id", "group_derivation": "prefix_before_:s",
            "mapping_sha256": "a" * 64, "provider_visible_values": False,
        }
        context.cohort.provenance["replacement_row_identity"] = {
            "output_identity_column": "stay_id", "mapping_file_sha256": "a" * 64,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
        }
    context.user_preferences.data_constraints = json.dumps(constraints)
    authority = build_baseline_authority(
        protocol_content_sha256="a" * 64, exposure_column=source, outcome_column="death",
        required_adjustment_columns=covariates, exposure_encoding="restricted_cubic_spline",
        dependence=context_dependence_authority(context),
    )
    groups = [
        {"id": "g3", "label": "Upper", "rule": {"summary": "value", "op": ">=", "value": 30}},
        {"id": "g1", "label": "Lower", "rule": {"summary": "value", "op": "<", "value": 20}},
        {"id": "g2", "label": "Middle", "rule": "otherwise"},
    ]
    reference, contrast = ("g1", "g2") if source == "assay" else ("g2", "g1")
    if not middle_band:
        groups = [
            {"id": "g2", "label": "Upper", "rule": {"summary": "value", "op": ">=", "value": 20}},
            {"id": "g1", "label": "Lower", "rule": {"summary": "value", "op": "<", "value": 20}},
        ]
    if source == "assay":
        groups = groups[1:]
    stated = read_stated_exposure_groupings({"groupings": [{
        "id": "x1", "concept": source, "window": None, "scale": "ordinal",
        "groups": groups, "unmeasured": {"handling": "exclude"}, "reference": reference, "contrast": contrast,
        "quote": "Compare synthetic measurement bands", "source": "question",
    }]})
    compiled = CompiledGroupings((CompiledGrouping(stated.groupings[0], "applied", None, "Synthetic source", {"value": source}),))
    record = {"schema_version": "easyicu.exposure_groupings_record/1", "planner": {"stated": stated.model_dump(mode="json")},
              "compiled": compiled.record()}
    raw = json.dumps(record).encode()
    (tmp_path / "exposure_groupings.json").write_bytes(raw)
    catalog = tmp_path / "cohort.parquet"
    frame = pd.DataFrame({v.name: pd.Series(dtype=v.dtype) for v in context.variables})
    frame.attrs["easyicu_planning_authority"] = {"kind": "metadata_only_planning_catalog", "patient_rows_read": False}
    frame.to_parquet(catalog, index=False)
    _declare_on_metadata(catalog, compiled.applied, groupings_record_sha256=hashlib.sha256(raw).hexdigest())
    group = compiled.applied[0].variable
    context.variables.append(ConceptDescriptor(
        name=group, role="other", dtype="int64", derived_from_concepts=[source],
        unit_normalization="exposure_group_ordinal", ordinal_levels=list(range(1, len(groups) + 1)), is_ordinal=True,
        analysis_window="icu_admission[0,0]h" if source == "assay" else None,
    ))
    context = context.model_copy(update={"primary_exposure": group})
    context.cohort.provenance.update({
        "evidence_stage": "metadata_only_planning",
        "exposure_group_contrasts": [{"variable": group, "reference": int(reference[1:]), "contrast": int(contrast[1:]), "stated": ["reference", "contrast"]}],
    })
    return context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority), catalog
