"""Project baseline-only study coordinates without reading patient rows."""

from __future__ import annotations

import json

from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.contracts.model_terms import level_spelling
from easyicu.research_agent.authority.declared_levels import closed_planning_levels_for
from easyicu.research_agent.planning.family_spec import BASELINE_ASSOCIATION_FAMILY_ID, build_family_spec_request
from easyicu.research_agent.planning.association_scope import baseline_association_context
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences


def compile_baseline_association_runtime_projection(
    *, study, sensitivity_specs, primary_exposure, primary_exposure_source, target_outcome,
    declared_covariates, covariate_operationalizations, target_is_event_status,
    universe_path, scientific_configuration_sha256, dependence=None, exposure_groupings=None,
):
    from .scientific_runtime_projection import (
        WebScientificRuntimeProjectionError, _bound_exposure_group, classify_variable,
        exposure_kind_for_dtype, signed_projection, load_verified_materialized_cohort_authority,
    )
    from .research_launch_scientific import launch_materialization_window
    from easyicu.research_agent.research_context.builder import _apply_materialized_column_metadata
    import pyarrow.parquet as pq
    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family

    design = study.get("analysis_design") or {}
    if (canonical_analysis_family(design.get("analysis_family")) != "association_study" or sensitivity_specs
        or not primary_exposure or not target_outcome or not target_is_event_status):
        return None
    schema = pq.read_schema(universe_path)
    if primary_exposure not in schema.names or target_outcome not in schema.names:
        return None
    verified = load_verified_materialized_cohort_authority(universe_path)
    bindings = verified.sidecar.files[0].columns if verified else {}
    grouped = _bound_exposure_group(exposure_groupings, primary_exposure=primary_exposure,
                                    primary_exposure_source=primary_exposure_source)
    source = primary_exposure_source or primary_exposure
    variables = []
    for field in schema:
        name = field.name
        original = source if name == primary_exposure else name
        hint = classify_variable(original, str(field.type))
        binding = bindings.get(name)
        _kind, levels = exposure_kind_for_dtype(
            primary_exposure=name, primary_exposure_source=original, dtype=str(field.type),
            published_levels=binding.metadata.allowed_values if binding else None,
        )
        group_fields = {}
        if grouped and name == primary_exposure:
            levels = tuple(str(code) for code in range(1, grouped.levels + 1))
            group_fields = dict(derived_from_concepts=[source], unit_normalization=grouped.transform)
        variables.append(ConceptDescriptor(
            name=name, dtype=str(field.type), role="outcome" if name == target_outcome else hint.role,
            source_concept=original,
            observed_domain={"levels": list(levels), "n_unique": len(levels), "is_binary": len(levels) == 2} if levels else None,
            **group_fields,
        ))
    if verified:
        variables = _apply_materialized_column_metadata(descriptors=variables, verified=verified)
    # All names come from the schema or the study's typed reading, never from rows.
    context = ResearchContext(
        research_question=str(study.get("research_question") or "Baseline association"),
        cohort=CohortDescriptor(cohort_name="baseline_runtime", database="synthetic", n_stays=0,
                                id_columns=["__runtime_identity"], outcome_columns=[target_outcome]),
        variables=[*variables, ConceptDescriptor(name="__runtime_identity", role="id", dtype="int64")],
        primary_exposure=primary_exposure, target_outcome=target_outcome,
        user_preferences=UserPreferences(
            covariate_selection=study.get("covariate_selection") or "planner_selectable",
            covariates=list(declared_covariates), covariate_operationalizations=dict(covariate_operationalizations),
            data_constraints=json.dumps({
                **{field: study.get(field) or [] for field in ("stated_adjustments", "stated_subgroups")},
                "materialization_window": launch_materialization_window(study),
            }),
        ),
    )
    if not baseline_association_context(context):
        return None
    try:
        request = build_family_spec_request(context, analysis_types=("association_study",),
                                            variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=())
        if request.family_id != BASELINE_ASSOCIATION_FAMILY_ID:
            return None
        subgroup_terms = []
        for name in request.subgroup_columns:
            levels = [level_spelling(level) for level in closed_planning_levels_for(name=name, variables={v.name: v for v in context.variables})]
            subgroup_terms.append(ModelTermSpec(name=name, role="covariate", coding="binary" if len(levels) == 2 else "categorical",
                                                transform="treatment_contrast", levels=levels, reference_level=levels[0]))
        levels = request.exposure_levels
        authority = build_baseline_authority(
            protocol_content_sha256=scientific_configuration_sha256,
            exposure_column=primary_exposure, outcome_column=target_outcome,
            exposure_levels=tuple(levels), exposure_reference_level=str(grouped.contrast.reference) if grouped else levels[0] if levels else None,
            primary_contrast_level=str(grouped.contrast.contrast) if grouped else levels[-1] if levels else None,
            required_adjustment_columns=tuple(request.exact_roster), subgroup_terms=tuple(subgroup_terms),
            dependence=dependence, exposure_encoding="treatment_contrast" if levels else "restricted_cubic_spline",
        )
    except ValueError as exc:
        raise WebScientificRuntimeProjectionError("web_baseline_association_incomplete", str(exc), details={"exposure": primary_exposure}) from exc
    return signed_projection(authority.model_dump(mode="json"), scientific_configuration_sha256=scientific_configuration_sha256)
