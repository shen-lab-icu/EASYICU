"""Source proposals compile into the very plan the existing authority approves."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.databases.research_approval import CapabilityApprovalError, require_approved_database_capabilities
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, ThresholdRequirement, assess_study
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from tests.support.source_substitutions import _approve, _review, context, plan


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
def test_exact_association_and_refusal_remain_distinct(database, tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context(database, outcome="icu_death"))
    compiled = plan(ctx)
    result = SimpleNamespace(context=ctx, plan=compiled, evidence=EvidenceStore(tmp_path))
    require_approved_database_capabilities(result, tmp_path)
    refused = compiled.model_copy(update={"steps": []})
    original = json.loads(ctx.user_preferences.data_constraints)
    original["database_study_requirements"]["concepts"] = ["d_dimer" if database == "sic" else "apixaban"]
    invalid = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(original))})
    with pytest.raises(CapabilityApprovalError, match="study_capability_refused"):
        require_approved_database_capabilities(SimpleNamespace(context=invalid, plan=refused, evidence=result.evidence), tmp_path)


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_unit_proposal_binds_native_icu_identity_and_requires_approval(database, tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    original = context(database, unit="hospitalization", outcome="icu_death")
    original = original.model_copy(update={"cohort": original.cohort.model_copy(update={"notes": "Synthetic cohort source note."})})
    ctx = bind_source_substitutions(original)
    binding = json.loads(ctx.user_preferences.data_constraints)["database_capability_binding"]
    assert binding["effective_analysis_unit"] == "icu_stay"
    assert binding["identifier"] == ctx.cohort.id_columns[0]
    assert "icu_stay" in ctx.cohort.notes
    assert "Synthetic cohort source note." in ctx.cohort.notes
    assert bind_source_substitutions(ctx, restoring=True) == ctx
    assert "hospitalization" not in ctx.cohort.id_columns
    population = compile_population(PopulationSpec(criteria=[]), ctx)
    assert population.record()["database_capability_binding"] == binding
    compiled = plan(ctx)
    evidence = EvidenceStore(tmp_path)
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence=evidence), tmp_path)
    require_approved_database_capabilities(_approve(tmp_path, evidence, ctx, compiled, _review(ctx, compiled)), tmp_path)
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=original, plan=compiled, evidence=evidence), tmp_path)


@pytest.mark.parametrize("database,requested,aligned", [("aumc", 65, 60), ("mimic", 95, 90), ("miiv", 95, 90), ("eicu", 95, 90)])
def test_age_proposal_is_compiled_before_approval(database, requested, aligned, tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    requirement = ThresholdRequirement(concept="age", operator=">=", value=requested)
    ctx = bind_source_substitutions(context(database, thresholds=(requirement,)))
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "age_years", "role": "include",
        "source": "question", "quote": "Include the requested older age group.", "min_years": requested}]})
    compiled = plan(ctx, spec)
    assert [(p.op, p.value) for p in compiled.cohort.inclusion] == [(">=", aligned)]
    require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled)), tmp_path)


@pytest.mark.parametrize("database", ["hirid", "aumc"])
def test_mortality_proposal_names_the_new_outcome_and_rejects_partial_application(database, tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    original = context(database, outcome="death")
    ctx = bind_source_substitutions(original)
    assert ctx.target_outcome == "icu_death"
    compiled = plan(ctx)
    require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled)), tmp_path)
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_review_mismatch"):
        require_approved_database_capabilities(SimpleNamespace(context=original, plan=compiled, evidence=EvidenceStore(tmp_path)), tmp_path)


def test_aumc_band_values_are_ordered_categories_not_continuous_covariates():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("aumc"))
    age, weight = (next(v for v in ctx.variables if v.name == name) for name in ("age", "weight"))
    assert age.is_ordinal and age.ordinal_levels == [30, 45, 55, 65, 75, 90]
    assert weight.is_ordinal and weight.ordinal_levels == [50, 65, 75, 85, 95, 105, 120]


def test_an_icu_outcome_is_not_fabricated_from_a_legacy_death_column():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("aumc", outcome="death")
    ctx = ctx.model_copy(update={"variables": [v for v in ctx.variables if v.name != "icu_death"]})
    with pytest.raises(ValueError, match="capability_substitution_requires_extraction"):
        bind_source_substitutions(ctx)


def test_aumc_source_still_distinguishes_the_dictionary_post_icu_window():
    from easyicu.databases.research_substitutions import source_decision

    profile = research_capability_profile("aumc")
    assert profile.mortality.meaning == "death_within_72h_after_icu_discharge"
    requirements = StudyRequirements(family="association_study", outcome="hospital_death")
    assert assess_study(profile, requirements).substitutions[0].substitute == "death_within_72h_after_icu_discharge"
    proposal = source_decision(profile, requirements)
    assert proposal.substitutions[0].substitute == "icu_death"
    assert proposal.substitutions[0].source_field.column == "destination"


@pytest.mark.parametrize("concept,requested,aligned,operator", [("age", 65, 60, ">="), ("weight", 73, 70, ">="), ("weight", 73, 70, "<")])
def test_band_threshold_selects_whole_source_bands(concept, requested, aligned, operator, tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.research_agent.cohort.schema import build_cohort

    requirement = ThresholdRequirement(concept=concept, operator=operator, value=requested)
    ctx = bind_source_substitutions(context("aumc", thresholds=(requirement,)))
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "measurement", "role": "include",
        "source": "question", "quote": "Use the requested baseline measurement band.", "concept": concept,
        "summary": "first", "window": {"start_hours": 0, "end_hours": 24},
        "op": operator, "value": requested, "unit": "years" if concept == "age" else "kg"}]})
    compiled = plan(ctx, spec)
    predicate, = compiled.cohort.inclusion
    assert (predicate.op, predicate.value) == (operator, aligned)
    values = [30, 55, 65, 75, 90] if concept == "age" else [50, 65, 75, 105, 120]
    rows = pd.DataFrame({"admissionid": range(len(values)), concept: values})
    selected = build_cohort(compiled.cohort, rows)
    expected = [v for v in values if (v >= aligned if operator == ">=" else v < aligned)]
    assert selected[concept].tolist() == expected
    require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled)), tmp_path)


def test_a_band_midpoint_is_not_compiled_without_a_disclosed_proposal():
    ctx = context("aumc")
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "age_years", "role": "include",
        "source": "question", "quote": "Include stays meeting the age threshold.", "min_years": 65}]})
    compiled = compile_population(spec, ctx)
    assert compiled.blocking[0].reason == "population_threshold_outside_domain"


def test_native_population_identity_is_required():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("hirid", unit="hospitalization")
    ctx = ctx.model_copy(update={"cohort": ctx.cohort.model_copy(update={"id_columns": ["invented_hospital_id"]})})
    with pytest.raises(ValueError, match="capability_substitution_identifier_unbound"):
        bind_source_substitutions(ctx)


def test_model_compile_and_matrix_keep_bands_as_ordered_treatment_categories(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep
    from easyicu.research_agent.planning.progressive_compiler import _compile_model_terms, ProgressivePlanCompileError
    from easyicu.research_agent.execution.model_matrix import compile_model_terms

    ctx = bind_source_substitutions(context("aumc", outcome="icu_death"))
    variables = {v.name: v for v in ctx.variables}
    variables["hr"] = ConceptDescriptor(name="hr", role="vital", dtype="float64")
    step = ProgressiveSkeletonStep(step_id="association", module_id="adjusted_association",
        planned_analysis_role="primary", objective="Estimate the declared association.", primary_exposure="hr",
        outcome="icu_death", outcome_type="binary",
        model_terms=[{"name": "hr", "role": "exposure", "coding": "continuous"},
            {"name": "age", "role": "covariate", "coding": "categorical", "reference_level_index": 0,
             "clinical_rationale": "Age bands are declared baseline adjustment categories."}])
    terms, covariates, *_ = _compile_model_terms(variables=variables, step=step, step_index=0)
    assert covariates == ["age"]
    age = terms[1]
    assert age.levels == ["30", "45", "55", "65", "75", "90"]
    rows = pd.DataFrame({"hr": [60, 70, 80, 65, 75, 85], "age": [30, 45, 55, 65, 75, 90]})
    matrix = compile_model_terms(rows, terms=terms, exposure="hr")
    assert list(matrix.design) == ["hr", "age__is_45", "age__is_55", "age__is_65", "age__is_75", "age__is_90"]
    assert set(matrix.design.age__is_65) == {0., 1.}
    continuous = step.model_terms[1].model_copy(update={"coding": "continuous"})
    with pytest.raises(ProgressivePlanCompileError, match="progressive_model_continuous_coding_declared_domain"):
        _compile_model_terms(variables=variables, step=step.model_copy(update={"model_terms": [step.model_terms[0], continuous]}), step_index=0)


@pytest.mark.parametrize("coding", ["continuous", "ordinal_linear"])
def test_execution_refuses_a_fake_continuous_or_linear_band_covariate(tmp_path, coding):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("aumc", outcome="icu_death"))
    compiled = plan(ctx)
    requirement = compiled.steps[0].model_requirements[0]
    term = {"name": "age", "role": "covariate", "coding": coding,
            "transform": "identity" if coding == "continuous" else "declared_level_index"}
    if coding == "ordinal_linear":
        term["levels"] = ["30", "45", "55", "65", "75", "90"]
    payload = requirement.model_dump(mode="json")
    payload.update(covariates=["age"], model_terms=[
        {"name": "weight", "role": "exposure", "coding": "continuous", "transform": "identity"}, term])
    step = AnalysisStep(step_id="association", intent="Estimate the association.", model_requirements=[payload],
        method="adjusted_association_models", expected_outputs=["table:adjusted_association_estimates"])
    compiled = compiled.model_copy(update={"steps": [step]})
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence=EvidenceStore(tmp_path)), tmp_path)


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
def test_synthetic_native_sources_reach_population_and_approval(database, tmp_path):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.research_agent.cohort.schema import build_cohort
    from tests.support.source_mortality import source

    original = context(database, unit="hospitalization" if database in {"hirid", "aumc", "sic"} else "icu_stay",
                       outcome="death" if database in {"hirid", "aumc"} else "icu_death")
    ctx = bind_source_substitutions(original)
    compiled = plan(ctx)
    try:
        frames = load_concepts(["icu_death"], database=database, data_path=source(tmp_path / database, database),
                               merge=False, verbose=False, concept_workers=1, parallel_workers=1)
        rows = frames["icu_death"] if isinstance(frames, dict) else frames
        selected = build_cohort(compiled.cohort, rows)
        assert selected.icu_death.iloc[:3].tolist() == [True, False, False]
        assert ctx.cohort.id_columns[0] in selected
        assert pd.isna(selected.icu_death.iloc[3])
        require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled)), tmp_path)
    finally:
        clear_global_loader()


def test_normal_preplanner_binding_reaches_prompt_and_is_idempotent(tmp_path):
    from easyicu.research_agent.orchestration.reviewed_requirements import bind_reviewed_requirements
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.agents.progressive_prompt_contracts import outline_study_anchors

    config = PipelineConfig(workdir=tmp_path)
    ctx = bind_reviewed_requirements(context("hirid", unit="hospitalization", outcome="death"), config)
    assert bind_reviewed_requirements(ctx, config, restoring=True) == ctx
    prompt = outline_study_anchors(ctx)
    assert "hospitalization_identifier_unavailable" in prompt
    assert "mortality_meaning_differs" in prompt
    assert "icu_death" in prompt


def test_removing_original_requirements_does_not_turn_an_approved_substitution_into_exact(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    compiled = plan(ctx)
    result = _approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled))
    stripped = ctx.model_copy(update={"user_preferences": UserPreferences()})
    result.context = stripped
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_review_mismatch"):
        require_approved_database_capabilities(result, tmp_path)


def test_restoring_a_context_cannot_rebind_changed_source_facts():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("aumc", unit="hospitalization", outcome="death"))
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints["database_capability_binding"]["decision"]["substitutions"][0]["source_field"]["column"] = "invented_id"
    drifted = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        bind_source_substitutions(drifted, restoring=True)
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        compile_population(PopulationSpec(criteria=[]), drifted)


def test_a_partially_changed_model_outcome_cannot_cross_approval(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    compiled = plan(ctx)
    step = compiled.steps[0]
    partial = step.model_copy(update={"model_requirements": [step.model_requirements[0].model_copy(update={"outcome": "death"})]})
    compiled = compiled.model_copy(update={"steps": [partial]})
    assert not _review(ctx, compiled).approval_allowed
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence=EvidenceStore(tmp_path)), tmp_path)


def test_full_progressive_compiler_seals_the_native_outcome_report_label():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload

    ctx = ResearchContext.model_validate_json(_context().model_dump_json().replace("outcome_flag", "icu_death"))
    skeleton = ProgressivePlanSkeleton.model_validate(json.loads(json.dumps(_payload()).replace("outcome_flag", "icu_death")))
    compiled, receipt = compile_progressive_plan(skeleton=skeleton, context=ctx)
    assert compiled.display_labels["icu_death"] == "ICU death"
    assert receipt.analysis_plan_sha256 == canonical_sha256(compiled.model_dump(mode="json"))
    assert all(r.outcome == "icu_death" for step in compiled.steps for r in step.model_requirements)


def test_binned_model_uses_observed_subset_without_inventing_absent_categories():
    from easyicu.databases.research_substitutions import bind_source_substitutions, source_model_term_errors
    from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep
    from easyicu.research_agent.planning.progressive_compiler import _compile_model_terms
    from easyicu.research_agent.execution.model_matrix import compile_model_terms

    ctx = bind_source_substitutions(context("aumc", outcome="icu_death"))
    variables = {v.name: v for v in ctx.variables}
    variables["age"] = variables["age"].model_copy(update={"observed_domain": {"levels": [65, 75, 90]}})
    variables["hr"] = ConceptDescriptor(name="hr", role="vital", dtype="float64")
    step = ProgressiveSkeletonStep(step_id="association", module_id="adjusted_association",
        planned_analysis_role="primary", objective="Estimate the declared association.", primary_exposure="hr",
        outcome="icu_death", outcome_type="binary",
        model_terms=[{"name": "hr", "role": "exposure", "coding": "continuous"},
            {"name": "age", "role": "covariate", "coding": "categorical", "reference_level_index": 0,
             "clinical_rationale": "Adjust using observed source age bands in their declared order."}])
    terms, covariates, *_ = _compile_model_terms(variables=variables, step=step, step_index=0)
    assert terms[1].levels == ["65", "75", "90"]
    matrix = compile_model_terms(pd.DataFrame({"hr": [60, 70, 80], "age": [65, 75, 90]}), terms=terms, exposure="hr")
    assert list(matrix.design) == ["hr", "age__is_75", "age__is_90"]
    compiled = plan(ctx)
    compiled.steps[0].model_requirements[0].model_terms = terms
    compiled.steps[0].model_requirements[0].covariates = covariates
    assert not source_model_term_errors(ctx, compiled)


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
def test_source_exposure_and_outcome_reach_full_progressive_compile_and_approval(database, tmp_path):
    from easyicu import load_concepts
    from easyicu.api.concepts import clear_global_loader
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.research_agent.cohort.schema import build_cohort
    from easyicu.research_agent.execution.model_matrix import compile_model_terms
    from tests.support.source_mortality import source

    try:
        rows = load_concepts(["age", "icu_death"], database=database,
            data_path=source(tmp_path / database, database), merge=True,
            verbose=False, concept_workers=1, parallel_workers=1)
        assert len(rows) == 4 and rows.age.notna().all()
        original = context(database, unit="hospitalization" if database in {"hirid", "aumc", "sic"} else "icu_stay",
                           outcome="death" if database in {"hirid", "aumc"} else "icu_death")
        variables = [v for v in original.variables if v.name != "weight"]
        variables = [v.model_copy(update={"observed_domain": {"levels": sorted(rows.age.unique().tolist())}})
                     if v.name == "age" else v for v in variables]
        ctx = bind_source_substitutions(original.model_copy(update={"primary_exposure": "age", "variables": variables,
            "research_question": "Estimate the association of recorded age categories with mortality."}))
        skeleton = ProgressivePlanSkeleton.model_validate({
            "schema_version": "easyicu.progressive_plan_skeleton/1", "analysis_type": "association_study",
            "rationale": "Estimate the prespecified source-category association in the native ICU population.",
            "cohort": {"name": "primary", "selection_mode": "all_input_rows", "inclusion": [], "exclusion": []},
            "display_labels": [{"key": "age", "value": "Recorded age category"},
                               {"key": "icu_death", "value": "ICU death"}],
            "steps": [{"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
                       "objective": "Bind the native ICU population."},
                      {"step_id": "association", "module_id": "adjusted_association", "planned_analysis_role": "primary",
                       "objective": "Estimate the association of source age categories with ICU death.",
                       "depends_on": ["cohort"], "raw_inputs": ["age", "icu_death"],
                       "scientific_action_id": "association.adjusted_association",
                       "primary_exposure": "age", "outcome": "icu_death", "outcome_type": "binary",
                       "primary_contrast_level_index": 1,
                       "model_terms": [{"name": "age", "role": "exposure", "coding": "categorical", "reference_level_index": 0}]}]})
        compiled, _ = compile_progressive_plan(skeleton=skeleton, context=ctx)
        selected = build_cohort(compiled.cohort, rows)
        assert len(selected) == 4
        assert compiled.steps[0].cohort_definition_spec.identity_column == ctx.cohort.id_columns[0]
        terms = compiled.steps[1].model_requirements[0].model_terms
        matrix = compile_model_terms(selected, terms=terms, exposure="age")
        assert matrix.design.shape == (4, 3)
        assert compiled.display_labels["icu_death"] == "ICU death"
        require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled)), tmp_path)
    finally:
        clear_global_loader()


def test_a_source_receipt_for_the_original_threshold_does_not_hide_the_aligned_predicate():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("aumc", thresholds=(ThresholdRequirement(concept="age", operator=">=", value=65),))
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints["source_selection"] = {"basis": "export_contract", "host_applied": [],
        "executed_cohort": {"age_min": 65}, "export_report": {
            "count_unit": "icu_stay", "source_total": 6, "selected": 4,
            "selected_before_concept_prefilter": 4, "concept_matches": None,
            "selected_before_icd": 4, "selected_before_cap": 4,
            "demographic_steps": [{"criterion": "age", "parameters": {"age_min": 65},
                "n_before": 6, "n_excluded": 2, "n_remaining": 4, "n_excluded_missing": 0}]}}
    ctx = bind_source_substitutions(ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))}))
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "age_years", "role": "include",
        "source": "question", "quote": "Include the requested older age group.", "min_years": 65}]})
    compiled = compile_population(spec, ctx)
    assert compiled.criteria[0].disposition == "applied_by_plan"
    assert compiled.cohort_definition().inclusion[0].value == 60
