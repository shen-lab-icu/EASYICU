from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    build_current_case_scientific_runtime_authority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.capability_ids import (
    LANDMARK_CATEGORICAL_ASSOCIATION_CAPABILITY_ID,
)
from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
    LandmarkCategoricalExecutionError,
    landmark_categorical_cohort_executor_code,
    landmark_categorical_primary_executor_code,
    landmark_eligibility_mask,
    run_landmark_categorical_cohort,
    run_landmark_categorical_primary,
)
from easyicu.research_agent.execution.runners.association_model_grid_executor import (
    association_model_grid_executor_owns_step,
    run_association_model_grid,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.planning.capability_registry import (
    resolve_primary_capability,
)
from easyicu.research_agent.planning.scientific_review import timing_design_closed
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260904)
    patients = 240
    n = patients * 2
    stage = np.tile(np.arange(4), n // 4)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (
        1.0 + np.exp(-(-3.2 + 0.38 * stage + 0.012 * (age - 64.0)))
    )
    death = rng.binomial(1, probability, size=n)
    early = (death == 1) & (rng.random(n) < 0.08)
    death_time = np.where(
        death == 1,
        np.where(early, 8.0, rng.uniform(30.0, 160.0, size=n)),
        np.nan,
    )
    followup = rng.uniform(36.0, 180.0, size=n)
    followup[rng.random(n) < 0.04] = 12.0
    return pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{patient}:s{stay}"
                for patient in range(patients)
                for stay in (1, 2)
            ],
            "aki_stage_strict": stage,
            "death": death,
            "death_time_hours": death_time,
            "hospital_followup_time_hours": followup,
            "age": age,
            "sex": sex,
        }
    )


def _projection(tmp_path):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
    landmark = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "landmark_24h",
            "axis": "timing",
            "strategy": "landmark",
            "landmark_hours": 24,
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
            "event_time_variable": "death_time_hours",
            "observation_duration_variable": "hospital_followup_time_hours",
            "observation_duration_unit": "hours",
        }
    )
    dependence = PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=(landmark,),
        primary_exposure="aki_stage_strict",
        primary_exposure_source="aki_stage",
        target_outcome="death",
        declared_covariates=("age", "sex"),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="a" * 64,
        dependence=dependence,
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    return universe, projection, authority


def _draft_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Compare KDIGO stages with in-hospital mortality.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_landmark_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death", "death_time_hours"],
                    "expected_outputs": [
                        "artifact:analysis_cohort",
                        "table:cohort_flow",
                    ],
                    "method": "cohort_definition_and_attrition",
                },
                {
                    "step_id": "primary_adjusted_association",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate adjusted stage contrasts.",
                    "inputs": [
                        "artifact:analysis_cohort",
                        "aki_stage_strict",
                        "death",
                        "age",
                        "sex",
                        "patient_stay_id",
                    ],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "sensitivity_spec_ids": ["landmark_24h"],
                    "model_requirements": [
                        {
                            "requirement_id": "primary_stage_model",
                            "outcome": "death",
                            "outcome_type": "binary",
                            "method_family": "statsmodels_logit_mle",
                            "exposure_source": "aki_stage_strict",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": ["age", "sex"],
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                {
                                    "name": "age",
                                    "role": "covariate",
                                    "coding": "continuous",
                                    "transform": "identity",
                                },
                                {
                                    "name": "sex",
                                    "role": "covariate",
                                    "coding": "binary",
                                    "levels": ["Female", "Male"],
                                    "reference_level": "Female",
                                    "transform": "treatment_contrast",
                                },
                            ],
                            "exposure_levels": ["0", "1", "2", "3"],
                            "exposure_reference_level": "0",
                            "primary_contrast_level": "3",
                            "dependence": {
                                "group_source": "patient_stay_id",
                                "group_derivation": "prefix_before_delimiter",
                                "delimiter": ":s",
                            },
                        }
                    ],
                },
                {
                    "step_id": "duplicate_landmark_sensitivity",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Repeat the landmark analysis.",
                    "inputs": [
                        "artifact:analysis_cohort",
                        "table:adjusted_association_estimates",
                    ],
                    "expected_outputs": ["table:sensitivity_landmark_24h"],
                    "method": "landmark_analysis",
                    "sensitivity_spec_ids": ["landmark_24h"],
                },
                {
                    "step_id": "report",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Report results.",
                    "inputs": [
                        "table:adjusted_association_estimates",
                        "table:sensitivity_landmark_24h",
                    ],
                    "expected_outputs": ["report:study"],
                    "method": "scientific_reporting",
                },
            ],
        }
    )


def test_signed_landmark_parent_composes_a_verified_exposure_grid(tmp_path) -> None:
    universe, projection, original = _projection(tmp_path)
    grid = build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.association_model_grid_runtime_authority/1",
        "authority_kind": "association_model_grid",
        "protocol_content_sha256": original.protocol_content_sha256,
        "plan_method": "verified_association_model_grid",
        "plan_intent": "Compare the prespecified exposure definitions on the landmark cohort.",
        "cohort_product": original.cohort_product,
        "parent_product": original.primary_product,
        "output_product": "table:exposure_definition_grid",
        "reference_variant_id": "reference",
        "metadata_columns": ["definition"],
        "output_aliases": {},
        "variants": [
            {"analysis_id": "reference", "metadata": {"definition": "primary"}},
            {
                "analysis_id": "alternate",
                "exposure_column": "aki_stage_alternate",
                "metadata": {"definition": "alternate"},
            },
        ],
    })
    body = original.model_dump(
        mode="json", exclude={"execution_contract_sha256", "association_model_grid"}
    )
    body["schema_version"] = (
        "easyicu.landmark_categorical_association_runtime_authority/2"
    )
    body["association_model_grid"] = grid.model_dump(mode="json")
    authority = build_current_case_scientific_runtime_authority(body)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)

    bound = authority.bind_plan(_draft_plan())
    authority.validate_plan(bound)
    grid_step = next(
        step for step in bound.steps
        if "table:exposure_definition_grid" in step.expected_outputs
    )
    assert association_model_grid_executor_owns_step(
        grid_step, plan=bound, authority=authority
    )
    selected = select_standard_executor(
        grid_step,
        plan=bound,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert selected is not None
    assert selected.analysis_kind == "association_model_grid"
    assert selected.consumed_input_keys == (
        "artifact:analysis_cohort", "table:adjusted_association_estimates"
    )
    compile(selected.code, "<landmark-exposure-grid>", "exec")

    frame = pd.read_parquet(universe)
    frame["aki_stage_alternate"] = frame["aki_stage_strict"].astype("float64")
    frame.loc[frame.index[:40], "aki_stage_alternate"] = np.nan
    frame.to_parquet(universe, index=False)
    cohort_dir = tmp_path / "grid_cohort"
    run_landmark_categorical_cohort(
        frame=frame,
        source_path=universe,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=cohort_dir,
    )
    cohort_path = cohort_dir / "analysis_cohort.parquet"
    eligible = pd.read_parquet(cohort_path)
    primary_step = authority.governed_primary_step(bound)
    primary_dir = tmp_path / "grid_primary"
    primary_summary = run_landmark_categorical_primary(
        frame=eligible,
        cohort_path=cohort_path,
        step=primary_step,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=primary_dir,
    )
    parent_path = primary_dir / primary_summary["output_files"][authority.primary_product]
    digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    input_key = authority.primary_product
    evidence_id = "signed_parent_estimates"
    binding = {
        "relative_path": parent_path.relative_to(tmp_path).as_posix(),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": evidence_id,
        "product": input_key.partition(":")[2],
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": evidence_id,
            "input_key": input_key,
            "product": input_key.partition(":")[2],
            "sha256": digest,
        },
        "product_contract": {
            "columns": list(pd.read_csv(parent_path).columns),
            "row_count": len(pd.read_csv(parent_path)),
        },
        "consumption_contract": {
            "input_key": input_key, "mode": "all_rows", "artifact_sha256": digest,
        },
    }
    grid_summary = run_association_model_grid(
        frame=eligible,
        cohort_path=cohort_path,
        authority=grid,
        runtime_projection_sha256=projection.projection_sha256,
        parent_requirement=primary_step.model_requirements[0],
        out_dir=tmp_path / "grid_results",
        run_dir=tmp_path,
        resolved_inputs={"step_id": grid_step.step_id, "inputs": {input_key: binding}},
        step_id=grid_step.step_id,
    )
    rows = {row["analysis_id"]: row for row in grid_summary["analysis_rows"]}
    assert rows["reference"]["fit_n"] == primary_summary["n_total"]
    assert rows["reference"]["exposure_missing_n"] == 0
    assert rows["alternate"]["exposure_missing_n"] > 0
    assert rows["alternate"]["fit_n"] < rows["reference"]["fit_n"]

    changed = grid_step.model_copy(update={"inputs": grid_step.inputs[:-1]})
    with pytest.raises(ValueError, match="model-grid plan drifted"):
        authority.validate_plan(bound.model_copy(update={"steps": [
            changed if step.step_id == grid_step.step_id else step
            for step in bound.steps
        ]}))


def test_plan_review_credits_only_signed_categorical_grid_variants(tmp_path) -> None:
    from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
    from easyicu.research_agent.planning.sensitivity_plan_shaping import (
        ensure_prespecified_sensitivity_steps,
    )
    from easyicu.research_agent.schema import UserPreferences
    from tests.research_agent.planning.scientific_review_fixtures import _context

    _, _, original = _projection(tmp_path)
    grid = build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.association_model_grid_runtime_authority/1",
        "authority_kind": "association_model_grid",
        "protocol_content_sha256": original.protocol_content_sha256,
        "plan_method": "verified_association_model_grid",
        "plan_intent": "Compare prespecified definitions and covariate forms.",
        "cohort_product": original.cohort_product,
        "parent_product": original.primary_product,
        "output_product": "table:association_sensitivity_grid",
        "reference_variant_id": "reference",
        "metadata_columns": ["axis", "source_spec_id"],
        "output_aliases": {},
        "variants": [
            {"analysis_id": "reference", "metadata": {
                "axis": "primary", "source_spec_id": "primary",
            }},
            {"analysis_id": "alternate", "exposure_column": "aki_stage_alternate",
             "metadata": {"axis": "exposure_definition", "source_spec_id": "alternate"}},
            {"analysis_id": "age_form", "nonlinear_terms": [{
                "source_column": "age", "basis": "natural_cubic_spline",
                "degrees_of_freedom": 3, "center_before_basis": True,
            }], "metadata": {"axis": "functional_form", "source_spec_id": "age_form"}},
        ],
    })
    body = original.model_dump(
        mode="json", exclude={"execution_contract_sha256", "association_model_grid"}
    )
    body["schema_version"] = (
        "easyicu.landmark_categorical_association_runtime_authority/2"
    )
    body["association_model_grid"] = grid.model_dump(mode="json")
    authority = build_current_case_scientific_runtime_authority(body)
    context = _context().model_copy(update={"user_preferences": UserPreferences(
        covariates=["age", "sex"],
        sensitivity_specs=[
            {"spec_id": "alternate", "axis": "exposure_definition",
             "strategy": "alternate_exposure",
             "execution_variables": ["aki_stage_alternate"]},
            {"spec_id": "age_form", "axis": "functional_form",
             "strategy": "restricted_cubic_spline", "execution_variables": ["age"]},
        ],
    )})
    shaped, _ = ensure_prespecified_sensitivity_steps(
        plan=_draft_plan(), context=context, runtime_authority=authority
    )
    assert not any(
        step.step_id == "sensitivity_age_form" for step in shaped.steps
    )
    bound = authority.bind_plan(shaped)

    unsigned = _sensitivity_facts(context, bound)
    signed = _sensitivity_facts(context, bound, runtime_authority=authority)

    assert {"alternate", "age_form"}.issubset(unsigned["missing_spec_ids"])
    assert {"alternate", "age_form"}.issubset(signed["executed_spec_ids"])
    assert "exposure_definition" in signed["typed_executable"]
    assert "functional_form" in signed["typed_executable"]
    assert signed["missing_spec_ids"] == []

    grid_step = grid.governed_step(bound, allow_signed_parent=True)
    changed = grid_step.model_copy(update={"inputs": grid_step.inputs[:-1]})
    drifted = bound.model_copy(update={"steps": [
        changed if step.step_id == grid_step.step_id else step
        for step in bound.steps
    ]})
    drifted_facts = _sensitivity_facts(context, drifted, runtime_authority=authority)
    assert {"alternate", "age_form"}.issubset(drifted_facts["missing_spec_ids"])


def test_generic_host_cohort_adoption_defers_to_signed_runtime_owner(tmp_path) -> None:
    from easyicu.research_agent.authority.run_input import (
        _declares_host_cohort_products,
    )
    from easyicu.research_agent.execution.cohort_adoption import (
        record_planned_host_cohort_checkpoint,
    )

    _, _, authority = _projection(tmp_path)
    draft = _draft_plan()
    assert _declares_host_cohort_products(draft.steps[0])
    plan = authority.bind_plan(draft)
    cohort_step = authority.governed_cohort_step(plan)

    # Identical logical output names do not transfer ownership of the signed
    # landmark filter to the generic locked-cohort materializer.
    assert not _declares_host_cohort_products(cohort_step)
    records = []
    preexecuted = set()
    findings = []
    record_planned_host_cohort_checkpoint(
        plan=plan,
        result={},
        cohort_path=tmp_path / "not-materialized.parquet",
        evidence=None,
        prompt_pack_version="test",
        llm_signature="test",
        run_dir=tmp_path,
        reason="test",
        gate_stamp={},
        per_step_records=records,
        preexecuted_step_ids=preexecuted,
        findings=findings,
    )
    assert records == []
    assert preexecuted == set()
    assert findings == []


def test_runtime_planner_constraints_preserve_indices_without_level_literals(tmp_path):
    _, _, authority = _projection(tmp_path)
    # Non-default coordinates demonstrate projection, not an answer-key default.
    authority = authority.model_copy(update={
        "exposure_levels": ("private_a", "private_b", "private_c", "private_d"),
        "exposure_reference_level": "private_b",
        "primary_contrast_level": "private_c",
    })
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    contract = runtime.planning_contract_context()
    coordinates = json.loads(contract.split("\n", 1)[1])
    assert coordinates["exposure_term"]["reference_level_index"] == 1
    assert coordinates["primary_contrast_level_index"] == 2
    assert coordinates["exposure_term"]["coding"] == "categorical"
    assert coordinates["covariates"] == ["age", "sex"]
    assert "private_" not in contract
    assert runtime.planning_contract_context() == authority.planning_contract_context()
    assert ScientificRuntimeAuthorities(
        trajectory=None, current_case=None,
    ).planning_contract_context() == ""


def test_disclosing_coordinates_does_not_allow_primary_contrast_drift(tmp_path):
    _, _, authority = _projection(tmp_path)
    authority.planning_contract_context()
    plan = _draft_plan()
    primary = plan.steps[1]
    changed_requirement = primary.model_requirements[0].model_copy(
        update={"primary_contrast_level": "1"},
    )
    plan = plan.model_copy(update={"steps": [
        plan.steps[0],
        primary.model_copy(update={"model_requirements": [changed_requirement]}),
        *plan.steps[2:],
    ]})
    with pytest.raises(ValueError, match="primary_contrast_level"):
        authority.bind_plan(plan)


def test_signed_landmark_categorical_plan_rebind_is_idempotent(tmp_path) -> None:
    _, _, authority = _projection(tmp_path)
    first = authority.bind_plan(_draft_plan())
    saved = AnalysisPlan.model_validate(first.model_dump(mode="json"))

    rebound, findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(saved)

    assert rebound.model_dump(mode="json") == saved.model_dump(mode="json")
    assert findings[0].detail["reason_code"] == (
        "landmark_categorical_association_host_compiled"
    )


def test_partial_signed_landmark_categorical_plan_still_fails_closed(tmp_path) -> None:
    _, _, authority = _projection(tmp_path)
    draft = _draft_plan()
    partially_signed = draft.model_copy(
        update={
            "steps": [
                draft.steps[0],
                draft.steps[1].model_copy(update={"method": authority.primary_method}),
                *draft.steps[2:],
            ]
        }
    )

    with pytest.raises(ValueError, match="signed cohort owner"):
        authority.bind_plan(partially_signed)


def test_signed_landmark_categorical_owner_filters_then_fits(tmp_path) -> None:
    universe, projection, authority = _projection(tmp_path)
    bound, findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(_draft_plan())

    assert findings[0].detail["reason_code"] == (
        "landmark_categorical_association_host_compiled"
    )
    assert "duplicate_landmark_sensitivity" not in {
        step.step_id for step in bound.steps
    }
    # Host binding is not complete until the public plan survives the same
    # serialize/rehydrate boundary used by normalized-plan authority.
    bound = AnalysisPlan.model_validate(bound.model_dump(mode="json"))
    primary = next(
        step for step in bound.steps if step.planned_analysis_role == "primary"
    )
    assert (
        primary.scientific_capability
        == LANDMARK_CATEGORICAL_ASSOCIATION_CAPABILITY_ID
    )
    capability = resolve_primary_capability(
        analysis_type=bound.analysis_type,
        plan=bound,
    )
    assert capability.failure_reason is None
    assert capability.owner_claimed is True
    assert capability.scientific_validation == "reportable"
    report = next(step for step in bound.steps if step.step_id == "report")
    assert "table:sensitivity_landmark_24h" not in report.inputs
    authority.validate_plan(bound)
    assert timing_design_closed(bound) is True

    cohort_step = authority.governed_cohort_step(bound)
    cohort_selection = select_standard_executor(
        cohort_step,
        plan=bound,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert cohort_selection is not None
    assert cohort_selection.analysis_kind == "signed_landmark_analysis_cohort"
    cohort_summary = run_landmark_categorical_cohort(
        frame=pd.read_parquet(universe),
        source_path=universe,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "cohort",
    )
    assert cohort_summary["n_analysis_cohort"] < cohort_summary["n_source"]
    eligible = pd.read_parquet(tmp_path / "cohort" / "analysis_cohort.parquet")
    assert (eligible["hospital_followup_time_hours"] >= 24).all()
    assert (
        (eligible["death"] == 0) | (eligible["death_time_hours"] > 24)
    ).all()
    assert cohort_step.method == "signed_landmark_analysis_cohort"

    primary = authority.governed_primary_step(bound)
    primary_selection = select_standard_executor(
        primary,
        plan=bound,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert primary_selection is not None
    assert primary_selection.analysis_kind == "adjusted_association_estimates"
    primary_summary = run_landmark_categorical_primary(
        frame=eligible,
        cohort_path=tmp_path / "cohort" / "analysis_cohort.parquet",
        step=primary,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "primary",
    )
    assert primary_summary["variance_estimator"] == "cluster_robust"
    assert primary_summary["cluster_count"] > 1
    assert len(primary_summary["model_contracts"]) == 1
    assert primary_summary["landmark_runtime_receipt"]["exposure_levels"] == [
        "0",
        "1",
        "2",
        "3",
    ]


@pytest.mark.parametrize("mutation", [None, "wrong_population", "wrong_plan", "changed_bytes"])
def test_categorical_absolute_risk_reuses_exact_primary_population(
    tmp_path, monkeypatch, mutation
) -> None:
    from easyicu.research_agent.execution.runners.primary_population_descriptive import (
        run_primary_population_risk,
    )
    from easyicu.research_agent.execution.runners.typed_input_binding import (
        TypedInputBindingError,
    )

    universe, projection, authority = _projection(tmp_path)
    risk = AnalysisStep(
        step_id="risk", planned_analysis_role="secondary",
        method="primary_population_absolute_risk_context",
        intent="Describe the primary model population.",
        inputs=["artifact:analysis_cohort", "table:adjusted_association_estimates"],
        expected_outputs=["table:absolute_risk_context"],
    )
    draft = _draft_plan()
    plan = authority.bind_plan(draft.model_copy(update={
        "steps": [*draft.steps[:2], risk, *draft.steps[2:]],
    }))
    authority.validate_plan(plan)
    risk = next(step for step in plan.steps if step.step_id == "risk")
    assert risk.runtime_outcome_contract is not None
    selected = select_standard_executor(
        risk, plan=plan, current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )
    assert selected is not None and selected.analysis_kind == risk.method
    wrong = risk.model_copy(update={"inputs": list(reversed(risk.inputs))})
    assert select_standard_executor(
        wrong, plan=plan, current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    ) is None
    with pytest.raises(ValueError, match="absolute-risk population"):
        authority.validate_plan(plan.model_copy(update={
            "steps": [wrong if step.step_id == "risk" else step for step in plan.steps],
        }))

    cohort_dir = tmp_path / "cohort"
    run_landmark_categorical_cohort(
        frame=pd.read_parquet(universe), source_path=universe,
        authority=authority, runtime_projection_sha256=projection.projection_sha256,
        out_dir=cohort_dir,
    )
    cohort_path = cohort_dir / "analysis_cohort.parquet"
    primary_dir = tmp_path / "primary"
    primary = authority.governed_primary_step(plan)
    summary = run_landmark_categorical_primary(
        frame=pd.read_parquet(cohort_path), cohort_path=cohort_path,
        step=primary, authority=authority,
        runtime_projection_sha256=projection.projection_sha256, out_dir=primary_dir,
    )
    table_path = primary_dir / summary["output_files"][authority.primary_product]
    if mutation == "wrong_population":
        table = pd.read_csv(table_path)
        table.loc[0, "n"] += 1
        table.to_csv(table_path, index=False)
    manifest = {"step_id": risk.step_id, "inputs": {}}
    for key, path in [(authority.cohort_product, cohort_path), (authority.primary_product, table_path)]:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["inputs"][key] = {
            "relative_path": str(path.relative_to(tmp_path)), "sha256": digest,
            "declared_kind": key.partition(":")[0], "evidence_kind": "table",
            "product": key.partition(":")[2], "evidence_id": key,
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        }
    if mutation == "changed_bytes":
        table = pd.read_csv(table_path)
        table.loc[0, "n"] += 1
        table.to_csv(table_path, index=False)
    (tmp_path / "resolved_inputs.json").write_text(json.dumps(manifest))
    (tmp_path / "manifest_partial.json").write_text(json.dumps({"plan_path": "analysis_plan.json"}))
    if mutation == "wrong_plan":
        plan = plan.model_copy(update={"steps": [
            step.model_copy(update={"intent": "changed"}) if step.step_id == "risk" else step
            for step in plan.steps
        ]})
    (tmp_path / "analysis_plan.json").write_text(plan.model_dump_json())
    (tmp_path / "research_context.json").write_text(json.dumps({"target_outcome": "death"}))
    out = tmp_path / "risk_out"
    for key, value in {
        "EASYICU_RUN_DIR": tmp_path, "EASYICU_STEP_ID": risk.step_id,
        "COHORT_PARQUET": cohort_path, "STEP_OUT_DIR": out,
    }.items():
        monkeypatch.setenv(key, str(value))
    if mutation is not None:
        failure = {
            "wrong_population": "does not match",
            "wrong_plan": "differs from the bound plan",
            "changed_bytes": "digest_mismatch",
        }[mutation]
        with pytest.raises((ValueError, TypedInputBindingError), match=failure):
            run_primary_population_risk(
                step=risk, authority=authority,
                runtime_projection_sha256=projection.projection_sha256,
                run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json",
            )
        assert not (out / "step_summary.json").exists()
        return
    result = run_primary_population_risk(
        step=risk, authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        run_dir=tmp_path, resolved_inputs=tmp_path / "resolved_inputs.json",
    )
    assert result["n_total"] == summary["n_total"]
    assert result["population_binding"]["event_n"] == summary["n_events"]
    assert set(pd.read_csv(out / "absolute_risk_context.csv")["population_n"]) == {
        summary["n_total"]
    }


@pytest.mark.parametrize("owner", ["cohort", "primary"])
@pytest.mark.parametrize("with_obligations", [False, True])
def test_landmark_generated_script_compiles_with_plausibility_scope(
    tmp_path, owner, with_obligations
) -> None:
    _, projection, authority = _projection(tmp_path)
    plan = authority.bind_plan(_draft_plan())
    if owner == "cohort":
        step = authority.governed_cohort_step(plan)
        renderer = landmark_categorical_cohort_executor_code
        column = "death"
    else:
        step = authority.governed_primary_step(plan)
        renderer = landmark_categorical_primary_executor_code
        column = "age"
    scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=(column,) if with_obligations else (),
        source_contracts_sha256="a" * 64,
        authority_kind="synthetic_raw_input_contracts",
    )

    code = renderer(
        step,
        plan=plan,
        authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
        plausibility_scope=scope,
    )

    compile(code, f"<{owner}-landmark-script>", "exec")


@pytest.mark.parametrize(
    ("column", "nonfinite", "message"),
    [
        ("event_time", np.inf, "event time is non-finite"),
        ("event_time", -np.inf, "event time is non-finite"),
        ("duration", np.inf, "observation duration is non-finite"),
        ("duration", -np.inf, "observation duration is non-finite"),
    ],
)
def test_landmark_eligibility_refuses_nonfinite_temporal_evidence(
    column, nonfinite, message
) -> None:
    frame = pd.DataFrame(
        {"death": [1, 0], "event_time": [48.0, np.nan], "duration": [72.0, 48.0]}
    )
    frame.loc[0 if column == "event_time" else 1, column] = nonfinite

    with pytest.raises(LandmarkCategoricalExecutionError, match=message):
        landmark_eligibility_mask(
            frame,
            outcome_column="death",
            event_time_column="event_time",
            observation_duration_column="duration",
            observation_duration_unit="hours",
            landmark_hours=24,
        )


def test_landmark_eligibility_preserves_missing_censoring_and_exact_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "death": [0, 0, 1, 1, 1],
            "event_time": [np.nan, np.nan, 24.0, 25.0, -1.0],
            "duration": [1.0, np.nan, 1.0, 1.0, 1.0],
        }
    )

    nonnegative, alive, observed = landmark_eligibility_mask(
        frame,
        outcome_column="death",
        event_time_column="event_time",
        observation_duration_column="duration",
        observation_duration_unit="days",
        landmark_hours=24,
    )

    assert (nonnegative & alive & observed).tolist() == [True, False, False, True, False]
