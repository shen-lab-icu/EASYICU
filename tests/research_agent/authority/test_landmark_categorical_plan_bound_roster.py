"""Plan-bound adjustment roster for the signed landmark categorical authority.

A ``planner_selectable`` categorical landmark study compiles a v3 contract
whose roster is empty and whose executable column domain is sealed from the
universe schema. The host seals the Planner-selected roster from the reviewed
primary model at bind time, re-signs the contract, and every later phase holds
that sealed digest. These tests run offline on a synthetic, case-neutral frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
    run_landmark_categorical_cohort,
    run_landmark_categorical_primary,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.resume_plan_migration import (
    _migrate_resume_scientific_runtime_binding,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
)

V3 = "easyicu.landmark_categorical_association_runtime_authority/3"


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    patients = 240
    n = patients * 2
    stage = np.tile(np.arange(4), n // 4)
    age = rng.normal(64.0, 11.0, size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    probability = 1.0 / (1.0 + np.exp(-(-3.2 + 0.38 * stage + 0.012 * (age - 64.0))))
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
                f"p{patient}:s{stay}" for patient in range(patients) for stay in (1, 2)
            ],
            "aki_stage_strict": stage,
            "aki_stage_alt": stage,
            "death": death,
            "death_time_hours": death_time,
            "hospital_followup_time_hours": followup,
            "age": age,
            "sex": sex,
            "charlson": rng.integers(0, 8, size=n).astype("float64"),
            "admit_time": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )


def _landmark_spec() -> PrespecifiedSensitivitySpec:
    return PrespecifiedSensitivitySpec.model_validate(
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


def _dependence() -> PlannedDependenceRequirement:
    return PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )


def _projection(tmp_path, *, extra_specs=(), declared_covariates=()):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "planner_selectable"},
        sensitivity_specs=(_landmark_spec(), *extra_specs),
        primary_exposure="aki_stage_strict",
        primary_exposure_source="aki_stage",
        target_outcome="death",
        declared_covariates=tuple(declared_covariates),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="b" * 64,
        dependence=_dependence(),
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    return universe, projection, authority


def _draft_plan(covariates: list[str] | None = None) -> AnalysisPlan:
    roster = ["age", "sex"] if covariates is None else covariates
    term_by_name = {
        "age": {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
        "charlson": {
            "name": "charlson",
            "role": "covariate",
            "coding": "continuous",
            "transform": "identity",
        },
        "sex": {
            "name": "sex",
            "role": "covariate",
            "coding": "binary",
            "levels": ["Female", "Male"],
            "reference_level": "Female",
            "transform": "treatment_contrast",
        },
    }
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
                    "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
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
                        *roster,
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
                            "covariates": list(roster),
                            "model_terms": [
                                {
                                    "name": "aki_stage_strict",
                                    "role": "exposure",
                                    "coding": "categorical",
                                    "levels": ["0", "1", "2", "3"],
                                    "reference_level": "0",
                                    "transform": "treatment_contrast",
                                },
                                *(
                                    term_by_name.get(
                                        name,
                                        {
                                            "name": name,
                                            "role": "covariate",
                                            "coding": "continuous",
                                            "transform": "identity",
                                        },
                                    )
                                    for name in roster
                                ),
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
                    "step_id": "report",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Report results.",
                    "inputs": ["table:adjusted_association_estimates"],
                    "expected_outputs": ["report:study"],
                    "method": "scientific_reporting",
                },
            ],
        }
    )


def test_planner_selectable_projection_seals_the_executable_domain_not_a_roster(tmp_path):
    _, projection, authority = _projection(tmp_path, declared_covariates=("age",))

    assert authority.schema_version == V3
    assert authority.required_adjustment_columns == ()
    assert authority.categorical_adjustment_columns == ()
    assert authority.adjustment_roster_sealed is False
    roster = authority.plan_bound_adjustment_roster
    assert roster is not None and roster.sealed is False
    assert roster.authority == "plan_primary_model"
    # Design coordinates and unsupported physical types never enter the domain;
    # a declared hint list does not pre-seal anything.
    assert set(roster.admissible_columns) == {"aki_stage_alt", "age", "sex", "charlson"}
    assert roster.admissible_categorical_columns == ("sex",)
    assert projection.authority["plan_bound_adjustment_roster"]["sealed"] is False


def test_unsealed_contract_fails_closed_everywhere(tmp_path):
    _, _, authority = _projection(tmp_path)
    plan = _draft_plan()

    for action in (authority.bind_plan, authority.validate_plan, authority.governed_primary_step):
        with pytest.raises(CurrentCaseScientificAuthorityError, match="has not sealed"):
            action(plan)
    contract = authority.planning_contract_context()
    coordinates = json.loads(contract.split("\n", 1)[1])
    assert "covariates" not in coordinates
    assert coordinates["adjustment_roster_authority"] == "plan_primary_model"
    assert set(coordinates["admissible_adjustment_columns"]) == {"aki_stage_alt", "age", "sex", "charlson"}
    assert "Planner-selected" in contract


def test_host_seals_the_planner_roster_and_resigns_the_contract(tmp_path):
    universe, projection, unsealed = _projection(tmp_path)
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)

    sealed_runtime = runtime.seal_for_plan(_draft_plan())
    sealed = sealed_runtime.current_case
    assert isinstance(sealed, LandmarkCategoricalAssociationRuntimeAuthority)
    assert sealed is not unsealed
    assert sealed.required_adjustment_columns == ("age", "sex")
    assert sealed.categorical_adjustment_columns == ("sex",)
    assert sealed.adjustment_roster_sealed is True
    assert sealed.execution_contract_sha256 != unsealed.execution_contract_sha256
    assert sealed.protocol_content_sha256 == unsealed.protocol_content_sha256
    # Sealed contracts are self-consistent through the JSON boundary the
    # generated executor script crosses.
    assert (
        load_current_case_scientific_runtime_authority(sealed.model_dump(mode="json"))
        == sealed
    )
    # Idempotent and deterministic.
    assert sealed_runtime.seal_for_plan(_draft_plan()) is sealed_runtime
    assert runtime.seal_for_plan(_draft_plan()).current_case == sealed
    coordinates = json.loads(sealed.planning_contract_context().split("\n", 1)[1])
    assert coordinates["covariates"] == ["age", "sex"]

    bound, findings = runtime.bind_plan(_draft_plan())
    detail = findings[0].detail
    assert detail["reason_code"] == "landmark_categorical_association_host_compiled"
    assert detail["adjustment_roster_authority"] == "plan_primary_model"
    assert detail["adjustment_roster"] == ["age", "sex"]
    assert detail["execution_contract_sha256"] == sealed.execution_contract_sha256
    assert detail["unsealed_execution_contract_sha256"] == unsealed.execution_contract_sha256
    primary = sealed.governed_primary_step(bound)
    assert sealed.plan_rule_ref in primary.icu_rule_refs
    assert unsealed.plan_rule_ref not in primary.icu_rule_refs
    assert tuple(primary.inputs) == (
        "artifact:analysis_cohort",
        "aki_stage_strict",
        "death",
        "age",
        "sex",
        "patient_stay_id",
    )
    sealed.validate_plan(bound)
    with pytest.raises(CurrentCaseScientificAuthorityError, match="has not sealed"):
        unsealed.validate_plan(bound)

    # The sealed contract, not the Web projection's unsealed one, owns execution.
    cohort_step = sealed.governed_cohort_step(bound)
    assert (
        select_standard_executor(
            cohort_step,
            plan=bound,
            current_case_scientific_runtime_authority=sealed.model_dump(mode="json"),
            scientific_runtime_projection_sha256=projection.projection_sha256,
        ).analysis_kind
        == "signed_landmark_analysis_cohort"
    )
    assert (
        select_standard_executor(
            cohort_step,
            plan=bound,
            current_case_scientific_runtime_authority=projection.authority,
            scientific_runtime_projection_sha256=projection.projection_sha256,
        )
        is None
    )
    cohort_summary = run_landmark_categorical_cohort(
        frame=pd.read_parquet(universe),
        source_path=universe,
        authority=sealed,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "cohort",
    )
    assert cohort_summary["landmark_runtime_receipt"]["execution_contract_sha256"] == (
        sealed.execution_contract_sha256
    )
    eligible = pd.read_parquet(tmp_path / "cohort" / "analysis_cohort.parquet")
    primary_summary = run_landmark_categorical_primary(
        frame=eligible,
        cohort_path=tmp_path / "cohort" / "analysis_cohort.parquet",
        step=primary,
        authority=sealed,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "primary",
    )
    assert primary_summary["variance_estimator"] == "cluster_robust"
    assert primary_summary["landmark_runtime_receipt"]["execution_contract_sha256"] == (
        sealed.execution_contract_sha256
    )


def test_saved_plan_reseals_to_the_recorded_digest_and_rejects_tampering(tmp_path):
    _, _, unsealed = _projection(tmp_path)
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)
    bound, _ = runtime.bind_plan(_draft_plan())
    saved = AnalysisPlan.model_validate(bound.model_dump(mode="json"))
    first_digest = runtime.seal_for_plan(_draft_plan()).current_case.execution_contract_sha256

    # Resume: the unsealed Web contract re-seals from the signed primary to the
    # digest the saved plan already references, and the plan is unchanged.
    rebound, findings = runtime.bind_plan(saved)
    assert rebound.model_dump(mode="json") == saved.model_dump(mode="json")
    assert findings[0].detail["execution_contract_sha256"] == first_digest
    resealed = runtime.seal_for_plan(saved).current_case
    assert resealed.execution_contract_sha256 == first_digest
    migrated, path, changed, _ = _migrate_resume_scientific_runtime_binding(
        plan=saved,
        resume_state={"per_step_records": []},
        resume_from_step_id=saved.steps[0].step_id,
        scientific_runtime_authorities=runtime.seal_for_plan(saved),
        run_dir=tmp_path,
        evidence=EvidenceStore(tmp_path),
        prompt_version="test",
        llm_signature="mock",
    )
    assert path is None and changed == ()
    assert migrated == saved

    # Tampering: editing the saved roster changes the sealed digest, so the
    # recorded rule ref no longer matches and validation fails closed.
    signed_primary = next(step for step in saved.steps if step.method == unsealed.primary_method)
    requirement = signed_primary.model_requirements[0]
    reduced = requirement.model_copy(
        update={
            "covariates": ["age"],
            "model_terms": [term for term in requirement.model_terms if term.name != "sex"],
        }
    )
    tampered = saved.model_copy(
        update={
            "steps": [
                signed_primary.model_copy(update={"model_requirements": [reduced]})
                if step is signed_primary
                else step
                for step in saved.steps
            ]
        }
    )
    tampered_runtime = runtime.seal_for_plan(tampered)
    assert tampered_runtime.current_case.execution_contract_sha256 != first_digest
    with pytest.raises(CurrentCaseScientificAuthorityError, match="signed scientific runtime digest"):
        tampered_runtime.validate_plan(tampered)


def test_roster_outside_the_sealed_domain_fails_closed(tmp_path):
    _, _, unsealed = _projection(tmp_path)
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)

    with pytest.raises(CurrentCaseScientificAuthorityError, match="admissible adjustment domain: admit_time"):
        runtime.seal_for_plan(_draft_plan(["age", "admit_time"]))
    with pytest.raises(
        CurrentCaseScientificAuthorityError,
        match="admissible adjustment domain: death_time_hours",
    ):
        runtime.bind_plan(_draft_plan(["death_time_hours"]))
    # An unadjusted Planner decision is a roster too: it seals to an empty set.
    empty = runtime.seal_for_plan(_draft_plan([])).current_case
    assert empty.required_adjustment_columns == ()
    assert empty.adjustment_roster_sealed is True


def test_plan_bound_grid_binds_alternate_exposures_and_refuses_prefixed_forms(tmp_path):
    alternate = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "alt_stage",
            "axis": "exposure_definition",
            "strategy": "alternate_exposure",
            "execution_variables": ["aki_stage_alt"],
        }
    )
    _, _, unsealed = _projection(tmp_path, extra_specs=(alternate,))
    assert unsealed.schema_version == V3
    assert unsealed.association_model_grid is not None
    assert "aki_stage_alt" not in unsealed.plan_bound_adjustment_roster.admissible_columns
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)
    bound, _ = runtime.bind_plan(_draft_plan(["age", "charlson"]))
    sealed = runtime.seal_for_plan(_draft_plan(["age", "charlson"])).current_case
    sealed.validate_plan(bound)
    grid_step = sealed.association_model_grid.governed_step(bound, allow_signed_parent=True)
    assert {"age", "charlson", "aki_stage_alt"} <= set(grid_step.inputs)

    form = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "age_restricted_cubic_spline",
            "axis": "functional_form",
            "strategy": "restricted_cubic_spline",
            "execution_variables": ["age"],
        }
    )
    with pytest.raises(WebScientificRuntimeProjectionError) as failure:
        _projection(tmp_path, extra_specs=(form,))
    assert failure.value.code == "web_model_grid_functional_form_requires_exact_roster"
    assert failure.value.details["spec_ids"] == ["age_restricted_cubic_spline"]


def test_exact_contracts_keep_their_signed_bytes_and_ignore_sealing(tmp_path):
    universe = tmp_path / "universe.parquet"
    _frame().to_parquet(universe, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "exact"},
        sensitivity_specs=(_landmark_spec(),),
        primary_exposure="aki_stage_strict",
        primary_exposure_source="aki_stage",
        target_outcome="death",
        declared_covariates=("age", "sex"),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="b" * 64,
        dependence=_dependence(),
    )
    exact = load_current_case_scientific_runtime_authority(projection.authority)
    assert exact.schema_version.endswith("/1")
    assert "plan_bound_adjustment_roster" not in {
        key for key, value in projection.authority.items() if value is not None
    }
    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=exact)
    assert runtime.seal_for_plan(_draft_plan()) is runtime
    _, findings = runtime.bind_plan(_draft_plan())
    assert "adjustment_roster" not in findings[0].detail
    with pytest.raises(ValueError, match="v1/v2 contracts name their roster"):
        load_current_case_scientific_runtime_authority(
            {
                **projection.authority,
                "plan_bound_adjustment_roster": {
                    "authority": "plan_primary_model",
                    "admissible_columns": ["age"],
                    "admissible_categorical_columns": [],
                    "sealed": False,
                },
            }
        )


def test_pipeline_holds_the_sealed_contract_for_the_reviewed_plan(tmp_path):
    """The pipeline swaps in the sealed authority before validation and review.

    A locked development plan stands in for the Planner (no Provider call); the
    persisted plan must reference the sealed digest, and the pipeline instance
    that later executes and reviews the run must hold that same sealed contract
    rather than the Web projection's unsealed one.
    """
    import hashlib

    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    universe = tmp_path / "universe.parquet"
    _frame().drop(columns=["admit_time"]).to_parquet(universe, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": "planner_selectable"},
        sensitivity_specs=(_landmark_spec(),),
        primary_exposure="aki_stage_strict",
        primary_exposure_source="aki_stage",
        target_outcome="death",
        declared_covariates=(),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="b" * 64,
        dependence=None,
    )
    unsealed = load_current_case_scientific_runtime_authority(projection.authority)
    draft = _draft_plan()
    primary = draft.steps[1]
    requirement = primary.model_requirements[0].model_copy(update={"dependence": None})
    draft = draft.model_copy(
        update={
            "steps": [
                draft.steps[0],
                primary.model_copy(
                    update={
                        "model_requirements": [requirement],
                        "inputs": [value for value in primary.inputs if value != "patient_stay_id"],
                    }
                ),
                *draft.steps[2:],
            ]
        }
    )
    plan_path = tmp_path / "locked_plan.json"
    plan_path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    client = ScriptedMockLLMClient([])
    pipeline = ResearchAgentPipeline(
        config=PipelineConfig(
            workdir=tmp_path / "pipeline",
            planner_only=True,
            development_diagnostic=True,
            require_human_plan_review=True,
            development_locked_analysis_plan_path=plan_path,
            development_locked_analysis_plan_sha256=hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest(),
            current_case_scientific_runtime_authority=projection.authority,
            scientific_runtime_projection_sha256=projection.projection_sha256,
            enable_memory=False,
            enable_replanning=False,
        ),
        services=PipelineServices(llm=client),
    )
    outcome = pipeline.run(
        question=draft.research_question,
        cohort=universe,
        database="synthetic",
        target_outcome="death",
        primary_exposure="aki_stage_strict",
        id_columns=["patient_stay_id"],
        user_preferences={
            "covariate_selection": "planner_selectable",
            "landmark_hours": 24.0,
            "sensitivity_specs": [_landmark_spec().model_dump(mode="json")],
            "data_constraints": json.dumps(
                {
                    "analysis_design": {
                        "analysis_family": "association_study",
                        "analysis_unit": "icu_stay",
                        "variance_estimator": "model_based",
                    },
                    "cohort": {"exclude_readmissions": False},
                }
            ),
        },
        stop_after_analysis=True,
    )
    assert type(outcome).__name__ == "HumanReviewPending"
    assert not client.calls

    sealed = pipeline._scientific_runtime_authorities.current_case
    assert isinstance(sealed, LandmarkCategoricalAssociationRuntimeAuthority)
    assert sealed.adjustment_roster_sealed is True
    assert sealed.required_adjustment_columns == ("age", "sex")
    assert sealed.execution_contract_sha256 != unsealed.execution_contract_sha256
    saved = json.loads(
        (Path(outcome.run_dir) / "analysis_plan.json").read_text(encoding="utf-8")
    )
    signed_primary = next(
        step for step in saved["steps"] if step["method"] == unsealed.primary_method
    )
    assert signed_primary["model_requirements"][0]["covariates"] == ["age", "sex"]
    assert signed_primary["icu_rule_refs"] == [sealed.plan_rule_ref]
    assert unsealed.plan_rule_ref not in signed_primary["icu_rule_refs"]
    sealed.validate_plan(AnalysisPlan.model_validate(saved))
    with pytest.raises(CurrentCaseScientificAuthorityError, match="has not sealed"):
        unsealed.validate_plan(AnalysisPlan.model_validate(saved))


# ---------------------------------------------------------------------------
# The continuous-exposure (restricted cubic spline) landmark contract seals its
# roster the same way; its signed primary carries the roster in its inputs.
# ---------------------------------------------------------------------------


def _spline_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260924)
    n = 400
    return pd.DataFrame(
        {
            "patient_stay_id": [f"p{i}:s1" for i in range(n)],
            "lact_max": rng.lognormal(0.6, 0.5, size=n),
            "death": rng.binomial(1, 0.2, size=n),
            "death_time_hours": rng.uniform(30.0, 200.0, size=n),
            "hospital_followup_time_hours": rng.uniform(40.0, 240.0, size=n),
            "age": rng.normal(63.0, 12.0, size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "charlson": rng.integers(0, 7, size=n).astype("float64"),
        }
    )


def _spline_projection(tmp_path, *, selection: str):
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        LandmarkSplineRuntimeAuthority,
    )

    universe = tmp_path / "spline_universe.parquet"
    _spline_frame().to_parquet(universe, index=False)
    spline = PrespecifiedSensitivitySpec.model_validate(
        {
            "spec_id": "easyicu_auto_primary_exposure_rcs",
            "axis": "functional_form",
            "strategy": "restricted_cubic_spline",
            "execution_variables": ["lact_max"],
        }
    )
    projection = compile_web_scientific_runtime_projection(
        study={"covariate_selection": selection},
        sensitivity_specs=(_landmark_spec(), spline),
        primary_exposure="lact_max",
        primary_exposure_source="lact_max",
        target_outcome="death",
        declared_covariates=("age", "sex") if selection == "exact" else (),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256="d" * 64,
        dependence=_dependence(),
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    return projection, authority


def _spline_draft(covariates: list[str]) -> AnalysisPlan:
    terms = [
        {"name": "lact_max", "role": "exposure", "coding": "continuous", "transform": "identity"},
        *(
            {
                "name": "sex",
                "role": "covariate",
                "coding": "binary",
                "levels": ["F", "M"],
                "reference_level": "F",
                "transform": "treatment_contrast",
            }
            if name == "sex"
            else {"name": name, "role": "covariate", "coding": "continuous", "transform": "identity"}
            for name in covariates
        ),
    ]
    return AnalysisPlan.model_validate(
        {
            "research_question": "Peak lactate and hospital mortality after a 24-hour landmark.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "define_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Define the analysis cohort.",
                    "inputs": ["death", "death_time_hours"],
                    "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
                    "method": "cohort_definition_and_attrition",
                },
                {
                    "step_id": "primary_adjusted_model",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the adjusted lactate association.",
                    "inputs": ["artifact:analysis_cohort", "lact_max", "death", *covariates, "patient_stay_id"],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "model_requirements": [
                        {
                            "requirement_id": "primary_lactate_model",
                            "outcome": "death",
                            "outcome_type": "binary",
                            "method_family": "statsmodels_logit_mle",
                            "exposure_source": "lact_max",
                            "analysis_role": "primary",
                            "analysis_set": "source_aware",
                            "covariates": list(covariates),
                            "model_terms": terms,
                            "dependence": {
                                "group_source": "patient_stay_id",
                                "group_derivation": "prefix_before_delimiter",
                                "delimiter": ":s",
                            },
                        }
                    ],
                },
                {
                    "step_id": "display",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the primary result.",
                    "inputs": ["table:adjusted_association_estimates"],
                    "expected_outputs": ["figure:primary_result"],
                    "method": "visualization",
                },
            ],
        }
    )


def test_spline_contract_seals_the_planner_roster_and_reseals_on_resume(tmp_path):
    projection, unsealed = _spline_projection(tmp_path, selection="planner_selectable")
    assert unsealed.schema_version.endswith("/5")
    assert unsealed.required_adjustment_columns == ()
    roster = unsealed.plan_bound_adjustment_roster
    assert roster is not None and roster.sealed is False
    assert set(roster.admissible_columns) == {"age", "sex", "charlson"}
    assert roster.admissible_categorical_columns == ("sex",)
    with pytest.raises(CurrentCaseScientificAuthorityError, match="has not sealed"):
        unsealed.bind_plan(_spline_draft(["age", "sex"]))

    runtime = ScientificRuntimeAuthorities(trajectory=None, current_case=unsealed)
    sealed = runtime.seal_for_plan(_spline_draft(["age", "sex"])).current_case
    assert sealed.required_adjustment_columns == ("age", "sex")
    assert sealed.categorical_adjustment_columns == ("sex",)
    assert sealed.execution_contract_sha256 != unsealed.execution_contract_sha256
    assert "age" in sealed.required_columns and "sex" in sealed.required_columns
    bound, findings = runtime.bind_plan(_spline_draft(["age", "sex"]))
    assert findings[0].detail["reason_code"] == "landmark_spline_host_compiled"
    assert findings[0].detail["adjustment_roster"] == ["age", "sex"]
    assert findings[0].detail["unsealed_execution_contract_sha256"] == unsealed.execution_contract_sha256
    sealed.validate_plan(bound)
    primary = sealed.governed_step(bound)
    assert primary.model_requirements == []
    assert list(primary.inputs) == ["artifact:analysis_cohort", *sealed.required_columns]

    # Resume: the signed primary carries the roster in its inputs, so the
    # unsealed Web contract re-seals to the recorded digest without a draft.
    saved = AnalysisPlan.model_validate(bound.model_dump(mode="json"))
    resealed = runtime.seal_for_plan(saved).current_case
    assert resealed.execution_contract_sha256 == sealed.execution_contract_sha256
    rebound, _ = runtime.bind_plan(saved)
    assert rebound.model_dump(mode="json") == saved.model_dump(mode="json")
    # An edited signed roster fails the rule-ref check instead of being accepted.
    signed = next(step for step in saved.steps if step.method == unsealed.plan_method)
    tampered_inputs = [value for value in signed.inputs if value != "sex"]
    tampered = saved.model_copy(
        update={
            "steps": [
                signed.model_copy(update={"inputs": tampered_inputs}) if step is signed else step
                for step in saved.steps
            ]
        }
    )
    # The owner ref and the rule ref both carry the digest; either check closes it.
    with pytest.raises(
        CurrentCaseScientificAuthorityError,
        match="drifted from signed authority|signed scientific runtime digest",
    ):
        runtime.seal_for_plan(tampered).validate_plan(tampered)
    with pytest.raises(CurrentCaseScientificAuthorityError, match="admissible adjustment domain: death_time_hours"):
        runtime.seal_for_plan(_spline_draft(["age", "death_time_hours"]))

    exact_projection, exact = _spline_projection(tmp_path, selection="exact")
    assert exact.schema_version.endswith("/4")
    assert exact.required_adjustment_columns == ("age", "sex")
    assert ScientificRuntimeAuthorities(trajectory=None, current_case=exact).seal_for_plan(
        _spline_draft(["age", "sex"])
    ).current_case is exact
