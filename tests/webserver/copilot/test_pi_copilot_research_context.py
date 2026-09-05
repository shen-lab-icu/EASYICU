"""Typed research-context compilation from reviewed Copilot study state."""

from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any
import pytest
from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.schema import UserPreferences
from easyicu.webserver import (
    agent_pipeline_runs,
    research_launch_scientific,
    research_pipeline_run_preparation,
)
from tests.webserver.copilot.research_workflow_fixtures import (
    complete_study as _complete_study,
)

from tests.webserver.copilot.research_workflow_fixtures import (
    _foundation_profile as _foundation_profile,
)


def test_web_data_foundation_profile_keeps_continuous_outcome_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="sex",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="los_icu",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="los_icu",
    )

    assert profile == {
        "allowed_modules": ("demographics", "outcome"),
        "static_concepts": ("age", "sex", "los_icu"),
        "outcome_concepts": (),
        "required_feature_concepts": (),
        "require_outcome": False,
        "primary_exposure_source_concept": None,
    }


def test_web_data_foundation_profile_keeps_all_candidate_plan_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="los_icu",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
        additional_outcomes=("los_icu", "death"),
    )

    assert profile["static_concepts"] == ("age", "los_icu")
    assert profile["outcome_concepts"] == ("death",)
    assert profile["require_outcome"] is True


def test_web_data_foundation_profile_keeps_event_outcome_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
    )

    assert profile["static_concepts"] == ("age",)
    assert profile["outcome_concepts"] == ("death",)
    assert profile["require_outcome"] is True


def test_web_data_foundation_profile_keeps_legacy_owner_declared_event_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="legacy-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=False,
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=False,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/legacy/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
    )

    assert profile["static_concepts"] == ("age",)
    assert profile["outcome_concepts"] == ("death",)
    assert profile["require_outcome"] is True


def test_web_data_foundation_materializes_typed_exposure_and_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="sex",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="sep3_sofa2",
                    file_name="sepsis3_sofa2.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome", "sepsis3_sofa2"],
        },
        target="death",
        primary_exposure="sep3_sofa2_max",
        covariates=("age", "sex"),
    )

    assert profile == {
        "allowed_modules": ("demographics", "outcome", "sepsis3_sofa2"),
        "static_concepts": ("age", "sex"),
        "outcome_concepts": ("death",),
        "required_feature_concepts": ("sep3_sofa2",),
        "require_outcome": True,
        "primary_exposure_source_concept": "sep3_sofa2",
    }


def test_web_data_foundation_materializes_sensitivity_support_without_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module
    from easyicu.research_agent.planning.sensitivity_authority import (
        normalize_prespecified_sensitivities,
    )

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="icu_readmission",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )
    specs = normalize_prespecified_sensitivities(
        [
            {
                "spec_id": "non_readmission_only",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            }
        ]
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={"modules": ["demographics", "outcome"]},
        target="death",
        covariates=(),
        sensitivity_specs=specs,
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")
    assert profile["required_feature_concepts"] == ()


def test_web_data_foundation_keeps_available_readmission_safety_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="icu_readmission",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome"],
            "covariate_selection": "planner_selectable",
        },
        target="death",
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")


def test_web_data_foundation_materializes_owner_readmission_indicator_for_first_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
                CatalogConcept(
                    concept_id="icu_readmission",
                    file_name="outcome.parquet",
                    typed_metadata=False,
                    column_role="event_status",
                ),
            ],
        ),
    )

    profile = research_launch_scientific._data_foundation_profile(
        export_path="/typed/demo",
        study={
            "modules": ["demographics", "outcome"],
            "cohort": {"exclude_readmissions": True},
        },
        target="death",
    )

    assert profile["static_concepts"] == ("age", "icu_readmission")
    assert profile["required_feature_concepts"] == ()


def test_web_data_foundation_rejects_first_stay_without_owner_indicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
                CatalogConcept(
                    concept_id="death",
                    file_name="outcome.parquet",
                    typed_metadata=True,
                    column_role="event_status",
                ),
            ],
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        research_launch_scientific._data_foundation_profile(
            export_path="/typed/demo",
            study={
                "modules": ["demographics", "outcome"],
                "cohort": {"exclude_readmissions": True},
            },
            target="death",
        )

    assert exc.value.code == "research_pipeline_readmission_indicator_unavailable"


def test_web_study_context_compiles_typed_sensitivity_authority() -> None:
    study = {
        **_complete_study(),
        "sensitivity_specs": [
            {
                "spec_id": "landmark_24h",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
                "exclude_negative_event_times": True,
            },
            {
                "spec_id": "non_readmission_only",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            },
        ],
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert compiled["landmark_hours"] == 24
    assert [spec.spec_id for spec in validated.sensitivity_specs] == [
        "landmark_24h",
        "non_readmission_only",
    ]


def test_web_study_context_drops_legacy_primary_cluster_duplicate() -> None:
    study = {
        **_complete_study(),
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "sensitivity_specs": [
            {
                "spec_id": "repeated_stays_cluster_robust",
                "axis": "repeated_stays",
                "strategy": "cluster_robust",
            },
            {
                "spec_id": "non_readmission_only",
                "axis": "repeated_stays",
                "strategy": "non_readmission_restriction",
                "execution_variables": ["icu_readmission"],
            },
        ],
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert [spec.spec_id for spec in validated.sensitivity_specs] == [
        "non_readmission_only"
    ]


def test_descriptive_timing_choice_declines_landmark_in_typed_preferences() -> None:
    study = {
        **_complete_study(),
        "confirmations": {
            **_complete_study()["confirmations"],
            "plan_timing_descriptive_only": True,
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert json.loads(validated.timing_and_design or "{}") == {
        "analysis_scope": "descriptive_distribution_only",
        "association_timing": "not_authorized",
        "landmark": "not_authorized",
    }


def test_web_data_foundation_resolves_only_issued_operational_exposure() -> None:
    acquisition = SimpleNamespace(
        analysis_columns={"sep3_sofa2": "sep3_sofa2_max"},
        materialized_columns=("stay_id", "sep3_sofa2_max", "death"),
    )

    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2_max",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        == "sep3_sofa2_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        == "sep3_sofa2_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="sep3_sofa2_mean",
            source_concept="sep3_sofa2",
            acquisition=acquisition,
        )
        is None
    )
    continuous = SimpleNamespace(
        analysis_columns={},
        materialized_columns=("stay_id", "lact_max", "death"),
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="lact",
            source_concept="lact",
            aggregation="max",
            acquisition=continuous,
        )
        == "lact_max"
    )
    assert (
        agent_pipeline_runs._resolve_materialized_primary_exposure(
            configured="lact",
            source_concept="lact",
            aggregation="mean",
            acquisition=continuous,
        )
        is None
    )


def test_web_data_foundation_rejects_unmaterialized_primary_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_available_catalog",
        lambda _path: AvailableCatalog(
            source="typed-demo",
            concepts=[
                CatalogConcept(
                    concept_id="age",
                    file_name="demographics.parquet",
                    typed_metadata=True,
                    column_role="value",
                ),
            ],
        ),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        research_launch_scientific._data_foundation_profile(
            export_path="/typed/demo",
            study={"modules": ["demographics"]},
            target=None,
            primary_exposure="missing_exposure",
        )

    assert exc.value.code == (
        "research_pipeline_primary_exposure_outside_configured_modules"
    )


def test_pipeline_factory_validates_execution_concepts_before_job_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def reject(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        raise agent_pipeline_runs.ResearchPipelineRunError(
            "research_pipeline_target_outside_configured_modules",
            "The configured outcome is not available in the selected feature modules.",
            details={
                "field": "execution_concepts.outcome",
                "concept_id": kwargs.get("target"),
            },
        )

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        reject,
    )
    study = {
        **_complete_study(),
        "outcome": "In-hospital mortality from the outcome module",
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "heart_rate",
            "covariates": ["age", "sex"],
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
            budget_mode="full_reviewed",
        )

    assert exc.value.code == "research_pipeline_target_outside_configured_modules"
    assert exc.value.details == {
        "field": "execution_concepts.outcome",
        "concept_id": "death",
    }
    assert calls[0]["target"] == "death"
    assert calls[0]["primary_exposure"] == "heart_rate"
    assert calls[0]["covariates"] == ("age", "sex")


def test_pipeline_factory_rejects_generic_sepsis_sofa2_before_job_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def unexpected_foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        unexpected_foundation,
    )
    study = {
        **_complete_study(),
        "question": "What is standard Sepsis-3 prevalence and mortality association?",
        "primary_exposure": "Sepsis-3 using SOFA-2",
        "modules": ["outcome", "sepsis3_sofa2"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa2",
            "covariates": [],
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "concept_explicit_selection_required"
    assert exc.value.details["canonical_alternative"] == "sep3_sofa1"
    assert foundation_called is False


def test_pipeline_factory_accepts_owner_confirmed_explicit_sepsis_concept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        **_complete_study(),
        "question": "What is standard Sepsis-3 prevalence and mortality?",
        "primary_exposure": "Sepsis-3 using SOFA-2",
        "modules": ["outcome", "sepsis3_sofa2"],
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa2",
            "covariates": [],
        },
        "confirmations": {
            "concept_selection_sep3_sofa2_authorized": True,
        },
    }
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_validate_analysis_design",
        lambda _study: {},
    )

    research_launch_scientific._validate_primary_concept_selection(
        study,
        "sep3_sofa2",
    )


def test_pipeline_factory_rejects_unimplemented_cluster_variance_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foundation_called = False

    def foundation(**_kwargs: Any) -> dict[str, Any]:
        nonlocal foundation_called
        foundation_called = True
        return _foundation_profile()

    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        foundation,
    )
    study = {
        **_complete_study(),
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "hospital_admission",
        },
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path="/typed/demo",
            study_context=study,
            project_root=None,
            provider={"provider": "openai", "external": True},
        )

    assert exc.value.code == "research_pipeline_cluster_unit_unsupported"
    assert exc.value.details == {
        "cluster_unit": "hospital_admission",
        "supported_cluster_units": ["patient"],
    }
    assert foundation_called is False


def test_web_study_context_compiles_to_strict_user_preferences() -> None:
    study = {
        **_complete_study(),
        "purpose": "Demo-only product validation.",
        "comparator": "Compare aggregate summaries by sex.",
        "confirmations": {
            "demo_only": True,
            "non_causal": True,
            "not_for_manuscript": True,
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert set(compiled) == {
        "extra_notes",
        "must_have_outputs",
        "data_constraints",
        "covariates",
        "covariate_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
        "covariate_operationalizations",
    }
    # A stated comparator is the estimand's reference group, so it travels as a
    # declaration the Planner may honour. Filed as `subgroup_sensitivity` it
    # reached the Planner as "Include subgroup/sensitivity requests: ...",
    # turning a reference group into extra analyses nobody requested; the
    # contrast itself belongs to PlannedModelRequirement.
    assert validated.subgroup_sensitivity is None
    assert validated.extra_notes == (
        "Demo-only product validation.\n"
        "Comparator stated by the researcher: Compare aggregate summaries by sex."
    )
    assert "not_for_manuscript" in str(validated.data_constraints)
    assert validated.timing_and_design is None
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["materialization_window"] == {
        "role": "outer_observation_window",
        "hours": 24,
        "anchor": "ICU admission",
    }
    assert validated.covariates == ["age", "sex"]
    assert validated.covariate_selection == "exact"
    assert validated.covariate_temporal_roles == {
        "age": "baseline_static",
        "sex": "baseline_static",
    }
    # An exact covariate roster with no reviewed column bindings compiles the
    # empty map, not a missing key: the Research Agent must be able to tell
    # "nothing was bound" from "this study never reached that decision".
    assert validated.covariate_operationalizations == {}


def test_reviewed_covariate_column_bindings_reach_the_research_agent() -> None:
    study = {
        **_complete_study(),
        "covariate_operationalizations": {"age": "age_years", "sex": "sex_female"},
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.covariate_operationalizations == {
        "age": "age_years",
        "sex": "sex_female",
    }


def test_web_typed_descriptive_family_overrides_free_text_risk_contrast_routing() -> None:
    study = {
        **_complete_study(),
        "question": (
            "What are the observed outcome risks and their risk difference "
            "between exposure groups?"
        ),
        "analysis_goal": (
            "Descriptive, unadjusted, noncausal absolute risks and risk difference."
        ),
        "analysis_design": {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.inferred_analysis_family == "descriptive_epidemiology"
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["analysis_design"]["analysis_family"] == (
        "descriptive_epidemiology"
    )


def test_web_descriptive_timing_choice_compiles_closed_analysis_family() -> None:
    study = {
        **_complete_study(),
        "question": "How common is Sepsis-3 and what is mortality in each group?",
        "analysis_goal": "描述暴露与结局分布，不估计时间对齐后的关联",
        "analysis_design": {
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "confirmations": {
            "extraction_completed": True,
            "plan_timing_descriptive_only": True,
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.inferred_analysis_family == "descriptive_epidemiology"
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["analysis_design"] == {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }


def test_legacy_descriptive_receipt_compiles_complete_counts_only_contract() -> None:
    study = {
        **_complete_study(),
        "analysis_design": {},
        "analysis_goal": "描述暴露与结局分布，不估计时间对齐后的关联",
        "confirmations": {
            **_complete_study()["confirmations"],
            "plan_timing_descriptive_only": True,
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)
    constraints = json.loads(str(validated.data_constraints))

    assert validated.inferred_analysis_family == "descriptive_epidemiology"
    assert constraints["analysis_design"] == {
        "analysis_family": "descriptive_epidemiology",
        "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }


def test_web_materialization_window_never_declares_clinical_time_zero() -> None:
    study = {
        **_complete_study(),
        "question": (
            "Classify Sepsis-3 at suspected-infection onset while materializing "
            "features over the first 24 hours after ICU admission."
        ),
        "primary_exposure": "sep3_sofa1",
        "execution_concepts": {
            "outcome": "death",
            "primary_exposure": "sep3_sofa1",
            "covariates": ["age", "sex"],
        },
    }

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert validated.timing_and_design is None
    constraints = json.loads(str(validated.data_constraints))
    assert constraints["materialization_window"]["anchor"] == "ICU admission"
    assert constraints["materialization_window"]["role"] == "outer_observation_window"


def _large_diagnosis_study(count: int) -> dict[str, Any]:
    study = _complete_study()
    study["cohort"] = {
        "preset": "adult_icu",
        "label": "Adult ICU stays with suspected infection",
        "comparison": (
            "Sepsis-3 positive versus Sepsis-3 negative at suspected-infection onset"
        ),
        "icd_include": (
            "Sepsis and septic shock diagnoses recorded during the index admission"
        ),
        "age_min": 18,
        "age_max": 89,
        "max_patients": 20000,
        "icd_enabled": True,
        "include_diagnoses": [
            f"A41.{index} Sepsis due to unspecified organism, variant {index}"
            for index in range(count)
        ],
        "exclude_diagnoses": [
            f"Z51.{index} Encounter for palliative care variant {index}"
            for index in range(8)
        ],
    }
    # The only place the repeated-stay signal lives for this study.
    study["confirmations"] = {"repeated_icu_stays_retained": True}
    return study


@pytest.mark.parametrize("count", [0, 12, 18, 28, 64])
def test_web_data_constraints_never_silently_drop_a_constraint_key(
    count: int,
) -> None:
    """A long cohort must not delete confirmations or the executed window.

    ``data_constraints`` is transported as one JSON string and is read
    downstream both as prompt text and by token-scanning scientific gates.
    Cutting the serialized text at a character offset used to remove whole
    trailing keys -- ``sort_keys`` sorts ``confirmations`` and
    ``materialization_window`` last -- so a study with enough ICD codes lost
    its repeated-stay signal and its executed window at the same time.
    """

    compiled = agent_pipeline_runs._research_user_preferences(
        _large_diagnosis_study(count)
    )
    validated = UserPreferences.model_validate(compiled)

    constraints = json.loads(str(validated.data_constraints))
    assert set(constraints) == {
        "analysis_design",
        "cohort",
        "confirmations",
        "materialization_window",
    }
    assert constraints["confirmations"] == {"repeated_icu_stays_retained": True}
    assert constraints["materialization_window"]["role"] == "outer_observation_window"
    assert "repeat" in str(validated.data_constraints).casefold()


def test_web_data_constraints_elide_list_items_visibly() -> None:
    """Anything actually dropped is dropped inside the structure, in the open."""

    compiled = agent_pipeline_runs._research_user_preferences(
        _large_diagnosis_study(64)
    )
    constraints = json.loads(str(compiled["data_constraints"]))
    included = constraints["cohort"]["include_diagnoses"]

    assert len(included) < 64
    assert included[-1] == f"[{64 - (len(included) - 1)} omitted]"
    # The marker must not be able to satisfy a gate's text scan on its own.
    assert "readmission" not in included[-1].casefold()


def test_web_oversized_data_constraints_fail_closed_instead_of_truncating() -> None:
    study = _complete_study()
    study["cohort"] = {
        name: "L" * 500
        for name in (
            "preset",
            "label",
            "review",
            "review_scope",
            "comparison",
            "source_type",
            "comparison_mode",
            "icd_include",
            "icd_exclude",
        )
    }

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as excinfo:
        agent_pipeline_runs._research_user_preferences(study)

    assert excinfo.value.code == "research_pipeline_data_constraints_too_large"
    assert excinfo.value.details["section_chars"]["cohort"] > 2_400


def test_web_study_context_preserves_an_explicit_empty_adjustment_set() -> None:
    study = _complete_study()
    study["covariates"] = []
    study["execution_concepts"] = {
        **study.get("execution_concepts", {}),
        "covariates": [],
    }
    study["covariate_rationales"] = {}
    study["covariate_temporal_roles"] = {}

    compiled = agent_pipeline_runs._research_user_preferences(study)

    assert compiled["covariates"] == []
    assert compiled["covariate_selection"] == "exact"
    assert UserPreferences.model_validate(compiled).covariates == []


def test_available_covariates_do_not_silently_become_an_exact_adjustment_set() -> None:
    study = _complete_study()
    study.pop("covariate_selection")

    compiled = agent_pipeline_runs._research_user_preferences(study)
    validated = UserPreferences.model_validate(compiled)

    assert compiled["covariates"] == ["age", "sex"]
    assert "covariate_selection" not in compiled
    assert validated.covariates == ["age", "sex"]
    assert validated.covariate_selection == "planner_selectable"


def test_invalid_web_adjustment_authority_fails_closed() -> None:
    study = _complete_study()
    study["covariate_selection"] = "suggested"

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as exc:
        agent_pipeline_runs._research_user_preferences(study)

    assert exc.value.code == "research_pipeline_covariate_selection_invalid"
