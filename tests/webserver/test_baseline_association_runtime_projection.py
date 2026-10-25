"""The baseline Web projection reads synthetic schema and signed coordinates only."""

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import load_current_case_scientific_runtime_authority
from easyicu.webserver.scientific_runtime_projection import compile_web_scientific_runtime_projection


def verified_column(monkeypatch, name, *, levels, window=None):
    """Supply synthetic facts at the verified-metadata loader's boundary."""
    from types import SimpleNamespace
    from easyicu.concept.metadata_projection import ConceptColumnMetadata, ConceptColumnRole
    from easyicu.concept.metadata_sidecar import ColumnMetadataBinding, DerivationWindow
    from easyicu.webserver import scientific_runtime_projection as owner

    metadata = ConceptColumnMetadata(
        column_name=name, source_concept=name, role=ConceptColumnRole.EVENT_STATUS if levels == [0, 1] else ConceptColumnRole.VALUE, aggregation=None,
        canonical_unit=None, accepted_units=(), extraction_bounds=None, analysis_plausibility_range=None,
        allowed_values=tuple(levels) if levels == [0, 1] else None, time_origin=None, time_unit=None,
        source_database="synthetic", dictionary_source_database="synthetic", source_resolution_chain=("synthetic",),
        available_databases=("synthetic",), source_declared_for_database=True,
        availability_basis="synthetic_fixture", source_lineage=(), description="Synthetic admission attribute",
        category=None, class_name=None, derived_from_concepts=(),
    )
    binding = ColumnMetadataBinding(metadata, derivation_window=DerivationWindow("icu_admission", *window) if window else None)
    verified = SimpleNamespace(sidecar=SimpleNamespace(files=[SimpleNamespace(columns={name: binding})]),
                               authority=SimpleNamespace(output_derivations=[]))
    monkeypatch.setattr(owner, "load_verified_materialized_cohort_authority", lambda path: verified)


@pytest.mark.parametrize("exposure", ["age", "sex"])
def test_baseline_projection_is_signed_without_a_landmark(tmp_path, exposure):
    path = tmp_path / "metadata.parquet"
    pd.DataFrame({"age": pd.Series(dtype="float64"), "sex": pd.Series(dtype="bool"),
                  "death": pd.Series(dtype="bool")}).to_parquet(path)
    projection = compile_web_scientific_runtime_projection(
        study={"analysis_design": {"analysis_family": "association_study"}, "covariate_selection": "exact"},
        sensitivity_specs=(), primary_exposure=exposure, primary_exposure_source=exposure,
        target_outcome="death", declared_covariates=(), covariate_operationalizations={},
        target_is_event_status=True, universe_path=path, scientific_configuration_sha256="a" * 64,
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.authority_kind == "baseline_adjusted_association"
    with pytest.raises(ValueError, match="digest"):
        load_current_case_scientific_runtime_authority({**projection.authority, "outcome_column": "tampered"})


def test_ungrouped_continuous_exposure_keeps_its_spline_authority(tmp_path):
    from tests.support.baseline_association import baseline_context, baseline_plan_attempt

    path = tmp_path / "continuous.parquet"
    pd.DataFrame({"age": pd.Series(dtype="float64"), "sex": pd.Series(dtype="bool"),
                  "death": pd.Series(dtype="bool")}).to_parquet(path, index=False)
    projection = compile_web_scientific_runtime_projection(
        study={"analysis_design": {"analysis_family": "association_study"}, "covariate_selection": "exact"},
        sensitivity_specs=(), primary_exposure="age", primary_exposure_source="age", target_outcome="death",
        declared_covariates=("sex",), covariate_operationalizations={}, target_is_event_status=True,
        universe_path=path, scientific_configuration_sha256="b" * 64,
    )
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.exposure_column == "age"
    assert authority.exposure_encoding == "restricted_cubic_spline"
    assert authority.exposure_levels == ()
    assert authority.exposure_reference_level is authority.primary_contrast_level is None
    plan = baseline_plan_attempt(baseline_context("age", covariates=("sex",)))[0].output
    sealed = authority.seal_adjustment_roster(plan)
    sealed.validate_plan(sealed.bind_plan(plan))


@pytest.mark.parametrize("exposure,covariates,subgroup", [
    ("sex", (), None), ("adm", ("age",), "sex"),
    ("chronic_flag", ("age", "sex", "weight"), "sex"), ("age", ("sex",), None),
])
def test_web_projection_keeps_adjustments_subgroups_and_baseline_windows(tmp_path, monkeypatch, exposure, covariates, subgroup):
    levels = [0, 1] if exposure == "chronic_flag" else [1, 2, 3] if exposure == "adm" else []
    verified_column(monkeypatch, exposure, levels=levels, window=(-1, 0) if exposure == "chronic_flag" else None)
    path = tmp_path / "baseline.parquet"
    pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in {
        "age": "float64", "sex": "bool", "adm": "int64", "chronic_flag": "bool", "weight": "float64", "death_icu": "bool",
    }.items()}).to_parquet(path)
    projection = compile_web_scientific_runtime_projection(
        study={"analysis_design": {"analysis_family": "association_study"}, "covariate_selection": "exact",
               "stated_subgroups": [{"variable": subgroup, "concept": subgroup, "words": "within each sex", "source_status": "available"}] if subgroup else []},
        sensitivity_specs=(), primary_exposure=exposure, primary_exposure_source=exposure,
        target_outcome="death_icu", declared_covariates=covariates, covariate_operationalizations={},
        target_is_event_status=True, universe_path=path, scientific_configuration_sha256="b" * 64,
    )
    assert projection is not None
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.required_adjustment_columns == covariates
    assert authority.exposure_encoding == ("restricted_cubic_spline" if exposure == "age" else "treatment_contrast")
    assert [term.name for term in authority.subgroup_terms] == ([subgroup] if subgroup else [])


@pytest.mark.parametrize("exposure", ["age", "chronic_flag"])
def test_a_physical_post_baseline_window_cannot_be_signed_as_baseline(tmp_path, monkeypatch, exposure):
    verified_column(monkeypatch, exposure, levels=None, window=(0, 24))
    path = tmp_path / "window.parquet"
    pd.DataFrame({exposure: pd.Series(dtype="float64"), "death": pd.Series(dtype="bool")}).to_parquet(path)
    assert compile_web_scientific_runtime_projection(
        study={"analysis_design": {"analysis_family": "association_study"}}, sensitivity_specs=(),
        primary_exposure=exposure, primary_exposure_source=exposure, target_outcome="death",
        declared_covariates=(), covariate_operationalizations={}, target_is_event_status=True,
        universe_path=path, scientific_configuration_sha256="c" * 64,
    ) is None


@pytest.mark.parametrize("reason", [
    "web_baseline_association_incomplete",
    "family_spec_stated_adjustment_unavailable",
    "family_spec_subgroup_unavailable",
])
def test_baseline_refusals_require_an_explicit_plan_edit(reason):
    from easyicu.webserver.stop_next_actions import STOP_POLICIES, next_actions_for_stop

    assert reason in STOP_POLICIES
    actions = next_actions_for_stop(reason)
    assert [action.kind for action in actions] == ["edit_plan"]
    assert actions[0].params.requires_plan_approval
