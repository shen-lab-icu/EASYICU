"""A bound grouping remains the exposure of a signed landmark study.

Only synthetic catalogs, exports and plans: the acquisition universe has the
source, while the run restages its copy with the grouping's level codes.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
    landmark_categorical_cohort_executor_code,
    run_landmark_categorical_primary,
)
from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    planning_run_groupings,
    run_exposure_grouping_phase,
)
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.research_launch_scientific import _runtime_projection_sensitivity_specs
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
    compile_web_scientific_runtime_projection,
)
from tests.support.native_outcome_export import typed_native_export


LANDMARK = PrespecifiedSensitivitySpec(
    spec_id="landmark_24h", axis="timing", strategy="landmark", landmark_hours=24,
    require_alive_at_landmark=True, exclude_negative_event_times=True,
    event_time_variable="death_time",
    observation_duration_variable="los_hosp",
    observation_duration_unit="days",
)


def _context(cohort, *, primary_exposure=None):
    return build_research_context(
        research_question="Compare synthetic measurement bands with hospital death.",
        cohort=cohort, cohort_name="synthetic", database="miiv",
        primary_exposure=primary_exposure, target_outcome="death",
        id_columns=("stay_id",), outcome_columns=("death",),
        user_preferences={"data_constraints": json.dumps({"materialization_window": {
            "role": "outer_observation_window", "anchor": "icu_admission", "hours": 24,
        }})},
    )


def _phase(cohort, *, answer=None, candidate=None):
    return run_exposure_grouping_phase(
        context=_context(cohort), cohort_path=cohort, run_dir=cohort.parent,
        planner=ScriptedMockLLMClient([json.dumps(answer)] if answer else []),
        rebuild_context=_context, capability_review_pending=False,
        trajectory_staged=False, emit_progress=lambda *_args, **_kwargs: None,
        candidate=candidate,
    )


def _groupings(tmp_path, *, concept="lact", scale="ordinal", count=1, keep_unmeasured=False):
    summary = "value" if concept == "bmi" else "max"
    answer = {"groupings": [{
        "id": "x1", "concept": concept, "scale": scale,
        "window": None if concept == "bmi" else {"start_hours": 0, "end_hours": 24},
        # Matching order differs from ordinal code order; neither declared
        # comparison is the first/last level default.
        "groups": [
            {"id": "g3", "label": "Upper band", "rule": {
                "summary": summary, "op": ">=", "value": 5,
            }},
            {"id": "g1", "label": "Lower band", "rule": {
                "summary": summary, "op": "<", "value": 2.5,
            }},
            {"id": "g2", "label": "Middle band", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "exclude"},
        "reference": "g2", "contrast": "g1",
        "quote": "Compare synthetic measurement bands", "source": "question",
    }]}
    if keep_unmeasured:
        answer["groupings"][0]["unmeasured"] = {"handling": "own_group", "label": "Unmeasured"}
    if count == 2:
        second = json.loads(json.dumps(answer["groupings"][0]))
        second["id"] = "x2"
        second["groups"][0]["rule"]["value"] = 4.5
        answer["groupings"].append(second)
    catalog = tmp_path / "planning/cohort.parquet"
    catalog.parent.mkdir(parents=True)
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"), concept: pd.Series(dtype="float64"),
        "age": pd.Series(dtype="float64"), "death": pd.Series(dtype="float64"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    frame.to_parquet(catalog, index=False)
    _phase(catalog, answer=answer)
    return planning_run_groupings(
        catalog.parent, cohort_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
    )


def _universe(tmp_path, source="lact_max"):
    path = tmp_path / "universe.parquet"
    pd.DataFrame({
        source: [1.0, 3.0, 6.0], "age": [40.0, 50.0, 60.0], "death": [0, 1, 0],
        "death_time": [np.nan, 48.0, np.nan], "los_hosp": [3.0, 2.0, 4.0],
    }).to_parquet(path, index=False)
    return path


def _project(universe, groupings, *, concept="lact", source="lact_max", specs=(LANDMARK,),
             selection="exact"):
    return compile_web_scientific_runtime_projection(
        study={"covariate_selection": selection}, sensitivity_specs=specs,
        primary_exposure=source, primary_exposure_source=concept,
        target_outcome="death", target_is_event_status=True,
        declared_covariates=("age",), covariate_operationalizations={},
        universe_path=universe, scientific_configuration_sha256="a" * 64,
        exposure_groupings=groupings,
    )


def _plan(variable, *, reference="2", contrast="1"):
    return AnalysisPlan.model_validate({
        "research_question": "Compare synthetic measurement bands with hospital death.",
        "analysis_type": "association_study",
        "steps": [
            {"step_id": "cohort", "planned_analysis_role": "auxiliary",
             "intent": "Define the landmark cohort.", "inputs": ["death"],
             "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
             "method": "cohort_definition_and_attrition"},
            {"step_id": "primary", "planned_analysis_role": "primary",
             "intent": "Estimate adjusted band contrasts.",
             "inputs": ["artifact:analysis_cohort", variable, "death", "age"],
             "expected_outputs": ["table:adjusted_association_estimates"],
             "method": "adjusted_association_models", "sensitivity_spec_ids": [LANDMARK.spec_id],
             "model_requirements": [{
                 "requirement_id": "primary_model", "outcome": "death", "outcome_type": "binary",
                 "method_family": "statsmodels_logit_mle", "exposure_source": variable,
                 "analysis_role": "primary", "analysis_set": "source_aware", "covariates": ["age"],
                 "model_terms": [
                     {"name": variable, "role": "exposure", "coding": "categorical",
                      "levels": ["1", "2", "3"], "reference_level": reference,
                      "transform": "treatment_contrast"},
                     {"name": "age", "role": "covariate", "coding": "continuous",
                      "transform": "identity"},
                 ],
                 "exposure_levels": ["1", "2", "3"], "exposure_reference_level": reference,
                 "primary_contrast_level": contrast,
             }]},
            {"step_id": "alternate", "planned_analysis_role": "sensitivity",
             "intent": "Compare an alternate exposure estimate with the primary.",
             "inputs": ["artifact:analysis_cohort", "table:adjusted_association_estimates"],
             "expected_outputs": ["table:alternate_estimates"],
             "method": "binary_exposure_sensitivity"},
        ],
    })


@pytest.mark.parametrize("concept,source", [("bmi", "bmi"), ("lact", "lact_max")])
@pytest.mark.parametrize("scale,kind,reference,contrast", [
    ("ordinal", "ordinal", "2", "1"), ("nominal", "categorical", "3", "2"),
])
def test_a_bound_grouping_seals_its_column_codes_and_declared_comparison(
    tmp_path, concept, source, scale, kind, reference, contrast,
):
    groupings = _groupings(tmp_path, concept=concept, scale=scale)
    universe = _universe(tmp_path, source)
    specs = _runtime_projection_sensitivity_specs(
        (LANDMARK,), primary_exposure=source, primary_exposure_source=concept,
        universe_path=universe, exposure_groupings=groupings,
    )
    projection = _project(universe, groupings, concept=concept, source=source, specs=specs)
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority)
    assert authority.exposure_column == f"{concept}_group_x1"
    assert authority.exposure_kind == kind
    assert authority.exposure_levels == ("1", "2", "3")
    assert authority.exposure_reference_level == reference
    assert authority.primary_contrast_level == contrast
    assert authority.exposure_column not in pq.read_schema(universe).names
    plan = _plan(authority.exposure_column, reference=reference, contrast=contrast)
    bound = authority.bind_plan(plan)
    authority.validate_plan(bound)
    assert bound.steps[-1] == plan.steps[-1]


def test_the_launch_does_not_add_a_spline_to_a_bound_grouping(tmp_path):
    groupings = _groupings(tmp_path)
    universe = _universe(tmp_path)
    for path in (universe, None):
        specs = _runtime_projection_sensitivity_specs(
            (LANDMARK,), primary_exposure="lact_max", primary_exposure_source="lact",
            universe_path=path, exposure_groupings=groupings,
        )
        assert specs == (LANDMARK,)


def test_a_nominal_grouping_keeps_the_unmeasured_code_out_of_its_comparison(tmp_path):
    projection = _project(
        _universe(tmp_path), _groupings(tmp_path, scale="nominal", keep_unmeasured=True),
    )
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.exposure_levels == ("1", "2", "3", "4")
    assert authority.exposure_reference_level == "3"
    assert authority.primary_contrast_level == "2"


def test_a_source_with_several_groupings_requires_the_exact_grouped_column(tmp_path):
    groupings = _groupings(tmp_path, count=2)
    universe = _universe(tmp_path)
    with pytest.raises(WebScientificRuntimeProjectionError) as raised:
        _project(universe, groupings)
    assert raised.value.code == "web_scientific_runtime_projection_ambiguous"
    projection = _project(universe, groupings, source="lact_group_x2")
    assert projection.authority["exposure_column"] == "lact_group_x2"


def test_an_unrelated_grouping_leaves_a_continuous_landmark_exposure_unchanged(tmp_path):
    groupings = _groupings(tmp_path)
    universe = _universe(tmp_path, "bmi")
    specs = _runtime_projection_sensitivity_specs(
        (LANDMARK,), primary_exposure="bmi", primary_exposure_source="bmi",
        universe_path=universe, exposure_groupings=groupings,
    )
    assert len(specs) == 2
    projection = _project(universe, groupings, source="bmi", concept="bmi", specs=specs)
    assert projection.authority["authority_kind"] == "landmark_spline_association"
    assert projection.authority["exposure_column"] == "bmi"


@pytest.mark.parametrize("target", ["lact", "lact_max", "lact_group_x1"])
@pytest.mark.parametrize("source", ["lact_max", "lact_group_x1"])
def test_an_explicit_spline_on_a_grouped_exposure_is_refused(tmp_path, target, source):
    groupings = _groupings(tmp_path)
    spline = PrespecifiedSensitivitySpec(
        spec_id="exposure_spline", axis="functional_form", strategy="restricted_cubic_spline",
        execution_variables=(target,),
    )
    with pytest.raises(WebScientificRuntimeProjectionError) as raised:
        _project(_universe(tmp_path), groupings, source=source, specs=(LANDMARK, spline))
    assert raised.value.code == "web_landmark_exposure_model_incompatible"


def test_a_grouped_landmark_still_requires_its_acquisition_source(tmp_path):
    groupings = _groupings(tmp_path)
    with pytest.raises(WebScientificRuntimeProjectionError) as raised:
        _project(_universe(tmp_path, "unrelated_measurement"), groupings)
    assert raised.value.code == "web_scientific_runtime_columns_missing"
    assert raised.value.details["missing_columns"] == ["lact_max"]


def test_the_grouped_source_cannot_enter_the_plan_bound_adjustment_roster(tmp_path):
    projection = _project(_universe(tmp_path), _groupings(tmp_path), selection="planner_selectable")
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    assert authority.plan_bound_adjustment_roster.admissible_columns == ("age",)
    sealed = authority.seal_adjustment_roster(_plan("lact_group_x1"))
    sealed.validate_plan(sealed.bind_plan(_plan("lact_group_x1")))


def test_the_signed_executor_reads_the_runs_restaged_grouped_cohort(tmp_path, monkeypatch):
    groupings = _groupings(tmp_path)
    n = 180
    rng = np.random.default_rng(901)
    stays = np.arange(1, n + 1)
    values = np.tile([1.0, 3.0, 6.0], n // 3)
    age = rng.uniform(35, 80, n)
    death = rng.binomial(1, 0.2 + 0.04 * values)
    death_hours = np.where(death, np.where(stays % 11 == 0, 12.0, 48.0), np.nan)
    source = typed_native_export(
        tmp_path / "export",
        statics=pd.DataFrame({"stay_id": stays, "age": age}),
        outcome=pd.DataFrame({
            "stay_id": stays, "charttime": np.zeros(n), "death": death,
            "death_time": death_hours, "los_hosp": np.full(n, 4.0),
        }),
        outcome_concepts=["death", "los_hosp"],
        longitudinal=pd.DataFrame({"stay_id": stays, "charttime": np.ones(n), "lact": values}),
        longitudinal_concepts=["lact"],
    )
    paths = materializer.materialize_to_parquet(
        tmp_path / "materialized", data_path=source, database="miiv",
        static_concepts=("age", "los_hosp"), feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    universe = paths["parquet"]
    projection = _project(universe, groupings)
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    bound = authority.bind_plan(_plan("lact_group_x1"))
    cohort = tmp_path / "run/cohort.parquet"
    stage_materialized_cohort_authority(universe, cohort, producer_implementation_sha256="b" * 64)
    _phase(cohort, candidate=groupings.candidate.model_dump(mode="json"))
    assert load_verified_materialized_cohort_authority(cohort) is not None
    assert "lact_group_x1" not in pq.read_schema(universe).names
    grouped = pd.read_parquet(cohort)
    assert grouped["lact_group_x1"].tolist() == np.tile([1, 2, 3], n // 3).tolist()
    # Run the actual generated cohort adapter with the run's input binding.
    cohort_out = tmp_path / "landmark"
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(cohort_out))
    code = landmark_categorical_cohort_executor_code(
        authority.governed_cohort_step(bound), plan=bound, authority=authority,
        runtime_projection_sha256=projection.projection_sha256,
    )
    exec(compile(code, "<synthetic-landmark-cohort>", "exec"), {})
    eligible_path = cohort_out / "analysis_cohort.parquet"
    eligible = pd.read_parquet(eligible_path)
    assert 0 < len(eligible) < n
    assert set(eligible["lact_group_x1"]) == {1, 2, 3}
    result = run_landmark_categorical_primary(
        frame=eligible, cohort_path=eligible_path, step=authority.governed_primary_step(bound),
        authority=authority, runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "primary",
    )
    assert result["n_total"] == len(eligible)
    assert result["exposure"] == "lact_group_x1"
    assert result["landmark_runtime_receipt"]["reference_level"] == "2"
    assert result["landmark_runtime_receipt"]["primary_contrast_level"] == "1"
