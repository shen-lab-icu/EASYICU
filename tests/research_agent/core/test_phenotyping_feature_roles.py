"""Only the explicit fit roster may affect the learned cluster assignments."""

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.phenotyping_features import require_phenotyping_features
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import (
    cross_sectional_phenotyping_executor_code, run_primary_phenotyping,
)
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError, ProgressivePlanSkeleton
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from .test_cross_sectional_phenotyping_executor import _context, _frame, _primary_step
from ..planning.progressive_planner_fixtures import _payload


def _skeleton(features):
    payload = _payload()
    primary = payload["steps"][5]
    primary.update(
        step_id="cluster", planned_analysis_role="primary", module_id="custom_analysis",
        depends_on=["01_cohort"], raw_inputs=["marker_a", "marker_b", "death"], product_inputs=[],
        outputs=[{"product_id": product, "semantic_role": "custom"} for product in
                 ("table:phenotype_profiles", "table:phenotype_assignments")],
        scientific_action_id="phenotyping.cluster_solution", custom_method="cross_sectional_phenotyping",
        sensitivity_spec_ids=[], phenotyping_feature_columns=features,
    )
    payload.update(analysis_type="trajectory_clustering", steps=[payload["steps"][0], primary],
                   display_labels=[], robustness_intents=[])
    return ProgressivePlanSkeleton.model_validate(payload)


def test_compiler_binds_the_fit_roster_separately_from_readable_columns():
    plan, _ = compile_progressive_plan(skeleton=_skeleton(["marker_a", "marker_b"]), context=_context(0))
    step = plan.steps[1]
    assert step.phenotyping_feature_columns == ["marker_a", "marker_b"]
    assert "death" in step.inputs
    assert "feature_columns=('marker_a', 'marker_b')" in cross_sectional_phenotyping_executor_code(step)


def test_transport_closes_feature_names_without_increasing_the_schema_budget():
    from easyicu.research_agent.agents.progressive_payload import progressive_structured_output_request

    request = progressive_structured_output_request(
        analysis_types=["trajectory_clustering"], variable_names=["marker_a", "marker_b", "death"],
        cohort_concept_ids=["marker_a"],
        scientific_action_ids=["phenotyping.cluster_solution", "phenotyping.cluster_stability"],
        allowed_literature_citation_keys=["strobe_2007"],
    )
    assert len(request.canonical_payload_json.encode()) < 12_000
    branches = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"]["anyOf"]
    custom = next(branch for branch in branches if branch["properties"]["module_id"].get("const") == "custom_analysis")
    array = next(branch for branch in custom["properties"]["phenotyping_feature_columns"]["anyOf"] if branch.get("type") == "array")
    assert array["items"]["enum"] == ["marker_a", "marker_b", "death"]


@pytest.mark.parametrize("features,reason", [
    (None, "progressive_phenotyping_feature_roster_missing"),
    (["marker_a", "death"], "progressive_phenotyping_feature_role_forbidden"),
])
def test_compiler_rejects_missing_roster_and_outcome_leakage(features, reason):
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(skeleton=_skeleton(features), context=_context(0))
    assert caught.value.reason_code == reason
    assert caught.value.path == "phenotyping_feature_columns"


def test_legacy_plan_cannot_render_by_guessing_a_roster():
    step = _primary_step().model_copy(update={"phenotyping_feature_columns": None})
    assert "phenotyping_feature_columns" not in step.model_dump(mode="json")
    with pytest.raises(ValueError, match="phenotyping_feature_roster_missing"):
        cross_sectional_phenotyping_executor_code(step)


def test_scientific_review_does_not_approve_an_unbound_fit_roster():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.research_agent.schema import AnalysisPlan

    step = _primary_step().model_copy(update={"phenotyping_feature_columns": None})
    review = build_plan_scientific_review(
        context=_context(0), plan=AnalysisPlan(research_question="Discover early phenotypes.",
                                            analysis_type="trajectory_clustering", steps=[step]),
        literature=None,
    )
    finding = next(item for item in review.findings if item.code == "PHENOTYPING_FIT_ROSTER_INVALID")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert not review.approval_allowed


def _run(tmp_path, frame, features=("marker_a", "marker_b")):
    tmp_path.mkdir()
    context = _context(len(frame))
    context.variables.extend([
        ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
        ConceptDescriptor(name="constant_marker", role=VariableRole.LAB, dtype="float64"),
    ])
    (tmp_path / "research_context.json").write_text(context.model_dump_json(), encoding="utf-8")
    source = tmp_path / "cohort.parquet"
    frame.to_parquet(source, index=False)
    return run_primary_phenotyping(
        frame=frame, declared_columns=tuple(frame.columns), feature_columns=features,
        typed_cohort_input="artifact:analysis_cohort", source_cohort=source,
        out_dir=tmp_path / "results", run_dir=tmp_path, step_id="cluster",
    )


def test_permuting_profile_variables_and_outcomes_cannot_change_clusters(tmp_path):
    frame = _frame().assign(age=np.linspace(18, 99, 360))
    first = _run(tmp_path / "first", frame)
    changed = frame.assign(age=frame.age.iloc[::-1].to_numpy() * 100, death=1 - frame.death)
    second = _run(tmp_path / "second", changed)
    assert first["feature_roster"] == second["feature_roster"] == ["marker_a", "marker_b"]
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "first/results/phenotype_assignments.csv"),
        pd.read_csv(tmp_path / "second/results/phenotype_assignments.csv"),
    )
    assert first["cluster_selection"] == second["cluster_selection"]
    assert first["source_cohort_sha256"] != second["source_cohort_sha256"]


@pytest.mark.parametrize("mutation", ["constant", "sparse", "nonnumeric", "missing"])
def test_a_declared_unusable_feature_is_not_silently_dropped(tmp_path, mutation):
    frame = _frame().assign(constant_marker=np.arange(360, dtype=float))
    if mutation == "constant":
        frame["constant_marker"] = 1.
    elif mutation == "sparse":
        frame.loc[19:, "constant_marker"] = np.nan
    elif mutation == "nonnumeric":
        frame["constant_marker"] = "not numeric"
    else:
        frame = frame.drop(columns="constant_marker")
    # A missing raw input fails before support checks. Both paths must refuse
    # instead of fitting just the two remaining valid features.
    with pytest.raises((ValueError, RuntimeError), match="phenotyping_declared_feature_unusable|phenotyping_feature_input_mismatch"):
        _run(tmp_path / "attempt", frame, ("marker_a", "marker_b", "constant_marker"))
    assert not (tmp_path / "attempt/results/phenotype_assignments.csv").exists()


def test_outcome_derived_column_cannot_hide_behind_an_other_role():
    descriptor = ConceptDescriptor(name="proxy", role=VariableRole.OTHER, dtype="float64", derived_from_concepts=["death"])
    with pytest.raises(ValueError, match="phenotyping_feature_role_forbidden"):
        require_phenotyping_features(
            ["marker_a", "proxy"], inputs=["marker_a", "proxy"],
            descriptors=[*_context(0).variables, descriptor], outcome_columns=["death"],
        )


def test_duplicate_row_identity_cannot_create_ambiguous_assignments(tmp_path):
    frame = _frame()
    frame.loc[1, "stay_id"] = frame.loc[0, "stay_id"]
    with pytest.raises(RuntimeError, match="complete unique typed row identity"):
        _run(tmp_path / "attempt", frame)
    assert not (tmp_path / "attempt/results/phenotype_assignments.csv").exists()
