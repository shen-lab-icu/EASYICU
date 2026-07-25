"""Plan-DAG contracts for monolithic and decomposed trajectory phenotyping."""

from __future__ import annotations

from copy import deepcopy

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.trajectory.plan_contract import (
    augment_trajectory_plan_products,
    evaluate_trajectory_plan_dag,
    trajectory_plan_contract_applies,
    trajectory_planner_contract_guide,
    trajectory_role_code_contract,
)


def _context(
    windows: dict[str, list[tuple[float, float]]],
) -> ResearchContext:
    variables = []
    for family, family_windows in windows.items():
        for start, end in family_windows:
            start_token = str(start).replace(".0", "").replace(".", "p")
            end_token = str(end).replace(".0", "").replace(".", "p")
            variables.append(
                ConceptDescriptor(
                    name=f"{family}_h{start_token}_{end_token}",
                    role=VariableRole.ORDINAL_SCORE,
                    dtype="float64",
                    is_ordinal=True,
                    fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                        family=family,
                        window_start_hours=start,
                        window_end_hours=end,
                        window_width_hours=end - start,
                        source_scale="ordinal",
                        representation_kind="fractional_window_summary",
                        observed_fractional_values=True,
                    ),
                )
            )
    return ResearchContext(
        research_question="Discover phenotypes from fixed-window physiology.",
        cohort=CohortDescriptor(
            cohort_name="trajectory-dag",
            database="test",
            n_patients=20,
            n_stays=20,
            id_columns=["stay_id"],
        ),
        variables=variables,
        target_outcome="death",
    )


def _split_plan(
    context: ResearchContext,
    *,
    selected_windows: list[str] | None = None,
    include_figure_step: bool = True,
) -> AnalysisPlan:
    window_inputs = selected_windows or [
        variable.name
        for variable in context.variables
        if variable.fixed_window_trajectory is not None
    ]
    steps = [
        AnalysisStep(
            step_id="01_cohort",
            intent="Create the locked analysis cohort.",
            method="cohort_definition_and_attrition",
            expected_outputs=["artifact:analysis_cohort"],
        ),
        AnalysisStep(
            step_id="02_missingness",
            intent="Audit longitudinal observation availability.",
            method="longitudinal_missingness_and_score_quality_audit",
            inputs=["artifact:analysis_cohort", *window_inputs],
            expected_outputs=["artifact:missingness_audit"],
        ),
        AnalysisStep(
            step_id="03_representation",
            intent="Build the agent-selected missingness-aware representation.",
            method="fixed_anchor_missingness_aware_feature_representation",
            inputs=[
                "artifact:analysis_cohort",
                "artifact:missingness_audit",
                *window_inputs,
            ],
            expected_outputs=["artifact:trajectory_features"],
        ),
        AnalysisStep(
            step_id="04_candidates",
            intent="Fit and compare the agent-selected candidate class models.",
            method="latent_class_trajectory_clustering",
            inputs=["artifact:analysis_cohort", "artifact:trajectory_features"],
            expected_outputs=[
                "table:cluster_selection",
                "artifact:candidate_cluster_fits",
            ],
        ),
    ]
    if include_figure_step:
        steps.append(
            AnalysisStep(
                step_id="04_candidates_figure",
                intent="Render the candidate-selection display.",
                method="latent_class_trajectory_clustering",
                inputs=["artifact:candidate_cluster_fits"],
                expected_outputs=["figure:phenotype_structure"],
            )
        )
    steps.extend(
        [
            AnalysisStep(
                step_id="05_stability",
                intent="Refit resamples and freeze the stable solution.",
                method="bootstrap_cluster_stability",
                inputs=[
                    "artifact:analysis_cohort",
                    "artifact:trajectory_features",
                    "artifact:candidate_cluster_fits",
                ],
                expected_outputs=[
                    "table:cluster_stability",
                    "artifact:stable_cluster_assignments",
                ],
            ),
            AnalysisStep(
                step_id="06_characterization",
                intent="Describe the stable clusters without causal claims.",
                method="descriptive_cluster_characterization",
                inputs=[
                    "artifact:trajectory_features",
                    "artifact:stable_cluster_assignments",
                ],
                expected_outputs=[
                    "table:cluster_characteristics",
                    "table:cluster_mortality_descriptive",
                ],
            ),
        ]
    )
    return AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=steps,
    )


def _kinds(evaluation) -> set[str]:
    return {
        str((finding.detail or {}).get("kind") or "") for finding in evaluation.findings
    }


def _free_named_split_plan(
    context: ResearchContext,
    *,
    selected_windows: list[str] | None = None,
) -> AnalysisPlan:
    windows = selected_windows or [variable.name for variable in context.variables]
    return AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="02_panel_preparation",
                intent="Align the selected repeated measures.",
                method="fixed_anchor_longitudinal_missingness_audit",
                inputs=windows,
                expected_outputs=[
                    "artifact:aligned_longitudinal_panel",
                    "artifact:trajectory_feature_manifest",
                ],
            ),
            AnalysisStep(
                step_id="03_functional_encoding",
                intent="Build the agent-selected representation.",
                method="missingness_aware_rank_preserving_functional_representation",
                inputs=[
                    "artifact:aligned_longitudinal_panel",
                    "artifact:trajectory_feature_manifest",
                ],
                expected_outputs=[
                    "artifact:trajectory_representation",
                    "table:scaling_summary",
                ],
            ),
            AnalysisStep(
                step_id="04_candidate_models",
                intent="Compare agent-selected candidate solutions.",
                method=(
                    "model_based_latent_class_clustering_with_bic_and_bootstrap_"
                    "stability"
                ),
                inputs=["artifact:trajectory_representation"],
                expected_outputs=[
                    "artifact:candidate_cluster_models",
                    "artifact:candidate_cluster_assignments",
                    "table:candidate_cluster_criteria",
                ],
            ),
            AnalysisStep(
                step_id="05_freeze",
                intent="Refit resamples and freeze the selected solution.",
                method=(
                    "model_based_latent_class_clustering_with_bic_and_bootstrap_"
                    "stability"
                ),
                inputs=[
                    "artifact:trajectory_representation",
                    "artifact:candidate_cluster_models",
                ],
                expected_outputs=[
                    "artifact:stability_freeze",
                    "artifact:cluster_assignments",
                    "table:cluster_stability",
                ],
            ),
            AnalysisStep(
                step_id="06_describe",
                intent="Describe the frozen groups.",
                method="descriptive_cluster_characterization_and_outcome_summary",
                inputs=[
                    "artifact:cluster_assignments",
                    "artifact:aligned_longitudinal_panel",
                ],
                expected_outputs=[
                    "table:cluster_characteristics",
                    "table:cluster_outcome_summary",
                ],
            ),
        ],
    )


def test_current_h3_shaped_split_dag_is_recognized_without_collapsing_steps():
    context = _context(
        {
            "physiology": [(0, 6), (6, 12), (12, 18)],
            "organ_state": [(0, 6), (6, 12), (12, 18)],
        }
    )
    plan = _split_plan(context)
    before = deepcopy(plan.model_dump(mode="json"))

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.applies is True
    assert evaluation.findings == ()
    assert evaluation.role_owners == {
        "representation": "03_representation",
        "candidate_selection": "04_candidates",
        "stability_freeze": "05_stability",
        "characterization": "06_characterization",
    }
    assert plan.model_dump(mode="json") == before
    assert "04_candidates_figure" not in evaluation.role_owners.values()


def test_free_method_and_product_names_use_method_family_plus_typed_structure():
    context = _context(
        {
            "unseen_organ_a": [(0, 6), (6, 12), (12, 18)],
            "unseen_organ_b": [(0, 6), (6, 12), (12, 18)],
        }
    )
    plan = _free_named_split_plan(context)

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.findings == ()
    assert evaluation.role_owners == {
        "representation": "03_functional_encoding",
        "candidate_selection": "04_candidate_models",
        "stability_freeze": "05_freeze",
        "characterization": "06_describe",
    }


def test_one_scientific_step_may_own_candidate_selection_and_stability():
    context = _context({"unseen_index": [(0, 4), (4, 8), (8, 12)]})
    plan = _free_named_split_plan(context)
    candidate = next(step for step in plan.steps if step.step_id == "04_candidate_models")
    candidate.intent = "Compare candidates, assess stability, and freeze the solution."
    candidate.expected_outputs.extend(
        [
            "artifact:stability_freeze",
            "artifact:stable_cluster_assignments",
            "table:cluster_stability",
            "table:cluster_assignments",
        ]
    )
    plan.steps = [step for step in plan.steps if step.step_id != "05_freeze"]
    characterization = next(step for step in plan.steps if step.step_id == "06_describe")
    characterization.inputs = [
        "artifact:stable_cluster_assignments",
        "artifact:aligned_longitudinal_panel",
    ]

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.findings == ()
    assert evaluation.role_owners["candidate_selection"] == "04_candidate_models"
    assert evaluation.role_owners["stability_freeze"] == "04_candidate_models"


def test_split_stability_owner_cannot_claim_selection_or_characterization_products():
    context = _context({"unseen_index": [(0, 4), (4, 8), (8, 12)]})
    plan = _free_named_split_plan(context)
    stability = next(step for step in plan.steps if step.step_id == "05_freeze")
    stability.expected_outputs.extend(
        [
            "table:cluster_number_selection",
            "table:cluster_sizes",
        ]
    )
    characterization = next(
        step for step in plan.steps if step.step_id == "06_describe"
    )
    characterization.expected_outputs.append("table:cluster_sizes")

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    mismatches = [
        finding.detail
        for finding in evaluation.findings
        if (finding.detail or {}).get("kind")
        == "trajectory_role_product_owner_mismatch"
    ]
    assert {
        (item["typed_product"], item["expected_owner_step_id"])
        for item in mismatches
    } >= {
        ("table:cluster_number_selection", "04_candidate_models"),
        ("table:cluster_sizes", "06_describe"),
    }
    duplicate = next(
        finding
        for finding in evaluation.findings
        if (finding.detail or {}).get("kind")
        == "trajectory_typed_product_producer_ambiguous"
    )
    assert duplicate.detail["typed_product"] == "table:cluster_sizes"
    assert duplicate.detail["producer_step_ids"] == ["05_freeze", "06_describe"]


def test_augmentation_migrates_only_redundant_split_role_outputs():
    context = _context({"unseen_index": [(0, 4), (4, 8), (8, 12)]})
    plan = _free_named_split_plan(context)
    stability = next(step for step in plan.steps if step.step_id == "05_freeze")
    stability.expected_outputs.extend(
        ["table:cluster_number_selection", "table:cluster_sizes"]
    )
    characterization = next(
        step for step in plan.steps if step.step_id == "06_describe"
    )
    characterization.expected_outputs.append("table:cluster_sizes")

    revised, findings = augment_trajectory_plan_products(
        plan=plan,
        context=context,
    )

    migrated_stability = next(
        step for step in revised.steps if step.step_id == "05_freeze"
    )
    migrated_characterization = next(
        step for step in revised.steps if step.step_id == "06_describe"
    )
    assert "table:cluster_number_selection" not in migrated_stability.expected_outputs
    assert "table:cluster_sizes" not in migrated_stability.expected_outputs
    assert migrated_characterization.expected_outputs.count("table:cluster_sizes") == 1
    assert "table:cluster_sizes" not in migrated_characterization.inputs
    assert evaluate_trajectory_plan_dag(plan=revised, context=context).findings == ()
    assert any(
        finding.detail.get("kind")
        == "trajectory_redundant_split_role_outputs_removed"
        for finding in findings
    )


def test_typed_table_cannot_be_consumed_from_the_same_step():
    context = _context({"unseen_index": [(0, 4), (4, 8), (8, 12)]})
    plan = _free_named_split_plan(context)
    characterization = next(
        step for step in plan.steps if step.step_id == "06_describe"
    )
    characterization.inputs.append("table:cluster_sizes")
    characterization.expected_outputs.append("table:cluster_sizes")

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert (
        "trajectory_typed_product_producer_not_preceding_consumer"
        in _kinds(evaluation)
    )


def test_upstream_panel_manifest_is_required_and_its_raw_windows_are_audited():
    context = _context({"unseen_index": [(0, 5), (5, 10), (10, 15)]})
    plan = _free_named_split_plan(
        context,
        selected_windows=["unseen_index_h0_5", "unseen_index_h10_15"],
    )

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    gap = next(
        finding
        for finding in evaluation.findings
        if (finding.detail or {}).get("kind") == "trajectory_internal_window_gap"
    )
    assert gap.detail["window_source_step_id"] == "02_panel_preparation"
    assert gap.detail["omitted_columns"] == ["unseen_index_h5_10"]

    representation = next(
        step for step in plan.steps if step.step_id == "03_functional_encoding"
    )
    representation.inputs.remove("artifact:trajectory_feature_manifest")
    without_manifest = evaluate_trajectory_plan_dag(plan=plan, context=context)
    assert "trajectory_window_manifest_missing" in _kinds(without_manifest)


def test_trajectory_planner_guide_exposes_canonical_roles_without_science_choices():
    context = _context({"unseen_index": [(2, 7), (7, 13)]})

    guide = trajectory_planner_contract_guide(
        context=context,
        analysis_type="trajectory_clustering",
    )

    assert "one scientific step may own both candidate selection" in guide
    assert "manifest:trajectory_window_manifest" in guide
    assert "manifest:cluster_selection" in guide
    assert "framework to choose a horizon, method, k, threshold" in guide


def test_complete_monolithic_agent_step_may_own_all_four_roles():
    context = _context({"unseen_burden": [(3, 9), (9, 15)]})
    windows = [variable.name for variable in context.variables]
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Create the locked cohort.",
                method="cohort_definition",
                expected_outputs=["artifact:analysis_cohort"],
            ),
            AnalysisStep(
                step_id="02_phenotyping",
                intent="Execute the agent-selected monolithic phenotype analysis.",
                method="gaussian_mixture_model",
                inputs=["artifact:analysis_cohort", *windows],
                expected_outputs=[
                    "artifact:trajectory_features",
                    "manifest:cluster_selection",
                    "table:cluster_assignments",
                    "table:cluster_stability",
                    "table:trajectory_profiles",
                ],
            ),
        ],
    )

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.findings == ()
    assert set(evaluation.role_owners.values()) == {"02_phenotyping"}


def test_artifact_only_trajectory_plan_cannot_escape_with_applies_false():
    context = _context({"physiology": [(0, 4), (4, 8)]})
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="artifact_export",
                intent="Export one candidate artifact.",
                method="artifact_export",
                inputs=[variable.name for variable in context.variables],
                expected_outputs=["artifact:candidate_cluster_fits"],
            )
        ],
    )

    assert trajectory_plan_contract_applies(plan=plan, context=context)
    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.applies is True
    assert _kinds(evaluation) == {"trajectory_role_missing"}
    assert all(
        finding.validator == "plan_contract" and finding.severity == "error"
        for finding in evaluation.findings
    )


def test_orphan_artifact_and_missing_adjacent_role_edge_fail_closed():
    context = _context({"physiology": [(0, 6), (6, 12)]})
    plan = _split_plan(context, include_figure_step=False)
    candidate = next(step for step in plan.steps if step.step_id == "04_candidates")
    candidate.inputs = ["artifact:unbound_trajectory_features"]

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert "trajectory_artifact_orphan" in _kinds(evaluation)
    assert "trajectory_role_edge_missing" in _kinds(evaluation)


def test_ambiguous_artifact_producer_is_not_resolved_by_plan_order():
    context = _context({"physiology": [(0, 6), (6, 12)]})
    plan = _split_plan(context, include_figure_step=False)
    plan.steps.insert(
        2,
        AnalysisStep(
            step_id="02_duplicate_export",
            intent="Export a duplicate artifact name.",
            method="artifact_export",
            expected_outputs=["artifact:trajectory_features"],
        ),
    )

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    matches = [
        finding
        for finding in evaluation.findings
        if (finding.detail or {}).get("kind")
        == "trajectory_artifact_producer_ambiguous"
    ]
    assert len(matches) == 1
    assert matches[0].detail["producer_step_ids"] == [
        "02_duplicate_export",
        "03_representation",
    ]


def test_artifact_cycle_and_nonpreceding_producer_are_both_reported():
    context = _context({"physiology": [(0, 6), (6, 12)]})
    plan = _split_plan(context, include_figure_step=False)
    representation = next(
        step for step in plan.steps if step.step_id == "03_representation"
    )
    representation.inputs.append("artifact:stable_cluster_assignments")

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert "trajectory_artifact_producer_not_preceding_consumer" in _kinds(evaluation)
    assert "trajectory_artifact_cycle" in _kinds(evaluation)


def test_mixed_figure_and_scientific_owner_is_rejected():
    context = _context({"physiology": [(0, 6), (6, 12)]})
    plan = _split_plan(context, include_figure_step=False)
    candidate = next(step for step in plan.steps if step.step_id == "04_candidates")
    candidate.expected_outputs.append("figure:cluster_selection")

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert "trajectory_role_owner_has_figure_output" in _kinds(evaluation)
    assert "trajectory_role_missing" in _kinds(evaluation)


def test_internal_window_gap_is_detected_without_filling_or_horizon_defaults():
    context = _context({"unseen_index": [(0, 5), (5, 9), (9, 14)]})
    plan = _split_plan(
        context,
        selected_windows=["unseen_index_h0_5", "unseen_index_h9_14"],
        include_figure_step=False,
    )
    before = deepcopy(plan.model_dump(mode="json"))

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    gap = next(
        finding
        for finding in evaluation.findings
        if (finding.detail or {}).get("kind") == "trajectory_internal_window_gap"
    )
    assert gap.detail["family"] == "unseen_index"
    assert gap.detail["omitted_columns"] == ["unseen_index_h5_9"]
    assert plan.model_dump(mode="json") == before


def test_short_contiguous_selected_subhorizon_is_valid():
    context = _context({"physiology": [(0, 6), (6, 12), (12, 18), (18, 24)]})
    selected = ["physiology_h6_12", "physiology_h12_18"]
    plan = _split_plan(
        context,
        selected_windows=selected,
        include_figure_step=False,
    )

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)

    assert evaluation.findings == ()


def test_plan_augmentation_distributes_only_schema_products_to_role_owners():
    context = _context({"physiology": [(0, 6), (6, 12), (12, 18)]})
    plan = _split_plan(context, include_figure_step=True)
    original = deepcopy(plan.model_dump(mode="json"))

    revised, findings = augment_trajectory_plan_products(
        plan=plan,
        context=context,
    )

    assert [step.step_id for step in revised.steps] == [
        step["step_id"] for step in original["steps"]
    ]
    assert [step.method for step in revised.steps] == [
        step["method"] for step in original["steps"]
    ]
    original_inputs = {
        step["step_id"]: step["inputs"] for step in original["steps"]
    }
    for step in revised.steps:
        expected_inputs = list(original_inputs[step.step_id])
        if step.step_id == "04_candidates":
            expected_inputs.append("manifest:trajectory_representation_schema")
        if step.step_id == "05_stability":
            expected_inputs.extend(
                [
                    "manifest:cluster_selection",
                    "manifest:trajectory_representation_schema",
                    "manifest:candidate_cluster_solution_schema",
                ]
            )
        assert step.inputs == expected_inputs
    outputs = {step.step_id: step.expected_outputs for step in revised.steps}
    assert "table:trajectory_membership" in outputs["03_representation"]
    assert (
        "manifest:trajectory_representation_schema"
        in outputs["03_representation"]
    )
    assert "manifest:cluster_selection" in outputs["04_candidates"]
    assert (
        "manifest:candidate_cluster_solution_schema" in outputs["04_candidates"]
    )
    assert "manifest:trajectory_missingness_policy" in outputs["05_stability"]
    assert "table:cluster_assignments" in outputs["05_stability"]
    assert "table:cluster_stability_assignments" in outputs["05_stability"]
    assert "table:trajectory_profiles" in outputs["06_characterization"]
    assert "table:cluster_sizes" in outputs["06_characterization"]
    assert "table:outcome_by_cluster" in outputs["06_characterization"]
    assert outputs["04_candidates_figure"] == ["figure:phenotype_structure"]
    assert findings[0].severity == "info"

    guidance = {
        step.step_id: trajectory_role_code_contract(context=context, step=step)
        for step in revised.steps
    }
    assert "REPRESENTATION ROLE" in guidance["03_representation"]
    assert "CANDIDATE-SELECTION ROLE" in guidance["04_candidates"]
    assert "STABILITY/FREEZE ROLE" in guidance["05_stability"]
    assert "CHARACTERIZATION ROLE" in guidance["06_characterization"]
    assert guidance["04_candidates_figure"] == ""
