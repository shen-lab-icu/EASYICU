from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.phenotyping_validation import (
    phenotyping_runtime_bundle_errors,
    phenotyping_runtime_receipt_valid,
)
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import (
    CLUSTER_STABILITY_PRODUCT,
    PHENOTYPE_ASSIGNMENTS_PRODUCT,
    PHENOTYPE_PROFILES_PRODUCT,
    PHENOTYPING_ANALYSIS_KIND,
    cross_sectional_phenotyping_executor_owns_step,
    run_phenotyping_diagnostic,
    run_primary_phenotyping,
)
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_figure_executor import (
    PHENOTYPING_FIGURE_ANALYSIS_KIND,
    PHENOTYPING_FIGURE_INPUTS,
    cross_sectional_phenotyping_figure_executor_owns_step,
    run_cross_sectional_phenotyping_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.scientific_action_catalog import (
    scientific_actions_for_analysis_type,
)
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    roles_covered_by_plan,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.robustness.panel import (
    build_robustness_panel_from_records,
    robustness_specs_sha,
    unexecuted_locked_spec_ids,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(n: int) -> ResearchContext:
    return ResearchContext(
        research_question="Discover early cross-sectional phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="phenotype_fixture",
            database="synthetic",
            n_stays=n,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="str"),
            ConceptDescriptor(name="marker_a", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="marker_b", role=VariableRole.VITAL, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 360
    group = np.repeat(np.arange(3), n // 3)
    return pd.DataFrame(
        {
            "stay_id": [f"stay_{index}" for index in range(n)],
            "marker_a": rng.normal(group * 2.5, 0.5),
            "marker_b": rng.normal((2 - group) * 2.0, 0.6),
            "death": rng.binomial(1, 0.2 + group * 0.1),
        }
    )


def _binding(key: str, frame: pd.DataFrame, path: Path, step_id: str) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "produced_by_step": f"producer_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "sha256": digest,
        },
        "step_id": step_id,
    }


def _primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_phenotypes",
        planned_analysis_role="primary",
        intent="Fit the Planner-selected early phenotype roster.",
        inputs=["stay_id", "marker_a", "marker_b", "death", "artifact:analysis_cohort"],
        expected_outputs=[PHENOTYPE_PROFILES_PRODUCT, PHENOTYPE_ASSIGNMENTS_PRODUCT],
        method="cross-sectional phenotyping",
        scientific_action_id="phenotyping.cluster_solution",
    )


def _write_robustness_lock(tmp_path: Path, variables: list[str]) -> RobustnessSpec:
    spec = RobustnessSpec(
        spec_id="complete_case_primary_features",
        axis="missing",
        description="Refit on the exact locked complete-case population.",
        missing_override={"strategy": "complete_case", "variables": variables},
    )
    payload = {
        "schema_version": "easyicu.robustness_specs/1",
        "locked_at": "2026-08-24T00:00:00+00:00",
        "spec_sha256": robustness_specs_sha([spec]),
        "specs": [spec.to_dict()],
    }
    (tmp_path / "robustness_specs_locked.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return spec


def test_phenotyping_actions_publish_one_exact_host_profile() -> None:
    actions = {
        action.action_id: action
        for action in scientific_actions_for_analysis_type("trajectory_clustering").actions
    }
    assert actions["phenotyping.cluster_solution"].execution_mode == "host_owned"
    assert actions["phenotyping.cluster_solution"].runtime_contract.outputs == (
        (PHENOTYPE_PROFILES_PRODUCT, "custom"),
        (PHENOTYPE_ASSIGNMENTS_PRODUCT, "custom"),
    )
    assert actions["phenotyping.cluster_stability"].runtime_contract.required_product_inputs == (
        PHENOTYPE_ASSIGNMENTS_PRODUCT,
    )


def test_phenotyping_owner_selects_only_the_exact_action_contract() -> None:
    step = _primary_step()
    assert cross_sectional_phenotyping_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Discover phenotypes.", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == PHENOTYPING_ANALYSIS_KIND
    widened = step.model_copy(update={"expected_outputs": [*step.expected_outputs, "table:extra"]})
    assert not cross_sectional_phenotyping_executor_owns_step(widened)


def test_exact_runtime_action_products_cover_their_published_article_roles() -> None:
    step = _primary_step()
    plan = AnalysisPlan(
        research_question="Discover phenotypes.",
        analysis_type="trajectory_clustering",
        steps=[step],
    )
    contract = build_article_analysis_contract(
        _context(360), analysis_type="trajectory_clustering"
    )
    covered = roles_covered_by_plan(plan, contract)
    assert {"phenotype_structure", "phenotype_profile"} <= covered


def test_phenotyping_workflow_is_outcome_excluding_typed_and_renderable(tmp_path: Path) -> None:
    frame = _frame()
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    primary_dir = tmp_path / "primary"
    summary = run_primary_phenotyping(
        frame=frame,
        declared_columns=("stay_id", "marker_a", "marker_b", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=primary_dir,
        run_dir=tmp_path,
        step_id="primary_phenotypes",
    )
    assert summary["authority_scope"] == "analysis_only"
    assert summary["feature_roster"] == ["marker_a", "marker_b"]
    assignments = pd.read_csv(primary_dir / "phenotype_assignments.csv")
    assert "feature__death" not in assignments
    assignment_binding = _binding(
        PHENOTYPE_ASSIGNMENTS_PRODUCT,
        assignments,
        primary_dir / "phenotype_assignments.csv",
        "stability",
    )
    selection_dir = tmp_path / "selection"
    selection_summary = run_phenotyping_diagnostic(
        action_id="phenotyping.k_selection",
        out_dir=selection_dir,
        run_dir=tmp_path,
        resolved_inputs={
            "step_id": "selection",
            "inputs": {PHENOTYPE_ASSIGNMENTS_PRODUCT: assignment_binding},
        },
        step_id="selection",
    )
    stability_dir = tmp_path / "stability"
    stability_summary = run_phenotyping_diagnostic(
        action_id="phenotyping.cluster_stability",
        out_dir=stability_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": "stability", "inputs": {PHENOTYPE_ASSIGNMENTS_PRODUCT: assignment_binding}},
        step_id="stability",
    )
    assert stability_summary["output_files"].keys() == {CLUSTER_STABILITY_PRODUCT}
    records = [
        {"step_summary": summary},
        {"step_summary": selection_summary},
        {"step_summary": stability_summary},
    ]
    assert phenotyping_runtime_receipt_valid(summary)
    assert phenotyping_runtime_bundle_errors(records) == []

    tampered = copy.deepcopy(records)
    tampered[2]["step_summary"]["cluster_stability"]["replicates"][0][
        "adjusted_rand_index"
    ] = 2.0
    assert "invalid ARI values" in " ".join(
        phenotyping_runtime_bundle_errors(tampered)
    )
    tampered_grid = copy.deepcopy(records)
    tampered_grid[1]["step_summary"]["cluster_selection"]["candidates"][0][
        "silhouette"
    ] += 0.01
    assert "candidate grid was not replayed" in " ".join(
        phenotyping_runtime_bundle_errors(tampered_grid)
    )

    product_paths = {
        PHENOTYPE_PROFILES_PRODUCT: primary_dir / "phenotype_profiles.csv",
        PHENOTYPE_ASSIGNMENTS_PRODUCT: primary_dir / "phenotype_assignments.csv",
        CLUSTER_STABILITY_PRODUCT: stability_dir / "cluster_stability.csv",
    }
    bindings = {}
    for key, path in product_paths.items():
        bindings[key] = _binding(key, pd.read_csv(path), path, "phenotype_figure")
    figure_step = AnalysisStep(
        step_id="phenotype_figure",
        planned_analysis_role="auxiliary",
        intent="Render phenotype profiles, sizes and stability.",
        inputs=list(PHENOTYPING_FIGURE_INPUTS),
        expected_outputs=["figure:phenotype_figure"],
        method="visualization",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in PHENOTYPING_FIGURE_INPUTS
        ],
    )
    assert cross_sectional_phenotyping_figure_executor_owns_step(
        figure_step, resolved_bindings=bindings
    )
    selected = select_standard_executor(
        figure_step,
        plan=AnalysisPlan(research_question="Discover phenotypes.", steps=[figure_step]),
        resolved_bindings=bindings,
    )
    assert selected is not None
    assert selected.analysis_kind == PHENOTYPING_FIGURE_ANALYSIS_KIND
    figure_dir = tmp_path / "figure"
    rendered = run_cross_sectional_phenotyping_figure(
        out_dir=figure_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": "phenotype_figure", "inputs": bindings},
        step_id="phenotype_figure",
        figure_product="phenotype_figure",
    )
    assert rendered["rendering_only"] is True
    assert (figure_dir / "phenotype_figure.figure_contract.json").is_file()
    assert (figure_dir / "phenotype_figure.svg").is_file()
    for key, parent_path in product_paths.items():
        product = key.partition(":")[2]
        source = pd.read_csv(figure_dir / f"{product}_source_data.csv")
        assert source["source_table"].eq(parent_path.name).all()
        assert source["source_step_id"].eq(f"producer_{product}").all()
        assert source["source_row_index"].tolist() == list(range(len(source)))


def test_phenotyping_owner_executes_locked_complete_case_sensitivity(
    tmp_path: Path,
) -> None:
    frame = _frame()
    frame.loc[::5, "marker_a"] = np.nan
    frame.loc[::7, "marker_b"] = np.nan
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    spec = _write_robustness_lock(tmp_path, ["marker_a", "marker_b"])

    summary = run_primary_phenotyping(
        frame=frame,
        declared_columns=("stay_id", "marker_a", "marker_b", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=tmp_path / "primary",
        run_dir=tmp_path,
        step_id="primary_phenotypes",
    )

    expected_n = int(frame[["marker_a", "marker_b"]].notna().all(axis=1).sum())
    row = summary["robustness_rows"][0]
    receipt = summary["scientific_runtime_receipt"][
        "complete_case_sensitivities"
    ][0]
    assert row["spec_id"] == spec.spec_id
    assert row["n"] == expected_n
    assert row["converged"] is True
    assert -1 <= row["point_estimate"] <= 1
    assert receipt["complete_case_variables"] == ["marker_a", "marker_b"]
    assert receipt["n_complete"] == expected_n
    assert receipt["n_bootstrap"] == 200
    assert (tmp_path / "primary/phenotyping_complete_case_sensitivity.csv").is_file()
    panel = build_robustness_panel_from_records(
        specs=[spec],
        per_step_records=[
            {
                "step_id": "primary_phenotypes",
                "status": "ok",
                "step_summary": summary,
                "step_summary_evidence_id": "primary_phenotypes",
            }
        ],
    )
    assert unexecuted_locked_spec_ids(panel) == []

    selection_summary = {
        "deterministic_standard_analysis": PHENOTYPING_ANALYSIS_KIND,
        "method": "deterministic_cross_sectional_phenotyping_diagnostic",
        "cluster_selection": summary["cluster_selection"],
    }
    assignments = pd.read_csv(tmp_path / "primary/phenotype_assignments.csv")
    assignment_path = tmp_path / "primary/phenotype_assignments.csv"
    stability_summary = run_phenotyping_diagnostic(
        action_id="phenotyping.cluster_stability",
        out_dir=tmp_path / "stability",
        run_dir=tmp_path,
        resolved_inputs={
            "step_id": "stability",
            "inputs": {
                PHENOTYPE_ASSIGNMENTS_PRODUCT: _binding(
                    PHENOTYPE_ASSIGNMENTS_PRODUCT,
                    assignments,
                    assignment_path,
                    "stability",
                )
            },
        },
        step_id="stability",
    )
    records = [
        {"step_summary": summary},
        {"step_summary": selection_summary},
        {"step_summary": stability_summary},
    ]
    assert phenotyping_runtime_bundle_errors(records) == []
    tampered = copy.deepcopy(records)
    tampered[0]["step_summary"]["robustness_rows"][0]["point_estimate"] += 0.01
    assert "disagrees with its robustness row" in " ".join(
        phenotyping_runtime_bundle_errors(tampered)
    )


def test_phenotyping_complete_case_spec_cannot_widen_the_feature_roster(
    tmp_path: Path,
) -> None:
    frame = _frame()
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    _write_robustness_lock(tmp_path, ["marker_a", "death"])

    with pytest.raises(RuntimeError, match="outside the primary feature roster: death"):
        run_primary_phenotyping(
            frame=frame,
            declared_columns=("stay_id", "marker_a", "marker_b", "death"),
            typed_cohort_input="artifact:analysis_cohort",
            source_cohort=cohort_path,
            out_dir=tmp_path / "primary",
            run_dir=tmp_path,
            step_id="primary_phenotypes",
        )
