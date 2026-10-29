"""Synthetic numeric oracles and governed dispatch for standalone diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.cluster_sizes import (
    CLUSTER_SIZE_OUTPUTS,
    ClusterSizeSpec,
)
from easyicu.research_agent.audits.aggregate_row import (
    unlabelled_aggregate_row_findings,
)
from easyicu.research_agent.contracts.covariate_balance import (
    BALANCE_OUTPUTS,
    CovariateBalanceSpec,
)
from easyicu.research_agent.execution.runners.cluster_sizes_executor import (
    cluster_sizes_executor_scaffold,
    cluster_sizes_executor_verdict,
    cluster_sizes_output_findings,
    run_cluster_sizes_from_bound,
)
from easyicu.research_agent.execution.runners.covariate_balance_executor import (
    covariate_balance_executor_scaffold,
    covariate_balance_executor_verdict,
    covariate_balance_output_findings,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.methods.cluster_sizes import compute_cluster_sizes
from easyicu.research_agent.methods.covariate_balance import compute_covariate_balance
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _balance_spec(**changes):
    return CovariateBalanceSpec.model_validate(
        {
            "unit_id_column": "unit",
            "analysis_unit": "encounter",
            "treatment_column": "treated",
            "weight_column": "weight",
            "covariates": [{"column": "x", "kind": "continuous"}],
            **changes,
        }
    )


def _balance_frame():
    return pd.DataFrame(
        {
            "unit": list("abcdef"),
            "treated": [1, 1, 1, 0, 0, 0],
            "weight": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 0.0, 1.0, 2.0],
        }
    )


def _size_spec(**changes):
    return ClusterSizeSpec.model_validate(
        {
            "unit_id_column": "unit",
            "analysis_unit": "encounter",
            "assignment_source": "table:phenotype_assignments",
            "assignment_unit_id_column": "unit_id",
            "label_column": "cluster",
            "expected_clusters": ["A", "B", "C"],
            "missing_label_policy": "exclude",
            **changes,
        }
    )


def _sizes_frames():
    return pd.DataFrame({"unit": list("abcde")}), pd.DataFrame(
        {"unit_id": ["d", "a", "c", "b"], "cluster": ["B", "A", "A", None]}
    )


def test_weighted_balance_matches_hand_computation_with_fixed_denominator():
    result = compute_covariate_balance(_balance_frame(), _balance_spec())
    row = result.rows[0]
    assert result.input_n == 6 and result.treated_n == result.control_n == 3
    assert row.mean_treated_before == 2 and row.mean_control_before == 1
    assert row.variance_treated_before == row.variance_control_before == 1
    assert row.smd_denominator == 1 and row.smd_before == 1
    assert row.mean_treated_after == 2 and row.mean_control_after == 0.75
    assert row.smd_after == 1.25
    assert row.ess_treated == row.ess_control == pytest.approx(16 / 6)
    assert row.exceeds_threshold_before is None
    assert result.model_dump(mode="json")["claim_ceiling"] == "analysis_only"


def test_weight_scale_does_not_change_smd_or_ess_and_zero_weights_stay_in_before():
    frame = _balance_frame()
    frame.loc[0, "weight"] = 0
    base = compute_covariate_balance(frame, _balance_spec()).rows[0]
    frame["weight"] *= 1e100
    scaled = compute_covariate_balance(frame, _balance_spec()).rows[0]
    assert scaled.smd_before == base.smd_before == 1
    assert scaled.smd_after == pytest.approx(base.smd_after)
    assert scaled.ess_treated == pytest.approx(base.ess_treated)
    assert scaled.n_treated == 3 and scaled.positive_weight_treated_n == 2
    assert scaled.zero_weight_treated_n == 1


def test_binary_and_declared_categorical_indicators_use_sample_variance():
    frame = _balance_frame().assign(
        binary=[1, 1, 0, 1, 0, 0], category=["A", "B", "A", "B", "B", "B"]
    )
    spec = _balance_spec(
        covariates=[
            {"column": "binary", "kind": "binary"},
            {"column": "category", "kind": "categorical", "levels": ["A", "B", "C"]},
        ]
    )
    rows = compute_covariate_balance(frame, spec).rows
    assert rows[0].smd_before == pytest.approx(1 / math.sqrt(3))
    assert [(row.covariate, row.level) for row in rows] == [
        ("binary", None),
        ("category", "A"),
        ("category", "B"),
        ("category", "C"),
    ]
    assert rows[-1].status_before == "constant_equal" and rows[-1].smd_before == 0


def test_per_covariate_missingness_and_no_weighted_request_are_explicit():
    frame = _balance_frame()
    frame.loc[0, "x"] = np.nan
    row = compute_covariate_balance(frame, _balance_spec(weight_column=None)).rows[0]
    assert row.n_treated == 2 and row.missing_treated_n == 1 and row.n_control == 3
    assert row.status_after == "not_requested" and row.smd_after is None
    assert row.mean_treated_after is None
    frame.loc[:2, "x"] = np.nan
    row = compute_covariate_balance(frame, _balance_spec()).rows[0]
    assert row.smd_before is None and row.status_before == "insufficient_observations"
    assert row.smd_after is None and row.status_after == "empty_weighted_group"
    assert row.ess_treated == 0


@pytest.mark.parametrize(
    "values,expected,status",
    [
        ([1.0] * 6, 0.0, "constant_equal"),
        ([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], None, "zero_variance_different_means"),
    ],
)
def test_constants_are_retained_without_fabricating_finite_smd(
    values, expected, status
):
    row = compute_covariate_balance(
        _balance_frame().assign(x=values), _balance_spec()
    ).rows[0]
    assert row.smd_before == expected and row.status_before == status
    assert row.smd_after == expected and row.status_after == status


@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, "bad"])
def test_invalid_weights_are_not_silently_dropped(bad):
    frame = _balance_frame().astype({"weight": object})
    frame.loc[0, "weight"] = bad
    with pytest.raises(ValueError, match="covariate_balance_"):
        compute_covariate_balance(frame, _balance_spec())


def test_empty_weighted_group_and_duplicate_units_are_rejected():
    frame = _balance_frame()
    frame.loc[:2, "weight"] = 0
    with pytest.raises(ValueError, match="empty_weighted_group"):
        compute_covariate_balance(frame, _balance_spec())
    frame = _balance_frame()
    frame.loc[0, "unit"] = frame.loc[1, "unit"]
    with pytest.raises(ValueError, match="analysis_unit_not_unique"):
        compute_covariate_balance(frame, _balance_spec())


def test_numeric_identity_aliases_cannot_be_counted_as_distinct_units():
    frame = _balance_frame()
    frame["unit"] = pd.Series([1, 1.0, "c", "d", "e", "f"], dtype=object)
    with pytest.raises(ValueError, match="analysis_unit_not_unique"):
        compute_covariate_balance(frame, _balance_spec())
    roster = pd.DataFrame({"unit": pd.Series([1, 1.0], dtype=object)})
    assignments = pd.DataFrame({"unit_id": [1], "cluster": ["A"]})
    with pytest.raises(ValueError, match="analysis_unit_not_unique"):
        compute_cluster_sizes(roster, assignments, _size_spec())


def test_nonfinite_covariates_single_treatment_group_and_unknown_category_fail_closed():
    for frame in (
        _balance_frame().assign(x=np.inf),
        _balance_frame().assign(treated=1),
    ):
        with pytest.raises(ValueError):
            compute_covariate_balance(frame, _balance_spec())
    with pytest.raises(ValueError, match="undeclared_levels"):
        compute_covariate_balance(
            _balance_frame().assign(x="C"),
            _balance_spec(
                covariates=[
                    {"column": "x", "kind": "categorical", "levels": ["A", "B"]}
                ]
            ),
        )


def test_ambiguous_category_or_cluster_label_types_are_not_merged():
    with pytest.raises(ValueError, match="ambiguous_level_types"):
        compute_covariate_balance(
            _balance_frame().assign(x=[1, "1", 0, "0", 1, "0"]),
            _balance_spec(
                covariates=[
                    {"column": "x", "kind": "categorical", "levels": ["0", "1"]}
                ]
            ),
        )
    frame, assignments = _sizes_frames()
    assignments["cluster"] = [1, "1", "A", None]
    with pytest.raises(ValueError, match="ambiguous_label_types"):
        compute_cluster_sizes(
            frame, assignments, _size_spec(expected_clusters=["1", "A"])
        )


def test_cluster_sizes_use_unique_cohort_units_and_keep_missing_denominators():
    frame, assignments = _sizes_frames()
    result = compute_cluster_sizes(frame, assignments, _size_spec())
    assert result.input_n == 5 and result.assignment_rows_n == 4
    assert result.assigned_n == 3 and result.missing_label_n == 2
    assert [row.n_units for row in result.rows] == [2, 1, 0]
    assert [row.proportion_assigned for row in result.rows] == pytest.approx(
        [2 / 3, 1 / 3, 0]
    )
    assert [row.proportion_all_units for row in result.rows] == [0.4, 0.2, 0]
    assert result.rows[-1].is_empty
    assert all(row.is_near_empty is None for row in result.rows)
    assert not result.degenerate_single_cluster


def test_near_empty_rule_is_explicit_strict_and_single_cluster_is_retained():
    frame, assignments = _sizes_frames()
    result = compute_cluster_sizes(
        frame, assignments, _size_spec(near_empty_n=2, near_empty_fraction=0.5)
    )
    assert [row.is_near_empty for row in result.rows] == [False, True, True]
    result = compute_cluster_sizes(frame, assignments.assign(cluster="A"), _size_spec())
    assert result.degenerate_single_cluster and result.observed_cluster_n == 1


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("duplicate_unit", "analysis_unit_not_unique"),
        ("duplicate_assignment", "analysis_unit_not_unique"),
        ("outside", "outside_cohort"),
        ("unknown_label", "undeclared_labels"),
        ("all_missing", "no_assigned_units"),
    ],
)
def test_cluster_sizes_reject_ambiguous_membership(mutation, reason):
    frame, assignments = _sizes_frames()
    if mutation == "duplicate_unit":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "duplicate_assignment":
        assignments = pd.concat([assignments, assignments.iloc[[0]]])
    elif mutation == "outside":
        assignments.loc[0, "unit_id"] = "outside"
    elif mutation == "unknown_label":
        assignments.loc[0, "cluster"] = "D"
    else:
        assignments["cluster"] = None
    with pytest.raises(ValueError, match=reason):
        compute_cluster_sizes(frame, assignments, _size_spec())


def test_missing_cluster_labels_require_explicit_exclusion():
    with pytest.raises(ValueError, match="missing_labels"):
        compute_cluster_sizes(
            *_sizes_frames(), _size_spec(missing_label_policy="error")
        )


def _step(kind: str):
    if kind == "balance":
        return AnalysisStep(
            step_id="balance",
            intent="Describe measured covariate balance.",
            method="covariate_balance",
            planned_analysis_role="secondary",
            scientific_action_id="causal_emulation.covariate_balance",
            inputs=["artifact:analysis_cohort", "unit", "treated", "weight", "x"],
            expected_outputs=list(BALANCE_OUTPUTS),
            covariate_balance_spec=_balance_spec(),
        )
    return AnalysisStep(
        step_id="sizes",
        intent="Count existing frozen cluster memberships.",
        method="cluster_sizes",
        planned_analysis_role="secondary",
        scientific_action_id="phenotyping.cluster_sizes",
        inputs=["artifact:analysis_cohort", "table:phenotype_assignments", "unit"],
        expected_outputs=list(CLUSTER_SIZE_OUTPUTS),
        cluster_size_spec=_size_spec(),
    )


def _binding(run_dir, key, frame, filename, step_id):
    path = run_dir / filename
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kind, _, product = key.partition(":")
    binding = {
        "relative_path": filename,
        "sha256": digest,
        "declared_kind": kind,
        "evidence_kind": "table",
        "product": product,
        "evidence_id": "ev_" + product,
        "produced_by_step": "producer_" + product,
        "step_id": step_id,
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
    }
    binding["identity_row"] = {
        "input_key": key,
        **{
            name: binding[name]
            for name in ("sha256", "declared_kind", "product", "evidence_id")
        },
    }
    return binding


def _bound_case(tmp_path, kind):
    step = _step(kind)
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    out_dir.mkdir(parents=True)
    frame = _balance_frame() if kind == "balance" else _sizes_frames()[0]
    bindings = {
        "artifact:analysis_cohort": _binding(
            run_dir, "artifact:analysis_cohort", frame, "cohort.parquet", step.step_id
        )
    }
    if kind == "sizes":
        assignments = _sizes_frames()[1].assign(
            source_cohort_sha256=bindings["artifact:analysis_cohort"]["sha256"],
            source_identity_column="unit",
        )
        bindings["table:phenotype_assignments"] = _binding(
            run_dir,
            "table:phenotype_assignments",
            assignments,
            "assignments.parquet",
            step.step_id,
        )
    resolved = run_dir / "resolved.json"
    resolved.write_text(
        json.dumps({"step_id": step.step_id, "inputs": bindings}), encoding="utf-8"
    )
    return step, out_dir, resolved, bindings


@pytest.mark.parametrize("kind", ["balance", "sizes"])
def test_standalone_diagnostics_select_execute_and_replay_through_host_pipeline(
    tmp_path, monkeypatch, kind
):
    step, out_dir, resolved, bindings = _bound_case(tmp_path, kind)
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    plan = AnalysisPlan(research_question="Synthetic descriptive checks.", steps=[step])
    selected = select_standard_executor(step, plan=plan)
    assert selected is not None
    assert selected.analysis_kind == (
        "covariate_balance" if kind == "balance" else "cluster_sizes"
    )
    assert set(selected.consumed_input_keys) == set(bindings)
    scaffold = (
        covariate_balance_executor_scaffold
        if kind == "balance"
        else cluster_sizes_executor_scaffold
    )(step)
    assert scaffold.body == "" and scaffold.has_host_regions
    assert scaffold.host_regions_intact(selected.code)
    exec(compile(selected.code, "<synthetic diagnostic>", "exec"), {})
    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    assert not unlabelled_aggregate_row_findings(step_id=step.step_id, out_dir=out_dir)
    validate = (
        covariate_balance_output_findings
        if kind == "balance"
        else cluster_sizes_output_findings
    )
    assert not validate(
        step=step,
        step_summary=summary,
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )
    assert validate(
        step=step,
        step_summary={**summary, "claim_ceiling": "paper_authorized"},
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )
    receipt_path = out_dir / (
        "covariate_balance_receipt.json"
        if kind == "balance"
        else "cluster_sizes_receipt.json"
    )
    receipt_text = receipt_path.read_text("utf-8")
    receipt_path.write_text("[]", encoding="utf-8")
    assert validate(
        step=step,
        step_summary=summary,
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )
    receipt_path.write_text(receipt_text, encoding="utf-8")
    bindings["artifact:analysis_cohort"]["product_contract"]["analysis_unit"] = (
        "subject"
    )
    assert validate(
        step=step,
        step_summary=summary,
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )
    del bindings["artifact:analysis_cohort"]["product_contract"]["analysis_unit"]
    table = out_dir / (
        "covariate_balance.csv" if kind == "balance" else "cluster_sizes.csv"
    )
    table.write_text(
        table.read_text("utf-8").replace("estimated", "fabricated")
        if kind == "balance"
        else table.read_text("utf-8").replace("0.4", "0.9"),
        encoding="utf-8",
    )
    assert validate(
        step=step,
        step_summary=summary,
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )


@pytest.mark.parametrize("kind", ["balance", "sizes"])
def test_declared_action_missing_spec_has_a_typed_declaration_gap(kind):
    step = _step(kind)
    field = "covariate_balance_spec" if kind == "balance" else "cluster_size_spec"
    step = step.model_copy(update={field: None})
    verdict = (
        covariate_balance_executor_verdict
        if kind == "balance"
        else cluster_sizes_executor_verdict
    )(step)
    assert not verdict.claimed and verdict.declaration_is_incomplete
    assert field in verdict.missing_declarations


@pytest.mark.parametrize("kind", ["balance", "sizes"])
def test_typed_source_tampering_blocks_before_calculation(tmp_path, monkeypatch, kind):
    step, out_dir, resolved, bindings = _bound_case(tmp_path, kind)
    monkeypatch.setenv("EASYICU_RUN_DIR", str(resolved.parent))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    selected = select_standard_executor(
        step, plan=AnalysisPlan(research_question="Test", steps=[step])
    )
    (resolved.parent / "cohort.parquet").write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="digest_mismatch"):
        exec(compile(selected.code, "<synthetic diagnostic>", "exec"), {})
    assert not (out_dir / "step_summary.json").exists()


def _trajectory_case(
    tmp_path,
    *,
    freeze_status="candidate_labels_frozen_stability_threshold_passed",
    producer="producer_cluster_assignments",
    selected_n_clusters=3,
):
    step, out_dir, resolved, bindings = _bound_case(tmp_path, "sizes")
    del bindings["table:phenotype_assignments"]
    frame, assignments = _sizes_frames()
    assignments = assignments.rename(columns={"unit_id": "unit"})
    bindings["table:cluster_assignments"] = _binding(
        resolved.parent,
        "table:cluster_assignments",
        assignments,
        "trajectory.parquet",
        step.step_id,
    )
    freeze_path = resolved.parent / "stability_freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.trajectory_stability_freeze/1",
                "freeze_status": freeze_status,
                "selected_n_clusters": selected_n_clusters,
            }
        ),
        encoding="utf-8",
    )
    bindings["artifact:stability_freeze"] = {
        "declared_kind": "artifact",
        "product": "stability_freeze",
        "relative_path": freeze_path.name,
        "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "produced_by_step": producer,
    }
    spec = _size_spec(
        assignment_source="table:cluster_assignments", assignment_unit_id_column="unit"
    )
    step = step.model_copy(
        update={
            "cluster_size_spec": spec,
            "inputs": [
                "artifact:analysis_cohort",
                "table:cluster_assignments",
                "artifact:stability_freeze",
                "unit",
            ],
        }
    )
    resolved.write_text(
        json.dumps({"step_id": step.step_id, "inputs": bindings}), encoding="utf-8"
    )
    return frame, step, out_dir, resolved, bindings


def test_trajectory_sizes_require_the_exact_frozen_owner_and_keep_unassigned_units(
    tmp_path,
):
    frame, step, out_dir, resolved, bindings = _trajectory_case(tmp_path)
    summary = run_cluster_sizes_from_bound(
        frame=frame,
        source_cohort=resolved.parent / "cohort.parquet",
        spec=step.cluster_size_spec,
        step_id=step.step_id,
        typed_cohort_input="artifact:analysis_cohort",
        run_dir=resolved.parent,
        resolved_inputs=resolved,
        out_dir=out_dir,
    )
    assert summary["assigned_n"] == 3 and summary["missing_label_n"] == 2
    assert summary["refit_performed"] is False
    assert not cluster_sizes_output_findings(
        step=step,
        step_summary=summary,
        resolved_input_bindings=bindings,
        out_dir=out_dir,
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        (
            {"freeze_status": "not_frozen_stability_threshold_failed"},
            "no_frozen_solution",
        ),
        ({"producer": "another_stability_owner"}, "producer_mismatch"),
        ({"selected_n_clusters": 4}, "differs_from_frozen_count"),
    ],
)
def test_trajectory_sizes_fail_closed_without_a_matching_frozen_solution(
    tmp_path, changes, reason
):
    frame, step, out_dir, resolved, bindings = _trajectory_case(tmp_path, **changes)
    with pytest.raises(ValueError, match=reason):
        run_cluster_sizes_from_bound(
            frame=frame,
            source_cohort=resolved.parent / "cohort.parquet",
            spec=step.cluster_size_spec,
            step_id=step.step_id,
            typed_cohort_input="artifact:analysis_cohort",
            run_dir=resolved.parent,
            resolved_inputs=resolved,
            out_dir=out_dir,
        )
    assert not (out_dir / "step_summary.json").exists()


@pytest.mark.parametrize("kind", ["balance", "sizes"])
def test_model_construct_does_not_bypass_supporting_role(kind):
    step = _step(kind).model_copy(update={"planned_analysis_role": "primary"})
    verdict = (
        covariate_balance_executor_verdict
        if kind == "balance"
        else cluster_sizes_executor_verdict
    )(step)
    assert not verdict.claimed and verdict.declaration_is_incomplete
