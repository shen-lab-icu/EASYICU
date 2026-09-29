"""Frozen trajectory classes are described on the run cohort, never refit or guessed.

The signed trajectory suite freezes its classes in the stability owner.  Its
labels cover only the stays with enough observed windows, and they are frozen
only when the stability owner's freeze record says so.  Synthetic stays only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.contracts.phenotype_comparison import (
    ASSIGNMENTS_PRODUCT,
    TRAJECTORY_ASSIGNMENTS_PRODUCT,
    TRAJECTORY_FREEZE_PRODUCT,
    TRAJECTORY_FROZEN_STATUS,
    TRAJECTORY_NO_SOLUTION_REASON,
    comparison_cohort_input,
    comparison_label_source,
    phenotype_comparison_output_findings,
)
from easyicu.research_agent.execution.runners.phenotype_comparison_executor import (
    phenotype_comparison_executor_code,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PhenotypeComparisonSpec,
    ResearchContext,
    TableOneVariableSpec,
    VariableRole,
)

STEP_ID = "05_frozen_class_description"
COHORT_KEY = "table:analysis_cohort"
NOT_FROZEN = "not_frozen_candidate_selection_failed_closed"


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe hospital mortality by frozen organ-support class.",
        cohort=CohortDescriptor(
            cohort_name="class_description_fixture",
            database="synthetic",
            n_stays=14,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="str"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={"is_binary": True, "levels": [0, 1]},
            ),
        ],
        target_outcome="death",
    )


def _spec() -> PhenotypeComparisonSpec:
    return PhenotypeComparisonSpec(
        identity_column="stay_id",
        outcome_columns=["death"],
        variables=[
            TableOneVariableSpec(
                name="death",
                variable_kind="categorical",
                summary="count_percent",
                test="none_descriptive_smd_only",
                levels=[0, 1],
            ),
            TableOneVariableSpec(
                name="age",
                variable_kind="continuous",
                summary="median_iqr",
                test="none_descriptive_smd_only",
            ),
        ],
    )


def _step(*, label_products=(TRAJECTORY_ASSIGNMENTS_PRODUCT, TRAJECTORY_FREEZE_PRODUCT)):
    return AnalysisStep(
        step_id=STEP_ID,
        planned_analysis_role="secondary",
        intent="Describe hospital mortality and age by frozen class.",
        method="descriptive_outcome_by_cluster",
        scientific_action_id="phenotyping.outcome_by_cluster",
        inputs=["stay_id", "death", "age", COHORT_KEY, *label_products],
        expected_outputs=["table:outcome_by_cluster"],
        phenotype_comparison_spec=_spec(),
    )


def _binding(run_dir: Path, key: str, path: Path, frame: pd.DataFrame | None) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kind, _, product = key.partition(":")
    binding = {
        "declared_kind": kind,
        "evidence_kind": "table" if frame is not None else "log",
        "product": product,
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "produced_by_step": f"producer_{product}",
        "product_contract": (
            {"columns": list(frame.columns), "row_count": len(frame)}
            if frame is not None
            else {}
        ),
        "identity_row": {
            "input_key": key,
            "declared_kind": kind,
            "product": product,
            "evidence_id": f"evidence_{product}",
            "sha256": digest,
        },
    }
    if frame is not None:
        binding["consumption_contract"] = {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        }
    return binding


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [f"s{i:02}" for i in range(14)],
            "age": [40.0, 50.0, 60.0, 70.0, 45.0, 55.0, 65.0, 75.0, None, 58.0,
                    61.0, 62.0, 63.0, 64.0],
            "death": [0, 0, 1, 0, 1, 1, 0, 1, 1, None, 0, 1, 0, 0],
        }
    )


def _labels() -> pd.DataFrame:
    # Ten of fourteen stays had enough observed windows to be clustered.
    return pd.DataFrame(
        {"stay_id": [f"s{i:02}" for i in (9, 3, 0, 1, 2, 4, 5, 6, 7, 8)],
         "cluster": [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]}
    )


def _inputs(tmp_path: Path, *, labels=None, freeze_status=TRAJECTORY_FROZEN_STATUS):
    run_dir = tmp_path / "run"
    cohort_dir = run_dir / "steps" / "04_host_bound_analysis_cohort" / "outputs"
    stability_dir = run_dir / "steps" / "02_stability" / "outputs"
    cohort_dir.mkdir(parents=True)
    stability_dir.mkdir(parents=True)
    cohort = _cohort()
    cohort_path = cohort_dir / "analysis_cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    labels = _labels() if labels is None else labels
    labels_path = stability_dir / "cluster_assignments.csv"
    labels.to_csv(labels_path, index=False)
    freeze_path = stability_dir / "stability_freeze.json"
    freeze_path.write_text(json.dumps({"freeze_status": freeze_status}), encoding="utf-8")
    (run_dir / "research_context.json").write_text(
        _context().model_dump_json(), encoding="utf-8"
    )
    manifest = {
        "step_id": STEP_ID,
        "inputs": {
            COHORT_KEY: _binding(run_dir, COHORT_KEY, cohort_path, cohort),
            TRAJECTORY_ASSIGNMENTS_PRODUCT: _binding(
                run_dir, TRAJECTORY_ASSIGNMENTS_PRODUCT, labels_path, labels
            ),
            TRAJECTORY_FREEZE_PRODUCT: _binding(
                run_dir, TRAJECTORY_FREEZE_PRODUCT, freeze_path, None
            ),
        },
    }
    return run_dir, manifest


def _execute(run_dir: Path, manifest: dict, monkeypatch) -> tuple[dict, Path]:
    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest_path))
    exec(phenotype_comparison_executor_code(_step()), {})
    return json.loads((out_dir / "step_summary.json").read_text("utf-8")), out_dir


def _findings(summary: dict, manifest: dict, out_dir: Path):
    return phenotype_comparison_output_findings(
        step=_step(),
        step_summary=summary,
        context=_context(),
        resolved_input_bindings=manifest["inputs"],
        out_dir=out_dir,
    )


def test_frozen_labels_describe_the_clustered_stays_and_count_the_rest(
    tmp_path, monkeypatch
):
    run_dir, manifest = _inputs(tmp_path)
    summary, out_dir = _execute(run_dir, manifest, monkeypatch)

    table = pd.read_csv(
        out_dir / "outcome_by_cluster.csv",
        dtype={"category": "string", "group": "string"},
    )
    death = table[(table.variable == "death") & (table.category == "1")].set_index("group")
    assert death.denominator_n.to_dict() == {"Overall": 10, "0": 5, "1": 5}
    assert death["count"].to_dict() == {"Overall": 5.0, "0": 2.0, "1": 3.0}
    # A missing outcome stays missing; it is not coded as a non-event.
    assert death.nonmissing_n.to_dict() == {"Overall": 9, "0": 5, "1": 4}
    # The four stays too sparse to cluster are excluded and counted on every row.
    assert table.group_missing_excluded_n.eq(4).all()
    assert set(table.variable_role) == {"outcome", "clinical_profile"}
    assert summary["n_rows"] == 10
    assert summary["n_not_clustered"] == 4
    assert summary["cluster_counts"] == {"0": 5, "1": 5}
    assert summary["assignment_source"] == TRAJECTORY_ASSIGNMENTS_PRODUCT
    assert summary["freeze_status"] == TRAJECTORY_FROZEN_STATUS
    assert summary["refit_performed"] is False
    assert _findings(summary, manifest, out_dir) == []


def test_a_freeze_record_without_a_frozen_class_describes_nothing(tmp_path, monkeypatch):
    run_dir, manifest = _inputs(
        tmp_path,
        labels=pd.DataFrame(columns=["stay_id", "cluster"]),
        freeze_status=NOT_FROZEN,
    )
    summary, out_dir = _execute(run_dir, manifest, monkeypatch)

    assert summary["status"] == "ok"
    assert summary["scientific_status"] == "failed_closed"
    assert summary["reason_code"] == TRAJECTORY_NO_SOLUTION_REASON
    assert summary["freeze_status"] == NOT_FROZEN
    assert pd.read_csv(out_dir / "outcome_by_cluster.csv").empty
    assert _findings(summary, manifest, out_dir) == []

    presented_as_result = {**summary, "scientific_status": "ok"}
    findings = _findings(presented_as_result, manifest, out_dir)
    assert [f.detail["reason"] for f in findings] == [
        "phenotype_comparison_no_solution_incoherent"
    ]

    # Rows under a record that froze no class describe a class that does not
    # exist, even with a receipt digest that matches them.
    table = out_dir / "outcome_by_cluster.csv"
    table.write_text("variable,group,count\ndeath,0,2\n", encoding="utf-8")
    rows_without_class = {
        **summary,
        "output_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    findings = _findings(rows_without_class, manifest, out_dir)
    assert [f.detail["reason"] for f in findings] == [
        "phenotype_comparison_no_solution_incoherent"
    ]


def test_candidate_labels_under_a_not_frozen_record_are_never_described(
    tmp_path, monkeypatch
):
    run_dir, manifest = _inputs(tmp_path, freeze_status=NOT_FROZEN)
    summary, out_dir = _execute(run_dir, manifest, monkeypatch)

    assert summary["scientific_status"] == "failed_closed"
    assert pd.read_csv(out_dir / "outcome_by_cluster.csv").empty


@pytest.mark.parametrize(
    "labels,reason",
    [
        (
            pd.DataFrame({"stay_id": ["s00", "s01", "foreign"], "cluster": [0, 1, 1]}),
            "phenotype_comparison_membership_mismatch",
        ),
        (
            pd.DataFrame({"unit_id": ["s00", "s01"], "cluster": [0, 1]}),
            "phenotype_comparison_source_mismatch",
        ),
        (
            pd.DataFrame({"stay_id": ["s00", "s00"], "cluster": [0, 1]}),
            "phenotype_comparison_identity_invalid",
        ),
    ],
)
def test_labels_outside_the_cohort_or_keyed_otherwise_fail_closed(
    tmp_path, monkeypatch, labels, reason
):
    run_dir, manifest = _inputs(tmp_path, labels=labels)
    with pytest.raises(ValueError, match=reason):
        _execute(run_dir, manifest, monkeypatch)
    assert not (run_dir / "steps" / STEP_ID / "outputs" / "outcome_by_cluster.csv").exists()


def test_a_freeze_record_changed_after_binding_is_refused(tmp_path, monkeypatch):
    run_dir, manifest = _inputs(tmp_path)
    freeze = run_dir / "steps" / "02_stability" / "outputs" / "stability_freeze.json"
    freeze.write_text(json.dumps({"freeze_status": TRAJECTORY_FROZEN_STATUS, "x": 1}))
    with pytest.raises(ValueError, match="phenotype_comparison_freeze_record_unverified"):
        _execute(run_dir, manifest, monkeypatch)


@pytest.mark.parametrize(
    "tamper,reason",
    [
        (lambda s: s.update(n_not_clustered=3), "phenotype_comparison_membership_mismatch"),
        (lambda s: s.update(freeze_status=NOT_FROZEN), "phenotype_comparison_receipt_mismatch"),
        (
            lambda s: s.update(assignment_source=ASSIGNMENTS_PRODUCT),
            "phenotype_comparison_receipt_mismatch",
        ),
    ],
)
def test_the_output_gate_closes_membership_and_freeze_receipts(
    tmp_path, monkeypatch, tamper, reason
):
    run_dir, manifest = _inputs(tmp_path)
    summary, out_dir = _execute(run_dir, manifest, monkeypatch)
    tamper(summary)
    assert [f.detail["reason"] for f in _findings(summary, manifest, out_dir)] == [reason]


def test_typed_inputs_name_exactly_one_label_source():
    assert comparison_label_source(_step()) == TRAJECTORY_ASSIGNMENTS_PRODUCT
    assert comparison_cohort_input(_step()) == COHORT_KEY
    cross_sectional = _step(label_products=(ASSIGNMENTS_PRODUCT,))
    assert comparison_label_source(cross_sectional) == ASSIGNMENTS_PRODUCT
    for label_products in (
        (TRAJECTORY_ASSIGNMENTS_PRODUCT,),
        (ASSIGNMENTS_PRODUCT, TRAJECTORY_ASSIGNMENTS_PRODUCT, TRAJECTORY_FREEZE_PRODUCT),
    ):
        with pytest.raises(ValueError, match="phenotype_comparison_typed_inputs_invalid"):
            comparison_label_source(_step(label_products=label_products))


def test_only_the_frozen_trajectory_source_changes_the_generated_call():
    assert "label_source='table:cluster_assignments'" in phenotype_comparison_executor_code(
        _step()
    )
    assert "label_source" not in phenotype_comparison_executor_code(
        _step(label_products=(ASSIGNMENTS_PRODUCT,))
    )


def test_the_selector_binds_the_cohort_labels_and_freeze_record():
    step = _step()
    plan = AnalysisPlan(
        research_question=_context().research_question,
        analysis_type="trajectory_clustering",
        steps=[step],
    )
    selected = select_standard_executor(step=step, plan=plan)
    assert selected.analysis_kind == "phenotype_comparison"
    assert set(selected.consumed_input_keys) == {
        COHORT_KEY,
        TRAJECTORY_ASSIGNMENTS_PRODUCT,
        TRAJECTORY_FREEZE_PRODUCT,
    }
