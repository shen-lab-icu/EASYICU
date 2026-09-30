"""The trajectory owner's eligibility flow is drawn by the host renderer.

The signed trajectory representation owner publishes ``table:cohort_flow`` as
``metric,n``: the input cohort, the rows meeting the minimum of observed
windows, the rows excluded for too few windows, and the rows clustered.  The
host's cohort-flow renderer accepted only its two ledger schemas, so every
trajectory plan's cohort accounting figure fell to a generated script.  It now
reads this flow as a three-stage ledger whose one exclusion is the exclusion
row, and exports every bound row.  Synthetic counts only.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.typed_binding import (
    _write_host_input_binding_receipts,
)
from easyicu.research_agent.audits.aggregate_row import (
    unlabelled_aggregate_row_findings,
)
from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
    COHORT_ACCOUNTING_COMPLETE,
    COHORT_FLOW_INPUT,
    _verified_flow,
    cohort_flow_figure_executor_owns_step,
    render_cohort_flow_axis,
    run_cohort_flow_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)

OWNER_STEP = "00_authority_compiled_trajectory_representation"
FLOW = [
    ("input_cohort", 120),
    ("meets_min_observed_windows", 100),
    ("excluded_insufficient_windows", 20),
    ("included_in_clustering", 100),
]
METRICS = sorted(metric for metric, _n in FLOW)
STAGE_LABELS = ["Source cohort", "Meets min observed windows", "Included in clustering"]


def _step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "07_cohort_accounting_figure",
            "planned_analysis_role": "auxiliary",
            "intent": "Render the trajectory cohort accounting.",
            "inputs": [COHORT_FLOW_INPUT],
            "expected_outputs": ["figure:cohort_flow"],
            "method": "visualization",
            "input_consumption_contracts": [
                ArtifactConsumptionContract(input_key=COHORT_FLOW_INPUT, mode="all_rows")
            ],
        }
    )


def _binding(tmp_path, rows=FLOW, *, columns=("metric", "n"), metrics=METRICS):
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / OWNER_STEP / "outputs" / "cohort_flow.csv"
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(source, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description="Trajectory window-eligibility flow.",
        source_path=source,
        evidence_id="trajectory_cohort_flow",
        produced_by_step=OWNER_STEP,
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    evidence_path = run_dir / record.relative_path
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    contract = {
        "schema_version": "easyicu.host_typed_product.v4",
        "tabular_format": "csv",
        "columns": list(frame.columns),
        "row_count": len(frame),
    }
    if metrics is not None:
        contract["categorical_values"] = {"metric": list(metrics)}
    binding = {
        "relative_path": str(evidence_path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": OWNER_STEP,
        "product": "cohort_flow",
        "identity_row": {
            "declared_kind": "table",
            "evidence_id": record.evidence_id,
            "input_key": COHORT_FLOW_INPUT,
            "produced_by_step": OWNER_STEP,
            "product": "cohort_flow",
            "sha256": digest,
        },
        "product_contract": contract,
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": COHORT_FLOW_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    manifest = {
        "schema_version": "2.1",
        "step_id": _step().step_id,
        "inputs": {COHORT_FLOW_INPUT: binding},
    }
    return run_dir, manifest, binding, evidence_path


def _render(run_dir, manifest, out_dir):
    return run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=_step().step_id,
        figure_product="cohort_flow",
    )


def test_the_trajectory_flow_is_drawn_without_a_generated_script(tmp_path) -> None:
    step = _step()
    run_dir, manifest, binding, evidence_path = _binding(tmp_path)
    assert cohort_flow_figure_executor_owns_step(
        step, resolved_bindings={COHORT_FLOW_INPUT: binding}
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings={COHORT_FLOW_INPUT: binding},
    )
    assert selection is not None
    assert selection.analysis_kind == "cohort_flow_figure"

    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = _render(run_dir, manifest, out_dir)

    assert summary["cohort_accounting_completeness"] == COHORT_ACCOUNTING_COMPLETE
    assert summary["paper_grade_cohort_accounting"] is True
    assert summary["rendering_mode"] == "sequential_attrition_flow"
    # Every bound row is consumed: three stages and the exclusion between them.
    assert summary["source_rows_consumed"] == 4
    assert summary["input_bindings"][0]["row_count"] == 4
    source_path = out_dir / "cohort_flow_source_data.csv"
    source = pd.read_csv(source_path)
    assert source["display_label"].tolist() == [*STAGE_LABELS, "Excluded"]
    assert source["metric"].tolist() == [
        "input_cohort",
        "meets_min_observed_windows",
        "included_in_clustering",
        "excluded_insufficient_windows",
    ]
    assert source["n"].tolist() == [120, 100, 100, 20]
    assert source["row_role"].tolist() == ["cohort_stage"] * 3 + ["exclusion"]
    assert source["source_row_index"].tolist() == [0, 1, 3, 2]
    assert "n_before" not in source and "step_order" not in source
    assert (
        FigureSourceDataValidator._compare_source_to_upstream(
            source_df=source, source_path=source_path, upstream_path=evidence_path
        )["ok"]
        is True
    )
    assert unlabelled_aggregate_row_findings(step_id=step.step_id, out_dir=out_dir) == []
    receipts = _write_host_input_binding_receipts(
        out_dir=out_dir,
        step_summary=summary,
        resolved_input_bindings={
            COHORT_FLOW_INPUT: {**binding, "absolute_path": str(evidence_path)}
        },
        consumed_input_keys=selection.consumed_input_keys,
    )
    assert receipts["input_bindings"][0]["row_count"] == 4


def test_the_drawing_shows_one_exclusion_between_the_first_two_stages(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _run_dir, _manifest, binding, evidence_path = _binding(tmp_path)
    frame = _verified_flow(evidence_path, binding)

    assert frame["predicate_kind"].tolist() == [
        "input_cohort",
        "meets_min_observed_windows",
        "included_in_clustering",
    ]
    assert frame["n_excluded"].tolist() == [0, 20, 0]
    assert frame["n_remaining"].tolist() == [120, 100, 100]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    try:
        render_cohort_flow_axis(ax, frame, STAGE_LABELS, complete=True)
        texts = [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)
    assert [text for text in texts if text.startswith("Excluded")] == ["Excluded (n = 20)"]
    for label in STAGE_LABELS:
        assert any(label in text for text in texts), label


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        pytest.param(
            [("input_cohort", 120), ("meets_min_observed_windows", 100),
             ("excluded_insufficient_windows", 15), ("included_in_clustering", 100)],
            "denominator arithmetic failed",
            id="exclusion_disagrees_with_the_stages",
        ),
        pytest.param(
            [("input_cohort", 120), ("meets_min_observed_windows", 100),
             ("excluded_insufficient_windows", 20), ("included_in_clustering", 90)],
            "denominator arithmetic failed",
            id="an_eligible_row_is_not_clustered",
        ),
        pytest.param(
            [("input_cohort", 80), ("meets_min_observed_windows", 100),
             ("excluded_insufficient_windows", -20), ("included_in_clustering", 100)],
            "contains invalid rows",
            id="a_negative_count",
        ),
        pytest.param(
            # Consistent, and still consistent once truncated to integers.
            [("input_cohort", 120.5), ("meets_min_observed_windows", 100.5),
             ("excluded_insufficient_windows", 20), ("included_in_clustering", 100.5)],
            "contains invalid rows",
            id="a_fractional_count",
        ),
        pytest.param(
            [*FLOW, ("input_cohort", 120)],
            "contains invalid rows",
            id="a_repeated_metric",
        ),
    ],
)
def test_counts_that_break_the_flow_identities_fail_closed(tmp_path, rows, reason) -> None:
    step = _step()
    run_dir, manifest, binding, _path = _binding(tmp_path, rows)
    # The typed contract still names the trajectory flow, so the host owns the
    # step; the bytes then fail its identities and nothing is drawn.
    assert cohort_flow_figure_executor_owns_step(
        step, resolved_bindings={COHORT_FLOW_INPUT: binding}
    )
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    with pytest.raises(ValueError, match=f"trajectory eligibility flow {reason}"):
        _render(run_dir, manifest, out_dir)
    assert not (out_dir / "step_summary.json").exists()
    assert not list(out_dir.glob("cohort_flow.*"))


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            {"rows": [("screened", 120), ("enrolled", 100)],
             "metrics": ["enrolled", "screened"]},
            id="other_metrics",
        ),
        pytest.param({"rows": FLOW, "metrics": None}, id="metrics_not_typed"),
        pytest.param(
            {"rows": FLOW, "metrics": [*METRICS, "died_before_window"]},
            id="an_extra_metric",
        ),
        pytest.param(
            {"rows": [(*row, "") for row in FLOW], "columns": ("metric", "n", "note")},
            id="an_extra_column",
        ),
    ],
)
def test_a_metric_table_that_is_not_the_trajectory_flow_is_not_claimed(
    tmp_path, shape
) -> None:
    _run_dir, _manifest, binding, _path = _binding(
        tmp_path,
        shape["rows"],
        columns=shape.get("columns", ("metric", "n")),
        metrics=shape.get("metrics", METRICS),
    )
    assert not cohort_flow_figure_executor_owns_step(
        _step(), resolved_bindings={COHORT_FLOW_INPUT: binding}
    )
