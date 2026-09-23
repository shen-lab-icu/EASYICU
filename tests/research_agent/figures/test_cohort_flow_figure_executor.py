from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    COHORT_ACCOUNTING_DENOMINATOR_ONLY,
    COHORT_FLOW_INPUT,
    cohort_flow_figure_executor_owns_step,
    run_cohort_flow_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)


def _step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "08_cohort_accounting_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render exact cohort accounting.",
        "inputs": [COHORT_FLOW_INPUT],
        "expected_outputs": ["figure:cohort_accounting"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(input_key=COHORT_FLOW_INPUT, mode="all_rows")
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, "universe", 140, 0, 140],
            [1, "adult", 140, 12, 128],
            [2, "first_icu_stay", 128, 8, 120],
        ],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )


def _binding(
    tmp_path: Path,
    *,
    frame: pd.DataFrame | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / "01_define_analysis_cohort" / "outputs" / "flow.csv"
    source.parent.mkdir(parents=True)
    frame = _frame() if frame is None else frame
    frame.to_csv(source, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description="Canonical cohort flow.",
        source_path=source,
        evidence_id="cohort_flow_parent",
        produced_by_step="01_define_analysis_cohort",
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    evidence_path = run_dir / record.relative_path
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    identity = {
        "declared_kind": "table",
        "evidence_id": record.evidence_id,
        "input_key": COHORT_FLOW_INPUT,
        "produced_by_step": "01_define_analysis_cohort",
        "product": "cohort_flow",
        "sha256": digest,
    }
    binding: dict[str, object] = {
        "relative_path": str(evidence_path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": "01_define_analysis_cohort",
        "product": "cohort_flow",
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": COHORT_FLOW_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    manifest: dict[str, object] = {
        "schema_version": "2.1",
        "step_id": _step().step_id,
        "inputs": {COHORT_FLOW_INPUT: binding},
    }
    return run_dir, manifest, binding


def test_exact_cohort_flow_selects_and_renders_without_llm(tmp_path: Path) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
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
    assert selection.consumed_input_keys == (COHORT_FLOW_INPUT,)

    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    source = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    assert source["n_remaining"].tolist() == [140, 128, 120]
    assert summary["source_rows_consumed"] == 3
    assert summary["cohort_accounting_completeness"] == COHORT_ACCOUNTING_COMPLETE
    assert summary["paper_grade_cohort_accounting"] is True
    assert summary["upstream_attrition_available"] is True
    assert summary["rendering_mode"] == "sequential_attrition_flow"
    assert (out_dir / "cohort_accounting.figure_contract.json").is_file()


def test_complete_flow_figure_draws_stages_exclusions_and_retention(
    tmp_path: Path,
) -> None:
    """The manuscript form is a flow diagram, not a labelled bar chart.

    Every drawn number is derived from the bound ledger: stage counts, per-stage
    exclusions and the retained share of the previous stage and of the universe.
    Labels are humanised without inventing clinical words, and the mechanical
    ``First_Icu_Stay`` spelling must not survive anywhere in the export.
    """

    step = _step()
    run_dir, manifest, _binding_row = _binding(tmp_path)
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    source = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    assert source["display_label"].tolist() == [
        "Source universe",
        "Adult",
        "Final \u00b7 First icu stay",
    ]
    svg = (out_dir / "cohort_accounting.svg").read_text(encoding="utf-8")
    assert "First_Icu_Stay" not in svg
    assert "n = 140" in svg and "n = 128" in svg and "n = 120" in svg
    assert "\u221212 excluded" in svg and "\u22128 excluded" in svg
    assert "91.4% of previous" in svg
    assert "93.8% of previous" in svg
    assert "85.7% of universe" in svg
    # The grey share sits directly under a red "excluded" count.  It is the
    # share that REMAINS, and it must say so: a bare "91.4% of previous" under
    # "-12 excluded" reads as though 91.4% had been excluded.
    assert "retained 91.4% of previous" in svg
    assert "retained 93.8% of previous" in svg
    contract = json.loads(
        (out_dir / "cohort_accounting.figure_contract.json").read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["metadata"]["paper_grade_cohort_accounting"] is True


def test_single_stage_flow_keeps_the_node_grammar_without_fake_shares(
    tmp_path: Path,
) -> None:
    step = _step()
    singleton = pd.DataFrame(
        [[0, "universe", 94_458, 0, 94_458]],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )
    run_dir, manifest, _binding_row = _binding(tmp_path, frame=singleton)
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    svg = (out_dir / "cohort_accounting.svg").read_text(encoding="utf-8")
    assert "n = 94,458" in svg
    assert "All bound input rows" in svg
    assert "of universe" not in svg
    assert "ICU stays remaining" not in svg


def test_retention_share_makes_no_claim_when_counts_are_not_monotone() -> None:
    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        _retention_text,
    )

    assert _retention_text(50, 100) == "50%"
    assert _retention_text(100, 100) == "100%"
    assert _retention_text(120, 100) is None
    assert _retention_text(10, 0) is None


def test_a_rounded_share_never_contradicts_the_exclusion_beside_it() -> None:
    """Two stays excluded from 48,971 must not read "retained 100%"."""

    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        _retention_text,
    )

    assert _retention_text(48_969, 48_971) == ">99.9%"
    assert _retention_text(1, 48_971) == "<0.1%"
    # Exact extremes and ordinary shares are unchanged.
    assert _retention_text(48_971, 48_971) == "100%"
    assert _retention_text(0, 48_971) == "0%"
    assert _retention_text(89_964, 94_418) == "95.3%"


@pytest.mark.parametrize(
    "source_input",
    [
        "table:landmark_population_flow",
        "table:matched_population_flow",
        "table:validation_population_flow",
    ],
)
def test_primary_population_flow_selects_and_renders_without_llm(
    tmp_path: Path,
    source_input: str,
) -> None:
    frame = pd.DataFrame(
        [
            ["source_cohort", 100, 0, "source"],
            ["alive_at_landmark", 80, 20, "alive"],
            ["complete_case_model_population", 60, 20, "complete"],
        ],
        columns=["stage", "n", "excluded_from_previous", "population_rule"],
    )
    step = _step(
        inputs=[source_input],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=source_input, mode="all_rows")
        ],
    )
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / "primary_model" / "outputs" / "flow.csv"
    source.parent.mkdir(parents=True)
    frame.to_csv(source, index=False)
    record = EvidenceStore(run_dir).register_file(
        kind="table",
        description="Primary model population flow.",
        source_path=source,
        evidence_id="primary_population_flow",
        produced_by_step="primary_model",
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    evidence_path = run_dir / record.relative_path
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    product = source_input.partition(":")[2]
    identity = {
        "declared_kind": "table",
        "evidence_id": record.evidence_id,
        "input_key": source_input,
        "produced_by_step": "primary_model",
        "product": product,
        "sha256": digest,
    }
    binding = {
        "relative_path": str(evidence_path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": "table",
        "evidence_kind": "table",
        "evidence_id": record.evidence_id,
        "produced_by_step": "primary_model",
        "product": product,
        "identity_row": identity,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": source_input,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
    }
    manifest = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "inputs": {source_input: binding},
    }

    assert cohort_flow_figure_executor_owns_step(
        step, resolved_bindings={source_input: binding}
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Test", steps=[step]),
        resolved_bindings={source_input: binding},
    )
    assert selection is not None
    assert selection.consumed_input_keys == (source_input,)
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    summary = run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
        source_input=source_input,
    )

    source_data = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    assert source_data["n"].tolist() == [100, 80, 60]
    assert "step_order" not in source_data and "n_before" not in source_data
    assert source_data["row_role"].tolist() == ["cohort_stage"] * 3
    assert (
        FigureSourceDataValidator._compare_source_to_upstream(
            source_df=source_data,
            source_path=out_dir / "cohort_accounting_source_data.csv",
            upstream_path=evidence_path,
        )["ok"]
        is True
    )
    assert (
        unlabelled_aggregate_row_findings(step_id=step.step_id, out_dir=out_dir) == []
    )
    assert summary["source_input"] == source_input
    assert summary["paper_grade_cohort_accounting"] is True
    bound = _write_host_input_binding_receipts(
        out_dir=out_dir,
        step_summary=summary,
        resolved_input_bindings={
            source_input: {**binding, "absolute_path": str(evidence_path)}
        },
        consumed_input_keys=selection.consumed_input_keys,
    )
    assert bound["input_bindings"] == [
        {
            "input_key": source_input,
            "loaded": True,
            "evidence_id": record.evidence_id,
            "sha256": digest,
            "row_count": 3,
        }
    ]


def test_sorted_flow_preserves_original_row_coordinates(tmp_path: Path) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path, frame=_frame().iloc[[2, 0, 1]])
    out = tmp_path / "sorted_figure"
    run_cohort_flow_figure(
        out_dir=out,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )
    path = out / "cohort_accounting_source_data.csv"
    source = pd.read_csv(path)
    assert source["source_row_index"].tolist() == [1, 2, 0]
    assert source["n_remaining"].tolist() == [140, 128, 120]
    assert source["display_label"].iloc[0] == "Source universe"
    assert (
        FigureSourceDataValidator._compare_source_to_upstream(
            source_df=source,
            source_path=path,
            upstream_path=run_dir / binding["relative_path"],
        )["ok"]
        is True
    )


def test_a_single_stage_that_is_not_the_universe_still_reports_the_gap(
    tmp_path: Path,
) -> None:
    """The honest wording depends on which one-row case this is.

    A lone universe row that excluded nobody records "no filter was applied".
    A lone row that is not the universe, or that did exclude rows, leaves the
    stages before it genuinely unaccounted, and must keep saying so.
    """
    step = _step()
    singleton = pd.DataFrame(
        [[0, "inclusion", 120_000, 25_542, 94_458]],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )
    run_dir, manifest, _binding_row = _binding(tmp_path, frame=singleton)
    out_dir = run_dir / "steps" / step.step_id / "outputs"

    run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    source = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    assert source["display_label"].tolist() == ["Analysis denominator only"]
    contract = json.loads(
        (out_dir / "cohort_accounting.figure_contract.json").read_text(encoding="utf-8")
    )
    assert "upstream eligibility and attrition are " in contract["core_claim"].lower()
    assert "no eligibility filter" not in contract["core_claim"].lower()
    assert "final number of bound analysis records" in contract["reader_caption"]
    assert "no eligibility filter" not in contract["reader_caption"]
    assert "Upstream eligibility and attrition are not recorded" not in (
        out_dir / "cohort_accounting.svg"
    ).read_text(encoding="utf-8")


def test_single_denominator_is_not_promoted_to_complete_cohort_accounting(
    tmp_path: Path,
) -> None:
    step = _step()
    singleton = pd.DataFrame(
        [[0, "universe", 94_458, 0, 94_458]],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )
    run_dir, manifest, _binding_row = _binding(tmp_path, frame=singleton)
    out_dir = run_dir / "steps" / step.step_id / "outputs"

    summary = run_cohort_flow_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=step.step_id,
        figure_product="cohort_accounting",
    )

    assert summary["cohort_accounting_completeness"] == (
        COHORT_ACCOUNTING_DENOMINATOR_ONLY
    )
    assert summary["paper_grade_cohort_accounting"] is False
    assert summary["upstream_attrition_available"] is False
    assert summary["rendering_mode"] == "denominator_only_node"
    source = pd.read_csv(out_dir / "cohort_accounting_source_data.csv")
    # The single stage IS the whole bound universe with nobody excluded, so
    # the figure names that fact instead of reporting a gap that does not
    # exist. The non-promotion guarantees above are what this test protects,
    # and they are unchanged.
    assert source["display_label"].tolist() == ["All bound input rows"]
    assert source["n_remaining"].tolist() == [94_458]
    contract = json.loads(
        (out_dir / "cohort_accounting.figure_contract.json").read_text(encoding="utf-8")
    )
    panel = contract["panels"][0]
    assert panel["metadata"]["paper_grade_cohort_accounting"] is False
    assert panel["metadata"]["accounting_completeness"] == (
        COHORT_ACCOUNTING_DENOMINATOR_ONLY
    )
    assert "must not be described as complete" in panel["review_risk"]
    claim = contract["core_claim"].lower()
    assert "no eligibility filter was applied" in claim
    assert "every bound input row is the analysis cohort" in claim
    assert "no eligibility filter was applied" in contract["reader_caption"]
    assert "not a complete participant-flow diagram" in contract["reader_caption"]
    assert "No eligibility filter was applied: every bound input row" not in (
        out_dir / "cohort_accounting.svg"
    ).read_text(encoding="utf-8")


def test_owner_and_runner_fail_closed_on_widening_or_arithmetic_drift(
    tmp_path: Path,
) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
    assert not cohort_flow_figure_executor_owns_step(
        _step(
            inputs=[COHORT_FLOW_INPUT, "table:other"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=COHORT_FLOW_INPUT, mode="all_rows"
                ),
                ArtifactConsumptionContract(input_key="table:other", mode="all_rows"),
            ],
        ),
        resolved_bindings={COHORT_FLOW_INPUT: binding},
    )
    path = run_dir / str(binding["relative_path"])
    frame = pd.read_csv(path)
    frame.loc[1, "n_remaining"] = 127
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding["sha256"] = digest
    binding["identity_row"]["sha256"] = digest
    binding["consumption_contract"]["artifact_sha256"] = digest
    with pytest.raises(ValueError, match="denominator arithmetic"):
        run_cohort_flow_figure(
            out_dir=run_dir / "steps" / step.step_id / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            figure_product="cohort_accounting",
        )


def test_runner_refuses_digest_drift(tmp_path: Path) -> None:
    step = _step()
    run_dir, manifest, binding = _binding(tmp_path)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authority binding|digest"):
        run_cohort_flow_figure(
            out_dir=run_dir / "steps" / step.step_id / "outputs",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            figure_product="cohort_accounting",
        )


def _staged_frame(stage_count: int, *, long_label: bool = False) -> pd.DataFrame:
    rows = [[0, "universe", 140, 0, 140]]
    remaining = 140
    for index in range(1, stage_count):
        excluded = 1 + index % 3
        before = remaining
        remaining = before - excluded
        label = (
            "excluded_patients_receiving_renal_replacement_therapy_or_vasopressors_"
            "within_the_first_twenty_four_hours_of_icu_admission"
            if long_label and index == stage_count // 2
            else f"stage_{index}_criterion"
        )
        rows.append([index, label, before, excluded, remaining])
    return pd.DataFrame(
        rows,
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )


def _drawn_text_rows(fig, ax):
    """Every text artist as an axes-fraction extent ``(x0, y0, x1, y1, str)``."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    rows = []
    for text in ax.texts:
        extent = text.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inverse.transform(
            [(extent.x0, extent.y0), (extent.x1, extent.y1)]
        )
        rows.append((x0, y0, x1, y1, text.get_text()))
    return rows


def _assert_no_text_overlap(rows) -> None:
    # Different columns may share y coordinates. Check every pair in both
    # dimensions so adjacent and non-adjacent actual collisions are caught.
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            x_overlap = min(first[2], second[2]) - max(first[0], second[0])
            y_overlap = min(first[3], second[3]) - max(first[1], second[1])
            assert not (x_overlap > 0.004 and y_overlap > 0.004), (
                f"text boxes overlap: {first[4]!r} and {second[4]!r}"
            )


def test_compact_panel_scales_type_so_stage_text_never_overlaps() -> None:
    """A composite sub-panel has a fixed height: type must shrink, not stack."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        render_cohort_flow_axis,
    )

    frame = _staged_frame(5)
    labels = [f"Stage {i}" for i in range(len(frame))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    ax = axes[0, 0]
    render_cohort_flow_axis(ax, frame, labels, compact=True)
    rows = _drawn_text_rows(fig, ax)
    assert rows, "the panel drew no text"
    _assert_no_text_overlap(rows)
    plt.close(fig)


@pytest.mark.parametrize("stage_count", [6, 12, 48])
def test_compact_panel_grows_to_keep_every_stage_legible(
    stage_count: int,
) -> None:
    """A fixed initial panel must not impose an execution limit on the ledger."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        render_cohort_flow_axis,
    )

    frame = _staged_frame(stage_count)
    labels = [f"Stage {i}" for i in range(len(frame))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    ax = axes[0, 0]
    render_cohort_flow_axis(ax, frame, labels, compact=True)
    assert fig.get_figheight() >= 7.0
    if stage_count == 6:
        assert fig.get_figheight() <= 8.75
    else:
        assert fig.get_figheight() > 7.0
    rows = _drawn_text_rows(fig, ax)
    _assert_no_text_overlap(rows)
    assert all(label in [row[4] for row in rows] for label in labels)
    count_texts = [row[4] for row in rows if row[4].startswith("n = ")]
    assert count_texts == [f"n = {value:,}" for value in frame["n_remaining"]]
    assert all(text.get_fontsize() >= 5.4 for text in ax.texts)
    plt.close(fig)


def test_long_labels_wrap_inside_their_stage_band() -> None:
    """A very long bound label wraps to fit its node instead of bleeding out."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        render_cohort_flow_axis,
    )

    frame = _staged_frame(10, long_label=True)
    labels = [f"Stage {i}" for i in range(len(frame))]
    labels[5] = (
        "Excluded patients receiving renal replacement therapy or "
        "vasopressors within the first twenty-four hours of intensive "
        "care unit admission after landmark assessment"
    )
    fig, ax = plt.subplots(figsize=(7.2, 12.3))
    render_cohort_flow_axis(ax, frame, labels)
    rows = _drawn_text_rows(fig, ax)
    _assert_no_text_overlap(rows)
    for x0, _y0, x1, _y1, _text in rows:
        assert x0 >= -0.01 and x1 <= 1.01, "text escaped the axes"
    plt.close(fig)


def test_extreme_stage_counts_and_exclusions_stay_inside_the_axes() -> None:
    """Wide count strings and long share annotations must not clip at the edge."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
        render_cohort_flow_axis,
    )

    frame = pd.DataFrame(
        [
            [0, "universe", 9_876_543, 0, 9_876_543],
            [1, "adult", 9_876_543, 8_765_432, 1_111_111],
            [2, "first_icu_stay", 1_111_111, 1_000_000, 111_111],
        ],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    render_cohort_flow_axis(ax, frame, ["Source", "Adult", "First stay"])
    rows = _drawn_text_rows(fig, ax)
    _assert_no_text_overlap(rows)
    for x0, _y0, x1, _y1, _text in rows:
        assert x0 >= -0.01 and x1 <= 1.01, "text escaped the axes"
    plt.close(fig)
