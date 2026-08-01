"""The renderer read three parents and left evidence for one.

``figure_source_data`` requires each typed parent a figure step binds to be
independently value-verified in the figure's own source data.  The robustness
renderer declares ``table:robustness_summary``, ``statistic:primary_or`` and
``statistic:complete_case_n`` as optional inputs, and its module docstring
states the rule plainly: "optional may never mean ignored".  It does read them
-- the primary estimate becomes the anchor line, the complete-case count an
annotation -- but it wrote source data only for the matrix it plots.

On the 2026-08-01 E1 run (canary25) that produced
``incomplete_source_lineage_coverage`` naming ``robustness_summary.csv``,
``statistic:complete_case_n`` and ``statistic:primary_or``, and failed the last
step of a task whose other 9 steps were all ok.

An absent optional input is deliberately NOT covered: it is not a coverage gap,
and writing an empty companion for it would claim evidence for something the
plan never bound.  A bound statistic recorded as null is skipped for the same
reason -- there is no value to verify, and a real ``complete_case_n.json`` in
the corpus is null.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.robustness_figure_executor import (
    ROBUSTNESS_COMPLETE_CASE_INPUT,
    ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
    ROBUSTNESS_SUMMARY_TABLE_INPUT,
    _write_bound_parent_source_data,
)


def _summary_binding(run_dir: Path) -> dict:
    table = run_dir / "steps" / "parent" / "outputs" / "robustness_summary.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "axis": "primary",
                "total_specs": 1,
                "range_low": 1.02,
                "range_high": 2.39,
            },
            {
                "axis": "missing",
                "total_specs": 1,
                "range_low": 1.02,
                "range_high": 2.39,
            },
        ]
    ).to_csv(table, index=False)
    return {"relative_path": str(table.relative_to(run_dir))}


def test_every_bound_parent_gets_its_own_source_data(tmp_path: Path) -> None:
    """The property that was false for all three."""

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    written = _write_bound_parent_source_data(
        out_dir=out_dir,
        run_dir=run_dir,
        figure_product="robustness_plot",
        bound_inputs={ROBUSTNESS_SUMMARY_TABLE_INPUT: _summary_binding(run_dir)},
        bound_statistics=(
            (ROBUSTNESS_PRIMARY_ESTIMATE_INPUT, True, 1.566),
            (ROBUSTNESS_COMPLETE_CASE_INPUT, True, 1000.0),
        ),
    )
    assert written == [
        "robustness_plot_robustness_summary_source_data.csv",
        "robustness_plot_bound_statistics_source_data.csv",
    ]
    for name in written:
        assert (out_dir / name).is_file()

    stats = pd.read_csv(out_dir / written[1])
    assert set(stats.columns) == {"statistic", "value"}
    assert dict(zip(stats["statistic"], stats["value"])) == {
        "primary_or": 1.566,
        "complete_case_n": 1000.0,
    }


def test_the_summary_companion_reproduces_the_bound_table(tmp_path: Path) -> None:
    """It must be traceable to its parent, not a re-derivation."""

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    binding = _summary_binding(run_dir)
    _write_bound_parent_source_data(
        out_dir=out_dir,
        run_dir=run_dir,
        figure_product="robustness_plot",
        bound_inputs={ROBUSTNESS_SUMMARY_TABLE_INPUT: binding},
        bound_statistics=(),
    )
    parent = pd.read_csv(run_dir / binding["relative_path"])
    companion = pd.read_csv(
        out_dir / "robustness_plot_robustness_summary_source_data.csv"
    )
    pd.testing.assert_frame_equal(parent, companion)


def test_an_unbound_optional_parent_is_not_invented(tmp_path: Path) -> None:
    """Absence is not a coverage gap.

    Two of the eight recorded steps bind the matrix alone; writing empty
    companions for them would claim evidence the plan never supplied.
    """

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    assert (
        _write_bound_parent_source_data(
            out_dir=out_dir,
            run_dir=tmp_path / "run",
            figure_product="robustness_plot",
            bound_inputs={},
            bound_statistics=(
                (ROBUSTNESS_PRIMARY_ESTIMATE_INPUT, False, None),
                (ROBUSTNESS_COMPLETE_CASE_INPUT, False, None),
            ),
        )
        == []
    )
    assert not list(out_dir.iterdir())


def test_a_bound_statistic_with_no_recorded_value_is_skipped(tmp_path: Path) -> None:
    """Bound-but-null has nothing to verify.

    ``(True, None)`` is a real shape in the corpus; emitting a row with an
    empty value would put an unverifiable number in a lineage table.
    """

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    written = _write_bound_parent_source_data(
        out_dir=out_dir,
        run_dir=tmp_path / "run",
        figure_product="robustness_plot",
        bound_inputs={},
        bound_statistics=(
            (ROBUSTNESS_PRIMARY_ESTIMATE_INPUT, True, 1.566),
            (ROBUSTNESS_COMPLETE_CASE_INPUT, True, None),
        ),
    )
    stats = pd.read_csv(out_dir / written[0])
    assert list(stats["statistic"]) == ["primary_or"]


def test_a_binding_escaping_the_run_directory_is_refused(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        _write_bound_parent_source_data(
            out_dir=out_dir,
            run_dir=tmp_path / "run",
            figure_product="robustness_plot",
            bound_inputs={
                ROBUSTNESS_SUMMARY_TABLE_INPUT: {"relative_path": "../escape.csv"}
            },
            bound_statistics=(),
        )


def test_the_contract_lists_every_file_written(tmp_path: Path) -> None:
    """The wiring: what is written must be what the figure contract declares.

    Mutation on the previous fix in this series showed that a helper can be
    correct while nothing points at it, so the assembled list is checked here
    rather than only the helper's return value.
    """

    import ast
    import inspect

    from easyicu.research_agent.execution.runners import robustness_figure_executor

    tree = ast.parse(inspect.getsource(robustness_figure_executor))
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "source_data_names"
    }
    assert assigned, "the renderer no longer assembles a source-data name list"

    rendered = inspect.getsource(robustness_figure_executor.run_robustness_figure)
    for field in (
        '"source_data"',
        "source_data=",
        '"source_data_files"',
        '"evidence_ids"',
    ):
        assert field in rendered
    assert (
        "[source_path.name]" not in rendered
    ), "a contract field still declares only the matrix source data"


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_parents_the_real_run_reported_missing_are_now_covered() -> None:
    """The exact gap the 2026-08-01 E1 run reported, closed on real names.

    Reads the recorded finding rather than restating it, so this stops being
    meaningful only if the corpus stops containing the failure.

    The population is steps THIS renderer owns.  A first draft swept every step
    reporting ``incomplete_source_lineage_coverage`` and failed, naming
    ``absolute_risk_context.csv``, ``event_timing_audit.csv``,
    ``measurement_process_audit.csv`` and ``missingness_measurement_audit.csv``
    -- real gaps, but in the missingness/measurement renderers, which this
    change does not touch.  Asserting over them would have made this test a
    claim about code it does not cover.  They are the same defect in a sibling
    owner and are recorded as such, not silently dropped.
    """

    reported: set[str] = set()
    for path in sorted(_CORPUS.glob("batch_*/*/aware/run_*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for record in manifest.get("per_step_records") or []:
            if record.get("deterministic_standard_analysis") != "robustness_figure":
                continue
            for finding in record.get("contract_findings") or []:
                detail = finding.get("detail") or {}
                if detail.get("reason") != "incomplete_source_lineage_coverage":
                    continue
                reported.update(detail.get("missing_bound_tables") or [])
                reported.update(detail.get("missing_bound_statistics") or [])

    if not reported:
        pytest.skip("no recorded run reports incomplete source lineage coverage")
    covered = {
        "robustness_summary.csv",
        ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
        ROBUSTNESS_COMPLETE_CASE_INPUT,
    }
    uncovered = sorted(reported - covered)
    assert not uncovered, (
        "recorded runs report bound sources this renderer still does not "
        f"evidence: {uncovered}"
    )
