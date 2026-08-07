"""The renderer refused products its own producer had written.

The deterministic replay owner registers eight products
(``ROBUSTNESS_REPLAY_OUTPUT_FILES``); the figure renderer's typed-input
capability listed five.  A plan binding one of the other three was refused --
silently, because declining is how a renderer says no -- and the figure went to
the Coder, whose hand-written source-data bundle then failed
``figure_source_data`` lineage and killed the step.

MEASURED 2026-08-03 over every recorded plan on disk, deduplicated to one row
per (run, step) at the highest revision: 364 visualization steps consume
``table:robustness_matrix``.  308 were admitted, 56 refused, and **38 of the 56
(68%) were refused only for products this renderer's own producer wrote**.  The
specification grid is the single largest cause at 22.

The three real spellings the grid is promised under -- ``table:robustness_grid``
(x20), ``table:specification_grid`` (x4) and the internal stem -- are why the
recognition is by CONTRACT and not by name.  The binding's ``product`` field
carries whichever name the Planner chose, so a check keyed on it would own two
spellings and abandon the third the next time the Planner invents one.

What stays refused is a table from a DIFFERENT producer.  The remaining 18
refusals bind ``table:adjusted_association_estimates``,
``table:exposure_outcome_distribution`` and friends: those steps are asking for
a composite figure this renderer does not draw, and claiming them would be a
promise it cannot keep.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _MATRIX_COLUMNS,
    _SPECIFICATION_GRID_COLUMNS,
    ROBUSTNESS_REPLAY_OUTPUT_FILES,
)
from easyicu.research_agent.execution.runners.robustness_figure_executor import (
    ROBUSTNESS_FIGURE_INPUT,
    _specification_grid_key,
    robustness_figure_consumed_input_keys,
    robustness_figure_executor_owns_step,
    run_robustness_figure,
)
from easyicu.research_agent.schema import AnalysisStep

_GRID_FILENAME = ROBUSTNESS_REPLAY_OUTPUT_FILES["specification_grid"]

#: The two spellings real plans promise the one grid file under, plus a third
#: nobody has used, because the point of the fix is that the name is not what
#: decides.
_REAL_GRID_KEYS = ("table:robustness_grid", "table:specification_grid")

#: One real grid row, from the 2026-08-03 e2 run.  ``json_normalize`` flattens
#: the override dicts, so the header is wider than the three guaranteed columns
#: -- which is why the contract check is containment.
_REAL_GRID_HEADER = [
    *_SPECIFICATION_GRID_COLUMNS,
    "cohort_override",
    "outcome_override",
    "missing_override.strategy",
    "missing_override.variables",
]
_REAL_GRID_ROW = {
    "spec_id": "complete_case_required_variables",
    "axis": "missing",
    "description": (
        "Repeat the locked primary estimand using complete cases for the "
        "primary exposure, outcome, and prespecified covariates without "
        "imputing the exposure or outcome."
    ),
    "cohort_override": "",
    "outcome_override": "",
    "missing_override.strategy": "complete_case",
    "missing_override.variables": "['lact_max', 'death', 'age', 'sex']",
}

_MATRIX_ROWS = [
    {
        "spec_id": "primary",
        "effect_scale": "OR",
        "point_estimate": "1.3414550655",
        "ci_low": "1.3286946705",
        "ci_high": "1.3543380076",
        "axis": "primary",
        "converged": "True",
    },
    {
        "spec_id": "complete_case_required_variables",
        "effect_scale": "OR",
        "point_estimate": "1.3414550655",
        "ci_low": "1.3286946705",
        "ci_high": "1.3543380076",
        "axis": "missing",
        "converged": "True",
    },
]


def _consumption(input_key: str) -> dict:
    return {
        "schema_version": "easyicu.artifact_consumption/1",
        "input_key": input_key,
        "mode": "all_rows",
        "role_column": None,
        "expected_roles": [],
    }


def _step(inputs, **overrides) -> AnalysisStep:
    payload = {
        "step_id": "06_robustness_replay_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Draw the locked robustness grid the replay owner refitted.",
        "inputs": list(inputs),
        "expected_outputs": ["figure:robustness_plot"],
        "method": "visualization",
        "input_consumption_contracts": [
            _consumption(key) for key in inputs if key.startswith("table:")
        ],
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def _write_csv(path: Path, header, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_binding(run_dir: Path) -> dict:
    relative = "inputs/robustness_matrix.csv"
    digest = _write_csv(run_dir / relative, _MATRIX_COLUMNS, _MATRIX_ROWS)
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": "robustness_matrix",
        "evidence_id": "ev_matrix",
        "sha256": digest,
        "relative_path": relative,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "columns": list(_MATRIX_COLUMNS),
            "row_count": len(_MATRIX_ROWS),
        },
        "consumption_contract": {
            "input_key": ROBUSTNESS_FIGURE_INPUT,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(_MATRIX_ROWS),
        },
        "identity_row": {
            "input_key": ROBUSTNESS_FIGURE_INPUT,
            "product": "robustness_matrix",
            "sha256": digest,
        },
    }


def _grid_binding(run_dir: Path, *, promised_product: str, filename=None) -> dict:
    """A binding shaped exactly like the host writes for the producer's grid."""

    relative = f"inputs/ev_grid__{filename or _GRID_FILENAME}"
    digest = _write_csv(run_dir / relative, _REAL_GRID_HEADER, [_REAL_GRID_ROW])
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        # The Planner's chosen name travels here, and it is NOT what the
        # renderer keys on.
        "product": promised_product,
        "evidence_id": "ev_grid",
        "sha256": digest,
        "relative_path": relative,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "columns": list(_REAL_GRID_HEADER),
            "row_count": 1,
        },
    }


@pytest.mark.parametrize("grid_key", _REAL_GRID_KEYS)
def test_a_step_binding_the_producers_grid_is_claimed(tmp_path: Path, grid_key) -> None:
    """The property that was false: 22 of 56 refusals were exactly this."""

    run_dir = tmp_path / "run"
    bindings = {
        ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
        grid_key: _grid_binding(run_dir, promised_product=grid_key.split(":", 1)[-1]),
    }
    step = _step([ROBUSTNESS_FIGURE_INPUT, grid_key])
    assert robustness_figure_executor_owns_step(step, resolved_bindings=bindings)
    assert grid_key in robustness_figure_consumed_input_keys(bindings)


def test_a_key_no_planner_has_used_is_claimed_on_its_contract(tmp_path: Path) -> None:
    """Recognition is by the producer's contract, not by a name set.

    The whole reason the grid was unreachable is that its key is whatever the
    Planner promised.  A check that enumerated today's two spellings would fail
    again on the third.
    """

    run_dir = tmp_path / "run"
    invented = "table:a_grid_spelling_no_planner_has_used_yet"
    bindings = {
        ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
        invented: _grid_binding(run_dir, promised_product="whatever_it_was_called"),
    }
    assert _specification_grid_key(bindings) == invented
    assert robustness_figure_executor_owns_step(
        _step([ROBUSTNESS_FIGURE_INPUT, invented]), resolved_bindings=bindings
    )


def test_a_table_from_another_producer_is_still_refused(tmp_path: Path) -> None:
    """The 18 refusals that must stay refused.

    ``table:adjusted_association_estimates`` is the largest of them.  A step
    binding it wants a composite figure this renderer does not draw, and
    claiming it would be a promise it cannot keep.
    """

    run_dir = tmp_path / "run"
    foreign = "table:adjusted_association_estimates"
    relative = "inputs/adjusted_association.csv"
    digest = _write_csv(
        run_dir / relative,
        ["model_label", "estimate", "ci_low", "ci_high"],
        [{"model_label": "primary", "estimate": "1.3"}],
    )
    bindings = {
        ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
        foreign: {
            "declared_kind": "table",
            "evidence_kind": "table",
            "product": "adjusted_association_estimates",
            "sha256": digest,
            "relative_path": relative,
            "product_contract": {"columns": ["model_label", "estimate"]},
        },
    }
    assert _specification_grid_key(bindings) is None
    assert not robustness_figure_executor_owns_step(
        _step([ROBUSTNESS_FIGURE_INPUT, foreign]), resolved_bindings=bindings
    )


def test_a_coder_table_carrying_the_grids_columns_is_refused(tmp_path: Path) -> None:
    """Header alone is not the producer.

    A Coder-written table can carry ``spec_id``/``axis``/``description``.  What
    it cannot do is be the file the producer wrote, which is why the stem is
    checked too -- the same reasoning that put the producer clause on the
    matrix.
    """

    run_dir = tmp_path / "run"
    key = "table:robustness_grid"
    bindings = {
        ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
        key: _grid_binding(
            run_dir,
            promised_product="robustness_grid",
            filename="coder_notes_about_specifications.csv",
        ),
    }
    assert _specification_grid_key(bindings) is None
    assert not robustness_figure_executor_owns_step(
        _step([ROBUSTNESS_FIGURE_INPUT, key]), resolved_bindings=bindings
    )


def test_two_keys_bound_to_one_grid_decline_rather_than_guess(tmp_path: Path) -> None:
    """Ambiguity must not buy ownership.

    One file promised as two products leaves the renderer unable to say which
    of the two the figure's lineage belongs to.  Declining is what already
    happened before the grid was readable at all, so this is not a regression
    -- it is the fail-closed edge of a widened capability.
    """

    run_dir = tmp_path / "run"
    bindings = {
        ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
        "table:robustness_grid": _grid_binding(
            run_dir, promised_product="robustness_grid"
        ),
        "table:specification_grid": _grid_binding(
            run_dir, promised_product="specification_grid"
        ),
    }
    assert _specification_grid_key(bindings) is None
    assert not robustness_figure_executor_owns_step(
        _step(
            [
                ROBUSTNESS_FIGURE_INPUT,
                "table:robustness_grid",
                "table:specification_grid",
            ]
        ),
        resolved_bindings=bindings,
    )


def test_the_grid_is_read_not_waved_through(tmp_path: Path) -> None:
    """Claiming a bound input obliges the renderer to use it.

    ``figure_source_data`` requires every bound parent to be independently
    value-verified, so the grid needs its own source-data companion; and the
    sentence the plan registered for each specification has to reach the
    reader.  It reaches them as a figure note, NOT as a tick label: measured
    over the 65 recorded descriptions they run 95 to 232 characters, and set on
    the axis they pushed the plot into the right quarter of the canvas.
    """

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    grid_key = "table:robustness_grid"
    manifest = {
        "step_id": "06_robustness_replay_figure",
        "inputs": {
            ROBUSTNESS_FIGURE_INPUT: _matrix_binding(run_dir),
            grid_key: _grid_binding(run_dir, promised_product="robustness_grid"),
        },
    }
    summary = run_robustness_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="06_robustness_replay_figure",
        figure_product="robustness_plot",
    )
    assert summary["status"] == "ok"

    companion = "robustness_plot_specification_grid_source_data.csv"
    assert companion in summary["source_data_files"]
    written = pd.read_csv(out_dir / companion)
    # Verbatim, so every plotted row traces to one parent row.
    assert list(written.columns) == _REAL_GRID_HEADER
    assert written.loc[0, "description"] == _REAL_GRID_ROW["description"]

    contract = pd.read_json(out_dir / "robustness_plot.figure_contract.json", typ="series")
    note = str(contract["statistics_note"])
    assert _REAL_GRID_ROW["description"] in note
    assert "complete case required variables" in note
