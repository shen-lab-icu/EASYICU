"""The robustness renderer draws the replay owner's grid, and only that.

This file exists because the renderer shipped without one, and the gap hid a
live defect. It claimed a step from ``table:robustness_matrix`` alone, on the
belief that only the deterministic replay owner writes that product. Measured
2026-07-31 against the five real matrices on disk: four were Coder-authored
under three different headers, and for those the renderer would have claimed
the step and then raised at load -- four steps the Coder was drawing turned
into four dead ones. Claiming is a promise to produce the figure, so the
producer check belongs at selection, not only inside the sandbox.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _MATRIX_COLUMNS,
)
from easyicu.research_agent.execution.runners.robustness_figure_executor import (
    ROBUSTNESS_FIGURE_INPUT,
    ROBUSTNESS_PRIMARY_EFFECT_INPUT,
    ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
    robustness_figure_executor_code,
    robustness_figure_executor_owns_step,
    run_robustness_figure,
)
from easyicu.research_agent.schema import AnalysisStep

#: Two real rows from
#: batch_20260729_..._e1_fresh27, the one recorded matrix the replay owner
#: actually wrote. The second specification did not converge, which is why the
#: "labelled, not dropped" behaviour has a real case to stand on.
_REAL_ROWS = [
    {
        "spec_id": "primary",
        "effect_scale": "OR",
        "point_estimate": "1.566375890701969",
        "ci_low": "1.024509145582385",
        "ci_high": "2.3948379978371683",
        "modeled_analytic_n": "1000",
        "axis": "primary",
        "converged": "True",
        "estimability_status": "estimated",
        "membership_n": "1000",
        "membership_executable": "True",
        "notes": "Primary analysis estimate (OR) from step_summary.",
    },
    {
        "spec_id": "alt_missing_complete_case",
        "effect_scale": "OR",
        "axis": "missing",
        "converged": "False",
        "estimability_status": "not_converged",
        "membership_n": "1000",
        "membership_executable": "True",
        "notes": "Strategy is supported by the deterministic estimator adapter.",
    },
]

#: One of the three headers the Coder really wrote for the same product.
_CODER_HEADER = [
    "comparison",
    "primary_n",
    "complete_case_n",
    "n_full",
    "primary_or",
    "complete_case_or",
    "absolute_or_difference",
    "note",
]


def _step(**overrides) -> AnalysisStep:
    payload = {
        "step_id": "07_robustness_sensitivity_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Draw the locked robustness grid the replay owner refitted.",
        "inputs": [ROBUSTNESS_FIGURE_INPUT],
        "expected_outputs": ["figure:robustness_plot"],
        "method": "visualization",
        "input_consumption_contracts": [
            {
                "schema_version": "easyicu.artifact_consumption/1",
                "input_key": ROBUSTNESS_FIGURE_INPUT,
                "mode": "all_rows",
                "role_column": None,
                "expected_roles": [],
            }
        ],
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def _producer_bindings(columns=None):
    return {
        ROBUSTNESS_FIGURE_INPUT: {
            "product_contract": {"columns": list(columns or _MATRIX_COLUMNS)}
        }
    }


def _write_bound_matrix(tmp_path: Path, rows, columns=None):
    columns = list(columns or _MATRIX_COLUMNS)
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    table = run_dir / "inputs" / "robustness_matrix.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = {
        "step_id": "07_robustness_sensitivity_figure",
        "inputs": {
            ROBUSTNESS_FIGURE_INPUT: {
                "declared_kind": "table",
                "evidence_kind": "table",
                "product": "robustness_matrix",
                "evidence_id": "ev_1",
                "sha256": digest,
                "relative_path": "inputs/robustness_matrix.csv",
                "product_contract": {
                    "schema_version": "easyicu.host_typed_product.v4",
                    "columns": columns,
                    "row_count": len(rows),
                },
                "consumption_contract": {
                    "input_key": ROBUSTNESS_FIGURE_INPUT,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                    "verified_row_count": len(rows),
                },
                "identity_row": {
                    "input_key": ROBUSTNESS_FIGURE_INPUT,
                    "product": "robustness_matrix",
                    "sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


def _bind_statistic(run_dir, manifest, input_key, value):
    product = input_key.split(":", 1)[1]
    path = run_dir / "inputs" / f"{product}.json"
    path.write_text(
        json.dumps({"statistic": product, "value": value}),
        encoding="utf-8",
    )
    manifest["inputs"][input_key] = {
        "declared_kind": "statistic",
        "evidence_kind": "statistic",
        "product": product,
        "evidence_id": f"ev_{product}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "relative_path": str(path.relative_to(run_dir)),
    }


def test_it_claims_a_step_bound_to_the_replay_owners_matrix():
    assert robustness_figure_executor_owns_step(
        _step(), resolved_bindings=_producer_bindings()
    )


def test_it_claims_the_normalized_primary_effect_input():
    step = _step(
        inputs=[ROBUSTNESS_FIGURE_INPUT, ROBUSTNESS_PRIMARY_EFFECT_INPUT],
        input_consumption_contracts=[
            {
                "schema_version": "easyicu.artifact_consumption/1",
                "input_key": ROBUSTNESS_FIGURE_INPUT,
                "mode": "all_rows",
                "role_column": None,
                "expected_roles": [],
            }
        ],
    )
    bindings = {
        **_producer_bindings(),
        ROBUSTNESS_PRIMARY_EFFECT_INPUT: {},
    }

    assert robustness_figure_executor_owns_step(step, resolved_bindings=bindings)


def test_the_figure_product_name_does_not_decide_ownership():
    for product in (
        "figure:robustness",
        "figure:robustness_sensitivity",
        "figure:complete_case_robustness",
        "figure:a_synonym_no_planner_has_used_yet",
    ):
        assert robustness_figure_executor_owns_step(
            _step(expected_outputs=[product]), resolved_bindings=_producer_bindings()
        ), product


def test_it_declines_a_matrix_the_coder_shaped():
    """The defect this file was written for: four real matrices look like this."""

    assert not robustness_figure_executor_owns_step(
        _step(), resolved_bindings=_producer_bindings(columns=_CODER_HEADER)
    )


def test_it_declines_a_coder_matrix_that_shares_the_columns_it_reads():
    """Reading the columns it needs is not the same as being the producer's.

    A Coder table can carry ``spec_id``/``point_estimate``/``ci_low`` and still
    mean something else -- a two-by-two audit, a complete-case comparison. What
    makes the grid interpretable is the whole locked contract.
    """

    partial = [
        "spec_id",
        "effect_scale",
        "point_estimate",
        "ci_low",
        "ci_high",
        "axis",
        "converged",
    ]
    assert not robustness_figure_executor_owns_step(
        _step(), resolved_bindings=_producer_bindings(columns=partial)
    )


def test_it_declines_when_no_binding_map_was_supplied():
    assert not robustness_figure_executor_owns_step(_step(), resolved_bindings=None)


def test_it_declines_a_step_that_also_freezes_a_clustering():
    assert not robustness_figure_executor_owns_step(
        _step(trajectory_stability_spec={"n_resamples": 10, "sample_fraction": 0.8}),
        resolved_bindings=_producer_bindings(),
    )


def test_the_entrypoint_does_not_re_derive_ownership():
    """It cannot see the bindings the selector had, so it must not try.

    Re-deriving here is what broke the wiring the moment the producer clause
    landed: the selector said yes, the builder said no, and the step died
    between them.
    """

    code = robustness_figure_executor_code(_step())
    assert "run_robustness_figure(" in code
    assert "figure_product='robustness_plot'" in code


def test_it_renders_the_real_grid_and_labels_what_did_not_converge(tmp_path):
    run_dir, manifest = _write_bound_matrix(tmp_path, _REAL_ROWS)
    summary = run_robustness_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="07_robustness_sensitivity_figure",
        figure_product="robustness_plot",
    )
    assert summary["status"] == "ok"
    assert summary["effect_scale"] == "OR"
    # The line at no effect. ``OR`` is the producer's own spelling and was
    # unrecognised until 2026-07-31, so this forest was drawn with no anchor
    # for a reader to judge an interval against, and nothing recorded that.
    assert summary["null_line_drawn"] is True
    assert summary["specifications_drawn"] == 1
    assert summary["specifications_not_estimable"] == 1
    assert summary["any_specification_not_estimable"] is True
    # The failed specification still travels with the figure.
    source = (tmp_path / "out" / "robustness_plot_source_data.csv").read_text()
    assert "alt_missing_complete_case" in source
    assert (tmp_path / "out" / "robustness_plot.png").is_file()
    contract = json.loads(
        (tmp_path / "out" / "robustness_plot.figure_contract.json").read_text()
    )
    assert contract["panels"][0]["metadata"]["chart_type"] == "sensitivity_forest"


def test_it_renders_the_normalized_primary_effect_anchor(tmp_path):
    run_dir, manifest = _write_bound_matrix(tmp_path, _REAL_ROWS)
    _bind_statistic(run_dir, manifest, ROBUSTNESS_PRIMARY_EFFECT_INPUT, 1.566)

    summary = run_robustness_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="07_robustness_sensitivity_figure",
        figure_product="robustness_plot",
    )

    assert summary["anchor_input_bound"] is True
    assert summary["anchor_line_drawn"] is True
    source = pd.read_csv(
        tmp_path / "out" / "robustness_plot_bound_statistics_source_data.csv"
    )
    assert source.to_dict("records") == [
        {"statistic": "primary_effect", "value": 1.566}
    ]


def test_it_refuses_conflicting_primary_effect_aliases(tmp_path):
    run_dir, manifest = _write_bound_matrix(tmp_path, _REAL_ROWS)
    _bind_statistic(run_dir, manifest, ROBUSTNESS_PRIMARY_EFFECT_INPUT, 1.566)
    _bind_statistic(run_dir, manifest, ROBUSTNESS_PRIMARY_ESTIMATE_INPUT, 9.999)

    with pytest.raises(ValueError, match="bindings disagree"):
        run_robustness_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="07_robustness_sensitivity_figure",
            figure_product="robustness_plot",
        )


def test_it_refuses_to_draw_a_matrix_the_replay_owner_did_not_write(tmp_path):
    """The render-time half, checked on its own rather than assumed."""

    run_dir, manifest = _write_bound_matrix(
        tmp_path,
        [{name: "1" for name in _CODER_HEADER}],
        columns=_CODER_HEADER,
    )
    with pytest.raises(ValueError, match="deterministic replay owner"):
        run_robustness_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="07_robustness_sensitivity_figure",
            figure_product="robustness_plot",
        )


def test_it_refuses_a_digest_that_does_not_match_the_bytes(tmp_path):
    run_dir, manifest = _write_bound_matrix(tmp_path, _REAL_ROWS)
    table = run_dir / "inputs" / "robustness_matrix.csv"
    table.write_text(table.read_text().replace("1.566", "9.999"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest verification failed"):
        run_robustness_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="07_robustness_sensitivity_figure",
            figure_product="robustness_plot",
        )


def test_it_refuses_a_grid_that_mixes_two_effect_scales(tmp_path):
    run_dir, manifest = _write_bound_matrix(
        tmp_path,
        [_REAL_ROWS[0], {**_REAL_ROWS[0], "spec_id": "alt", "effect_scale": "RD"}],
    )
    with pytest.raises(ValueError, match="mixes effect scales"):
        run_robustness_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="07_robustness_sensitivity_figure",
            figure_product="robustness_plot",
        )
