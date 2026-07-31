"""The adjusted-association renderer draws the host's own fit, and only that.

The defect this renderer exists to avoid is not a crash. Across the recorded
corpus ``adjusted_association_estimates.csv`` carries twelve distinct headers,
because the Coder invented one per run: ``estimate`` in some, ``odds_ratio`` in
others. A renderer that read the input key and guessed which column held the
effect would draw a number under the wrong label and nothing would fail. So the
tests below are mostly about *declining*, and every one of them is written so
that deleting the clause it covers makes it fail.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
)
from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import (
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
    adjusted_association_figure_executor_code,
    adjusted_association_figure_executor_owns_step,
    run_adjusted_association_figure,
)
from easyicu.research_agent.execution.runners.effect_scale import (
    describe_effect_scale,
)
from easyicu.research_agent.schema import AnalysisStep

#: One real host-produced row, copied from
#: batch_20260730_..._canonical9_full02/e2_lactate_mortality. Keeping a real row
#: rather than a tidy invented one is deliberate: it is what the producer
#: actually writes, semicolon-joined covariates and all.
_REAL_ROW = {
    "fit_status": "fitted",
    "estimate": "1.3572170056325177",
    "ci_low": "1.2396563297252763",
    "ci_high": "1.4859263460432757",
    "effect_scale": "odds_ratio",
    "exposure": "lact_max",
    "requirement_id": "primary_logistic_lact_max_death",
    "outcome": "death",
    "covariates": "age;sex",
    "estimator_kind": "logistic",
    "analysis_set": "complete_case",
    "n": "515",
    "n_events": "102",
    "standard_error": "0.046226408621338526",
    "notes": "sex treatment-coded against 'Female'",
}

#: One of the twelve headers the Coder really wrote for the same product.
_CODER_HEADER = [
    "model_id",
    "outcome",
    "exposure",
    "effect_scale",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "n",
    "event_n",
    "fit_status",
    "feasibility_note",
]


def _step(**overrides) -> AnalysisStep:
    payload = {
        "step_id": "08_adjusted_effect_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Draw the adjusted association the model owner fitted.",
        "inputs": [ADJUSTED_ASSOCIATION_FIGURE_INPUT],
        "expected_outputs": ["figure:adjusted_effect"],
        "method": "visualization",
        "input_consumption_contracts": [
            {
                "schema_version": "easyicu.artifact_consumption/1",
                "input_key": ADJUSTED_ASSOCIATION_FIGURE_INPUT,
                "mode": "all_rows",
                "role_column": None,
                "expected_roles": [],
            }
        ],
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def _host_bindings(columns=None):
    return {
        ADJUSTED_ASSOCIATION_FIGURE_INPUT: {
            "product_contract": {
                "columns": list(columns or ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)
            }
        }
    }


def _write_bound_table(tmp_path: Path, rows, columns=None) -> tuple[Path, dict]:
    """Write a bound estimates CSV and the manifest the sandbox would receive."""

    columns = list(columns or ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    table = run_dir / "inputs" / "adjusted_association_estimates.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = {
        "step_id": "08_adjusted_effect_figure",
        "inputs": {
            ADJUSTED_ASSOCIATION_FIGURE_INPUT: {
                "declared_kind": "table",
                "evidence_kind": "table",
                "product": "adjusted_association_estimates",
                "evidence_id": "ev_1",
                "sha256": digest,
                "relative_path": "inputs/adjusted_association_estimates.csv",
                "product_contract": {
                    "schema_version": "easyicu.host_typed_product.v4",
                    "columns": columns,
                    "row_count": len(rows),
                },
                "consumption_contract": {
                    "input_key": ADJUSTED_ASSOCIATION_FIGURE_INPUT,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                    "verified_row_count": len(rows),
                },
                "identity_row": {
                    "input_key": ADJUSTED_ASSOCIATION_FIGURE_INPUT,
                    "product": "adjusted_association_estimates",
                    "sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


# --------------------------------------------------------------------------
# ownership
# --------------------------------------------------------------------------


def test_it_claims_a_step_bound_to_the_hosts_own_estimates_table():
    assert adjusted_association_figure_executor_owns_step(
        _step(), resolved_bindings=_host_bindings()
    )


def test_the_figure_product_name_does_not_decide_ownership():
    """Eighteen names for this figure appear in the corpus; none may be keyed on."""

    for product in (
        "figure:primary_effect",
        "figure:adjusted_association_forest",
        "figure:primary_sepsis_mortality",
        "figure:a_name_no_planner_has_used_yet",
    ):
        assert adjusted_association_figure_executor_owns_step(
            _step(expected_outputs=[product]), resolved_bindings=_host_bindings()
        ), product


def test_it_declines_a_table_the_coder_shaped():
    """The whole point: a familiar key over an unfamiliar header is a decline."""

    assert not adjusted_association_figure_executor_owns_step(
        _step(), resolved_bindings=_host_bindings(columns=_CODER_HEADER)
    )


def test_a_coder_header_that_merely_contains_estimate_and_ci_is_not_enough():
    """Overlapping a few column names is not the producer's contract.

    This is the header a name-matching renderer would have accepted: it has an
    ``estimate``, a ``ci_low`` and a ``ci_high``, and nothing that says on what
    scale or adjusted for what.
    """

    assert not adjusted_association_figure_executor_owns_step(
        _step(),
        resolved_bindings=_host_bindings(
            columns=["estimate", "ci_low", "ci_high", "effect_scale", "outcome"]
        ),
    )


def test_a_producer_may_add_columns_without_losing_its_renderer():
    assert adjusted_association_figure_executor_owns_step(
        _step(),
        resolved_bindings=_host_bindings(
            columns=[*ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS, "future_diagnostic"]
        ),
    )


def test_it_declines_when_no_binding_map_was_supplied():
    """Without the bound contract the renderer cannot know whose table it is."""

    assert not adjusted_association_figure_executor_owns_step(
        _step(), resolved_bindings=None
    )


def test_it_declines_when_the_binding_carries_no_product_contract():
    assert not adjusted_association_figure_executor_owns_step(
        _step(),
        resolved_bindings={ADJUSTED_ASSOCIATION_FIGURE_INPUT: {"sha256": "x" * 64}},
    )


def test_it_declines_a_second_table_it_cannot_read():
    """Three recorded steps bind a sensitivity or absolute-risk table too.

    Both are Coder-authored at six headers between them, so the renderer cannot
    draw them -- and a step that binds one is asking for a figure that shows it.
    """

    step = _step(
        inputs=[ADJUSTED_ASSOCIATION_FIGURE_INPUT, "table:absolute_risk_context"],
        input_consumption_contracts=[
            {
                "schema_version": "easyicu.artifact_consumption/1",
                "input_key": key,
                "mode": "all_rows",
                "role_column": None,
                "expected_roles": [],
            }
            for key in (
                ADJUSTED_ASSOCIATION_FIGURE_INPUT,
                "table:absolute_risk_context",
            )
        ],
    )
    assert not adjusted_association_figure_executor_owns_step(
        step, resolved_bindings=_host_bindings()
    )


def test_a_rendering_step_cannot_declare_a_model_or_a_table_one_at_all():
    """The two guards this predicate does NOT carry, and why it need not.

    An earlier draft refused ``model_requirements`` and ``table_one_spec`` here.
    Both are unreachable: ``AnalysisStep`` itself refuses them on a
    visualization step whose sole output is one figure, so the clauses were
    deleted rather than left reading as protection. This test is what makes
    that deletion safe -- if the schema ever stops refusing them, it fails and
    the guards have to come back.
    """

    with pytest.raises(ValueError, match="model_requirements"):
        _step(
            model_requirements=[
                {
                    "requirement_id": "primary",
                    "outcome": "death",
                    "outcome_type": "binary",
                    "method_family": "logistic_regression",
                    "exposure_source": "lact_max",
                    "analysis_role": "primary",
                    "analysis_set": "complete_case",
                }
            ]
        )
    with pytest.raises(ValueError, match="table:table_one"):
        _step(
            table_one_spec={
                "group_by": "arm",
                "group_levels": ["a", "b"],
                "variables": [
                    {
                        "name": "age",
                        "variable_kind": "continuous",
                        "summary": "median_iqr",
                        "test": "mann_whitney_or_kruskal",
                    }
                ],
            }
        )


def test_it_declines_a_step_that_also_freezes_a_clustering():
    """The one scientific declaration that CAN reach this predicate."""

    assert not adjusted_association_figure_executor_owns_step(
        _step(trajectory_stability_spec={"n_resamples": 10, "sample_fraction": 0.8}),
        resolved_bindings=_host_bindings(),
    )


def test_it_declines_a_step_promising_two_figures():
    assert not adjusted_association_figure_executor_owns_step(
        _step(expected_outputs=["figure:adjusted_effect", "figure:something_else"]),
        resolved_bindings=_host_bindings(),
    )


def test_the_entrypoint_names_the_declared_product_and_step():
    code = adjusted_association_figure_executor_code(
        _step(expected_outputs=["figure:primary_effect"])
    )
    assert "run_adjusted_association_figure(" in code
    assert "figure_product='primary_effect'" in code
    assert "step_id='08_adjusted_effect_figure'" in code


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_it_renders_the_real_row_and_reports_what_it_drew(tmp_path):
    run_dir, manifest = _write_bound_table(tmp_path, [_REAL_ROW])
    summary = run_adjusted_association_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="08_adjusted_effect_figure",
        figure_product="adjusted_effect",
    )
    assert summary["status"] == "ok"
    assert summary["exposure"] == "lact_max"
    assert summary["outcome"] == "death"
    assert summary["effect_scale"] == "odds_ratio"
    assert summary["estimates_drawn"] == 1
    assert summary["estimates_not_drawn"] == 0
    # An odds ratio is multiplicative, so the axis has to be.
    assert summary["axis_scale"] == "log"
    assert summary["adjustment_note"] == "Adjusted for age, sex."
    assert (tmp_path / "out" / "adjusted_effect.png").is_file()
    contract = json.loads(
        (tmp_path / "out" / "adjusted_effect.figure_contract.json").read_text()
    )
    assert "age" in json.dumps(contract) and "sex" in json.dumps(contract)


def test_an_unadjusted_model_says_so_rather_than_going_quiet(tmp_path):
    """A silent caption cannot be told apart from an adjustment nobody recorded."""

    run_dir, manifest = _write_bound_table(tmp_path, [{**_REAL_ROW, "covariates": ""}])
    summary = run_adjusted_association_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="08_adjusted_effect_figure",
        figure_product="adjusted_effect",
    )
    assert summary["adjustment_note"] == (
        "Unadjusted: the model declared no covariates."
    )


def test_a_failed_fit_that_still_carries_numbers_is_not_drawn(tmp_path):
    """``fit_status`` decides, not whether the row happens to hold numbers.

    This is the case the guard exists for and the one a careless test misses. A
    separation-detected logistic fit does not leave the estimate blank -- it
    leaves an enormous one with an interval to match. Plotting it would put the
    study's most extreme point on the figure as though it were an estimate.
    An earlier version of this test blanked the numbers too, so deleting the
    ``fit_status`` check still passed: the row was undrawable for the other
    reason.
    """

    rows = [
        _REAL_ROW,
        {
            **_REAL_ROW,
            "fit_status": "separation_detected",
            "estimate": "2.9e7",
            "ci_low": "1.0e-8",
            "ci_high": "8.4e22",
            "analysis_set": "source_aware",
        },
    ]
    run_dir, manifest = _write_bound_table(tmp_path, rows)
    summary = run_adjusted_association_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="08_adjusted_effect_figure",
        figure_product="adjusted_effect",
    )
    assert summary["estimates_drawn"] == 1
    assert summary["estimates_not_drawn"] == 1
    # Both rows still travel with the figure, so a reader can see the failure.
    source = (tmp_path / "out" / "adjusted_effect_source_data.csv").read_text()
    assert "separation_detected" in source
    assert summary["source_rows_consumed"] == 2


def test_an_unrecognised_scale_keeps_a_linear_axis(tmp_path):
    """No null line and no log transform for a scale nobody declared the shape of."""

    run_dir, manifest = _write_bound_table(
        tmp_path, [{**_REAL_ROW, "effect_scale": "restricted_mean_survival_days"}]
    )
    summary = run_adjusted_association_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="08_adjusted_effect_figure",
        figure_product="adjusted_effect",
    )
    assert summary["effect_scale_recognised"] is False
    assert summary["axis_scale"] == "linear"


def test_it_refuses_to_draw_a_table_the_host_did_not_write(tmp_path):
    """The render-time half of the ownership clause, checked on its own.

    Selection decides who runs; this decides what may be drawn, and it does not
    take selection's answer on trust.
    """

    run_dir, manifest = _write_bound_table(
        tmp_path,
        [{name: "1" for name in _CODER_HEADER}],
        columns=_CODER_HEADER,
    )
    with pytest.raises(ValueError, match="host model owner"):
        run_adjusted_association_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="08_adjusted_effect_figure",
            figure_product="adjusted_effect",
        )


def test_it_refuses_a_table_that_mixes_two_exposures(tmp_path):
    """Two exposures is two figures; picking one would contradict a plotted row."""

    run_dir, manifest = _write_bound_table(
        tmp_path,
        [_REAL_ROW, {**_REAL_ROW, "exposure": "sofa_max"}],
    )
    with pytest.raises(ValueError, match="distinct exposure"):
        run_adjusted_association_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="08_adjusted_effect_figure",
            figure_product="adjusted_effect",
        )


def test_it_refuses_a_digest_that_does_not_match_the_bytes(tmp_path):
    run_dir, manifest = _write_bound_table(tmp_path, [_REAL_ROW])
    table = run_dir / "inputs" / "adjusted_association_estimates.csv"
    table.write_text(table.read_text().replace("1.357", "9.999"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest verification failed"):
        run_adjusted_association_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="08_adjusted_effect_figure",
            figure_product="adjusted_effect",
        )


def test_it_refuses_a_manifest_belonging_to_another_step(tmp_path):
    run_dir, manifest = _write_bound_table(tmp_path, [_REAL_ROW])
    manifest["step_id"] = "09_some_other_step"
    with pytest.raises(ValueError, match="does not belong to this step"):
        run_adjusted_association_figure(
            out_dir=tmp_path / "out",
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id="08_adjusted_effect_figure",
            figure_product="adjusted_effect",
        )


# --------------------------------------------------------------------------
# the shared effect-scale owner
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,null_value,multiplicative",
    [
        ("odds_ratio", 1.0, True),
        ("hazard_ratio", 1.0, True),
        ("risk_difference", 0.0, False),
        ("coefficient", 0.0, False),
    ],
)
def test_a_recognised_scale_reports_its_null_and_its_geometry(
    name, null_value, multiplicative
):
    scale = describe_effect_scale(name)
    assert scale.null_value == null_value
    assert scale.multiplicative is multiplicative
    assert scale.recognised is True


def test_an_unrecognised_scale_abstains_rather_than_assuming_one():
    """Drawing the wrong null is a claim about the result; abstaining is visible."""

    scale = describe_effect_scale("restricted_mean_survival_days")
    assert scale.null_value is None
    assert scale.multiplicative is False
    assert scale.recognised is False


def test_the_producers_own_abbreviation_is_recognised():
    """``OR`` is what the deterministic replay owner writes, uppercased.

    It was unrecognised until 2026-07-31, so every robustness forest fed by
    that owner was drawn with no line at no effect -- the reader had nothing to
    judge an interval against.
    """

    scale = describe_effect_scale("OR")
    assert scale.null_value == 1.0
    assert scale.multiplicative is True


def test_a_unit_qualifier_does_not_change_the_scale():
    """``odds_ratio_per_1_mmol_per_l`` is a real recorded value, and is an OR."""

    scale = describe_effect_scale("odds_ratio_per_1_mmol_per_l")
    assert scale.null_value == 1.0
    assert scale.multiplicative is True


def test_a_qualifier_on_an_unknown_head_stays_unknown():
    """Matching on a fragment is the defect this whole file exists to avoid."""

    assert describe_effect_scale("days_alive_per_100_admissions").null_value is None
    assert describe_effect_scale("_per_protocol").null_value is None
