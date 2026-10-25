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
import json
from dataclasses import replace
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


def _publish_synthetic_replay(
    tmp_path, monkeypatch, *, declared_identity=True, contract_updates=None, variant_updates=None,
):
    """Exercise publication with aggregate fits; never fit or read cohort rows."""
    from easyicu.research_agent.execution.runners import deterministic_robustness as owner
    from easyicu.research_agent.schema import RobustnessSpec
    from tests.support.robustness_sources import write_structured_source_authority

    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    summary = record["step_summary"]
    contract = summary["model_contracts"][0]
    if declared_identity:
        contract.update(
            estimand_id="synthetic_endpoint_adjusted_odds",
            estimand_label="Adjusted odds of the recorded endpoint",
            contrast_id="raised_vs_usual",
            contrast_label="Raised marker vs usual marker",
            effect_unit="category contrast",
            ci_method="Wald",
            confidence_level=0.95,
            dependence={"variance_estimator": "cluster_robust", "cluster_unit": "patient"},
            cluster_count=80,
        )
    contract.update(contract_updates or {})
    # Keep the actual source resolver and its digest checks in this test.
    summary_path = run_dir / "steps/01_primary_model/outputs/step_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt = next(item for item in evidence if item["evidence_id"] == "stat_primary")
    (run_dir / receipt["relative_path"]).write_bytes(summary_path.read_bytes())
    receipt["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    source = owner._find_structured_primary_model_source(
        records=[record], evidence_records=evidence, run_dir=run_dir,
    )
    assert source is not None
    primary, coefficients, _, error = owner._structured_model_row(
        spec_id="primary", axis="primary", outputs_dir=source["outputs_dir"],
        coefficient_path=source["coefficient_path"], contract=contract,
        evidence_id=source["coefficient_evidence_id"], note_prefix="Synthetic aggregate fit.",
    )
    assert error is None
    spec = RobustnessSpec(
        spec_id="complete_case", axis="missing", description="Complete-case specification",
        missing_override={"strategy": "complete_case", "variables": ["exposure", "outcome"]},
    )
    variant = replace(primary, spec_id=spec.spec_id, axis=spec.axis, independent_variant=False)
    replay = {
        "variant_contracts": [{**contract, "spec_id": spec.spec_id,
                               "analysis_set": "complete_case", "analysis_role": "sensitivity",
                               **(variant_updates or {})}],
        "variant_coefficients": [{**row, "spec_id": spec.spec_id} for row in coefficients],
    }
    monkeypatch.setattr(owner, "_load_locked_specs", lambda *a, **k: ([spec], "2026-01-01"))
    monkeypatch.setattr(owner, "_load_json_object", lambda *a: {
        "primary_exposure": "exposure", "target_outcome": "outcome",
    })
    monkeypatch.setattr(owner, "_load_frame", lambda *a: pd.DataFrame())
    monkeypatch.setattr(owner, "_load_primary_cohort", lambda *a: None)
    monkeypatch.setattr(owner, "_membership_audit", lambda **k: [])
    monkeypatch.setattr(owner, "_fit_structured_robustness_rows", lambda **k: (
        [primary, variant], [], replay,
    ))
    out = run_dir / "steps/replay/outputs"
    owner._run_robustness_preflight(
        out_dir=out, run_dir=run_dir,
        authority_payload={"per_step_records": [record], "evidence": evidence},
        authority_error="", context_path=run_dir / "context.json",
        lock_path=run_dir / "lock.json", cohort_path=run_dir / "unused.parquet",
        universe_path=None,
    )
    return out, pd.read_csv(out / "robustness_matrix.csv")


def test_replay_publishes_the_bound_models_identity_and_interval_semantics(tmp_path, monkeypatch):
    out, matrix = _publish_synthetic_replay(tmp_path, monkeypatch)
    for key, value in {
        "estimand_id": "synthetic_endpoint_adjusted_odds",
        "estimand_label": "Adjusted odds of the recorded endpoint",
        "contrast_id": "raised_vs_usual",
        "contrast_label": "Raised marker vs usual marker",
        "effect_unit": "category contrast",
        "ci_method": "Wald", "confidence_level": 0.95,
        "variance_estimator": "cluster_robust", "cluster_unit": "patient", "cluster_count": 80,
    }.items():
        assert key in matrix, f"The replay dropped {key} from its verified model contract"
        assert matrix[key].tolist() == [value, value]
    assert matrix["point_estimate"].tolist() == [1.4, 1.4]
    assert matrix["ci_low"].tolist() == [1.1, 1.1]
    assert matrix["ci_high"].tolist() == [1.8, 1.8]
    assert matrix.loc[1, "independent_variant"] == False  # noqa: E712
    assert (out / "sensitivity_comparison.csv").read_bytes() == (out / "robustness_matrix.csv").read_bytes()


def test_replay_summary_preserves_the_documented_equivalence(tmp_path, monkeypatch):
    out, _matrix = _publish_synthetic_replay(tmp_path, monkeypatch)
    summary = json.loads((out / "step_summary.json").read_text())
    variant = summary["robustness_rows"][1]
    assert variant["independent_variant"] is False
    assert variant["contrast_id"] == "raised_vs_usual"


def test_figure_carries_published_identity_into_its_caption_without_export(tmp_path, monkeypatch):
    from easyicu.research_agent.execution.runners import robustness_figure_executor as figure

    out, matrix = _publish_synthetic_replay(tmp_path, monkeypatch)
    run_dir = tmp_path / "figure_run"
    binding = _matrix_binding(run_dir)
    matrix_path = run_dir / binding["relative_path"]
    matrix_path.write_bytes((out / "robustness_matrix.csv").read_bytes())
    digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    binding["sha256"] = digest
    binding["identity_row"]["sha256"] = digest
    binding["consumption_contract"]["artifact_sha256"] = digest
    binding["product_contract"]["columns"] = list(matrix.columns)
    captured = {}

    def capture(fig, path, *, contract, **kwargs):
        captured["contract"] = contract.to_dict()
        captured["texts"] = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
        return {}

    monkeypatch.setattr(figure, "save_publication_figure", capture)
    figure.run_robustness_figure(
        out_dir=tmp_path / "figure_out", run_dir=run_dir,
        resolved_inputs={"step_id": "figure", "inputs": {ROBUSTNESS_FIGURE_INPUT: binding}},
        step_id="figure", figure_product="robustness_plot",
    )
    caption = captured["contract"]["reader_caption"]
    assert "Raised marker vs usual marker" in caption
    assert "Adjusted odds of the recorded endpoint" in caption
    assert "95% Wald confidence intervals" in caption
    assert "patient-cluster-robust" in caption and "80 clusters" in caption
    assert "not an independent variant" in captured["texts"].replace("\n", " ")
    assert "not authorized as directly comparable" in caption
    assert "Contrast / unit not declared" not in captured["texts"]


def test_or_scale_alone_does_not_supply_missing_identity(tmp_path, monkeypatch):
    from easyicu.research_agent.figures.robustness import assess_robustness_effect_comparability

    _out, matrix = _publish_synthetic_replay(tmp_path, monkeypatch, declared_identity=False)
    for key in ("estimand_id", "contrast_id", "contrast_label", "effect_unit", "confidence_level"):
        assert key not in matrix
    # Isolate identity from independence: even otherwise independent OR rows
    # are insufficient to authorize a common effect axis.
    matrix["independent_variant"] = True
    assert not assess_robustness_effect_comparability(matrix).authorized


def test_variant_keeps_its_own_estimand_and_interval_metadata(tmp_path, monkeypatch):
    _out, matrix = _publish_synthetic_replay(tmp_path, monkeypatch, variant_updates={
        "estimand_id": "secondary_endpoint", "ci_method": "Percentile bootstrap",
        "confidence_level": 0.9, "dependence": None,
        "variance_estimator": "bootstrap", "cluster_count": None,
    })
    assert matrix["estimand_id"].tolist() == ["synthetic_endpoint_adjusted_odds", "secondary_endpoint"]
    assert matrix["ci_method"].tolist() == ["Wald", "Percentile bootstrap"]
    assert matrix["confidence_level"].tolist() == [0.95, 0.9]
    assert matrix["variance_estimator"].tolist() == ["cluster_robust", "bootstrap"]
    assert pd.isna(matrix.loc[1, "cluster_count"])
    assert pd.isna(matrix.loc[1, "cluster_unit"])
