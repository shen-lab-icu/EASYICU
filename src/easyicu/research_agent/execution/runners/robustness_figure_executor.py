"""Deterministic renderer for the robustness matrix the replay owner emits.

The producing side of this pair is already deterministic: the robustness
replay owner refits the locked specification grid and writes
``table:robustness_matrix`` with a fixed column contract. Drawing that table
is arithmetic-free -- one row per specification, its point estimate and
interval on the declared effect scale -- and yet it was left to the Coder to
write from scratch on every study.

CLAIMED BY WHAT THE STEP CONSUMES, NOT BY WHAT THE FIGURE IS CALLED.
Across two recorded nine-task runs, eight visualization steps consume
``table:robustness_matrix`` and emit one figure, under five different product
names: ``robustness_plot``, ``robustness``, ``robustness_sensitivity``,
``robustness_replay`` and ``complete_case_robustness``. A renderer keyed on a
name set would own whichever names happened to be listed and abandon the next
synonym the Planner invents; the input contract is the same in all eight, so
that is what decides. The rule this follows is the one a name-keyed check
keeps breaking: anchor on what is exclusive to the work, not on its label.

THE EXTRA BINDINGS ARE READ, NOT EXCUSED. Two of the eight steps bind the
matrix alone; six bind five typed inputs -- the matrix,
``table:robustness_summary``, and the statistics ``statistic:primary_or``,
``statistic:complete_case_n`` and ``statistic:robustness_summary``. The first
draft of this renderer read only the matrix and was refused by
``TypedInputCapability`` for the other four, which was the right refusal: a
step binding the primary estimate is asking for a figure that SHOWS the
primary estimate, and a renderer that quietly dropped it would publish a
figure answering a different question from the one the plan registered. A
robustness forest without its anchor is exactly that figure.

So they are declared optional -- the claim being about this renderer's code
path, that it draws a correct figure with or without each -- and every one
that is bound is actually drawn: the primary estimate as the anchor line the
specifications are compared against, the complete-case count as an
annotation. A bound statistic whose recorded value is null is annotated as
not reported rather than skipped; a real ``complete_case_n.json`` in the
corpus carries exactly that.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...figures.publication import (
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .deterministic_robustness import _MATRIX_COLUMNS
from .effect_scale import describe_effect_scale
from .figure_input_capability import TypedInputCapability

__all__ = [
    "ROBUSTNESS_FIGURE_INPUT",
    "robustness_figure_executor_code",
    "robustness_figure_executor_owns_step",
    "run_robustness_figure",
]


ROBUSTNESS_FIGURE_INPUT = "table:robustness_matrix"

#: The columns this renderer reads. The replay owner's contract is wider; a
#: figure that required every one of its columns would break the next time the
#: producer gained a diagnostic field, so the check is containment, not
#: equality. Every column below is read, and none of them is optional.
_READ_COLUMNS = (
    "spec_id",
    "effect_scale",
    "point_estimate",
    "ci_low",
    "ci_high",
    "axis",
    "converged",
)


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]*", product)
    ):
        return None
    return product


#: Statistics and companion tables the recorded steps bind alongside the
#: matrix. Optional because the figure is correct without each of them, and
#: read whenever present -- see the module docstring on why "optional" may
#: never mean "ignored".
ROBUSTNESS_PRIMARY_ESTIMATE_INPUT = "statistic:primary_or"
ROBUSTNESS_COMPLETE_CASE_INPUT = "statistic:complete_case_n"

ROBUSTNESS_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset({ROBUSTNESS_FIGURE_INPUT}),
    optional=frozenset(
        {
            "table:robustness_summary",
            "statistic:robustness_summary",
            ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
            ROBUSTNESS_COMPLETE_CASE_INPUT,
        }
    ),
)


#: The replay owner's own header, imported rather than restated, so a producer
#: that changes its contract changes this gate with it.
_PRODUCER_CONTRACT_COLUMNS = frozenset(_MATRIX_COLUMNS)


def _binding_is_producer_contract(binding: Any) -> bool:
    """Whether this binding is the deterministic replay owner's own matrix."""

    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    if not isinstance(contract, Mapping):
        return False
    columns = contract.get("columns")
    if not isinstance(columns, list) or not all(
        isinstance(value, str) for value in columns
    ):
        return False
    return _PRODUCER_CONTRACT_COLUMNS <= set(columns)


def robustness_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether every scientific choice is already fixed upstream.

    No clause names a figure product. What makes the step renderable is that
    it consumes the replay owner's matrix and promises exactly one figure; the
    specification grid, effect scale, estimates and intervals were all decided
    by the plan and computed by the producer.

    THE PRODUCER CLAUSE IS NOT OPTIONAL, and its absence was a live defect.
    This renderer first shipped claiming the step from the input key alone, on
    the belief that only the deterministic replay owner ever writes
    ``robustness_matrix``. Measured 2026-07-31 against the five real matrices
    on disk, that was false: four were Coder-authored under three different
    headers -- a two-by-two audit table, a complete-case comparison row -- and
    for those the renderer would have claimed the step and then raised at load,
    turning four steps the Coder was drawing successfully into four dead ones.
    Claiming a step is a promise to produce its figure, so the check that the
    bound table is really the producer's belongs here, before the promise, not
    only inside the sandbox after it.
    """

    products = [_figure_product(value) for value in step.expected_outputs]
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and ROBUSTNESS_FIGURE_CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        # A renderer that also froze a clustering would be choosing science.
        # ``model_requirements`` and ``table_one_spec`` are not checked because
        # ``AnalysisStep`` already refuses both on a visualization step whose
        # sole output is one figure (verified 2026-07-31); a guard the type
        # system enforces reads as protection while protecting nothing.
        and step.trajectory_stability_spec is None
    ):
        return False
    if not isinstance(resolved_bindings, Mapping):
        return False
    return _binding_is_producer_contract(resolved_bindings.get(ROBUSTNESS_FIGURE_INPUT))


def robustness_figure_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    # Ownership is NOT re-derived here. The selector consulted this owner with
    # the step's resolved bindings; a second evaluation without them cannot see
    # what the selector saw and would answer differently -- which it did, once
    # the producer clause landed. What this builder checks is its own input:
    # that the step names exactly one canonical figure product to render.
    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("The step is not owned by the robustness renderer")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.robustness_figure_executor import (
            run_robustness_figure,
        )

        run_robustness_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _canonical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_matrix(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> tuple[pd.DataFrame, Mapping[str, Any], Mapping[str, Any]]:
    """Verify and read the bound robustness matrix, and hand back all bindings.

    The bindings travel with it so the optional statistics this figure draws
    are read from the same verified manifest rather than a second read of it.
    """

    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("resolved-input manifest carries no bindings")
    binding = inputs.get(ROBUSTNESS_FIGURE_INPUT)
    if not isinstance(binding, dict):
        raise ValueError("the robustness matrix binding is absent")

    expected_sha256 = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    product_contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(product_contract, dict)
        or not isinstance(consumption, dict)
        or not isinstance(identity, dict)
        or binding.get("declared_kind") != "table"
        or binding.get("evidence_kind") != "table"
        or binding.get("product") != "robustness_matrix"
        or identity.get("input_key") != ROBUSTNESS_FIGURE_INPUT
        or identity.get("product") != "robustness_matrix"
        or identity.get("sha256") != expected_sha256
        or consumption.get("input_key") != ROBUSTNESS_FIGURE_INPUT
        # Every specification the grid locked has to be drawn: a figure that
        # showed a chosen subset would report a different sensitivity analysis
        # from the one the plan registered.
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError("robustness matrix authority binding is incomplete")

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError("robustness matrix binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("robustness matrix must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("robustness matrix digest verification failed")

    # The clause selection already applied, re-asked of the manifest the
    # sandbox actually received. Selection decides who runs; this decides what
    # may be drawn, and it does not take the earlier answer on trust.
    if not _binding_is_producer_contract(binding):
        raise ValueError(
            "robustness matrix was not written by the deterministic replay owner"
        )

    columns = product_contract.get("columns")
    row_count = product_contract.get("row_count")
    if (
        not isinstance(columns, list)
        or not set(_READ_COLUMNS).issubset({str(name) for name in columns})
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError("robustness matrix product contract is unsupported")

    frame = pd.read_csv(path)
    if not set(_READ_COLUMNS).issubset(set(frame.columns)) or len(frame) != row_count:
        raise ValueError("robustness matrix bytes disagree with its product contract")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("robustness matrix changed while it was being read")
    return frame, binding, inputs


def _load_statistic(
    *,
    run_dir: Path,
    inputs: Mapping[str, Any],
    input_key: str,
) -> tuple[bool, float | None]:
    """Return whether the statistic was bound, and its value when recorded.

    ``(False, None)`` means the plan did not bind it; ``(True, None)`` means it
    was bound and its recorded value is null, which a real
    ``complete_case_n.json`` in the corpus is. The two are not the same thing
    on the figure -- one is absent, the other is reported as not estimated --
    so they are not collapsed here.
    """

    binding = inputs.get(input_key)
    if binding is None:
        return False, None
    if not isinstance(binding, dict):
        raise ValueError(f"{input_key} binding is not a mapping")
    expected_sha256 = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(relative_path, str)
        or not relative_path
        or binding.get("declared_kind") != "statistic"
        or binding.get("evidence_kind") != "statistic"
    ):
        raise ValueError(f"{input_key} authority binding is incomplete")
    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError(f"{input_key} binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{input_key} must be a regular bound file")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError(f"{input_key} digest verification failed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "value" not in payload:
        raise ValueError(f"{input_key} sidecar records no value")
    return True, _finite(payload.get("value"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validated_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    """Return the drawable rows, the effect scale, and whether any was dropped.

    A specification whose refit did not converge has no interval to draw. It is
    kept in the figure as a labelled gap rather than deleted, because a
    sensitivity analysis that silently omits the variants that failed reads as
    though every variant agreed.
    """

    scales = {
        str(value).strip()
        for value in frame["effect_scale"].tolist()
        if str(value).strip()
    }
    if len(scales) != 1:
        raise ValueError(
            "robustness matrix mixes effect scales; one figure cannot carry two axes"
        )
    effect_scale = scales.pop()

    rows = frame.copy()
    rows["__estimate"] = [_finite(value) for value in rows["point_estimate"]]
    rows["__low"] = [_finite(value) for value in rows["ci_low"]]
    rows["__high"] = [_finite(value) for value in rows["ci_high"]]
    rows["__drawable"] = [
        estimate is not None and low is not None and high is not None and low <= high
        for estimate, low, high in zip(
            rows["__estimate"], rows["__low"], rows["__high"]
        )
    ]
    if not rows["__drawable"].any():
        raise ValueError("no robustness specification carries a drawable interval")
    labels = [str(value).strip() for value in rows["spec_id"].tolist()]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("robustness specifications must carry unique non-empty ids")
    rows["__label"] = labels
    return rows, effect_scale, bool((~rows["__drawable"]).any())


def _reader_label(value: str) -> str:
    return str(value).replace("_", " ").strip()


def run_robustness_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> Mapping[str, Any]:
    """Render the locked specification grid and write its figure contract."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", figure_product or ""):
        raise ValueError("figure product must be one canonical lowercase token")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, binding, bound_inputs = _load_matrix(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )
    rows, effect_scale, has_gap = _validated_rows(frame)
    anchor_bound, anchor_value = _load_statistic(
        run_dir=Path(run_dir),
        inputs=bound_inputs,
        input_key=ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
    )
    complete_case_bound, complete_case_n = _load_statistic(
        run_dir=Path(run_dir),
        inputs=bound_inputs,
        input_key=ROBUSTNESS_COMPLETE_CASE_INPUT,
    )

    source_path = out_dir / f"{figure_product}_source_data.csv"
    frame.to_csv(source_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    height_mm = max(58.0, 16.0 + 7.4 * len(rows))
    fig, ax = plt.subplots(figsize=(120 / 25.4, height_mm / 25.4))

    positions = list(range(len(rows)))
    drawn = [index for index, ok in enumerate(rows["__drawable"]) if ok]
    estimates = [rows["__estimate"].iloc[i] for i in drawn]
    lows = [rows["__low"].iloc[i] for i in drawn]
    highs = [rows["__high"].iloc[i] for i in drawn]
    ax.errorbar(
        estimates,
        drawn,
        xerr=[
            [estimate - low for estimate, low in zip(estimates, lows)],
            [high - estimate for estimate, high in zip(estimates, highs)],
        ],
        fmt="o",
        color=palette["blue"],
        ecolor=palette["neutral"],
        elinewidth=1.0,
        capsize=2.0,
        markersize=4.2,
    )
    null_value = describe_effect_scale(effect_scale).null_value
    if null_value is not None:
        ax.axvline(
            null_value,
            color=palette["neutral"],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
    # The anchor the specifications are compared against. Drawn whenever the
    # plan bound it, because binding it is what asks for it to be shown.
    if anchor_value is not None:
        ax.axvline(
            anchor_value,
            color=palette["blue"],
            linewidth=1.0,
            zorder=0,
            label="primary estimate",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=6.1)
    ax.set_yticks(positions)
    ax.set_yticklabels([_reader_label(label) for label in rows["__label"]])
    ax.invert_yaxis()
    ax.set_xlabel(_reader_label(effect_scale))
    ax.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for index, ok in enumerate(rows["__drawable"]):
        if ok:
            continue
        # Named, not dropped.
        ax.text(
            0.5,
            index,
            "not estimable",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="center",
            fontsize=6.1,
            color=palette["neutral"],
        )
    if complete_case_bound:
        ax.set_title(
            "Complete-case n: "
            + (
                f"{int(complete_case_n):,}"
                if complete_case_n is not None
                else "not reported"
            ),
            loc="left",
            pad=4,
            fontsize=6.4,
        )
    fig.subplots_adjust(left=0.42, right=0.97, bottom=0.14, top=0.92)

    contract = make_figure_contract(
        figure_id=figure_product,
        title=f"Robustness of the primary estimate ({_reader_label(effect_scale)})",
        # Stated, not inferred: the claim a robustness figure makes is about
        # agreement across the locked grid, never about the effect itself.
        core_claim=(
            "Whether the registered effect estimate holds across every "
            "pre-specified analytic variant."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Locked specification grid",
                "role": "robustness",
                "claim": (
                    "One row per locked robustness specification, showing its "
                    "point estimate and confidence interval on the declared "
                    "effect scale. Specifications whose refit did not converge "
                    "are labelled rather than omitted."
                ),
                "evidence_ids": [source_path.name],
                "metadata": {
                    "chart_type": "forest_interval_robustness",
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "Estimates and intervals are reproduced from the bound robustness "
            "matrix without recomputation. The executor introduces no cohort, "
            "exposure, outcome, specification, missing-data or modeling "
            "decision, and draws every locked specification."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_robustness_figure",
        "analysis_family": "robustness",
        "deterministic_standard_analysis": "robustness_figure",
        "rendering_only": True,
        "source_input": ROBUSTNESS_FIGURE_INPUT,
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": int(len(frame)),
        "source_table": "robustness_matrix.csv",
        "effect_scale": effect_scale,
        # Whether the figure carries a line at no effect. Recorded because its
        # absence is exactly what a reader of the figure cannot see: an
        # unrecognised scale spelling silently removed the anchor every
        # interval is judged against, and nothing said so.
        "null_line_drawn": null_value is not None,
        "specifications_drawn": int(len(drawn)),
        "specifications_not_estimable": int(len(rows) - len(drawn)),
        "any_specification_not_estimable": bool(has_gap),
        "primary_estimate_bound": bool(anchor_bound),
        "primary_estimate_drawn": anchor_value is not None,
        "complete_case_n_bound": bool(complete_case_bound),
        "complete_case_n": complete_case_n,
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
