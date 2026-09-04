"""Deterministic rendering of one digest-bound cohort-flow table.

The cohort-definition owner has already fixed every eligibility predicate and
count.  This renderer verifies those exact bytes and draws the remaining
denominator after each recorded step; it never reloads the cohort or invents
another inclusion rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...contracts.figure_plan import COHORT_FLOW_FIGURE_PANELS, COHORT_FLOW_INPUT
from ...figures.publication import (
    PALETTE_CLINICAL,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep

__all__ = [
    "COHORT_FLOW_INPUT",
    "COHORT_ACCOUNTING_COMPLETE",
    "COHORT_ACCOUNTING_DENOMINATOR_ONLY",
    "cohort_flow_figure_executor_code",
    "cohort_flow_figure_executor_owns_step",
    "run_cohort_flow_figure",
]


COHORT_ACCOUNTING_COMPLETE = "sequential_attrition_ledger"
COHORT_ACCOUNTING_DENOMINATOR_ONLY = "analysis_denominator_only"
_REQUIRED_COLUMNS = (
    "step_order",
    "predicate_kind",
    "n_before",
    "n_excluded",
    "n_remaining",
)
_MODEL_FLOW_REQUIRED_COLUMNS = (
    "stage",
    "n",
    "excluded_from_previous",
    "population_rule",
)
_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")


def _method_head(value: Any) -> str:
    return str(value or "").strip().casefold().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not _PRODUCT_ID.fullmatch(product):
        return None
    return product


def _population_flow_input(step: AnalysisStep) -> str | None:
    if len(step.inputs) != 1:
        return None
    input_key = str(step.inputs[0])
    kind, separator, product = input_key.partition(":")
    if kind != "table" or not separator:
        return None
    if product != "cohort_flow" and not product.endswith("population_flow"):
        return None
    return input_key


def _binding_is_cohort_flow(binding: Any, *, input_key: str) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    product = input_key.partition(":")[2]
    supported_columns = bool(
        isinstance(columns, list)
        and (
            set(_REQUIRED_COLUMNS).issubset(set(columns))
            or set(_MODEL_FLOW_REQUIRED_COLUMNS).issubset(set(columns))
        )
    )
    return bool(
        binding.get("declared_kind") == "table"
        and binding.get("evidence_kind") == "table"
        and binding.get("product") == product
        and supported_columns
        and isinstance(consumption, Mapping)
        and consumption.get("mode") == "all_rows"
    )


def cohort_flow_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the exact typed parent fully fixes the figure."""

    products = [_figure_product(value) for value in step.expected_outputs]
    contracts = list(step.input_consumption_contracts or [])
    source_input = _population_flow_input(step)
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and source_input is not None
        and len(products) == 1
        and products[0] is not None
        and len(contracts) == 1
        and contracts[0].input_key == source_input
        and contracts[0].mode == "all_rows"
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == {source_input}
        and _binding_is_cohort_flow(
            resolved_bindings.get(source_input), input_key=source_input
        )
    )


def cohort_flow_figure_executor_code(step: AnalysisStep) -> str:
    source_input = _population_flow_input(step)
    if source_input is None or len(step.expected_outputs) != 1:
        raise ValueError("cohort-flow figure requires its one exact input and output")
    product = _figure_product(step.expected_outputs[0])
    if product is None:
        raise ValueError("cohort-flow figure output is not a typed figure")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.cohort_flow_figure_executor import (
            run_cohort_flow_figure,
        )

        run_cohort_flow_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            source_input={source_input!r},
        )
        """
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binding(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    source_input: str,
) -> tuple[Path, Mapping[str, Any]]:
    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    inputs = payload.get("inputs") if isinstance(payload, Mapping) else None
    if payload.get("step_id") != step_id or not isinstance(inputs, Mapping):
        raise ValueError("resolved-input manifest does not belong to this step")
    if set(inputs) != {source_input}:
        raise ValueError("cohort-flow input binding is absent or widened")
    binding = inputs[source_input]
    if not _binding_is_cohort_flow(binding, input_key=source_input):
        raise ValueError("cohort-flow input has no supported host contract")
    expected_sha = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(identity, Mapping)
        or identity.get("input_key") != source_input
        or identity.get("product") != source_input.partition(":")[2]
        or identity.get("sha256") != expected_sha
    ):
        raise ValueError("cohort-flow authority binding is incomplete")
    base = Path(run_dir).resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("cohort-flow input escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("cohort-flow input is not a safe CSV file")
    if _sha256(path) != expected_sha:
        raise ValueError("cohort-flow input digest verification failed")
    return path, binding


def _verified_flow(path: Path, binding: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    contract = binding["product_contract"]
    expected_rows = contract.get("row_count")
    if (
        list(frame.columns) != list(contract.get("columns") or [])
        or isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows < 1
        or len(frame) != expected_rows
    ):
        raise ValueError("cohort-flow bytes disagree with their contract")
    if set(_MODEL_FLOW_REQUIRED_COLUMNS).issubset(frame.columns):
        counts = pd.to_numeric(frame["n"], errors="coerce")
        excluded = pd.to_numeric(frame["excluded_from_previous"], errors="coerce")
        if (
            counts.isna().any()
            or excluded.isna().any()
            or (counts < 0).any()
            or (excluded < 0).any()
            or not (counts % 1 == 0).all()
            or not (excluded % 1 == 0).all()
        ):
            raise ValueError("model population flow contains invalid counts")
        counts = counts.astype("int64")
        excluded = excluded.astype("int64")
        labels = frame["stage"].fillna("").astype(str).str.strip()
        rules = frame["population_rule"].fillna("").astype(str).str.strip()
        if labels.eq("").any() or rules.eq("").any() or labels.duplicated().any():
            raise ValueError("model population flow contains invalid stage semantics")
        before = counts.shift(1, fill_value=int(counts.iloc[0])).astype("int64")
        if int(excluded.iloc[0]) != 0 or not (before - excluded).eq(counts).all():
            raise ValueError("model population flow denominator arithmetic failed")
        frame = frame.assign(
            step_order=range(len(frame)),
            predicate_kind=labels,
            n_before=before,
            n_excluded=excluded,
            n_remaining=counts,
        )

    numeric: dict[str, pd.Series] = {}
    for column in ("step_order", "n_before", "n_excluded", "n_remaining"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or not (values % 1 == 0).all():
            raise ValueError(f"cohort-flow has invalid {column} values")
        numeric[column] = values.astype("int64")
    if numeric["step_order"].duplicated().any():
        raise ValueError("cohort-flow step_order values are not unique")
    frame = frame.assign(**numeric).sort_values("step_order", kind="stable")
    if frame["step_order"].tolist() != list(range(len(frame))):
        raise ValueError("cohort-flow step_order values are not contiguous from zero")
    labels = frame["predicate_kind"].fillna("").astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError("cohort-flow has an empty predicate label")
    if not (frame["n_before"] - frame["n_excluded"]).eq(
        frame["n_remaining"]
    ).all():
        raise ValueError("cohort-flow denominator arithmetic failed")
    if len(frame) > 1 and not frame["n_before"].iloc[1:].reset_index(drop=True).eq(
        frame["n_remaining"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise ValueError("cohort-flow denominator sequence is discontinuous")
    return frame.reset_index(drop=True)


def _accounting_completeness(frame: pd.DataFrame) -> str:
    """Classify only what the bound ledger itself can prove.

    A singleton row proves one denominator, not upstream eligibility or
    attrition.  It remains useful as an analysis-denominator display but must
    never be promoted to complete participant-flow accounting.
    """

    return (
        COHORT_ACCOUNTING_COMPLETE
        if len(frame) > 1
        else COHORT_ACCOUNTING_DENOMINATOR_ONLY
    )


def _unfiltered_universe(frame: pd.DataFrame) -> bool:
    """Is the single stage the whole bound universe, with nothing excluded?

    A one-row ledger has two very different causes and the figure used to
    report only the pessimistic one. When the row IS the universe row and it
    excluded nobody, the ledger is not missing upstream attrition -- it is
    recording that no eligibility filter was applied, so every bound input row
    is the analysis cohort. Saying "upstream attrition unavailable" there
    reports a gap that does not exist and hides the fact a reader most needs:
    this study declared no inclusion or exclusion criterion.
    """

    if len(frame) != 1:
        return False
    row = frame.iloc[0]
    if str(row.get("predicate_kind") or "").strip().casefold() != "universe":
        return False
    try:
        return int(row.get("n_excluded") or 0) == 0
    except (TypeError, ValueError):
        return False


def _display_labels(frame: pd.DataFrame, *, complete: bool) -> list[str]:
    if not complete:
        return [
            "All bound input rows"
            if _unfiltered_universe(frame)
            else "Analysis denominator only"
        ]
    labels: list[str] = []
    for index, row in frame.iterrows():
        kind = str(row.get("predicate_kind") or "").strip()
        if index == 0:
            labels.append("Source universe")
            continue
        concept = str(row.get("concept_id") or "").strip()
        label = (
            f"{kind.title()}: {concept}"
            if concept and kind.casefold() in {"inclusion", "exclusion"}
            else kind.replace("_", " ").strip().title()
        )
        if index == len(frame) - 1:
            label = f"Final · {label}"
        labels.append(label)
    return labels


def run_cohort_flow_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    source_input: str = COHORT_FLOW_INPUT,
) -> dict[str, Any]:
    """Verify and render every row in the canonical cohort-flow artifact."""

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, binding = _load_binding(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        source_input=source_input,
    )
    frame = _verified_flow(path, binding)
    completeness = _accounting_completeness(frame)
    complete = completeness == COHORT_ACCOUNTING_COMPLETE
    unfiltered_universe = (not complete) and _unfiltered_universe(frame)
    display_labels = _display_labels(frame, complete=complete)
    source = frame.copy()
    source.insert(0, "accounting_completeness", completeness)
    source.insert(0, "display_label", display_labels)
    source.insert(0, "source_step_id", binding.get("produced_by_step"))
    source.insert(0, "source_table", path.name)
    source.insert(0, "source_row_index", range(len(source)))
    source_path = out_dir / f"{figure_product}_source_data.csv"
    source.to_csv(source_path, index=False)

    apply_publication_style()
    height = max(3.2, 0.42 * len(frame) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, height))
    if complete:
        positions = list(range(len(frame)))
        bars = ax.barh(
            positions,
            frame["n_remaining"],
            color=PALETTE_CLINICAL["blue"],
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(display_labels)
        ax.invert_yaxis()
        ax.set_xlabel("ICU stays remaining")
        ax.set_title("Cohort accounting", loc="left")
        ax.grid(axis="x", color=PALETTE_CLINICAL["neutral_light"], linewidth=0.6)
        for bar, remaining, excluded in zip(
            bars, frame["n_remaining"], frame["n_excluded"]
        ):
            suffix = f"  (-{int(excluded):,})" if int(excluded) else ""
            ax.annotate(
                f"{int(remaining):,}{suffix}",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
    else:
        # A ONE-STAGE FLOW IS STILL A FLOW.
        #
        # This branch used to switch the axes off and centre three lines of
        # text, so a manuscript figure slot shipped a caption card. The stage
        # is real and countable, so it is drawn on the same axis the
        # multi-stage ledger uses; what changes is only how many stages there
        # are, and the note underneath says which of the two one-row cases
        # this is.
        denominator = int(frame.iloc[0]["n_remaining"])
        unfiltered = unfiltered_universe
        # A lone bar drawn at the multi-stage height fills the panel; keep it
        # at the thickness a stage has when the ledger has several.
        bar = ax.barh(
            [0], [denominator], height=0.42, color=PALETTE_CLINICAL["blue"]
        )[0]
        ax.set_yticks([0])
        ax.set_yticklabels(display_labels)
        ax.set_ylim(-0.9, 0.9)
        ax.invert_yaxis()
        ax.set_xlabel("ICU stays remaining")
        ax.set_title(
            "Cohort accounting · single stage",
            loc="left",
        )
        ax.grid(axis="x", color=PALETTE_CLINICAL["neutral_light"], linewidth=0.6)
        ax.annotate(
            f"{denominator:,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
        ax.annotate(
            "No eligibility filter was applied: every bound input row is the\n"
            "analysis cohort."
            if unfiltered
            else "Upstream eligibility and attrition are not recorded in the\n"
            "bound ledger.",
            xy=(0.0, -0.24),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=7,
            color=PALETTE_CLINICAL["neutral"],
        )
    fig.tight_layout()
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The figure reproduces the sequential source-to-final denominator "
            "ledger from the digest-verified cohort-flow table."
            if complete
            else (
                "The bound cohort-flow table records one stage: no eligibility "
                "filter was applied, so every bound input row is the analysis "
                "cohort. Attrition upstream of the bound universe is outside "
                "this ledger."
                if unfiltered_universe
                else "The bound cohort-flow table proves only the final "
                "analysis denominator; upstream eligibility and attrition are "
                "unavailable."
            )
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=92.0,
        panels=[
            {
                "panel_id": COHORT_FLOW_FIGURE_PANELS[0].panel_id,
                "title": (
                    "Cohort accounting"
                    if complete
                    else (
                        "Cohort accounting · single stage"
                        if unfiltered_universe
                        else "Analysis denominator only"
                    )
                ),
                "role": COHORT_FLOW_FIGURE_PANELS[0].article_role,
                "claim": (
                    "Every displayed source, eligibility, and final count comes "
                    "from the bound sequential cohort flow."
                    if complete
                    else "The single displayed count is the bound analysis "
                    "denominator and does not establish upstream attrition."
                ),
                "evidence_ids": [str(binding.get("evidence_id") or "")],
                "review_risk": (
                    None
                    if complete
                    else "Upstream attrition is unavailable; this panel must not "
                    "be described as complete participant-flow accounting."
                ),
                "metadata": {
                    "article_role": COHORT_FLOW_FIGURE_PANELS[0].article_role,
                    "chart_type": COHORT_FLOW_FIGURE_PANELS[0].chart_type,
                    "source_products": list(
                        (source_input,)
                    ),
                    "source_data": [source_path.name],
                    "accounting_completeness": completeness,
                    "paper_grade_cohort_accounting": complete,
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "All bound attrition rows are preserved. The renderer introduces no "
            "cohort filter, imputation, or denominator change."
            if complete
            else "Only one bound denominator row was available. No exclusion "
            "counts or upstream eligibility stages were inferred."
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
    if _sha256(path) != binding.get("sha256"):
        raise ValueError("cohort-flow input changed while it was rendered")
    figure_files = [item.name for key, item in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cohort_flow_figure",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": "cohort_flow_figure",
        "rendering_only": True,
        "source_input": source_input,
        "source_step_id": binding.get("produced_by_step"),
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": len(frame),
        "cohort_accounting_completeness": completeness,
        "paper_grade_cohort_accounting": complete,
        "upstream_attrition_available": complete,
        "rendering_mode": (
            "sequential_attrition_bars" if complete else "denominator_only_node"
        ),
        "input_bindings": [
            {
                "input_key": source_input,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "loaded": True,
                "row_count": len(frame),
            }
        ],
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
