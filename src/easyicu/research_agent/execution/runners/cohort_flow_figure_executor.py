"""Deterministic rendering of one digest-bound cohort-flow table.

The cohort-definition owner has already fixed every eligibility predicate and
count.  This renderer verifies those exact bytes and draws the sequential
flow, with a side box for every exclusion and the share its stage retained;
it never reloads the cohort or invents another inclusion rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Sequence

import pandas as pd

from ...contracts.figure_plan import COHORT_FLOW_FIGURE_PANELS, COHORT_FLOW_INPUT
from ...figures.publication import (
    PALETTE_CLINICAL,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from ._shared import figure_product as _figure_product, method_head as _method_head

__all__ = [
    "COHORT_FLOW_INPUT",
    "COHORT_ACCOUNTING_COMPLETE",
    "COHORT_ACCOUNTING_DENOMINATOR_ONLY",
    "cohort_flow_figure_executor_code",
    "cohort_flow_figure_executor_owns_step",
    "render_cohort_flow_axis",
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
    if not (frame["n_before"] - frame["n_excluded"]).eq(frame["n_remaining"]).all():
        raise ValueError("cohort-flow denominator arithmetic failed")
    if (
        len(frame) > 1
        and not frame["n_before"]
        .iloc[1:]
        .reset_index(drop=True)
        .eq(frame["n_remaining"].iloc[:-1].reset_index(drop=True))
        .all()
    ):
        raise ValueError("cohort-flow denominator sequence is discontinuous")
    # Keep CSV row coordinates through display sorting for source-data joins.
    return frame


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


def _humanize_token(value: Any) -> str:
    """Format a bound predicate/concept token for readers, without inventing.

    ``str.title()`` over ``first_icu_stay`` produced mechanical
    ``First_Icu_Stay`` labels on a manuscript figure.  Only whitespace and
    underscore normalisation plus sentence capitalisation happen here; no
    clinical word is added or translated.
    """

    text = re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip()
    return f"{text[:1].upper()}{text[1:]}" if text else ""


def _display_labels(frame: pd.DataFrame, *, complete: bool) -> list[str]:
    """Reader names for each stage, from the ledger's own predicates.

    The first stage is the ledger's source cohort whatever its predicate is
    called.  The last stage is marked by its drawing, not by a prefix.
    """
    if not complete:
        return [
            "All bound input rows"
            if _unfiltered_universe(frame)
            else "Analysis denominator only"
        ]
    labels: list[str] = []
    for index, (_, row) in enumerate(frame.iterrows()):
        kind = str(row.get("predicate_kind") or "").strip()
        if index == 0:
            labels.append("Source cohort")
            continue
        concept = _humanize_token(row.get("concept_id"))
        label = (
            f"{_humanize_token(kind)} · {concept}"
            if concept and kind.casefold() in {"inclusion", "exclusion"}
            else _humanize_token(kind)
        )
        labels.append(label)
    return labels


def _retention_text(numerator: int, denominator: int) -> str | None:
    """Display-only retained share; ``None`` when the share is unprovable.

    A numerator above its denominator would print a misleading >100% share, so
    a non-monotone series keeps its counts and makes no share claim.
    """

    if denominator <= 0 or numerator > denominator:
        return None
    value = 100.0 * numerator / denominator
    # Rounding must not contradict the exclusion printed beside it: two stays
    # excluded from 48,971 is not "retained 100%", and one stay kept of many
    # is not "retained 0%".
    if numerator < denominator and value >= 99.95:
        return ">99.9%"
    if 0 < numerator and value < 0.05:
        return "<0.1%"
    if abs(value - round(value)) < 0.05:
        return f"{round(value)}%"
    return f"{value:.1f}%"


def _share_parts(count: int, previous: int, universe: int) -> list[str]:
    """The provable retained shares of one stage, each naming its base."""

    return [
        f"{share} of {scope}"
        for share, scope in (
            (_retention_text(count, previous), "previous"),
            (_retention_text(count, universe), "universe"),
        )
        if share
    ]


def _share_note(parts: Sequence[str], *, separator: str) -> str:
    """The grey note drawn under an exclusion, stating what the share is.

    It sits directly beneath a red "-N excluded" line, so a bare "95.3% of
    previous" reads as the excluded share -- the opposite of what it is.  The
    fit pass measures and the draw pass prints this same string, so the two
    can never disagree about its width.
    """

    return "retained " + separator.join(parts) if parts else ""


def _axes_size_pt(ax: Any) -> tuple[float, float]:
    """The axes' drawable rectangle in points, for legibility arithmetic."""

    figure = ax.get_figure()
    box = ax.get_position()
    width_in, height_in = figure.get_size_inches()
    return box.width * float(width_in) * 72.0, box.height * float(height_in) * 72.0


def _wrapped_lines(label: str, wrap: int) -> int:
    return max(1, len(textwrap.wrap(str(label), wrap)))


def _flow_type_scale(
    *,
    labels: Sequence[str],
    counts: Sequence[int],
    excluded: Sequence[int],
    panel_width_pt: float,
    panel_height_pt: float,
    step: float,
    width: float,
    side_x: float,
    compact: bool,
    base_label: float,
    base_count: float,
    base_note: float,
    base_wrap: int,
) -> tuple[float, int, float, int] | None:
    """Largest type scale at which every stage still clears its neighbours.

    Each stage owns a vertical pitch band of ``step`` axes units. The node
    block (wrapped label + count line) must stay inside its band and the
    side annotations (exclusion count, retained-share notes) inside their own
    column's pitch band. Type shrinks toward a legible floor; below it the stage
    set cannot be drawn without overlap at this height. Return ``None`` so
    the caller can enlarge the canvas before drawing the stage set.
    """

    box_width_pt = width * panel_width_pt
    side_width_pt = max(0.0, (1.0 - side_x - 0.02)) * panel_width_pt
    scale = 1.0
    while True:
        label_fs = base_label * scale
        count_fs = base_count * scale
        note_fs = base_note * scale
        if label_fs < 4.5 or count_fs < 5.0 or note_fs < 4.0:
            return None
        wrap = min(
            max(14, int(round(base_wrap / scale))),
            max(14, int(box_width_pt * 0.92 / (label_fs * 0.5))),
        )
        label_line_pt = label_fs * 1.25
        count_line_pt = count_fs * 1.25
        note_line_pt = note_fs * 1.25
        max_node_pt = 0.0
        for label in labels:
            node_pt = (
                _wrapped_lines(label, wrap) * label_line_pt + count_line_pt + 3.0
            )
            max_node_pt = max(max_node_pt, node_pt)
        node_frac = max_node_pt / panel_height_pt
        if node_frac > step * 0.85:
            scale -= 0.05
            continue
        # The band leftover after the node text is what the inter-stage
        # annotations may use; the box stays a visual container sized to the
        # text so it never pretends the text is smaller than it is.
        height = min(0.24, max(step * 0.62, node_frac * 1.12))
        height = min(height, step * 0.9)
        # Notes sit to the right of the boxes, so they may share vertical
        # coordinates with node text. Only neighbouring note blocks compete
        # for the same column. Charging them to the narrow gap between boxes
        # needlessly stretches even a six-stage composite to a very tall page.
        note_offset = min(0.008, step * 0.05)
        note_band_pt = (step * 0.95 - 2 * note_offset) * panel_height_pt
        # Annotation tiers, least load-bearing first: derived retained-share
        # notes may be dropped under space pressure because every count stays
        # on the nodes and in the source data; the per-stage exclusion note is
        # bound-ledger arithmetic and never degrades silently.
        for draw_shares in (True, False):
            shares_lines = 2 if compact else 1
            max_up_pt = 0.0
            max_down_pt = 0.0
            for index in range(1, len(labels)):
                max_up_pt = max(
                    max_up_pt, note_line_pt if excluded[index] else 0.0
                )
                if draw_shares and excluded[index]:
                    parts = _share_parts(
                        counts[index], counts[index - 1], counts[0]
                    )
                    if parts:
                        joined = _share_note(parts, separator=" \u00b7 ")
                        if (
                            not compact
                            and shares_lines == 1
                            and len(joined) * note_fs * 0.5 > side_width_pt * 0.92
                        ):
                            shares_lines = 2
                        max_down_pt = max(
                            max_down_pt,
                            note_line_pt * (shares_lines if len(parts) == 2 else 1),
                        )
            if max_up_pt + max_down_pt + 2 * _NOTE_BOX_PAD_PT <= note_band_pt:
                return scale, wrap, height, shares_lines if draw_shares else 0
        scale -= 0.05


def _add_flow_node(
    ax: Any,
    y: float,
    label: str,
    count: int,
    *,
    height: float,
    x: float,
    width: float,
    label_fontsize: float = 8.5,
    count_fontsize: float = 9.0,
    wrap: int = 36,
    final: bool = False,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle=(
                f"round,pad={min(0.006, height * 0.05)},"
                f"rounding_size={min(0.012, height * 0.15)}"
            ),
            linewidth=1.4 if final else 0.9,
            edgecolor=PALETTE_CLINICAL["blue"],
            facecolor=PALETTE_CLINICAL["blue_soft"],
            alpha=0.85 if final else 0.45,
            zorder=2,
        )
    )
    wrapped = textwrap.fill(str(label), wrap)
    _panel_w, panel_h_pt = _axes_size_pt(ax)
    label_block = _wrapped_lines(label, wrap) * label_fontsize * 1.25 / panel_h_pt
    count_block = count_fontsize * 1.25 / panel_h_pt
    inner_gap = 2.0 / panel_h_pt
    text_block = label_block + inner_gap + count_block
    ax.text(
        x,
        y + text_block / 2 - label_block / 2,
        wrapped,
        ha="center",
        va="center",
        fontsize=label_fontsize,
        color=PALETTE_CLINICAL["baseline"],
        zorder=3,
    )
    ax.text(
        x,
        y - text_block / 2 + count_block / 2,
        f"n = {int(count):,}",
        ha="center",
        va="center",
        fontsize=count_fontsize,
        fontweight="bold",
        color=PALETTE_CLINICAL["blue"],
        zorder=3,
    )


# Space between a side box's edge and its text, in points.
_NOTE_BOX_PAD_PT = 3.0


def _add_exclusion_box(
    ax: Any,
    *,
    middle: float,
    hub_x: float,
    side_x: float,
    note_offset: float,
    excluded: int,
    shares: str,
    fontsize: float,
) -> None:
    """A participant-flow side box: the exclusion count over the retained shares.

    A line joins it to the arrow between the two stages it separates.  The
    text keeps the coordinates the fit pass measured; the box only frames it.
    """

    from matplotlib.patches import FancyBboxPatch

    _panel_w, panel_h_pt = _axes_size_pt(ax)
    line = fontsize * 1.25 / panel_h_pt
    pad = _NOTE_BOX_PAD_PT / panel_h_pt
    share_lines = shares.count("\n") + 1 if shares else 0
    top = middle + note_offset + line + pad
    bottom = middle - note_offset - share_lines * line - pad
    left = side_x - 0.012
    ax.plot(
        [hub_x, left],
        [middle, middle],
        color=PALETTE_CLINICAL["neutral"],
        linewidth=0.6,
        zorder=0,
    )
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            0.995 - left,
            top - bottom,
            boxstyle="round,pad=0,rounding_size=0.006",
            linewidth=0.6,
            edgecolor=PALETTE_CLINICAL["neutral_light"],
            facecolor="white",
            zorder=1,
        )
    )
    ax.text(
        side_x,
        middle + note_offset,
        f"Excluded (n = {excluded:,})",
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=PALETTE_CLINICAL["red"],
        zorder=3,
    )
    if shares:
        ax.text(
            side_x,
            middle - note_offset,
            shares,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=PALETTE_CLINICAL["neutral"],
            zorder=3,
        )


def render_cohort_flow_axis(
    ax: Any,
    frame: pd.DataFrame,
    labels: list[str],
    *,
    complete: bool = True,
    compact: bool = False,
) -> None:
    """Draw the bound ledger as a top-to-bottom participant-flow diagram.

    Every box height, arrow, exclusion count and percentage is computed from
    the verified frame; the axis carries no prose claim of its own.  A lone
    denominator is drawn with the same node grammar instead of a caption card.
    ``compact`` shrinks type and shortens the share notes for the small
    sub-panel of a multi-panel publication figure; ``n_excluded`` is optional
    because composite contracts guarantee only ``n_remaining``.
    """

    from matplotlib.patches import FancyArrowPatch

    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    counts = [int(value) for value in frame["n_remaining"].tolist()]
    if not counts:
        return
    if "n_excluded" in frame:
        excluded = [int(value) for value in frame["n_excluded"].tolist()]
    else:
        # Composite contracts guarantee only ``n_remaining``. The drop between
        # adjacent bound counts is the same arithmetic the ledger itself
        # proves, so it may be annotated; an increase makes no exclusion claim.
        excluded = [0]
        for previous, current in zip(counts, counts[1:]):
            drop = previous - current
            excluded.append(drop if drop > 0 else 0)
    body_fontsize = 6.0 if compact else 8.5
    count_fontsize = 6.4 if compact else 9.0
    note_fontsize = 5.4 if compact else 7.5
    wrap = 24 if compact else 36
    hub_x = 0.36 if compact else 0.40
    side_x = 0.66 if compact else 0.74
    width = 0.56 if compact else 0.52
    if not complete:
        _add_flow_node(
            ax,
            0.55,
            labels[0],
            counts[0],
            height=0.26,
            x=0.5,
            width=width,
            label_fontsize=body_fontsize,
            count_fontsize=count_fontsize,
            wrap=wrap,
        )
        return
    stages = len(frame)
    top, bottom = 0.93, 0.07
    step = (top - bottom) / stages
    note_offset = min(0.008, step * 0.05)
    while True:
        panel_width_pt, panel_height_pt = _axes_size_pt(ax)
        if panel_width_pt <= 0 or panel_height_pt <= 0:
            raise ValueError("cohort-flow panel must have positive dimensions")
        fitted = _flow_type_scale(
            labels=labels,
            counts=counts,
            excluded=excluded,
            panel_width_pt=panel_width_pt,
            panel_height_pt=panel_height_pt,
            step=step,
            width=width,
            side_x=side_x,
            compact=compact,
            base_label=body_fontsize,
            base_count=count_fontsize,
            base_note=note_fontsize,
            base_wrap=wrap,
        )
        if fitted is not None and fitted[0] == 1.0 and fitted[3]:
            break
        # A valid ledger must not fail just because a composite panel started
        # too short. Grow the shared canvas before drawing any flow artists;
        # all axes keep their grid positions and every stage keeps its count
        # and intended font size. Callers record the resulting canvas height.
        figure = ax.get_figure()
        figure.set_size_inches(
            figure.get_figwidth(), figure.get_figheight() * 1.25, forward=False
        )
    scale, wrap, height, shares_lines = fitted
    body_fontsize *= scale
    count_fontsize *= scale
    note_fontsize *= scale
    universe = counts[0]
    node_x = 0.5 if stages == 1 else hub_x
    for index, count in enumerate(counts):
        y = top - step * (index + 0.5)
        _add_flow_node(
            ax,
            y,
            labels[index],
            count,
            height=height,
            x=node_x,
            width=width,
            label_fontsize=body_fontsize,
            count_fontsize=count_fontsize,
            wrap=wrap,
            final=stages > 1 and index == stages - 1,
        )
        if index == 0:
            continue
        y_previous = top - step * (index - 0.5)
        ax.add_patch(
            FancyArrowPatch(
                (hub_x, y_previous - height / 2),
                (hub_x, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=11 if not compact else 8,
                linewidth=0.9 if not compact else 0.7,
                color=PALETTE_CLINICAL["baseline"],
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=1,
            )
        )
        if not excluded[index]:
            # A stage that excluded nobody has no exclusion to report.
            continue
        # A one-line share note that would overflow the axes edge is stacked;
        # the fit pass already proved the taller block still clears the gap.
        separator = "\n" if compact or shares_lines == 2 else " \u00b7 "
        _add_exclusion_box(
            ax,
            middle=(y_previous + y) / 2,
            hub_x=hub_x,
            side_x=side_x,
            note_offset=note_offset,
            excluded=excluded[index],
            shares=(
                _share_note(
                    _share_parts(count, counts[index - 1], universe),
                    separator=separator,
                )
                if shares_lines
                else ""
            ),
            fontsize=note_fontsize,
        )


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
    # The normalized columns above are plotting coordinates, not new emitted
    # results. Preserve the upstream value columns exactly so every exported
    # number remains independently verifiable against its bound source.
    source = frame.loc[:, list(binding["product_contract"]["columns"])].copy()
    if "row_role" not in source:
        source["row_role"] = "cohort_stage"
    source.insert(0, "accounting_completeness", completeness)
    source.insert(0, "display_label", display_labels)
    source.insert(0, "source_step_id", binding.get("produced_by_step"))
    source.insert(0, "source_table", path.name)
    source.insert(0, "source_row_index", frame.index.tolist())
    source_path = out_dir / f"{figure_product}_source_data.csv"
    source.to_csv(source_path, index=False)

    apply_publication_style()
    # The caption names the figure, so the canvas holds only the flow.
    height = max(2.4, 0.8 * len(frame) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, height))
    render_cohort_flow_axis(ax, frame, display_labels, complete=complete)
    fig.tight_layout(pad=0.4)
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
        height_mm=float(fig.get_figheight()) * 25.4,
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
                    "source_products": list((source_input,)),
                    "source_data": [source_path.name],
                    "accounting_completeness": completeness,
                    "paper_grade_cohort_accounting": complete,
                },
            }
        ],
        source_data=[source_path.name],
        reader_caption=(
            "Cohort accounting. Counts reproduce the bound sequential ledger of "
            "records entering, excluded from, and remaining after each recorded "
            "eligibility stage; no additional selection is applied by the figure. "
            "The ledger begins at the bound input universe, not necessarily the "
            "entire source database."
            if complete
            else (
                "Analysis denominator. The single bar shows all bound input "
                "records; no eligibility filter was applied within this ledger. "
                if unfiltered_universe
                else "Analysis denominator. The single bar "
                "shows the final number of bound analysis records. "
            )
            + "Earlier eligibility stages and exclusions are unavailable; "
            "this is not a complete participant-flow diagram."
        ),
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
            "sequential_attrition_flow" if complete else "denominator_only_node"
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
