"""Deterministic renderer for the adjusted association the host itself fitted.

This draws the paper's headline number: one adjusted effect estimate, its
interval, on the scale the model declared. Every quantity is already computed
and locked by :mod:`adjusted_association_executor`; drawing it introduces no
cohort, exposure, outcome, adjustment, missing-data or modelling decision.

WHY THIS ONE IS GATED ON THE PRODUCER, NOT ONLY ON THE INPUT KEY.
The robustness renderer can claim a step from ``table:robustness_matrix``
alone, because only a deterministic host owner ever emits that product, so its
header is fixed by construction. ``table:adjusted_association_estimates`` is
not like that. Measured across the recorded corpus (2026-07-31): 17 real files
carry **12 distinct headers**, because until the host owner landed the Coder
invented one per run -- ``estimate`` in some, ``odds_ratio`` in others,
``event_n`` against ``n_events``, ``std_error`` against ``standard_error``.

A renderer that read the input key and then guessed which column held the
effect would be the allowlist defect in a new place, and the failure mode is
the worst kind: it does not crash, it draws a number under the wrong label.
So ownership additionally requires that the *bound product contract* carries
the host owner's own locked header. When the Coder produced the table the
renderer declines and the ordinary coder path draws it, exactly as before --
declining costs nothing, mislabelling an odds ratio costs the paper.

Reachability, measured before this was written: of the recorded visualization
steps consuming this key, only those in runs where the host owner produced the
table carry the locked header. That set is small today because the owner is
new, and it is every future run: the header is fixed the moment the host does
the fit.

NOTHING ELSE IS READABLE, SO NOTHING ELSE IS ADMITTED. Three recorded steps
bind a second table alongside the estimates -- a sensitivity grid, an
absolute-risk table. Both are Coder-authored and were measured at six distinct
headers between them, so this renderer cannot draw them, and per
:class:`TypedInputCapability` an input it cannot read is a refusal rather than
something to ignore: a step binding a second table is asking for a figure that
shows it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping, Optional

import pandas as pd

from ...figures.publication import (
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from ...numeric_scalars import coerce_optional_finite_float as _finite
from .adjusted_association_executor import ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS
from .effect_scale import describe_effect_scale
from .figure_input_capability import TypedInputCapability

__all__ = [
    "ADJUSTED_ASSOCIATION_FIGURE_INPUT",
    "adjusted_association_figure_executor_code",
    "adjusted_association_figure_executor_owns_step",
    "run_adjusted_association_figure",
]


ADJUSTED_ASSOCIATION_FIGURE_INPUT = "table:adjusted_association_estimates"

#: The producer's own header, imported rather than restated. A second copy here
#: would be a header this renderer believes in that the producer has stopped
#: emitting, and the two would disagree silently.
_HOST_CONTRACT_COLUMNS = frozenset(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)

#: The columns this renderer reads and draws. Every one is in the host
#: contract; the check against a binding is containment, so the producer may
#: gain a diagnostic field without breaking the figure.
_READ_COLUMNS = (
    "fit_status",
    "estimate",
    "ci_low",
    "ci_high",
    "effect_scale",
    "exposure",
    "outcome",
    "covariates",
    "estimator_kind",
    "analysis_set",
    "n",
    "n_events",
    # What each ROW is, as opposed to what the model is. The producer writes
    # these four; reading none of them is what made three contrasts share one
    # label. Safe to require: the ownership check above already refuses any
    # binding whose contract does not declare the producer's full column set,
    # so a table reaching here has them.
    "exposure_level",
    "reference_level",
    "contrast",
    "is_primary_contrast",
)

ADJUSTED_ASSOCIATION_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset({ADJUSTED_ASSOCIATION_FIGURE_INPUT}),
    # ``one_per_role`` is still complete-table consumption: the host has
    # verified that the bound table contains exactly one row for every role
    # named by the Planner and no other rows.  This renderer draws every one
    # of those verified rows; it never chooses a role itself.
    supported_consumption_modes=frozenset({"all_rows", "one_per_role"}),
)

#: ``fit_status`` values the producer writes for a model that actually fitted.
#: Anything else is reported on the figure instead of being drawn as a point --
#: a failed fit rendered as a dot is indistinguishable from a real estimate.
_FITTED_STATUSES = frozenset({"fitted", "ok", "converged", "success"})


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


def _binding_is_host_contract(binding: Any) -> bool:
    """Whether this binding is the host owner's own locked estimates table.

    Read at selection time from the host's typed-input binding map. The same
    fact is verified again from the manifest inside the sandbox: selection
    decides who runs, and the renderer still refuses to draw a table that is
    not the one selection was promised.
    """

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
    return _HOST_CONTRACT_COLUMNS <= set(columns)


def adjusted_association_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the host both fitted this estimate and can draw it.

    No clause names a figure product: across the recorded corpus these steps
    emit eighteen different names for the same figure. What decides is that the
    step consumes the estimates table, promises exactly one figure, and -- the
    clause that keeps this honest -- that the bound table is the one the host's
    own model owner wrote.
    """

    products = [_figure_product(value) for value in step.expected_outputs]
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and ADJUSTED_ASSOCIATION_FIGURE_CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        # A renderer that also froze a clustering would be choosing science,
        # not drawing a result. This is the only one of the three scientific
        # declarations that can actually reach here, and it was checked rather
        # than assumed (2026-07-31): ``AnalysisStep`` already refuses
        # ``model_requirements`` unless the step's method is
        # ``adjusted_association_models`` and it promises
        # ``table:adjusted_association_estimates``, and refuses
        # ``table_one_spec`` unless it promises ``table:table_one`` -- both
        # impossible for a visualization step whose sole output is one figure.
        # Those two clauses were written here first and deleted once measured;
        # a guard the type system already enforces reads as protection while
        # protecting nothing. The other figure renderers still carry the same
        # two dead clauses, which is a separate sweep.
        and step.trajectory_stability_spec is None
    ):
        return False
    if not isinstance(resolved_bindings, Mapping):
        return False
    return _binding_is_host_contract(
        resolved_bindings.get(ADJUSTED_ASSOCIATION_FIGURE_INPUT)
    )


def adjusted_association_figure_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("the step is not owned by the adjusted association renderer")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import (
            run_adjusted_association_figure,
        )

        run_adjusted_association_figure(
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


def _load_estimates(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Verify and read the bound estimates table, host contract and all."""

    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("resolved-input manifest carries no bindings")
    binding = inputs.get(ADJUSTED_ASSOCIATION_FIGURE_INPUT)
    if not isinstance(binding, dict):
        raise ValueError("the adjusted association estimates binding is absent")

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
        or binding.get("product") != "adjusted_association_estimates"
        or identity.get("input_key") != ADJUSTED_ASSOCIATION_FIGURE_INPUT
        or identity.get("product") != "adjusted_association_estimates"
        or identity.get("sha256") != expected_sha256
        or consumption.get("input_key") != ADJUSTED_ASSOCIATION_FIGURE_INPUT
        # Every estimate the model registered has to be drawn. ``one_per_role``
        # is legal only when its verified roster covers every bound row; it is
        # not permission for this renderer to select a subset.
        or consumption.get("mode") not in {"all_rows", "one_per_role"}
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError("adjusted association authority binding is incomplete")

    # The clause selection already applied, re-asked of the manifest the
    # sandbox actually received. Selection decides who runs; this decides what
    # may be drawn, and it does not take the earlier answer on trust.
    if not _binding_is_host_contract(binding):
        raise ValueError(
            "adjusted association estimates were not written by the host model owner"
        )

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError("estimates binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("adjusted association estimates must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("adjusted association estimates digest verification failed")

    row_count = product_contract.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError("adjusted association product contract is unsupported")

    frame = pd.read_csv(path)
    if not set(_READ_COLUMNS).issubset(set(frame.columns)) or len(frame) != row_count:
        raise ValueError("estimates bytes disagree with the product contract")
    if consumption.get("mode") == "one_per_role":
        role_column = str(consumption.get("role_column") or "")
        expected_roles = consumption.get("expected_roles")
        verified_counts = consumption.get("verified_role_counts")
        observed_roles = (
            [str(value) for value in frame[role_column].tolist()]
            if role_column in frame.columns
            else []
        )
        observed_counts = {
            str(role): observed_roles.count(str(role))
            for role in expected_roles or ()
        }
        if (
            not role_column
            or role_column not in frame.columns
            or not isinstance(expected_roles, list)
            or not expected_roles
            or not isinstance(verified_counts, dict)
            or verified_counts != {str(role): 1 for role in expected_roles}
            or len(expected_roles) != row_count
            or observed_counts != {str(role): 1 for role in expected_roles}
            or set(observed_roles) != {str(role) for role in expected_roles}
        ):
            raise ValueError("one-per-role association authority is incomplete")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError("estimates changed while they were being read")
    return frame, binding


def _text(value: Any) -> str:
    """The text of a cell that may be missing, with missing meaning empty.

    Every reader of this table goes through here because pandas' ``NaN`` is
    both truthy and stringifies to ``"nan"``: ``str(value or "")`` on an empty
    CSV cell yields the word "nan", so an unadjusted model captioned itself
    "Adjusted for nan." and an absent exposure would have labelled the axis.
    A fabricated word on a figure is worse than a blank, and it is invisible.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _sole(frame: pd.DataFrame, column: str) -> str:
    """The one value of a column the whole table must agree on.

    Exposure, outcome and effect scale describe the figure's axis and title. A
    table carrying two of any of them is two figures, and picking one would put
    a title on a plot that contradicts a row inside it.
    """

    values = {_text(value) for value in frame[column].tolist() if _text(value)}
    if len(values) != 1:
        raise ValueError(
            f"adjusted association estimates carry {len(values)} distinct {column} "
            "values; one figure cannot carry two"
        )
    return values.pop()


def _row_label(row: Mapping[str, Any], index: int) -> str:
    """Name what the ROW is before what the whole model is.

    ``analysis_set`` and ``estimator_kind`` describe the model, so they are
    identical on every row of it.  Used alone they render a four-level ordinal
    exposure as three rows reading ``complete_case``, and no reader can tell
    which one is stage 3 versus stage 0.

    The producer already writes what each row is: ``contrast``,
    ``exposure_level`` and ``reference_level`` are three of the twenty columns in
    its own product contract, and this renderer read none of them.  MEASURED
    over the recorded corpus: 99 emitted estimates tables carry those columns
    and 33 have more than one row -- 32 with three, one with four.

    Model-level fields stay as the fallback, which is what keeps the 66 one-row
    tables labelled exactly as before.
    """

    contrast = _text(row.get("contrast"))
    if contrast:
        return contrast
    level = _text(row.get("exposure_level"))
    reference = _text(row.get("reference_level"))
    if level and reference:
        return f"{level} vs {reference}"
    if level:
        return level
    for column in ("analysis_set", "estimator_kind"):
        value = _text(row.get(column))
        if value:
            return value
    return f"estimate {index + 1}"


def _is_primary_contrast(row: Mapping[str, Any]) -> bool:
    return _text(row.get("is_primary_contrast")).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _primary_contrast_label(frame: Any, labels: Any) -> Optional[str]:
    """The label of the one row the study's claim is about, when it says so.

    Returns None rather than guessing when no row declares itself primary or
    when more than one does -- a figure that named the wrong row as the headline
    would be worse than one that names none.
    """

    marked = [
        str(label)
        for row, label in zip(frame.to_dict("records"), labels, strict=True)
        if _is_primary_contrast(row)
    ]
    return marked[0] if len(marked) == 1 else None


def _reader_label(value: str) -> str:
    return str(value).replace("_", " ").strip()


#: Round multipliers a reader can hold in their head. A ratio axis is labelled
#: with whichever of these the data actually spans, never with the powers of ten
#: matplotlib defaults to -- over a 1.0-1.5 odds-ratio span those render as
#: overlapping "1.2 x 10^0" strings that no journal would take.
_RATIO_TICK_CANDIDATES = (
    0.1,
    0.125,
    0.2,
    0.25,
    0.33,
    0.5,
    0.67,
    0.8,
    0.9,
    1.0,
    1.1,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    8.0,
    10.0,
    20.0,
    50.0,
)


def _plain_number(value: float, _position: int = 0) -> str:
    """Format a ratio tick the way a clinician reads it: 1.5, not 1.5x10^0."""

    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _ratio_ticks(lows: list[float], highs: list[float], scale: Any) -> list[float]:
    """Round ratio ticks spanning the drawn intervals and the null.

    Always includes the null when the scale has one, so a reader can see which
    side of no-effect an interval sits on without measuring against a gridline
    that was never labelled.
    """

    span_low = min([*lows, *(v for v in [scale.null_value] if v)])
    span_high = max([*highs, *(v for v in [scale.null_value] if v)])
    ticks = [
        value
        for value in _RATIO_TICK_CANDIDATES
        if span_low * 0.97 <= value <= span_high * 1.03
    ]
    if scale.null_value is not None and scale.null_value not in ticks:
        ticks.append(scale.null_value)
    # Two ticks cannot show a reader the shape of an axis; falling back to the
    # observed endpoints is honest -- they are values that exist in the data.
    if len(ticks) < 2:
        ticks = sorted({round(span_low, 2), round(span_high, 2)})
    return sorted(set(ticks))


def _adjustment_note(covariates: Any) -> str:
    """Say what the model adjusted for, including when the answer is nothing.

    An adjusted-effect figure whose caption is silent about the adjustment set
    is the one a reader cannot check against the protocol. ``covariates`` is
    the producer's own semicolon-joined field.
    """

    terms = [term.strip() for term in _text(covariates).split(";") if term.strip()]
    if not terms:
        return "Unadjusted: the model declared no covariates."
    return "Adjusted for " + ", ".join(_reader_label(term) for term in terms) + "."


def run_adjusted_association_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> Mapping[str, Any]:
    """Render the host's own adjusted estimate and write its figure contract."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", figure_product or ""):
        raise ValueError("figure product must be one canonical lowercase token")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, binding = _load_estimates(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )

    exposure = _sole(frame, "exposure")
    outcome = _sole(frame, "outcome")
    scale = describe_effect_scale(_sole(frame, "effect_scale"))

    rows = frame.copy()
    rows["__estimate"] = [_finite(value) for value in rows["estimate"]]
    rows["__low"] = [_finite(value) for value in rows["ci_low"]]
    rows["__high"] = [_finite(value) for value in rows["ci_high"]]
    rows["__fitted"] = [
        str(value or "").strip().lower() in _FITTED_STATUSES
        for value in rows["fit_status"]
    ]
    rows["__drawable"] = [
        fitted
        and estimate is not None
        and low is not None
        and high is not None
        and low <= estimate <= high
        for fitted, estimate, low, high in zip(
            rows["__fitted"], rows["__estimate"], rows["__low"], rows["__high"]
        )
    ]
    if not rows["__drawable"].any():
        raise ValueError("no adjusted association estimate carries a drawable interval")
    rows["__label"] = [
        _row_label(row, index) for index, row in enumerate(frame.to_dict("records"))
    ]

    source_path = out_dir / f"{figure_product}_source_data.csv"
    frame.to_csv(source_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    # Keep the axis label and the model declaration in separate physical
    # bands.  The earlier 33 mm one-row canvas made those two independent text
    # groups overlap even though both were individually legible.  The model
    # declaration is part of the scientific contract, so dropping it is not a
    # layout fix; reserve enough height for both instead.
    height_mm = 42.0 + 7.0 * len(rows)
    fig, ax = plt.subplots(figsize=(120 / 25.4, height_mm / 25.4))

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
        elinewidth=1.1,
        capsize=2.4,
        markersize=4.6,
    )
    if scale.null_value is not None:
        ax.axvline(
            scale.null_value,
            color=palette["neutral"],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
    # A ratio scale is symmetric in the multiplicative sense, so halving and
    # doubling belong at equal distances from the null. Only a recognised ratio
    # scale gets this; an unrecognised one keeps a linear axis rather than a
    # transform nobody declared.
    if scale.multiplicative and all(value > 0 for value in lows):
        from matplotlib.ticker import FixedLocator, NullFormatter

        ax.set_xscale("log")
        # Matplotlib's default log formatter writes 1.2 as "1.2 x 10^0", which
        # over a typical odds-ratio span is both unreadable and overlapping.
        # A clinical reader wants plain multipliers, so the ticks are the round
        # ratios that fall inside the span the data actually occupies.
        ax.xaxis.set_major_locator(FixedLocator(_ratio_ticks(lows, highs, scale)))
        ax.xaxis.set_major_formatter(_plain_number)
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([_reader_label(label) for label in rows["__label"]])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(f"{_reader_label(scale.name)} ({_reader_label(exposure)})")
    ax.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)

    # The number the paper quotes, printed beside the interval it came from, so
    # a reader never has to infer a value off the axis.
    for index in drawn:
        estimate = rows["__estimate"].iloc[index]
        low = rows["__low"].iloc[index]
        high = rows["__high"].iloc[index]
        ax.annotate(
            f"{estimate:.2f} ({low:.2f}–{high:.2f})",
            xy=(1.0, index),
            xycoords=ax.get_yaxis_transform(),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.2,
            annotation_clip=False,
        )
    for index, ok in enumerate(rows["__drawable"]):
        if ok:
            continue
        # Named with the producer's own status, not dropped: a model that did
        # not fit is a result, and a figure that omitted the row would read as
        # though the estimate had never been requested.
        ax.text(
            0.5,
            index,
            _text(frame["fit_status"].iloc[index]) or "not estimated",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="center",
            fontsize=6.2,
            color=palette["neutral"],
        )

    total_n = _finite(frame["n"].iloc[0])
    total_events = _finite(frame["n_events"].iloc[0])
    subtitle = _reader_label(outcome)
    if total_n is not None and total_events is not None:
        subtitle += f" — {int(total_events):,} of {int(total_n):,}"
    ax.set_title(subtitle, loc="left", pad=4, fontsize=6.6)

    # The adjustment set belongs on the figure, not only in its contract. A
    # forest of adjusted estimates whose plotted surface never says what was
    # adjusted for is the figure a reader cannot check against the protocol,
    # and the caption travels separately from the image.
    adjustment = _adjustment_note(frame["covariates"].iloc[0])
    estimator = _reader_label(_text(frame["estimator_kind"].iloc[0]))
    association_kind = "unadjusted" if adjustment.startswith("Unadjusted:") else "adjusted"
    model_note = f"{adjustment} {estimator[:1].upper()}{estimator[1:]} model."
    fig.text(
        0.02,
        0.04,
        model_note,
        fontsize=5.9,
        color=palette["neutral"],
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.30, right=0.72, bottom=0.36, top=0.88)
    contract = make_figure_contract(
        figure_id=figure_product,
        title=(
            f"{association_kind.capitalize()} association between {_reader_label(exposure)} and "
            f"{_reader_label(outcome)}"
        ),
        core_claim=(
            f"The {association_kind} {_reader_label(scale.name)} for "
            f"{_reader_label(exposure)} on {_reader_label(outcome)}, with its "
            "confidence interval, as fitted and locked by the host model owner."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": f"{association_kind.capitalize()} effect estimate",
                "role": "primary_effect",
                "claim": (
                    f"Point estimate and confidence interval on the "
                    f"{_reader_label(scale.name)} scale. {adjustment} "
                    "Estimates that did not fit are labelled with the "
                    "producer's status rather than omitted."
                ),
                "evidence_ids": [source_path.name],
                "metadata": {
                    "chart_type": f"forest_interval_{association_kind}_association",
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "Estimates, intervals, effect scale and adjustment set are "
            "reproduced from the bound estimates table without recomputation. "
            f"{adjustment} The executor introduces no cohort, exposure, "
            "outcome, adjustment, missing-data or modeling decision."
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
        "method": "deterministic_adjusted_association_figure",
        "analysis_family": "association",
        "deterministic_standard_analysis": "adjusted_association_figure",
        "rendering_only": True,
        "source_input": ADJUSTED_ASSOCIATION_FIGURE_INPUT,
        "source_evidence_id": binding.get("evidence_id"),
        "source_sha256": binding.get("sha256"),
        "source_rows_consumed": int(len(frame)),
        "source_table": "adjusted_association_estimates.csv",
        "exposure": exposure,
        "outcome": outcome,
        "effect_scale": scale.name,
        "effect_scale_recognised": bool(scale.recognised),
        "axis_scale": "log" if ax.get_xscale() == "log" else "linear",
        "estimates_drawn": int(len(drawn)),
        "estimates_not_drawn": int(len(rows) - len(drawn)),
        # What the reader sees against each interval, and which of them the
        # study's claim is about. Without these a staged figure is three
        # unlabelled points and a caption that cannot say which is which.
        "row_labels": [str(label) for label in rows["__label"]],
        "primary_contrast_label": _primary_contrast_label(frame, rows["__label"]),
        "adjustment_note": adjustment,
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
