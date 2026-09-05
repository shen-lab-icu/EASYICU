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
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ...contracts.figure_plan import PlannedFigurePanelSpec
from ...contracts.ownership_verdict import OwnershipVerdict
from ...figures.presentation import (
    create_presented_axes,
    finish_presented_figure,
    presentation_from_panels,
)

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    configure_ratio_axis,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from ...numeric_scalars import coerce_optional_finite_float as _finite
from ...contracts.model_terms import level_spelling
from .adjusted_association_executor import ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS
from .exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
)
from .effect_scale import describe_effect_scale
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import load_typed_input

__all__ = [
    "ADJUSTED_ASSOCIATION_FIGURE_INPUT",
    "ASSOCIATION_OVERVIEW_FIGURE_INPUTS",
    "association_overview_figure_executor_code",
    "association_overview_figure_executor_owns_step",
    "adjusted_association_figure_executor_code",
    "adjusted_association_figure_executor_owns_step",
    "run_adjusted_association_figure",
    "run_association_overview_figure",
]


ADJUSTED_ASSOCIATION_FIGURE_INPUT = "table:adjusted_association_estimates"
ASSOCIATION_OVERVIEW_FIGURE_INPUTS = (
    "table:exposure_outcome_distribution",
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
)

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


def _association_panels(planned: Sequence[PlannedFigurePanelSpec | Mapping[str, Any]], *, figure_product: str, overview: bool = False) -> list[PlannedFigurePanelSpec]:
    """Bind only chart/source combinations actually drawn by this renderer."""
    specs = [PlannedFigurePanelSpec.model_validate(panel) for panel in planned]
    presentation_from_panels(specs)
    contracts = [
        (
            ADJUSTED_ASSOCIATION_FIGURE_INPUT,
            {
                "forest",
                "forest_plot",
                "forest_interval_adjusted_association",
                "forest_interval_unadjusted_association",
            },
            {"primary_effect", "primary_estimand", "primary_result"},
            "forest",
            "primary_effect",
        )
    ]
    if overview:
        contracts.insert(
            0,
            (
                ASSOCIATION_OVERVIEW_FIGURE_INPUTS[0],
                {"event_rate_panel"},
                {"absolute_risk_context", "descriptive_result"},
                "event_rate_panel",
                "absolute_risk_context",
            ),
        )
    if not specs:
        return [
            PlannedFigurePanelSpec(
                panel_id=chr(65 + index),
                figure_output=f"figure:{figure_product}",
                article_role=role,
                chart_type=chart,
                source_products=[source],
            )
            for index, (source, _, _, chart, role) in enumerate(contracts)
        ]
    if (
        len(specs) != len(contracts)
        or len({panel.placement for panel in specs}) != 1
        or len({panel.panel_id for panel in specs}) != len(specs)
    ):
        raise ValueError(
            "unsupported_planned_figure_design: panel count or split placement"
        )
    matched = []
    for source, charts, roles, _, _ in contracts:
        candidates = [
            panel
            for panel in specs
            if panel.figure_output == f"figure:{figure_product}"
            and panel.source_products == [source]
            and panel.chart_type in charts
            and panel.article_role in roles
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"unsupported_planned_figure_design: {source} requires charts={sorted(charts)} roles={sorted(roles)}"
            )
        matched.append(candidates[0])
    return matched


def association_figure_design_verdict(
    step: AnalysisStep, *, overview: bool = False
) -> OwnershipVerdict:
    kind = "association_overview_figure" if overview else "adjusted_association_figure"
    try:
        _association_panels(
            step.figure_panels,
            figure_product=str(step.expected_outputs[0]).partition(":")[2],
            overview=overview,
        )
    except (ValueError, IndexError) as exc:
        return OwnershipVerdict.wrong_shape(kind, reason=str(exc))
    return OwnershipVerdict.wrong_shape(
        kind, reason="typed input or execution contract is not owned by this renderer"
    )


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
    try:
        _association_panels(step.figure_panels, figure_product=products[0])
    except ValueError:
        return False
    return _binding_is_host_contract(
        resolved_bindings.get(ADJUSTED_ASSOCIATION_FIGURE_INPUT)
    )


def association_overview_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Own a two-table overview when both upstream owners are verified."""

    products = [_figure_product(value) for value in step.expected_outputs]
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and set(step.inputs) == set(ASSOCIATION_OVERVIEW_FIGURE_INPUTS)
        and len(step.inputs) == len(ASSOCIATION_OVERVIEW_FIGURE_INPUTS)
        and len(products) == 1
        and products[0] is not None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(ASSOCIATION_OVERVIEW_FIGURE_INPUTS)
        and _binding_is_host_contract(
            resolved_bindings.get(ADJUSTED_ASSOCIATION_FIGURE_INPUT)
        )
    ):
        return False
    try:
        _association_panels(
            step.figure_panels, figure_product=products[0], overview=True
        )
    except ValueError:
        return False
    distribution = resolved_bindings.get(ASSOCIATION_OVERVIEW_FIGURE_INPUTS[0])
    contract = distribution.get("product_contract") if isinstance(distribution, Mapping) else None
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return isinstance(columns, list) and set(
        EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS
    ) <= set(columns)


def association_overview_figure_executor_code(
    step: AnalysisStep,
    *,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    product = _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    if product is None:
        raise ValueError("association overview has no canonical figure product")
    panels = _association_panels(
        step.figure_panels, figure_product=product, overview=True
    )
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import run_association_overview_figure

        run_association_overview_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            panel_specs={[panel.model_dump(mode="json") for panel in panels]!r},
            display_labels={dict(display_labels or {})!r},
        )
        """
    ).strip()


def adjusted_association_figure_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("the step is not owned by the adjusted association renderer")
    panels = _association_panels(step.figure_panels, figure_product=product)
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
            panel_specs={[panel.model_dump(mode="json") for panel in panels]!r},
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


def run_association_overview_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    panel_specs: list[dict[str, Any]] | None = None,
    display_labels: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Render absolute group outcomes beside the host-owned adjusted effect."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", figure_product or ""):
        raise ValueError("figure product must be one canonical lowercase token")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = _association_panels(
        panel_specs or [], figure_product=figure_product, overview=True
    )
    estimates, estimate_binding = _load_estimates(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )
    distribution = load_typed_input(
        input_key=ASSOCIATION_OVERVIEW_FIGURE_INPUTS[0],
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
        require_consumption_contract=True,
        consumption_mode="all_rows",
        minimum_row_count=2,
    )
    groups = distribution.frame.loc[
        distribution.frame["row_role"].astype(str) == "exposure_level"
    ].copy()
    if len(groups) < 2:
        raise ValueError("distribution has fewer than two exposure-level rows")
    for column in (
        "exposure_pct",
        "exposure_ci_low_pct",
        "exposure_ci_high_pct",
        "outcome_rate_pct",
        "ci_low_pct",
        "ci_high_pct",
    ):
        groups[column] = pd.to_numeric(groups[column], errors="coerce")
        if groups[column].isna().any():
            raise ValueError(f"distribution column {column!r} is not finite")

    exposure = _sole(estimates, "exposure")
    outcome = _sole(estimates, "outcome")
    scale = describe_effect_scale(_sole(estimates, "effect_scale"))
    rows = estimates.copy()
    rows["__estimate"] = [_finite(value) for value in rows["estimate"]]
    rows["__low"] = [_finite(value) for value in rows["ci_low"]]
    rows["__high"] = [_finite(value) for value in rows["ci_high"]]
    rows["__label"] = [
        _row_label(row, index) for index, row in enumerate(estimates.to_dict("records"))
    ]
    drawable = [
        index
        for index, (estimate, low, high) in enumerate(
            zip(rows["__estimate"], rows["__low"], rows["__high"], strict=True)
        )
        if estimate is not None
        and low is not None
        and high is not None
        and low <= estimate <= high
    ]
    if not drawable:
        raise ValueError("adjusted association table has no drawable interval")

    distribution_source = out_dir / f"{figure_product}_distribution_source_data.csv"
    estimate_source = out_dir / f"{figure_product}_association_source_data.csv"
    distribution.frame.to_csv(distribution_source, index=False)
    estimates.to_csv(estimate_source, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    palette = apply_publication_style(font_size=7.0)
    presentation = presentation_from_panels(panels)
    if presentation is None:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    else:
        fig, axes, palette = create_presented_axes(2, presentation)
    ax = axes[0]
    positions = np.arange(len(groups), dtype=float)
    width = 0.36
    prevalence = groups["exposure_pct"].to_numpy(dtype=float)
    risks = groups["outcome_rate_pct"].to_numpy(dtype=float)
    ax.bar(
        positions - width / 2,
        prevalence,
        width,
        yerr=[
            prevalence - groups["exposure_ci_low_pct"].to_numpy(dtype=float),
            groups["exposure_ci_high_pct"].to_numpy(dtype=float) - prevalence,
        ],
        capsize=2,
        label="Exposure prevalence",
        color=palette["blue_soft"],
    )
    ax.bar(
        positions + width / 2,
        risks,
        width,
        yerr=[
            risks - groups["ci_low_pct"].to_numpy(dtype=float),
            groups["ci_high_pct"].to_numpy(dtype=float) - risks,
        ],
        capsize=2,
        label="Outcome risk",
        color=palette["orange"],
    )
    group_labels = [
        str(
            (display_labels or {}).get(
                f"{exposure}={level_spelling(value)}"
            )
            or value
        )
        for value in groups["exposure_level"]
    ]
    ax.set_xticks(positions, group_labels)
    ax.set_xlabel(_reader_label(exposure))
    ax.set_ylabel("Percent")
    ax.set_title("Absolute prevalence and outcome risk", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=6)
    add_panel_label(ax, "A", x=-0.13, y=1.04, fontsize=presentation.font_size * 1.1 if presentation else 11)

    ax = axes[1]
    values = [rows["__estimate"].iloc[index] for index in drawable]
    lows = [rows["__low"].iloc[index] for index in drawable]
    highs = [rows["__high"].iloc[index] for index in drawable]
    y = list(range(len(drawable)))
    ax.errorbar(
        values,
        y,
        xerr=[
            [value - low for value, low in zip(values, lows, strict=True)],
            [high - value for value, high in zip(values, highs, strict=True)],
        ],
        fmt="o",
        color=palette["blue"],
        ecolor=palette["neutral"],
        capsize=2.4,
    )
    if scale.null_value is not None:
        ax.axvline(scale.null_value, color=palette["neutral"], linestyle="--", linewidth=0.8)
    if scale.multiplicative and all(value > 0 for value in lows):
        configure_ratio_axis(
            ax, lows=lows, highs=highs, null_value=scale.null_value
        )
    ax.set_yticks(y, [_reader_label(rows["__label"].iloc[index]) for index in drawable])
    ax.set_xlabel(_reader_label(scale.name))
    ax.set_title("Adjusted association", loc="left", pad=8)
    add_panel_label(ax, "B", x=-0.13, y=1.04, fontsize=presentation.font_size * 1.1 if presentation else 11)
    adjustment = _adjustment_note(estimates["covariates"].iloc[0])
    if presentation is None:
        fig.text(0.51, 0.01, adjustment, ha="center", va="bottom", fontsize=5.8)
    else:
        fig.supxlabel(
            textwrap.fill(
                adjustment,
                width=max(30, int(presentation.width_mm / presentation.font_size * 3)),
            ),
            fontsize=presentation.font_size * 0.85,
        )
        finish_presented_figure(fig, presentation, base_font_size=7.0)

    evidence_ids = [
        str(distribution.evidence_id or ""),
        str(estimate_binding.get("evidence_id") or ""),
    ]
    contract = make_figure_contract(
        figure_id=figure_product,
        width_mm=float(fig.get_figwidth() * 25.4),
        height_mm=float(fig.get_figheight() * 25.4),
        title=f"{_reader_label(exposure)} and {_reader_label(outcome)}",
        core_claim=(
            "The absolute exposure/outcome distribution and the adjusted "
            "association are shown together from two digest-bound host tables."
        ),
        panels=[
            {
                "panel_id": panels[0].panel_id,
                "title": "Absolute distribution",
                "role": panels[0].article_role,
                "claim": "Exposure prevalence and observed outcome risk by declared exposure level.",
                "evidence_ids": [evidence_ids[0]],
                "metadata": {
                    "source_data": [distribution_source.name],
                    "chart_type": panels[0].chart_type,
                    "source_products": panels[0].source_products,
                },
            },
            {
                "panel_id": panels[1].panel_id,
                "title": "Adjusted association",
                "role": panels[1].article_role,
                "claim": f"Host-fitted effect estimate and confidence interval. {adjustment}",
                "evidence_ids": [evidence_ids[1]],
                "metadata": {
                    "source_data": [estimate_source.name],
                    "chart_type": panels[1].chart_type,
                    "source_products": panels[1].source_products,
                },
            },
        ],
        source_data=[distribution_source.name, estimate_source.name],
        statistics_note=(
            "All plotted values are direct projections of registered tables; "
            "the renderer performs no fitting, filtering, or denominator choice."
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
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "association",
        "method": "deterministic_association_overview_figure",
        "deterministic_standard_analysis": "association_overview_figure",
        "rendering_only": True,
        "figure_presentation": presentation.model_dump(mode="json")
        if presentation
        else None,
        "source_inputs": list(ASSOCIATION_OVERVIEW_FIGURE_INPUTS),
        "source_evidence_ids": dict(
            zip(ASSOCIATION_OVERVIEW_FIGURE_INPUTS, evidence_ids, strict=True)
        ),
        "source_data_files": [distribution_source.name, estimate_source.name],
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
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


def run_adjusted_association_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    panel_specs: list[dict[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Render the host's own adjusted estimate and write its figure contract."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", figure_product or ""):
        raise ValueError("figure product must be one canonical lowercase token")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = _association_panels(panel_specs or [], figure_product=figure_product)
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
    presentation = presentation_from_panels(panels)
    if presentation is None:
        fig, ax = plt.subplots(figsize=(120 / 25.4, height_mm / 25.4))
    else:
        fig, axes, palette = create_presented_axes(1, presentation)
        ax = axes[0]

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
        configure_ratio_axis(
            ax, lows=lows, highs=highs, null_value=scale.null_value
        )

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
    if panels[0].chart_type.startswith("forest_interval_") and panels[0].chart_type != f"forest_interval_{association_kind}_association":
        raise ValueError("unsupported_planned_figure_design: chart adjustment declaration disagrees with the source")
    model_note = f"{adjustment} {estimator[:1].upper()}{estimator[1:]} model."
    if presentation is None:
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
    else:
        fig.supxlabel(
            textwrap.fill(
                model_note,
                width=max(30, int(presentation.width_mm / presentation.font_size * 3)),
            ),
            fontsize=presentation.font_size * 0.85,
        )
    if presentation is not None:
        finish_presented_figure(fig, presentation, base_font_size=7.0)
    contract = make_figure_contract(
        figure_id=figure_product,
        width_mm=float(fig.get_figwidth() * 25.4),
        height_mm=float(fig.get_figheight() * 25.4),
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
                "panel_id": panels[0].panel_id,
                "title": f"{association_kind.capitalize()} effect estimate",
                "role": panels[0].article_role,
                "claim": (
                    f"Point estimate and confidence interval on the "
                    f"{_reader_label(scale.name)} scale. {adjustment} "
                    "Estimates that did not fit are labelled with the "
                    "producer's status rather than omitted."
                ),
                "evidence_ids": [source_path.name],
                "metadata": {
                    "chart_type": panels[0].chart_type
                    if panel_specs
                    else f"forest_interval_{association_kind}_association",
                    "source_products": panels[0].source_products,
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
        "figure_presentation": presentation.model_dump(mode="json")
        if presentation
        else None,
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
