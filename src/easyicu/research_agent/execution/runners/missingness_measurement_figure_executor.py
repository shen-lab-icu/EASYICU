"""Deterministic renderer for a closed missingness / measurement-process pair.

The executor consumes exactly the two digest-bound audit tables declared by one
direct parent under ``all_rows`` contracts and renders the availability and
measurement-process context the parent already measured.  It does not read the
cohort, choose variables, redefine missingness, or fit a model.

Each parent publishes one wide row per audited variable, and the thing this
executor exists to get right is that **one row carries two denominators**.
``eligible_n`` and ``not_applicable_n`` partition the cohort; within the
eligible stays, ``measured_one_n`` and ``value_missing_n`` partition again.
Both are re-derived here.  The published missing share is stated over the
*cohort*, so a variable that applies to only part of it can report a small
missing share while being unobservable for most stays -- that number is drawn
as the parent computed it and the variable is marked, never silently rescaled
to whichever denominator would look tidier.
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
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...contracts.figure_plan import (
    DATA_QUALITY_FIGURE_PANELS,
    MEASUREMENT_PROCESS_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    measurement_availability_figure_panels,
    resolve_data_quality_figure_inputs,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from ...schema import AnalysisStep
from ...numeric_scalars import coerce_optional_finite_float as _finite
from .deterministic_missingness import measurement_audit_product_filename
from .figure_input_capability import TypedInputCapability

__all__ = [
    "MEASUREMENT_PROCESS_AUDIT_INPUT",
    "MEASUREMENT_MISSINGNESS_FIGURE_INPUT",
    "MISSINGNESS_MEASUREMENT_AUDIT_INPUT",
    "MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND",
    "MISSINGNESS_MEASUREMENT_FIGURE_INPUTS",
    "missingness_measurement_figure_declaration_verdict",
    "missingness_measurement_figure_executor_code",
    "missingness_measurement_figure_executor_owns_step",
    "measurement_missingness_figure_executor_code",
    "measurement_missingness_figure_executor_owns_step",
    "run_measurement_missingness_figure",
    "run_missingness_measurement_figure",
]


DATA_QUALITY_FIGURE_REQUIRED_INPUTS = (
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    MEASUREMENT_PROCESS_AUDIT_INPUT,
)
# The one-panel renderer consumes the same typed audit product as panel A of
# the two-panel renderer.  Keep a role-specific alias for call-site clarity,
# but never invent a second Planner input key for the same product.
MEASUREMENT_MISSINGNESS_FIGURE_INPUT = MISSINGNESS_MEASUREMENT_AUDIT_INPUT
MISSINGNESS_MEASUREMENT_FIGURE_INPUTS = DATA_QUALITY_FIGURE_REQUIRED_INPUTS
#: One stable name for this owner across the claim, the decline and the trace.
MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND = "missingness_measurement_figure"
#: A figure product id is a Planner-owned *label*, not a capability claim.  What
#: this renderer can draw is fixed by its two required typed inputs and their
#: verified schemas, both checked below; the id never selects a panel, a
#: variable or a transform.  An allow-list of spellings therefore refuses work
#: this renderer is competent to do: a plan naming the two audit tables exactly,
#: under the right contracts, was declined for the spelling of its label alone.
#:
#: The id is still constrained, because it becomes a path segment and a
#: filename stem: ``out_dir / figure_product`` and ``{figure_product}.png``.
#: The length bound is set by that use -- the longest suffix this module
#: appends is ``_source_missingness_panel_source_data.csv`` -- so a legal id
#: cannot produce an ENAMETOOLONG failure after the run has already paid for
#: the analysis.
_FIGURE_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")


def _is_safe_figure_product_id(value: Any) -> bool:
    """Whether ``value`` is a legal, path-safe figure product id."""

    return bool(_FIGURE_PRODUCT_ID.fullmatch(str(value or "")))


# The columns this renderer READS from each parent, not the parent's full
# schema.  The previous contract demanded exact equality with a 13-column
# long-format table (``metric``/``level``/``count``/``summary_value`` ...) that
# the deterministic producer never emitted: that shape came from an
# LLM-generated audit step, and when ``deterministic_missingness`` took
# ownership of the product it emitted a 27-column wide table with zero columns
# in common.  Nothing in the tree has produced the old schema since, so this
# executor could not succeed at all -- it failed inside host code with
# "product contract is unsupported".
#
# Exact equality is also the wrong contract for a consumer even when it
# matches: it makes any column the producer ADDS a fatal error in a renderer
# that never reads it.  A consumer states what it needs; the digest binding
# already pins exactly which bytes were drawn, and the arithmetic below is
# re-derived rather than trusted.
_AUDIT_COLUMNS = (
    "variable",
    "label",
    "n_total",
    "eligible_n",
    "not_applicable_n",
    "measured_one_n",
    "value_missing_n",
    "value_missing_pct",
    "indicator_semantics",
)
_PROCESS_COLUMNS = (
    "variable",
    "n_total",
    "eligible_n",
    "measured_one_n",
    "repeat_measured_n",
)
#: Measurement-process measures drawn on panel B.  Each is a count of *stays*
#: and is therefore commensurable with ``n_total``; that is the property the
#: panel's shared 0-100% scale depends on, and it is re-checked per row.
#: ``measurement_total_n`` / ``measurement_count_max`` /
#: ``measurement_count_median_when_measured`` are deliberately excluded: they
#: count measurements, not stays (a real run reported 24,179 measurements over
#: 1,000 stays), so drawing them on this scale would be arithmetic nonsense.
#: A measure the producer stops emitting fails closed rather than silently
#: dropping a column from the reader's grid.
_PROCESS_MEASURES = (
    ("eligible_n", "Applicable"),
    ("measured_one_n", "Measured >=1"),
    ("repeat_measured_n", "Measured >1"),
)
_PRODUCT_BY_INPUT = {
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT: "missingness_measurement_audit",
    MEASUREMENT_PROCESS_AUDIT_INPUT: "measurement_process_audit",
}
_COLUMNS_BY_INPUT = {
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT: _AUDIT_COLUMNS,
    MEASUREMENT_PROCESS_AUDIT_INPUT: _PROCESS_COLUMNS,
}


def _measurement_missingness_input(
    step: AnalysisStep,
    *,
    plan: Any | None = None,
) -> str | None:
    """Return the sole typed product backed by the measurement audit owner."""

    if len(step.inputs or ()) != 1:
        return None
    input_key = str(step.inputs[0] or "").strip()
    kind, separator, product = input_key.partition(":")
    if kind != "table" or not separator:
        return None
    if (
        measurement_audit_product_filename(product)
        == "missingness_measurement_audit.csv"
    ):
        return input_key
    producers = [
        candidate
        for candidate in getattr(plan, "steps", ()) or ()
        if input_key in {str(value) for value in candidate.expected_outputs}
        and candidate.measurement_audit_spec is not None
        and candidate.measurement_audit_spec.audit_for(product)
        == "measurement_missingness"
    ]
    if len(producers) != 1:
        return None
    return input_key


def _columns_read(
    input_key: str,
    *,
    audit_role: str | None = None,
) -> tuple[str, ...]:
    if audit_role == "measurement_process" or (
        audit_role is None and input_key == MEASUREMENT_PROCESS_AUDIT_INPUT
    ):
        return _PROCESS_COLUMNS
    kind, separator, _product = str(input_key or "").partition(":")
    if kind == "table" and separator:
        return _AUDIT_COLUMNS
    raise ValueError(f"{input_key} is not a supported measurement-audit product")


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not _is_safe_figure_product_id(product):
        return None
    return product


#: ``run_missingness_measurement_figure`` indexes both bindings and builds a
#: panel from each, so neither may be declared optional.  A plan that names
#: only the missingness audit is refused here and stays refused: the E1
#: protocol layer is what requires the process audit be produced, and this
#: renderer is not the place to paper over a plan that did not promise it.
MISSINGNESS_MEASUREMENT_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
)


def _binding_carries_the_columns_read(
    binding: Any,
    input_key: str,
    *,
    audit_role: str | None = None,
) -> bool:
    """Whether the bound table really has the columns this renderer indexes.

    The loader below already asks this and raises when the answer is no. Asking
    it again HERE, before the claim, is the difference between declining a step
    and killing it: claiming is a promise to produce the figure, so a renderer
    that claims and then raises turns a step the Coder was drawing successfully
    into a dead one.

    Measured 2026-07-31 over the real files on disk: 23 recorded
    ``missingness_measurement_audit`` tables carry 5 distinct headers and only
    16 have every column this renderer reads; ``measurement_process_audit`` is
    16 of 20. Those 11 files are steps this renderer would have claimed and
    then failed. The same gap in the robustness renderer was worth 4 dead steps
    before it was closed.
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
    return set(_columns_read(input_key, audit_role=audit_role)).issubset(set(columns))


def missingness_measurement_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: Any | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether every scientific choice is fixed by the typed contract."""

    products = [_figure_product(value) for value in step.expected_outputs]
    input_by_role = resolve_data_quality_figure_inputs(
        step.inputs,
        steps=getattr(plan, "steps", ()) or (),
    )
    if input_by_role is None:
        return False
    required_inputs = frozenset(input_by_role.values())
    all_row_inputs = {
        str(contract.input_key)
        for contract in step.input_consumption_contracts or ()
        if contract.mode == "all_rows"
    }
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and {str(value) for value in step.inputs} == set(required_inputs)
        and all_row_inputs == set(required_inputs)
        and len(products) == 1
        and products[0] is not None
        # ``model_requirements`` and ``table_one_spec`` are not checked here:
        # ``AnalysisStep`` already refuses both on a visualization step whose
        # sole output is one figure (verified 2026-07-31), so guarding them
        # would read as protection while protecting nothing.
        and step.trajectory_stability_spec is None
    ):
        return False
    if not isinstance(resolved_bindings, Mapping):
        return False
    return all(
        _binding_carries_the_columns_read(
            resolved_bindings.get(input_key),
            input_key,
            audit_role=role,
        )
        for role, input_key in input_by_role.items()
    )


def measurement_missingness_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: Any | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    """Own a one-panel figure whose only authority is one full audit table."""

    products = [_figure_product(value) for value in step.expected_outputs]
    input_key = _measurement_missingness_input(step, plan=plan)
    contracts = list(step.input_consumption_contracts or ())
    if not (
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and input_key is not None
        and len(contracts) == 1
        and contracts[0].input_key == input_key
        and contracts[0].mode == "all_rows"
        and len(products) == 1
        and products[0] is not None
        and step.trajectory_stability_spec is None
    ):
        return False
    if not isinstance(resolved_bindings, Mapping):
        return False
    return _binding_carries_the_columns_read(
        resolved_bindings.get(input_key),
        input_key,
    )


def missingness_measurement_figure_declaration_verdict(
    step: AnalysisStep,
    *,
    plan: Any,
) -> OwnershipVerdict:
    """Report the one input a step must add for this owner to draw its figure.

    This renderer needs both audit tables because it builds a panel from each.
    A plan that names only one leaves it unclaimed, and the step falls to the
    Coder -- which is how a figure the host can draw deterministically ends up
    as a hand-written source-data table the traceability validator refuses.

    MEASURED over every recorded run: 59 figure steps name at least one of the
    two audit tables.  19 name both and this owner can claim them.  40 name one
    -- and they split in two, which is the whole reason this function has a
    guard rather than firing on all 40:

    * 9 declare one table whose producing step ALSO produces the other.  The
      plan can close the gap by adding one string to the figure step, and
      nothing about the science changes: the same parent, the same digest, the
      same two tables already on disk.  Those are what this reports.
    * 31 name a table whose sibling no step in the plan produces at all.
      Closing those means asking the parent for a different analysis, which is
      a scientific choice this owner does not get to make -- the same boundary
      the distribution owner had to be narrowed to after canary33.

    m1's ``09_missingness_audit_figure`` is the first kind: its parent declares
    both tables and writes both files, the figure step names one, and the
    renderer sat idle while the Coder produced a source-data table whose
    columns could not be traced back to any upstream vector.
    """

    products = [_figure_product(value) for value in step.expected_outputs or []]
    if not (
        len(products) == 1
        and products[0] is not None
        and step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and step.trajectory_stability_spec is None
    ):
        return OwnershipVerdict.wrong_shape(
            MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND,
            reason=(
                "the step is not one auxiliary visualization promising a single "
                "figure, so this owner could not draw it however it were declared"
            ),
        )

    declared = {str(value or "").strip() for value in step.inputs or []}
    named = [key for key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS if key in declared]
    if len(named) != 1:
        return OwnershipVerdict.wrong_shape(
            MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND,
            reason=(
                "the step does not name exactly one of this owner's two audit "
                "tables, so the gap is not a single missing input declaration"
            ),
        )
    have = named[0]
    missing = next(key for key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS if key != have)

    # The gap is only reportable when the plan can close it here. If NO step
    # produces the sibling table, adding it to this figure's inputs names an
    # artifact nobody writes; asking a step to produce it is asking for a
    # different analysis, and that belongs to the Planner directive rather than
    # to a refusal raised at this step.
    #
    # WIDENED 2026-08-04 from "the same step produces both" to "any step
    # produces it". The first version read the renderer's docstring, which
    # speaks of "one direct parent" -- but no code requires that: ``owns_step``
    # asks only that both keys resolve to bindings carrying the columns it
    # reads, and each binding is digest-pinned to its own producer. The
    # narrower rule was a restriction I introduced, not one the renderer has.
    #
    # It cost a real case immediately. e2/verify20 planned the two audits as
    # SEPARATE steps (04 and 05), its figure named one, the verdict stayed
    # silent, the step fell to the Coder and died on source-data traceability.
    # Measured over every recorded plan, the split is 9 same-step, 1
    # different-step, 31 produced-by-nobody -- so this widening reports exactly
    # one more case and still refuses the 31 it must.
    producer_of = {}
    for candidate in getattr(plan, "steps", None) or []:
        for output in getattr(candidate, "expected_outputs", None) or []:
            producer_of.setdefault(str(output or "").strip(), candidate)
    if producer_of.get(have) is None or producer_of.get(missing) is None:
        return OwnershipVerdict.wrong_shape(
            MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND,
            reason=(
                f"no step in this plan produces {missing!r}, so this owner "
                "could not be given the second table however this step were "
                "declared"
            ),
        )
    parent = producer_of[missing]

    return OwnershipVerdict.incomplete_declaration(
        MISSINGNESS_MEASUREMENT_FIGURE_ANALYSIS_KIND,
        missing=(missing,),
        reason=(
            f"the host draws this figure deterministically from {have!r} and "
            f"{missing!r} together, and step "
            f"{str(getattr(parent, 'step_id', '') or '')!r} already produces "
            f"both. This step names only {have!r}, so the renderer cannot be "
            f"given the second panel's table: declare {missing!r} beside it. "
            "Without it the figure is written by the Coder, whose source-data "
            "table renames the audited columns and cannot be traced back to "
            "the parent it came from"
        ),
    )


def missingness_measurement_figure_executor_code(
    step: AnalysisStep,
    *,
    plan: Any | None = None,
) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    # Ownership is NOT re-derived here. The selector consulted this owner with
    # the step's resolved bindings; a second evaluation without them cannot see
    # what the selector saw and would answer differently. What this builder
    # checks is its own input: one canonical figure product to render.
    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError(
            "The step is not owned by the missingness/measurement renderer"
        )
    input_by_role = resolve_data_quality_figure_inputs(
        step.inputs,
        steps=getattr(plan, "steps", ()) or (),
    )
    if input_by_role is None:
        raise ValueError(
            "The step has no closed missingness/measurement audit pair"
        )
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
            run_missingness_measurement_figure,
        )

        run_missingness_measurement_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            missingness_input={input_by_role['measurement_missingness']!r},
            process_input={input_by_role['measurement_process']!r},
        )
        """
    ).strip()


def measurement_missingness_figure_executor_code(
    step: AnalysisStep,
    *,
    plan: Any | None = None,
) -> str:
    """Return the sandbox entrypoint for one digest-bound audit figure."""

    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("The step is not owned by the measurement-missingness renderer")
    input_key = _measurement_missingness_input(step, plan=plan)
    if input_key is None:
        raise ValueError("The step has no supported measurement-audit input")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
            run_measurement_missingness_figure,
        )

        run_measurement_missingness_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            input_key={input_key!r},
        )
        """
    ).strip()


def _canonical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_one_binding(
    *,
    run_dir: Path,
    inputs: Mapping[str, Any],
    input_key: str,
    audit_role: str | None = None,
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    binding = inputs.get(input_key)
    if not isinstance(binding, dict):
        raise ValueError(f"{input_key} binding is absent")
    _kind, _separator, product = input_key.partition(":")
    expected_columns = list(_columns_read(input_key, audit_role=audit_role))
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
        or binding.get("product") != product
        or identity.get("input_key") != input_key
        or identity.get("product") != product
        or identity.get("sha256") != expected_sha256
        or consumption.get("input_key") != input_key
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError(f"{input_key} authority binding is incomplete")

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError(f"{input_key} binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"{input_key} must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError(f"{input_key} digest verification failed")

    row_count = product_contract.get("row_count")
    declared_columns = product_contract.get("columns")
    if (
        not isinstance(declared_columns, list)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError(f"{input_key} product contract is unsupported")
    absent = [name for name in expected_columns if name not in declared_columns]
    if absent:
        raise ValueError(
            f"{input_key} product contract omits the columns this figure reads: "
            + ", ".join(absent)
        )
    frame = pd.read_csv(path)
    if list(frame.columns) != declared_columns or len(frame) != row_count:
        raise ValueError(f"{input_key} bytes disagree with its product contract")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError(f"{input_key} changed while it was being read")
    # The digest-verified EvidenceStore copy preserves the producer basename as
    # ``<evidence_id>__<basename>``.  Use that bound basename rather than a
    # second alias table: typed MeasurementAuditSpec may authorize a valid
    # product id the legacy filename shim has never seen.
    source_filename = path.name.split("__", 1)[1] if "__" in path.name else path.name
    return frame, binding, source_filename


def _load_bindings(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_by_role: Mapping[str, str] | None = None,
) -> dict[str, tuple[pd.DataFrame, Mapping[str, Any], str]]:
    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    roles = dict(
        input_by_role
        or {
            "measurement_missingness": MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
            "measurement_process": MEASUREMENT_PROCESS_AUDIT_INPUT,
        }
    )
    if set(roles) != {"measurement_missingness", "measurement_process"} or len(
        set(roles.values())
    ) != 2:
        raise ValueError("audit-table role bindings are incomplete or ambiguous")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(roles.values()):
        raise ValueError("exact audit-table bindings are absent or widened")
    return {
        input_key: _load_one_binding(
            run_dir=run_dir,
            inputs=inputs,
            input_key=input_key,
            audit_role=role,
        )
        for role, input_key in roles.items()
    }


def _load_single_measurement_binding(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_key: str,
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {input_key}:
        raise ValueError("exact measurement-missingness binding is absent or widened")
    return _load_one_binding(
        run_dir=run_dir,
        inputs=inputs,
        input_key=input_key,
    )


def _integer(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _percentage_matches(value: Any, count: int, denominator: int) -> bool:
    parsed = _finite(value)
    if parsed is None:
        return False
    return math.isclose(
        parsed,
        100.0 * count / denominator,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _validate_audit_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Re-derive every audited variable's accounting from its own counts.

    The parent publishes one wide row per variable.  Two partitions hold and
    both are checked, because they answer different questions and a figure that
    silently picked one would mislabel the other: ``eligible_n`` and
    ``not_applicable_n`` partition the cohort, and within the eligible stays
    ``measured_one_n`` and ``value_missing_n`` partition again.  The published
    ``value_missing_pct`` is stated against the *cohort*, while reader-facing
    missingness must use the eligible denominator.  Conditional fields are
    therefore rendered as available, eligible-but-missing, and not applicable
    rather than allowing event prevalence to masquerade as measurement rate.
    """

    per_variable: dict[str, dict[str, Any]] = {}
    for index, row in frame.iterrows():
        variable = _text(row["variable"])
        if not variable:
            raise ValueError(f"missingness audit row {index} names no variable")
        if variable in per_variable:
            raise ValueError(f"variable {variable!r} appears twice in the audit")
        counts: dict[str, int] = {}
        for name in (
            "n_total",
            "eligible_n",
            "not_applicable_n",
            "measured_one_n",
            "value_missing_n",
        ):
            value = _integer(row[name])
            if value is None:
                raise ValueError(
                    f"variable {variable!r} has no whole-stay count for {name!r}"
                )
            counts[name] = value
        cohort = counts["n_total"]
        if cohort <= 0:
            raise ValueError(f"variable {variable!r} has no positive cohort size")
        if counts["eligible_n"] + counts["not_applicable_n"] != cohort:
            raise ValueError(
                f"variable {variable!r} eligible and not-applicable counts do not "
                "partition the cohort"
            )
        if counts["measured_one_n"] + counts["value_missing_n"] != counts["eligible_n"]:
            raise ValueError(
                f"variable {variable!r} measured and missing counts do not "
                "partition its eligible stays"
            )
        if not _percentage_matches(
            row["value_missing_pct"], counts["value_missing_n"], cohort
        ):
            raise ValueError(
                f"variable {variable!r} missing percentage does not reconcile "
                "against the cohort it is stated over"
            )
        semantics = _text(row["indicator_semantics"])
        if counts["not_applicable_n"] and semantics != "conditional_event_time":
            raise ValueError(
                f"variable {variable!r} has not-applicable stays without "
                "conditional event-time semantics"
            )
        eligible = counts["eligible_n"]
        per_variable[variable] = {
            "denominator": cohort,
            "eligible": eligible,
            "not_applicable": counts["not_applicable_n"],
            "missing": counts["value_missing_n"],
            "missing_pct": 100.0 * counts["value_missing_n"] / cohort,
            "missing_within_applicable_pct": (
                100.0 * counts["value_missing_n"] / eligible if eligible else 0.0
            ),
            "available": counts["measured_one_n"],
            "available_pct": 100.0 * counts["measured_one_n"] / cohort,
            "not_applicable_pct": 100.0 * counts["not_applicable_n"] / cohort,
            "conditional": semantics == "conditional_event_time",
            "indicator_semantics": semantics,
        }
    if not per_variable:
        raise ValueError("the missingness audit table audits no variable")
    return per_variable


def _validate_process_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn one wide process row per variable into verified panel cells.

    Every drawn measure must be a count of stays bounded by the cohort, and
    the measurement funnel must narrow: a stay measured more than once was
    measured at least once, and a stay measured at all was eligible.  A row
    that violates the nesting is not a smaller bar, it is a different
    denominator, so it fails closed instead of being drawn.
    """

    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in frame.iterrows():
        variable = _text(row["variable"])
        if not variable:
            raise ValueError(f"measurement-process row {index} names no variable")
        if variable in seen:
            raise ValueError(
                f"variable {variable!r} appears twice in the measurement-process audit"
            )
        seen.add(variable)
        cohort = _integer(row["n_total"])
        if cohort is None or cohort <= 0:
            raise ValueError(
                f"measurement-process row {index} has no positive cohort size"
            )
        counts: dict[str, int] = {}
        for column, _label in _PROCESS_MEASURES:
            value = _integer(row[column])
            if value is None or value > cohort:
                raise ValueError(
                    f"variable {variable!r} reports {column!r} as something other "
                    "than a stay count within its cohort"
                )
            counts[column] = value
        if not (
            counts["repeat_measured_n"]
            <= counts["measured_one_n"]
            <= counts["eligible_n"]
        ):
            raise ValueError(
                f"variable {variable!r} measurement counts do not nest: "
                "repeatedly measured stays must be a subset of measured stays, "
                "which must be a subset of eligible stays"
            )
        for column, label in _PROCESS_MEASURES:
            cells.append(
                {
                    "variable": variable,
                    "process_measure": column,
                    "level": "",
                    "column": label,
                    "count": counts[column],
                    "denominator": cohort,
                    "percentage": 100.0 * counts[column] / cohort,
                }
            )
    if not cells:
        raise ValueError("the measurement-process table audits no variable")
    return cells


def _reader_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip()
    return cleaned if cleaned else "Variable"


def _write_source_projection(
    rows: pd.DataFrame,
    *,
    path: Path,
    source_table: str,
) -> None:
    export = rows.copy()
    export.insert(0, "source_row_index", export.index.astype(int))
    export.insert(1, "source_table", source_table)
    export.to_csv(path, index=False)


def run_measurement_missingness_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    input_key: str = MEASUREMENT_MISSINGNESS_FIGURE_INPUT,
) -> Mapping[str, Any]:
    """Render one complete typed missingness audit without model-authored lineage."""

    if not _is_safe_figure_product_id(figure_product):
        raise ValueError("unsafe or malformed figure product id")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_frame, audit_binding, audit_table = _load_single_measurement_binding(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        input_key=input_key,
    )
    per_variable = _validate_audit_rows(audit_frame)

    source_path = out_dir / f"{figure_product}_source_data.csv"
    _write_source_projection(
        audit_frame,
        path=source_path,
        source_table=audit_table,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    display = audit_frame.copy()
    display["_missing_pct"] = [
        per_variable[_text(value)]["missing_within_applicable_pct"]
        for value in display["variable"]
    ]
    display = display.sort_values("_missing_pct", ascending=True)
    variables = [_text(value) for value in display["variable"]]
    labels = [
        _reader_label(_text(label) or variable)
        + (" (conditional)" if per_variable[variable]["conditional"] else "")
        for variable, label in zip(variables, display["label"])
    ]
    missing_pct = [per_variable[name]["missing_pct"] for name in variables]
    available_pct = [per_variable[name]["available_pct"] for name in variables]
    not_applicable_pct = [
        per_variable[name]["not_applicable_pct"] for name in variables
    ]
    height_mm = max(82.0, 31.0 + 7.0 * len(variables))
    panel_template = measurement_availability_figure_panels(input_key)[0]
    fig, ax = plt.subplots(figsize=(183 / 25.4, height_mm / 25.4))
    positions = list(range(len(variables)))
    ax.barh(
        positions,
        available_pct,
        color=palette["blue_soft"],
        label="Value available",
        height=0.62,
    )
    ax.barh(
        positions,
        missing_pct,
        left=available_pct,
        color=palette["orange"],
        label="Missing among applicable stays",
        height=0.62,
    )
    ax.barh(
        positions,
        not_applicable_pct,
        left=[available + missing for available, missing in zip(available_pct, missing_pct)],
        color=palette["neutral_light"],
        label="Not applicable",
        height=0.62,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the declared cohort (%)")
    ax.set_title("Data availability and applicability", loc="left", pad=4)
    ax.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.58, 1.10),
        ncol=3,
        fontsize=6.2,
        columnspacing=1.2,
    )
    for y, name in zip(positions, variables):
        entry = per_variable[name]
        if entry["conditional"]:
            annotation = (
                f"{entry['eligible']:,} applicable; "
                f"{entry['missing']:,}/{entry['eligible']:,} missing"
            )
        else:
            annotation = f"{entry['missing']:,}/{entry['eligible']:,} missing"
        ax.text(
            98.0,
            y,
            annotation,
            va="center",
            ha="right",
            fontsize=5.9,
            color=palette["blue"],
        )
    if any(entry["conditional"] for entry in per_variable.values()):
        ax.set_xlabel(
            "Share of the declared cohort (%)\n"
            "Conditional event times are applicable only when the event occurs"
        )
    add_panel_label(ax, "A", x=-0.20, y=1.02)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.20, top=0.76)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Availability, eligible-stay missingness, and non-applicability are "
            "rendered separately from every row of one digest-verified audit table."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=height_mm,
        panels=[
            {
                "panel_id": panel_template.panel_id,
                "title": "Data availability and applicability",
                "role": "data_quality",
                "claim": (
                    "For every audited variable, available and missing stays "
                    "partition the applicable denominator; non-applicable stays "
                    "remain a separate cohort partition."
                ),
                "evidence_ids": [source_path.name],
                "metadata": {
                    "article_role": panel_template.article_role,
                    "chart_type": panel_template.chart_type,
                    "source_products": [input_key],
                    "source_data": [source_path.name],
                },
            }
        ],
        source_data=[source_path.name],
        statistics_note=(
            "The executor consumes every row under the all_rows contract and "
            "re-derives both cohort and eligible-stay count partitions. It "
            "introduces no variable selection, imputation, denominator, or "
            "modeling decision."
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
        "method": "deterministic_measurement_missingness_figure",
        "analysis_family": "data_quality",
        "deterministic_standard_analysis": "measurement_missingness_figure",
        "rendering_only": True,
        "source_inputs": [input_key],
        "source_evidence_ids": {
            input_key: audit_binding.get("evidence_id")
        },
        "source_sha256": {
            input_key: audit_binding.get("sha256")
        },
        "source_rows_consumed": {
            input_key: int(len(audit_frame))
        },
        "audited_variable_count": int(len(per_variable)),
        "source_data_files": [source_path.name],
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
        "input_bindings": [
            {
                "input_key": input_key,
                "evidence_id": audit_binding.get("evidence_id"),
                "sha256": audit_binding.get("sha256"),
                "loaded": True,
                "row_count": int(len(audit_frame)),
            }
        ],
        "export_qa": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_missingness_measurement_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    missingness_input: str = MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    process_input: str = MEASUREMENT_PROCESS_AUDIT_INPUT,
) -> Mapping[str, Any]:
    """Render the verified missingness/measurement pair and write its contract."""

    # The selector parses this id out of ``figure:<id>``; this entry point is
    # public, so it re-checks rather than trusting its caller.  Everything below
    # interpolates it into a path.
    if not _is_safe_figure_product_id(figure_product):
        raise ValueError("unsafe or malformed figure product id")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bindings = _load_bindings(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        input_by_role={
            "measurement_missingness": missingness_input,
            "measurement_process": process_input,
        },
    )
    audit_frame, audit_binding, audit_table = bindings[missingness_input]
    process_frame, process_binding, process_table = bindings[process_input]
    per_variable = _validate_audit_rows(audit_frame)
    process_cells = _validate_process_rows(process_frame)

    audit_source = out_dir / f"{figure_product}_missingness_source_data.csv"
    process_source = out_dir / f"{figure_product}_measurement_process_source_data.csv"
    missingness_panel_source = (
        out_dir / f"{figure_product}_source_missingness_panel_source_data.csv"
    )
    _write_source_projection(audit_frame, path=audit_source, source_table=audit_table)
    _write_source_projection(
        process_frame,
        path=process_source,
        source_table=process_table,
    )
    # Every panel projection is a verbatim row subset of its parent, keeping the
    # parent's own values and row positions.  A derived column (an availability
    # complement, a reshaped coverage grid) cannot be traced back to any single
    # upstream row, so it is computed for validation and reported in the step
    # summary rather than published as if it were source data.
    missing_rows = audit_frame.sort_values("value_missing_pct", ascending=False)
    _write_source_projection(
        missing_rows,
        path=missingness_panel_source,
        source_table=audit_table,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    variables = [_text(value) for value in missing_rows["variable"]]
    # Declared order, not alphabetical: the three measures are a funnel, and
    # sorting them by name would put "Measured >1" before "Measured >=1".
    columns = [label for _column, label in _PROCESS_MEASURES]
    grid_variables = sorted({str(cell["variable"]) for cell in process_cells})
    height_mm = max(86.0, 26.0 + 5.4 * max(len(variables), len(grid_variables)))
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, height_mm / 25.4),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    positions = list(range(len(variables)))
    missing_pct = pd.to_numeric(missing_rows["value_missing_pct"]).to_numpy()
    missing_counts = (
        pd.to_numeric(missing_rows["value_missing_n"]).astype(int).to_numpy()
    )
    zero_missing_display = bool(len(missing_counts)) and all(
        int(value) == 0 for value in missing_counts
    )
    if zero_missing_display:
        # A zero-only missingness bar chart has no visible marks and can look
        # like a rendering failure.  Re-express the same audited counts as
        # completeness among eligible stays: 100% is derived exactly from
        # measured + missing = eligible, never invented.  An all-not-applicable
        # row has no eligible denominator and remains an explicit 0/N/A mark.
        plotted_values = [
            (
                100.0
                * float(per_variable[name]["available"])
                / float(per_variable[name]["eligible"])
                if int(per_variable[name]["eligible"]) > 0
                else 0.0
            )
            for name in variables
        ]
    else:
        plotted_values = [float(value) for value in missing_pct]
    bars = ax_a.barh(
        positions,
        plotted_values,
        color=palette["blue_soft"],
        height=0.62,
    )
    ax_a.set_yticks(positions)
    # A variable that does not apply to every stay is marked, because its
    # missing share is stated over the whole cohort: without the mark a
    # conditional variable observed for a tenth of the cohort reads as if it
    # were almost completely observed.
    ax_a.set_yticklabels(
        [
            _reader_label(name)
            + (" †" if per_variable.get(name, {}).get("conditional") else "")
            for name in variables
        ]
    )
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel(
        "Eligible stays with a source value (%)"
        if zero_missing_display
        else "Stays with no source value (% of cohort)"
    )
    ax_a.set_title(
        "Source completeness" if zero_missing_display else "Source missingness",
        loc="left",
        pad=4,
    )
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for variable, bar, percentage, plotted, missing_n in zip(
        variables,
        bars,
        missing_pct,
        plotted_values,
        missing_counts,
    ):
        if zero_missing_display:
            eligible_n = int(per_variable[variable]["eligible"])
            label = (
                "0 missing / 100% complete"
                if eligible_n > 0
                else "0 eligible / completeness N/A"
            )
            x = min(float(plotted) - 1.0, 97.0) if eligible_n > 0 else 1.0
            horizontal_alignment = "right" if eligible_n > 0 else "left"
        else:
            label = f"{float(percentage):.1f}%  n={int(missing_n):,}"
            x = min(float(percentage) + 1.0, 97.0)
            horizontal_alignment = "left" if percentage < 88 else "right"
        ax_a.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=horizontal_alignment,
            fontsize=6.1,
        )
    if any(entry["conditional"] for entry in per_variable.values()) and not (
        zero_missing_display
    ):
        ax_a.set_xlabel(
            "Stays with no source value (% of cohort)\n"
            "† applies to only part of the cohort; see panel B"
        )
    add_panel_label(ax_a, "A", x=-0.30, y=1.02)

    matrix = [
        [
            next(
                (
                    float(cell["percentage"])
                    for cell in process_cells
                    if cell["variable"] == variable and cell["column"] == column
                ),
                float("nan"),
            )
            for column in columns
        ]
        for variable in grid_variables
    ]
    frame = pd.DataFrame(matrix, index=grid_variables, columns=columns)
    image = ax_b.imshow(
        frame.to_numpy(dtype=float),
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )
    ax_b.set_xticks(range(len(columns)))
    ax_b.set_xticklabels(
        columns,
        rotation=35,
        ha="right",
    )
    ax_b.set_yticks(range(len(grid_variables)))
    ax_b.set_yticklabels([_reader_label(name) for name in grid_variables])
    ax_b.set_title("Measurement-process coverage", loc="left", pad=4)
    for row_index, variable in enumerate(grid_variables):
        for column_index, column in enumerate(columns):
            value = frame.iat[row_index, column_index]
            if math.isnan(float(value)):
                ax_b.text(
                    column_index,
                    row_index,
                    "–",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=palette["neutral"],
                )
                continue
            ax_b.text(
                column_index,
                row_index,
                f"{float(value):.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if float(value) >= 55.0 else palette["blue"],
            )
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.032, pad=0.02)
    colorbar.set_label("Share of the cohort (%)", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.8)
    add_panel_label(ax_b, "B", x=-0.30, y=1.02)
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.22, top=0.88, wspace=0.72)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Source availability and measurement-process coverage are rendered "
            "from two digest-verified parent audit tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=height_mm,
        panels=[
            {
                "panel_id": DATA_QUALITY_FIGURE_PANELS[0].panel_id,
                "title": (
                    "Source completeness"
                    if zero_missing_display
                    else "Source missingness"
                ),
                "role": "data_quality",
                "claim": (
                    (
                        "Every audited variable has zero missing source values; "
                        "the visible bars report 100% completeness among eligible "
                        "stays, while rows with no eligible stays remain N/A."
                    )
                    if zero_missing_display
                    else (
                        "Each audited variable's source-missingness share is the "
                        "parent table's own value, restated over the cohort it was "
                        "computed against; measured and missing counts partition "
                        "the variable's eligible stays, and eligible plus "
                        "not-applicable stays partition the cohort."
                    )
                ),
                "evidence_ids": [missingness_panel_source.name],
                "metadata": {
                    "article_role": DATA_QUALITY_FIGURE_PANELS[0].article_role,
                    "chart_type": DATA_QUALITY_FIGURE_PANELS[0].chart_type,
                    "source_products": [missingness_input],
                    "source_data": [missingness_panel_source.name],
                    "zero_missing_completeness_display": zero_missing_display,
                },
            },
            {
                "panel_id": DATA_QUALITY_FIGURE_PANELS[1].panel_id,
                "title": "Measurement-process coverage",
                "role": "data_quality",
                "claim": (
                    "Applicable, measured-at-least-once and measured-more-than-"
                    "once stays are each shown as a share of the same cohort, "
                    "so the three columns read as one narrowing funnel."
                ),
                "evidence_ids": [process_source.name],
                "metadata": {
                    "article_role": DATA_QUALITY_FIGURE_PANELS[1].article_role,
                    "chart_type": DATA_QUALITY_FIGURE_PANELS[1].chart_type,
                    "source_products": [process_input],
                    "source_data": [process_source.name],
                },
            },
        ],
        source_data=[
            audit_source.name,
            process_source.name,
            missingness_panel_source.name,
        ],
        statistics_note=(
            "Percentages are recomputed from the sealed integer counts. Both "
            "partitions are re-derived per variable (eligible + not applicable "
            "= cohort; measured + missing = eligible) and the measurement "
            "counts are required to nest. Measurement totals and per-stay "
            "measurement medians are deliberately not drawn: they count "
            "measurements, not stays, and are not commensurable with this "
            "scale. The executor validates all source rows and introduces no "
            "cohort, variable, missing-data, or modeling decision."
            + (
                " Because every missing count is zero, panel A deterministically "
                "shows eligible-stay completeness (100% when an eligible "
                "denominator exists) instead of an invisible zero-length "
                "missingness bar."
                if zero_missing_display
                else ""
            )
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
    source_files = [
        audit_source.name,
        process_source.name,
        missingness_panel_source.name,
    ]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_missingness_measurement_figure",
        "analysis_family": "data_quality",
        "deterministic_standard_analysis": "missingness_measurement_figure",
        "rendering_only": True,
        "source_inputs": [missingness_input, process_input],
        "source_evidence_ids": {
            missingness_input: audit_binding.get("evidence_id"),
            process_input: process_binding.get("evidence_id"),
        },
        "source_sha256": {
            missingness_input: audit_binding.get("sha256"),
            process_input: process_binding.get("sha256"),
        },
        "source_rows_consumed": {
            missingness_input: int(len(audit_frame)),
            process_input: int(len(process_frame)),
        },
        "audited_variable_count": int(len(per_variable)),
        "zero_missing_completeness_display": zero_missing_display,
        "measurement_process_cell_count": int(len(process_cells)),
        "source_data_files": source_files,
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
        "input_bindings": [
            {
                "input_key": input_key,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "loaded": True,
                "row_count": int(len(loaded)),
            }
            for input_key, (loaded, binding, _) in bindings.items()
        ],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
