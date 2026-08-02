"""Deterministic preflight for pre-specified robustness comparisons.

The generated runner consumes the plan-time robustness lock and completed-step
manifest, then emits the table shape expected by the existing deterministic
sensitivity renderer.  It is deliberately conservative: a missing primary
estimate blocks the step, relaxed cohort variants require the pre-filter
universe, and repeated aggregations of the same stay-level scalar outcome are
reported as non-independent instead of being re-fit as duplicate evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.planned_role import unique_verified_primary_record
from ...cohort.schema import build_cohort, coerce_cohort_definition
from ...robustness.estimators import (
    _data_with_predicate_aliases,
    fit_robustness_rows_from_records,
)
from ...robustness.primary_effect import (
    _extract_primary_effect_payload_from_records,
    _primary_effect_payload_is_complete,
)
from .typed_input_binding import contained_regular_file
from ...robustness.membership import (
    _identifier_column,
    _membership_audit,
    replay_locked_memberships,
)
from ...robustness.panel import (
    PRIMARY_SPEC_ID,
    RobustnessPanelRow,
    RobustnessSpec,
    _assert_lock_matches_evidence_anchor,
    robustness_specs_sha,
    validate_robustness_specs,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from ...contracts.result_envelope import (
    MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY,
    MODEL_SUMMARY_COVARIATE_KEYS,
    MODEL_SUMMARY_EXPOSURE_KEYS,
    MODEL_SUMMARY_OUTCOME_KEYS,
    model_summary_analysis_definition,
    model_summary_coefficient_filename,
)
from ...planning.robustness_contract import (
    COMPLETE_CASE_VARIABLES_KEY,
    complete_case_variables,
)
from ...schema import AnalysisStep, spec_backs_every_declared_product
from ...contracts.host_scaffold import HostScaffoldedScript
from .plausibility_receipt import render_standard_plausibility_receipt_code

__all__ = [
    "ROBUSTNESS_REPLAY_OUTPUT_FILES",
    "declared_robustness_product_registrations",
    "replay_locked_memberships",
    "ROBUSTNESS_REPLAY_ANALYSIS_KIND",
    "ROBUSTNESS_REPLAY_OUTPUT_KINDS",
    "robustness_replay_declaration_verdict",
    "robustness_replay_spec_is_emittable",
    "robustness_sensitivity_preflight_code",
    "robustness_sensitivity_preflight_scaffold",
]

# The one declaration of what this replay can answer: output -> canonical file.
#
# Keyed on the answer, not on the filename.  Several of these frames are copied
# to additional filenames below (``sensitivity_comparison.csv`` is byte-for-byte
# ``robustness_matrix.csv``; ``membership_change_summary.csv`` is written three
# times), because a fail-closed gate in ``gates/contract.py`` resolves declared
# spec/denominator tables by *filename* and those aliases are what it looks for.
# The copies therefore stay.  What does not stay is the possibility of promising
# a reader two products and satisfying it with one answer twice: a declaration
# keyed here cannot name the same output for two products.
#: This owner's one name, in its verdict and in the selector trace.  A
#: retyped kind literal is how two layers end up disagreeing about which
#: owner produced an artifact (task #95/N6).
ROBUSTNESS_REPLAY_ANALYSIS_KIND = "robustness_replay"

#: The product kinds this replay writes.  One declaration, read by both the
#: emittability check and the declaration-gap verdict, so the two can never
#: disagree about what this runner can produce -- and so ``figure`` staying out
#: of it is a single fact rather than two places that must be edited together.
ROBUSTNESS_REPLAY_OUTPUT_KINDS: frozenset = frozenset({"table", "statistic", "log"})

ROBUSTNESS_REPLAY_OUTPUT_FILES: Mapping[str, str] = MappingProxyType(
    {
        "robustness_matrix": "robustness_matrix.csv",
        "robustness_summary": "robustness_summary.csv",
        "specification_grid": "sensitivity_specification_grid.csv",
        "membership_change": "membership_change_summary.csv",
        "outcome_label_executability": "outcome_label_executability.csv",
        "missingness_strategy_notes": "missingness_strategy_notes.txt",
        "primary_effect": "primary_or.json",
        "complete_case_n": "complete_case_n.json",
    }
)


#: The ``kind`` half of every product identity this runner registers.
#:
#: ``result_envelope`` requires each ``output_files`` key to be a ``kind:name``
#: identity and drops anything else as ``invalid_product_identity``.  A dropped
#: product is then missing from the canonical envelope while still present in
#: the summary, so the bounded-metric shadow reports it as a declared product
#: that is absent and fails the step closed -- after the science has already
#: been computed and written.
#:
#: Measured over every recorded run: 490 product registrations across all other
#: producers use a valid identity and 191 do not, and every one of the 191 came
#: from this runner.  A single E1 step registered 17 products and lost all 17.
#:
#: Kinds follow how real plans declare the same product where they declare it
#: at all (``robustness_matrix`` table 259x, ``primary_or`` statistic 241x,
#: ``complete_case_n`` statistic 247x, ``missingness_strategy_notes`` log 244x,
#: ``outcome_label_executability`` table 9x) and the artifact's own form
#: otherwise: a row-bearing csv is a table, a scalar json is a statistic, and a
#: written record is a log.
#: The name this step gives the primary coefficients it copies into its own
#: outputs.  One spelling, used by the copy and by the matrix row that points
#: at it, because those two disagreeing is what made the row unreadable.
_PRIMARY_COEFFICIENT_COPY_NAME = "coefficients.csv"


_ROBUSTNESS_PRODUCT_KINDS: Dict[str, str] = {
    "coefficients": "table",
    "cohort_definition_overlap_attrition": "table",
    "cohort_overlap_and_attrition": "table",
    "complete_case_n": "statistic",
    "membership_change_summary": "table",
    "missingness_strategy_notes": "log",
    "missingness_strategy_notes_json": "log",
    "model_replay_index": "log",
    "model_summaries": "table",
    "outcome_label_executability": "table",
    "primary_or": "statistic",
    "robustness_matrix": "table",
    "robustness_summary": "table",
    "robustness_variant_coefficients": "table",
    "sensitivity_comparison": "table",
    "sensitivity_specification_grid": "table",
    "sensitivity_specification_matrix": "table",
}


def canonical_robustness_output_files(
    product_files: Mapping[str, str]
) -> Dict[str, str]:
    """Map this runner's product names onto canonical ``kind:name`` identities.

    Fails closed on a product with no declared kind: a new artifact must say
    what it is before it can be registered, because the alternative is exactly
    the silent drop this exists to stop.
    """

    canonical: Dict[str, str] = {}
    for name, filename in product_files.items():
        kind = _ROBUSTNESS_PRODUCT_KINDS.get(str(name))
        if kind is None:
            raise ValueError(
                "robustness product has no declared kind and cannot be "
                f"registered: {name!r}"
            )
        canonical[f"{kind}:{name}"] = filename
    return canonical


def robustness_replay_spec_is_emittable(step: AnalysisStep) -> bool:
    """Whether the step's typed replay declaration is one this runner can emit.

    ``AnalysisStep`` has enforced what is malformed on its own terms (one
    product per output, and no entry naming a product the step never declares).
    Coverage is asked here instead, because a coverage shortfall has a safe
    answer -- nobody claims the step -- while the same rule as a schema
    validator made a real fresh run's own sealed plan unreadable and killed it
    before its first step.

    Nothing about the step's ``method`` label or its product names is consulted,
    which is the entire point.  The runner's method allowlist is three strings;
    over the recorded corpus 182 robustness steps that were neither figures nor
    claimed by the agent-owned validation gate were turned away by it, 62 for
    saying ``prespecified_sensitivity_analysis``.  Widening that list would be
    worse than the gap, because this replay executes an already-locked grid and
    a differently-scienced sensitivity analysis is not it -- so the Planner
    declares the claim instead of the host guessing it from a label.
    """

    spec = step.robustness_replay_spec
    if spec is None:
        return False
    if not all(item.output in ROBUSTNESS_REPLAY_OUTPUT_FILES for item in spec.products):
        return False
    return spec_backs_every_declared_product(
        step.expected_outputs,
        spec=spec,
        lookup="output_for",
        allowed_kinds=ROBUSTNESS_REPLAY_OUTPUT_KINDS,
    )


def declared_robustness_product_registrations(
    step: AnalysisStep | None,
) -> Dict[str, str]:
    """The identity the plan promised -> the file this runner writes for it.

    The Planner is told, in the replanner directive that publishes this
    contract: "Name the step and its products whatever your reader should see;
    the ``output`` field is what the execution layer reads."  ``product_id`` is
    a label, ``output`` is the claim.  The execution layer did not read
    ``output`` at all -- it registered its own internal file stems -- so a plan
    that took the directive at its word promised a product that was written to
    disk and registered under a name nobody had asked for.

    Measured 2026-08-01 over the 32 distinct recorded steps carrying a replay
    spec: 28 are emittable, 22 happen to label every product exactly as the
    stem is spelled, and **6 do not**.  ``table:robustness_grid`` x4 and
    ``table:specification_grid`` x1 both resolve to
    ``sensitivity_specification_grid.csv``; ``statistic:primary_effect`` x1
    resolves to ``primary_or.json``.  Each of the 6 raises
    ``declared_product_missing`` on a file sitting in its own output directory,
    which costs two LLM contract repairs and then kills the step -- canary32's
    E1 lost the replay, its figure, the robustness figure and the missingness
    figure to exactly this, after the deterministic runner had already written
    a complete and valid ``status: ok`` result.

    Only the promised identity is returned.  The runner's internal stems keep
    their own registrations because a step that declares no spec still has
    nothing else, so this is "the contract name when the plan declared one",
    not a second spelling of one contract.

    The promised identity goes into ``output_files`` and NOWHERE else.  It was
    also written, bare, into ``aliases`` -- incidentally, not by design; the
    commit that introduced it argues only about ``output_files``.  But
    ``aliases`` is the runner's own internal-stem map, the exact set
    ``canonical_robustness_output_files`` is called with and the one population
    marker that says "this runner wrote this summary".  Putting a
    Planner-chosen id in it gave one key two meanings: the 2026-08-02 m1 run
    recorded ``robustness_grid`` there, a name ``_ROBUSTNESS_PRODUCT_KINDS``
    does not declare and would raise on.  The envelope reads ``output_files``,
    so the second write bought nothing.
    """

    # No ``step is None`` guard: ``getattr`` already answers ``None`` for it and
    # the spec check below returns.  Mutation 2026-08-01 proved an explicit one
    # protects nothing.
    spec = getattr(step, "robustness_replay_spec", None)
    if spec is None:
        return {}
    registrations: Dict[str, str] = {}
    for declared in step.expected_outputs or []:
        kind, sep, product_id = str(declared or "").strip().partition(":")
        if not sep or kind not in ROBUSTNESS_REPLAY_OUTPUT_KINDS:
            continue
        output = spec.output_for(product_id)
        if output is None:
            continue
        filename = ROBUSTNESS_REPLAY_OUTPUT_FILES.get(output)
        if filename is None:
            # ``robustness_replay_spec_is_emittable`` already refuses this
            # step; registering a promise no file backs would be worse than
            # the missing product it replaces.
            continue
        registrations[f"{kind}:{product_id}"] = filename
    return registrations


def robustness_replay_declaration_verdict(step: AnalysisStep) -> OwnershipVerdict:
    """Report a step this replay could execute if the Planner declared it.

    Measured 2026-07-30 over the recorded plans (623 distinct step shapes): 20
    declare a product this runner is the registered emitter of and fill no
    ``robustness_replay_spec`` at all, so every one goes to the Coder to invent
    a sensitivity grid.  An unspecified grid is not a weaker sensitivity
    analysis; it is an undeclared one, and which specifications a paper reports
    is a pre-specified choice rather than something code decides at run time.

    The gap is keyed on the products the step promises, never on its ``method``
    label -- see :func:`robustness_replay_spec_is_emittable`, which already
    settled that question and must not be reopened: widening a label allowlist
    would hand a *differently-scienced* sensitivity analysis to a runner that
    replays an already-locked grid.

    Two boundaries, both measured rather than assumed:

    * The kinds come from :data:`ROBUSTNESS_REPLAY_OUTPUT_KINDS`, the same
      constant emittability uses.  Including ``figure:`` would over-claim --
      ``figure:robustness_summary`` is the *sole* trigger on 5 recorded steps
      and this runner writes no figures.
    * A coverage shortfall *is* reported, as of 2026-07-30.  The earlier design
      deliberately left it silent -- "the Planner did answer" -- and the cost of
      that safe answer has now been measured: on today's post-fix plans, 5 of
      the 6 robustness steps carrying a spec declare one it does not back, and
      every one of them goes to the Coder without the host ever saying why.
      The shortfall names the missing entries, so it is a declaration gap in
      exactly the sense this verdict exists to report.

      It composes with the product-promise gate rather than fighting it.  A step
      whose promise also collides (``statistic:x`` plus ``table:x``) receives
      both directives in one replan: remove the surplus kind, then map what
      remains.  Satisfying them in that order is possible -- verified against
      the real ``07_missingness_robustness_replay``, whose five surviving
      products are all outputs this runner emits.

    This owner does not claim.  No recorded step is emittable today, so a claim
    path could not be exercised by any real plan, and the runner is already
    reachable as a preflight substitute before the Coder is asked.  Reporting
    the gap is what the plan-time gate needs; moving the routing is a separate,
    characterised change for when a real run first produces an emittable spec.
    """

    if step.robustness_replay_spec is not None:
        if robustness_replay_spec_is_emittable(step):
            return OwnershipVerdict.wrong_shape(
                ROBUSTNESS_REPLAY_ANALYSIS_KIND,
                reason=(
                    "the step declares a robustness replay spec that backs every "
                    "product it promises; there is no declaration gap to report"
                ),
            )
        unbacked = sorted(
            product
            for product in (
                str(value or "").strip().partition(":")[2]
                for value in step.expected_outputs or []
            )
            if product and step.robustness_replay_spec.output_for(product) is None
        )
        if not unbacked:
            # Every promised product IS named, so the shortfall is something
            # this verdict cannot close by asking for more entries -- a kind
            # outside the runner's surface, or one bare name promised twice,
            # which the product-promise gate owns. Reporting a missing entry
            # here would demand work that leaves the step exactly as unowned.
            return OwnershipVerdict.wrong_shape(
                ROBUSTNESS_REPLAY_ANALYSIS_KIND,
                reason=(
                    "the spec names every promised product yet still does not "
                    "back the step, so the gap is in how the products are "
                    "promised rather than in this declaration"
                ),
            )
        return OwnershipVerdict.incomplete_declaration(
            ROBUSTNESS_REPLAY_ANALYSIS_KIND,
            # One entry per unbacked product, not the field path. The path was
            # already in the plan -- with entries in it -- so the plan-time
            # directive ("add the exact field(s) it left undeclared to the step
            # that already exists") resolved to "add ``products``", which was
            # present, and the forced replan changed nothing. The bracket
            # spelling matters: ``_declared_choice`` cuts the path at ``[``, so
            # a product the Planner named after a scientific choice stays in
            # the index position and cannot delete that choice from the
            # "do not change the science to satisfy this" prohibition.
            missing=tuple(
                f"robustness_replay_spec.products[{name!r}]" for name in unbacked
            ),
            reason=(
                "the step declares a robustness replay spec that names no entry "
                "for "
                + ", ".join(repr(name) for name in unbacked)
                + ", so the host cannot tell which replay output each of those "
                "promised products is and the Coder would have to decide"
            ),
        )
    promised = sorted(
        value
        for value in (str(item or "").strip() for item in step.expected_outputs or [])
        if value.partition(":")[0] in ROBUSTNESS_REPLAY_OUTPUT_KINDS
        and value.partition(":")[2] in ROBUSTNESS_REPLAY_OUTPUT_FILES
    )
    if not promised:
        return OwnershipVerdict.wrong_shape(
            ROBUSTNESS_REPLAY_ANALYSIS_KIND,
            reason=(
                "the step promises no product this replay is the registered "
                "emitter of"
            ),
        )
    return OwnershipVerdict.incomplete_declaration(
        ROBUSTNESS_REPLAY_ANALYSIS_KIND,
        missing=("robustness_replay_spec",),
        reason=(
            "the step promises "
            + ", ".join(repr(name) for name in promised)
            + ", which this replay emits, but declares no robustness_replay_spec "
            "-- so the specification grid it would report is undeclared and the "
            "Coder would have to choose one"
        ),
    )


_MATRIX_COLUMNS = [
    "spec_id",
    "effect_scale",
    "point_estimate",
    "ci_low",
    "ci_high",
    "modeled_analytic_n",
    "axis",
    "converged",
    "model_contract_n",
    "event_n",
    "model_id",
    "source_model_id",
    "exposure_source",
    "exposure_expression",
    "exposure_role",
    "analysis_role",
    "analysis_set",
    "baseline_missing_policy",
    "fit_status",
    "fit_method",
    "replay_mode",
    "coefficient_source_table",
    "coefficient_term",
    "model_contract_source",
    "source_script_sha256",
    "estimability_status",
    "membership_n",
    "membership_executable",
    "outcome_executable",
    "independent_variant",
    "notes",
]
_MEMBERSHIP_COLUMNS = [
    "spec_id",
    "axis",
    "membership_source",
    "universe_n",
    "primary_membership_n",
    "variant_membership_n",
    "overlap_n",
    "inflow_n",
    "outflow_n",
    "membership_delta_n",
    "membership_executable",
    "notes",
]
_OUTCOME_COLUMNS = [
    "spec_id",
    "target_column",
    "aggregation",
    "data_shape",
    "event_timing_column",
    "event_timing_available",
    "outcome_executable",
    "independent_variant",
    "notes",
]
_SUPPORTED_MISSING_STRATEGIES = {
    "complete_case",
    "mean_imputation",
    "median_imputation",
}
_STRUCTURED_MISSING_STRATEGIES = {
    "complete_case",
    "source_aware_categories_no_imputation",
}
_AUTHORITY_SNAPSHOT_SCHEMA = "easyicu.run_artifact_authority_snapshot/1"
_AUTHORITY_SNAPSHOT_ENV = "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT"
_AUTHORITY_SNAPSHOT_SHA_ENV = "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256"
_AUTHORITY_ERROR_ENV = "EASYICU_RUN_ARTIFACT_AUTHORITY_ERROR"


def robustness_sensitivity_preflight_code(
    step: AnalysisStep | None = None,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return the assembled deterministic robustness runner script.

    Equal by construction to ``robustness_sensitivity_preflight_scaffold(...)
    .assembled()``; kept so every existing caller and test sees the same bytes
    while the boundary between host-owned and agent-owned regions becomes
    explicit.
    """

    return robustness_sensitivity_preflight_scaffold(
        step,
        plausibility_scope=plausibility_scope,
    ).assembled()


def robustness_sensitivity_preflight_scaffold(
    step: AnalysisStep | None = None,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> HostScaffoldedScript:
    """Return the runner split into host prologue, agent body, host epilogue.

    The prologue carries the sealed plausibility scope and the pin binding the
    resolved contracts to the step authority; the epilogue writes the receipt
    into ``step_summary.json``. Both are host property. Only the body -- the
    call that actually does the robustness work -- is the model's to replace,
    which is exactly what fresh17 step 07 proved it must be: asked to repair
    this script, the model rewrote the scope and deleted the pin.

    The robustness runner can consume the pre-filter universe when a locked
    cohort variant needs it.  A flag-only plausibility receipt therefore audits
    that exact frame when present, otherwise the locked analysis cohort.  The
    comparisons and the canonical ``step_summary.json`` write stay visible in
    the generated source so the static obligation gate can prove both.
    """

    if plausibility_scope is not None:
        if step is None:
            raise ValueError(
                "a robustness plausibility scope requires an exact analysis step"
            )
        plausibility_scope.require_step(step.step_id)
    plausibility_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="plausibility_frame",
        )
        if plausibility_scope is not None
        else ""
    )
    persist_plausibility_audit = bool(
        plausibility_scope is not None and plausibility_scope.expected_columns
    )
    declared_registrations = declared_robustness_product_registrations(step)
    # One read-modify-write, not two.  The plausibility receipt and the
    # promised-product registration both patch ``step_summary.json`` after the
    # body has written it; a second block would be a second canonical write for
    # the static obligation gate to reason about.
    receipt_persistence = (
        (
            textwrap.dedent(
                """
                summary_path = Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                """
            ).strip()
            + (
                "\n" + 'summary["plausibility_audit"] = plausibility_audit'
                if persist_plausibility_audit
                else ""
            )
            + (
                "\n"
                + textwrap.dedent(
                    f"""
                    declared_product_files = {declared_registrations!r}
                    registered_products = summary.setdefault("output_files", {{}})
                    for product_identity, product_filename in (
                        declared_product_files.items()
                    ):
                        product_path = (
                            Path(os.environ["STEP_OUT_DIR"]) / product_filename
                        )
                        if not product_path.is_file():
                            continue
                        registered_products.setdefault(
                            product_identity, product_filename
                        )
                    """
                ).strip()
                if declared_registrations
                else ""
            )
            + "\n"
            + textwrap.dedent(
                """
                summary_path.write_text(
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                """
            ).strip()
        )
        if persist_plausibility_audit or declared_registrations
        else ""
    )
    prologue = (
        textwrap.dedent(
            """
            import hashlib
            import json
            import os
            from pathlib import Path

            import pandas as pd

            from easyicu.research_agent.execution.runners.deterministic_robustness import (
                _run_robustness_preflight_from_env,
            )
            """
        ).strip()
        + (
            "\n\n"
            + textwrap.dedent(
                """
                plausibility_source_path = Path(
                    os.environ.get("EASYICU_UNIVERSE_PARQUET")
                    or os.environ["COHORT_PARQUET"]
                )
                plausibility_frame = pd.read_parquet(plausibility_source_path)
                """
            ).strip()
            + "\n\n"
            + plausibility_code
            if plausibility_code
            else ""
        )
    )
    return HostScaffoldedScript(
        prologue=prologue,
        body="_run_robustness_preflight_from_env()",
        epilogue=receipt_persistence,
    )


def _run_robustness_preflight_from_env() -> None:
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    run_dir = Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2])
    context_path = Path(
        os.environ.get("EASYICU_RESEARCH_CONTEXT") or run_dir / "research_context.json"
    )
    lock_path = Path(
        os.environ.get("EASYICU_ROBUSTNESS_SPECS_LOCKED")
        or run_dir / "robustness_specs_locked.json"
    )
    cohort_path = Path(os.environ["COHORT_PARQUET"])
    universe_raw = os.environ.get("EASYICU_UNIVERSE_PARQUET")
    universe_path = Path(universe_raw) if universe_raw else None
    authority_payload: Optional[Dict[str, Any]] = None
    authority_error = str(os.environ.get(_AUTHORITY_ERROR_ENV) or "").strip()
    try:
        snapshot_raw = str(os.environ.get(_AUTHORITY_SNAPSHOT_ENV) or "").strip()
        snapshot_sha256 = str(os.environ.get(_AUTHORITY_SNAPSHOT_SHA_ENV) or "").strip()
        if not snapshot_raw or not snapshot_sha256:
            raise ValueError(
                authority_error
                or "host-selected run artifact authority snapshot is unavailable"
            )
        authority_payload = _load_authority_snapshot(
            path=Path(snapshot_raw),
            expected_sha256=snapshot_sha256,
            run_dir=run_dir,
        )
        authority_error = ""
    except Exception as exc:
        authority_error = str(exc)
    _run_robustness_preflight(
        out_dir=out_dir,
        run_dir=run_dir,
        authority_payload=authority_payload,
        authority_error=authority_error,
        context_path=context_path,
        lock_path=lock_path,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )


def _run_robustness_preflight(
    *,
    out_dir: Path,
    run_dir: Path,
    authority_payload: Optional[Dict[str, Any]],
    authority_error: str,
    context_path: Path,
    lock_path: Path,
    cohort_path: Path,
    universe_path: Optional[Path],
) -> None:
    import pandas as pd  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    blocking_reasons: List[str] = []
    warnings: List[str] = []

    try:
        specs, locked_at = _load_locked_specs(lock_path, run_dir=run_dir)
    except Exception as exc:
        specs, locked_at = [], None
        blocking_reasons.append(f"Locked robustness specifications unavailable: {exc}")

    try:
        context_payload = _load_json_object(context_path)
        context = _to_namespace(context_payload)
    except Exception as exc:
        context_payload = {}
        context = SimpleNamespace()
        blocking_reasons.append(f"Research context unavailable: {exc}")

    manifest_payload = authority_payload or {}
    if authority_payload is None:
        blocking_reasons.append(
            "Current run artifact authority unavailable: "
            + (authority_error or "no digest-bound authority snapshot was supplied")
        )
    records = manifest_payload.get("per_step_records") or []
    if not isinstance(records, list):
        records = []
        blocking_reasons.append("Run manifest per_step_records is not a list")

    cohort = None
    universe = None
    try:
        cohort = _load_frame(cohort_path)
    except Exception as exc:
        blocking_reasons.append(f"Analysis cohort unavailable: {exc}")
    if universe_path is not None:
        try:
            universe = _load_frame(universe_path)
        except Exception as exc:
            warnings.append(f"Universe path could not be loaded: {exc}")

    exposure = str(context_payload.get("primary_exposure") or "").strip()
    outcome = str(context_payload.get("target_outcome") or "").strip()
    if not exposure or not outcome:
        blocking_reasons.append(
            "Research context must declare primary_exposure and target_outcome"
        )

    reported_primary = _extract_primary_effect_payload_from_records(
        records,
        preferred_predictor=exposure or None,
    )

    structured_source = _find_structured_primary_model_source(
        records=records,
        run_dir=run_dir,
        evidence_records=(
            manifest_payload.get("evidence")
            if isinstance(manifest_payload.get("evidence"), list)
            else []
        ),
    )
    primary = reported_primary
    if structured_source is not None:
        primary, primary_authority_errors = _structured_primary_effect_payload(
            source=structured_source,
            reported_payload=reported_primary,
            preferred_predictor=exposure or None,
        )
        blocking_reasons.extend(primary_authority_errors)
    if not _complete_primary_payload(primary):
        blocking_reasons.append(
            "A completed primary estimate with point estimate and confidence interval "
            "is required before robustness comparison"
        )

    membership_rows = _membership_audit(
        specs=specs,
        cohort=cohort,
        universe=universe,
        context=context,
        exposure=exposure,
    )
    outcome_rows = _outcome_executability_audit(
        specs=specs,
        data=universe if universe is not None else cohort,
        primary_outcome=outcome,
        exact_primary_replay_available=structured_source is not None,
    )
    missing_rows = _missing_strategy_audit(
        specs,
        structured_source_aware_available=structured_source is not None,
    )

    effect_scale = str((primary or {}).get("effect_measure") or "").upper()
    fitted_rows: List[RobustnessPanelRow] = []
    structured_replay: Dict[str, Any] = {}
    executable_specs: List[RobustnessSpec] = []
    if _complete_primary_payload(primary):
        fitted_rows.append(_primary_panel_row(primary or {}))

    primary_cohort = _load_primary_cohort(run_dir)
    if not blocking_reasons and cohort is not None and effect_scale == "OR":
        executable_specs = _executable_specs(
            specs=specs,
            membership_rows=membership_rows,
            outcome_rows=outcome_rows,
            missing_rows=missing_rows,
        )
        fit_data = (
            universe if universe is not None and primary_cohort is not None else cohort
        )
        if structured_source is not None:
            adapter_rows, adapter_warnings, structured_replay = (
                _fit_structured_robustness_rows(
                    specs=executable_specs,
                    primary_payload=primary or {},
                    source=structured_source,
                    data=fit_data,
                    primary_data=cohort,
                    context=context,
                    out_dir=out_dir,
                )
            )
        else:
            adapter_rows, adapter_warnings = fit_robustness_rows_from_records(
                specs=executable_specs,
                per_step_records=records,
                primary_cohort=primary_cohort,
                data=fit_data,
                context=context,
                exposure=exposure,
                outcome=outcome,
                run_dir=run_dir,
                allow_implicit_cohort_refit=False,
            )
        warnings.extend(adapter_warnings)
        fitted_rows = adapter_rows
    elif not blocking_reasons and effect_scale != "OR":
        warnings.append(
            f"Deterministic variant fitting is not available for {effect_scale or 'an unlabeled'} "
            "effect scale; the validated primary estimate is retained and variants fail closed"
        )

    unexecutable_specs = _unexecutable_locked_spec_ids(
        specs=specs,
        membership_rows=membership_rows,
        outcome_rows=outcome_rows,
        missing_rows=missing_rows,
    )
    if unexecutable_specs:
        blocking_reasons.append(
            "Locked robustness specifications are not executable under the "
            "registered analysis contract: " + ", ".join(unexecutable_specs)
        )
    if _complete_primary_payload(primary) and specs and effect_scale != "OR":
        blocking_reasons.append(
            "Locked robustness specifications require variant estimates, but "
            "deterministic fitting is unavailable for primary effect scale "
            f"{effect_scale or 'unlabeled'}"
        )

    row_by_id = {row.spec_id: row for row in fitted_rows}
    missing_estimates = [
        spec.spec_id
        for spec in executable_specs
        if not _panel_row_has_verifiable_estimate(row_by_id.get(spec.spec_id))
    ]
    if missing_estimates:
        blocking_reasons.append(
            "Executable locked robustness specifications did not emit verifiable "
            "estimates: " + ", ".join(missing_estimates)
        )
    membership_by_id = {row["spec_id"]: row for row in membership_rows}
    outcome_by_id = {row["spec_id"]: row for row in outcome_rows}
    missing_by_id = {row["spec_id"]: row for row in missing_rows}
    matrix_rows: List[Dict[str, Any]] = []

    ordered_ids = [PRIMARY_SPEC_ID, *[spec.spec_id for spec in specs]]
    for spec_id in ordered_ids:
        spec = next((item for item in specs if item.spec_id == spec_id), None)
        axis = "primary" if spec is None else spec.axis
        row = row_by_id.get(spec_id)
        if row is None:
            row = RobustnessPanelRow(
                spec_id=spec_id,
                axis=axis,
                n=0,
                point_estimate=None,
                ci_low=None,
                ci_high=None,
                se=None,
                evidence_id="",
                converged=False,
                notes="",
            )
        membership = membership_by_id.get(spec_id, {})
        outcome_audit = outcome_by_id.get(spec_id, {})
        missing_audit = missing_by_id.get(spec_id, {})
        notes = _join_notes(
            row.notes,
            membership.get("notes") if axis == "cohort" else None,
            outcome_audit.get("notes") if axis == "outcome" else None,
            missing_audit.get("notes") if axis == "missing" else None,
            "; ".join(blocking_reasons) if blocking_reasons else None,
        )
        converged = bool(
            row.converged
            and _finite(row.point_estimate)
            and _finite(row.ci_low)
            and _finite(row.ci_high)
        )
        model_trace = _matrix_model_trace(
            spec_id=spec_id,
            spec=spec,
            structured_source=structured_source,
            structured_replay=structured_replay,
        )
        if structured_source is not None and converged:
            missing_trace = [
                field
                for field in (
                    "model_id",
                    "event_n",
                    "exposure_expression",
                    "coefficient_source_table",
                    "coefficient_term",
                    "model_contract_source",
                )
                if model_trace.get(field) in (None, "")
            ]
            contract_n = model_trace.get("model_contract_n")
            if not _finite(contract_n) or int(float(contract_n)) != int(row.n):
                missing_trace.append("model_contract_n_matches_modeled_n")
            membership_n = membership.get("variant_membership_n")
            if (
                axis == "cohort"
                and _finite(membership_n)
                and int(row.n) > int(float(membership_n))
            ):
                missing_trace.append("modeled_n_within_replayed_membership")
            if missing_trace:
                trace_error = (
                    f"{spec_id}: fitted sensitivity estimate lacks an unambiguous "
                    "model-contract trace (" + ", ".join(missing_trace) + ")"
                )
                if trace_error not in blocking_reasons:
                    blocking_reasons.append(trace_error)
                notes = _join_notes(notes, trace_error)
                converged = False
        independent_variant = outcome_audit.get("independent_variant")
        if converged:
            estimability_status = "estimated"
        elif independent_variant is False:
            estimability_status = "not_independent"
        elif outcome_audit.get("outcome_executable") is False:
            estimability_status = "not_executable"
        else:
            estimability_status = "not_converged"
        matrix_rows.append(
            {
                "spec_id": spec_id,
                "effect_scale": effect_scale or None,
                "point_estimate": row.point_estimate if converged else None,
                "ci_low": row.ci_low if converged else None,
                "ci_high": row.ci_high if converged else None,
                # ``0`` is not a valid stand-in for "no model was fit".  Keep
                # the denominator blank for non-independent/non-executable
                # specifications while retaining a positive attempted-model N
                # when a fit genuinely failed to converge.
                "modeled_analytic_n": int(row.n) if row.n and row.n > 0 else None,
                **model_trace,
                "axis": axis,
                "converged": converged,
                "estimability_status": estimability_status,
                "membership_n": membership.get("variant_membership_n"),
                "membership_executable": membership.get("membership_executable"),
                "outcome_executable": outcome_audit.get("outcome_executable"),
                "independent_variant": independent_variant,
                "notes": notes,
                "se": row.se,
                "evidence_id": row.evidence_id,
            }
        )

    matrix = pd.DataFrame(matrix_rows)
    for column in _MATRIX_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = None
    matrix = matrix[[*_MATRIX_COLUMNS, "se", "evidence_id"]]
    for count_column in (
        "modeled_analytic_n",
        "model_contract_n",
        "event_n",
        "membership_n",
    ):
        if count_column in matrix.columns:
            matrix[count_column] = pd.to_numeric(
                matrix[count_column], errors="coerce"
            ).astype("Int64")
    membership_frame = pd.DataFrame(membership_rows, columns=_MEMBERSHIP_COLUMNS)
    outcome_frame = pd.DataFrame(outcome_rows, columns=_OUTCOME_COLUMNS)
    summary_frame = _robustness_summary(matrix)
    specification_frame = pd.json_normalize(
        [spec.to_dict() for spec in specs],
        sep=".",
    )

    matrix_path = out_dir / "robustness_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    shutil.copyfile(matrix_path, out_dir / "sensitivity_comparison.csv")
    membership_path = out_dir / "membership_change_summary.csv"
    membership_frame.to_csv(membership_path, index=False)
    shutil.copyfile(membership_path, out_dir / "cohort_overlap_and_attrition.csv")
    shutil.copyfile(
        membership_path,
        out_dir / "cohort_definition_overlap_attrition.csv",
    )
    outcome_frame.to_csv(out_dir / "outcome_label_executability.csv", index=False)
    summary_frame.to_csv(out_dir / "robustness_summary.csv", index=False)
    specification_path = out_dir / "sensitivity_specification_grid.csv"
    specification_frame.to_csv(specification_path, index=False)
    shutil.copyfile(
        specification_path,
        out_dir / "sensitivity_specification_matrix.csv",
    )
    missingness_notes = {"strategies": missing_rows, "warnings": warnings}
    (out_dir / "missingness_strategy_notes.json").write_text(
        json.dumps(
            missingness_notes,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    note_lines = [
        "Prespecified missing-data robustness strategies",
        *[
            f"- {row.get('spec_id')}: {row.get('strategy') or 'unspecified'}; "
            f"executable={bool(row.get('strategy_executable'))}; {row.get('notes') or ''}"
            for row in missing_rows
        ],
        *[f"WARNING: {warning}" for warning in warnings],
    ]
    (out_dir / "missingness_strategy_notes.txt").write_text(
        "\n".join(note_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    primary_row = next(
        (row for row in matrix_rows if row["spec_id"] == PRIMARY_SPEC_ID),
        None,
    )
    complete_case_n = _complete_case_n(matrix_rows, specs)
    (out_dir / "primary_or.json").write_text(
        json.dumps(
            {
                "statistic": "primary_or",
                "value": (
                    primary_row.get("point_estimate")
                    if primary_row is not None and effect_scale == "OR"
                    else None
                ),
                "ci_low": primary_row.get("ci_low") if primary_row else None,
                "ci_high": primary_row.get("ci_high") if primary_row else None,
                "effect_scale": effect_scale or None,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "complete_case_n.json").write_text(
        json.dumps(
            {"statistic": "complete_case_n", "value": complete_case_n},
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    robustness_rows = [
        {
            "spec_id": row["spec_id"],
            "axis": row["axis"],
            "n": row["modeled_analytic_n"],
            "point_estimate": row["point_estimate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "se": row["se"],
            "evidence_id": row["evidence_id"],
            "converged": row["converged"],
            "notes": row["notes"],
        }
        for row in matrix_rows
    ]
    product_files = {
        "robustness_matrix": "robustness_matrix.csv",
        "sensitivity_comparison": "sensitivity_comparison.csv",
        "membership_change_summary": "membership_change_summary.csv",
        "cohort_overlap_and_attrition": "cohort_overlap_and_attrition.csv",
        "cohort_definition_overlap_attrition": (
            "cohort_definition_overlap_attrition.csv"
        ),
        "sensitivity_specification_grid": "sensitivity_specification_grid.csv",
        "sensitivity_specification_matrix": "sensitivity_specification_matrix.csv",
        "robustness_summary": "robustness_summary.csv",
        "outcome_label_executability": "outcome_label_executability.csv",
        "missingness_strategy_notes": "missingness_strategy_notes.txt",
        "missingness_strategy_notes_json": "missingness_strategy_notes.json",
        "primary_or": "primary_or.json",
        "complete_case_n": "complete_case_n.json",
    }
    if structured_source is not None:
        inherited_files = _copy_structured_primary_contract_artifacts(
            source=structured_source,
            out_dir=out_dir,
        )
        product_files.update(inherited_files)
    if structured_replay.get("replay_index_file"):
        product_files["model_replay_index"] = structured_replay["replay_index_file"]
    if structured_replay.get("variant_coefficients_file"):
        product_files["robustness_variant_coefficients"] = structured_replay[
            "variant_coefficients_file"
        ]
    # ``output_files`` is the registration the canonical envelope reads and so
    # must carry identities; ``aliases`` keeps the bare product names it has
    # always carried, which downstream readers match on by substring.
    output_files = canonical_robustness_output_files(product_files)
    summary = {
        "step_id": out_dir.parent.name,
        "analysis_family": "robustness_sensitivity",
        "status": "blocked" if blocking_reasons else "ok",
        "blocking_reason": "; ".join(blocking_reasons) or None,
        "primary_predictor": exposure or None,
        "target_outcome": outcome or None,
        "primary_or": (
            primary_row.get("point_estimate")
            if primary_row is not None and effect_scale == "OR"
            else None
        ),
        "primary_effect": primary_row.get("point_estimate") if primary_row else None,
        "primary_effect_scale": effect_scale or None,
        "primary_ci_low": primary_row.get("ci_low") if primary_row else None,
        "primary_ci_high": primary_row.get("ci_high") if primary_row else None,
        "complete_case_n": complete_case_n,
        "n_locked_specs": len(specs),
        "n_converged_variants": sum(
            bool(row["converged"]) and row["spec_id"] != PRIMARY_SPEC_ID
            for row in matrix_rows
        ),
        "locked_at": locked_at,
        "robustness_rows": robustness_rows,
        "robustness_panel": {"rows": robustness_rows},
        "outputs": list(dict.fromkeys(product_files.values())),
        "output_files": output_files,
        "aliases": product_files,
        "warnings": warnings,
        "limitations": [
            "Same stay-level scalar outcome labels are not treated as independent variants.",
            "Event timing is used only when an explicit outcome-time column exists.",
        ],
    }
    if structured_source is not None:
        source_summary = structured_source["summary"]
        summary.update(
            {
                # Name the contract this replay ACTUALLY selected as primary,
                # rather than copying a field the upstream summary is not
                # obliged to write.  ``_primary_contract_from_summary`` already
                # resolves it -- by ``primary_model_id`` when the parent wrote
                # one, otherwise by ``analysis_role`` -- so the replay always
                # knows the answer even when the parent left the field out.
                #
                # It matters downstream, not here: the figure lineage check
                # filters the parent's ``model_contracts`` by exact equality
                # with this field and has no such fallback, so an empty value
                # matches nothing and the primary row of the robustness figure
                # is reported as an untraceable model contract.  Measured over
                # every recorded run: 10 of 10 robustness summaries carrying
                # model contracts left this empty -- it has never once been
                # populated -- while each named exactly one primary contract.
                # Each side is stripped BEFORE the choice: a parent that wrote
                # only whitespace is not naming anything, and letting it win
                # the ``or`` would suppress the fallback and republish nothing.
                "primary_model_id": (
                    str(source_summary.get("primary_model_id") or "").strip()
                    or str(
                        structured_source["primary_contract"].get("model_id") or ""
                    ).strip()
                    or None
                ),
                "primary_exposure": source_summary.get("primary_exposure")
                or structured_source["primary_contract"].get("exposure_source"),
                "analysis_cohort_n": structured_source["primary_contract"].get("n"),
                # The same number, under the name every reader of a locked
                # cohort already uses.  ``CrossStepCohortLockValidator`` asks a
                # fixed-cohort step to restate the N its parent locked and
                # reads it by trying a closed list of spellings;
                # ``analysis_cohort_n`` -- which the manuscript and figure
                # layers read -- is not on that list, so the gate saw a step
                # reporting no cohort count at all and failed it closed while
                # the correct value sat in the summary.  Both sibling
                # deterministic producers already say ``n_total``; this one was
                # the odd one out.
                "n_total": structured_source["primary_contract"].get("n"),
                "model_contracts": source_summary.get("model_contracts") or [],
                "robustness_model_contracts": structured_replay.get(
                    "variant_contracts", []
                ),
                "primary_model_replay": {
                    "source_step_id": structured_source["step_id"],
                    "source_analysis_script": str(structured_source["script_path"]),
                    "source_script_sha256": structured_source["script_sha256"],
                    "mode": "exact_registered_primary_model_code",
                },
            }
        )
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _safe_step_id(step_id: str) -> bool:
    return bool(
        step_id
        and step_id not in {".", ".."}
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", step_id)
    )


def _coefficient_filename_from_summary(summary: Dict[str, Any]) -> Optional[str]:
    """Resolve the exact registered coefficient companion name.

    The spellings a model summary may use are published in one place, because
    this reader and the owners that write the summary are different modules
    and had drifted: the reader consulted only
    ``diagnostic_companions.coefficients`` and otherwise assumed a fixed
    ``coefficients.csv``, while the deterministic association owner writes
    ``coefficient_table`` and a Coder-written summary wrote ``coefficient_file``.
    No run has ever produced a file called ``coefficients.csv``, so that
    assumption resolved to nothing and took the exact-replay path down with it.
    """

    return model_summary_coefficient_filename(summary)


def _coefficient_path_from_summary(
    *,
    summary: Dict[str, Any],
    outputs_dir: Path,
    containment_root: Path,
) -> Optional[Path]:
    filename = _coefficient_filename_from_summary(summary)
    if filename is None:
        return None
    return contained_regular_file(outputs_dir / filename, containment_root)


def _primary_contract_from_summary(
    summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Select exactly one primary model contract without prose inference."""

    contracts = summary.get("model_contracts")
    if not isinstance(contracts, list) or not contracts:
        return None
    primary_model_id = str(summary.get("primary_model_id") or "").strip()
    if primary_model_id:
        candidates = [
            contract
            for contract in contracts
            if isinstance(contract, dict)
            and str(contract.get("model_id") or "") == primary_model_id
        ]
    else:
        candidates = [
            contract
            for contract in contracts
            if isinstance(contract, dict)
            and str(contract.get("analysis_role") or "").lower() == "primary"
            and str(contract.get("exposure_role") or "primary").lower() == "primary"
        ]
    if len(candidates) != 1:
        return None
    return dict(candidates[0])


def _matching_active_evidence_id(
    *,
    evidence_by_id: Dict[str, Dict[str, Any]],
    evidence_ids: set[str],
    run_root: Path,
    evidence_root: Path,
    step_id: str,
    expected_sha256: str,
    kind: Optional[str] = None,
    required_script_evidence_id: Optional[str] = None,
    expected_logical_name: Optional[str] = None,
) -> Optional[str]:
    """Find an active, hashed evidence copy for one source artefact."""

    for evidence_id in evidence_ids:
        record = evidence_by_id.get(evidence_id)
        if not isinstance(record, dict):
            continue
        if str(record.get("produced_by_step") or "") != step_id:
            continue
        record_kind = str(record.get("kind") or "").strip().lower()
        if kind is not None and record_kind != kind:
            continue
        if kind is None and record_kind == "code":
            continue
        if (
            required_script_evidence_id is not None
            and str(record.get("script_evidence_id") or "")
            != required_script_evidence_id
        ):
            continue
        if str(record.get("sha256") or "").strip() != expected_sha256:
            continue
        relative_path = Path(str(record.get("relative_path") or ""))
        if (
            not relative_path.parts
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            continue
        evidence_path = contained_regular_file(run_root / relative_path, run_root)
        if evidence_path is None:
            continue
        try:
            evidence_path.relative_to(evidence_root.resolve())
        except ValueError:
            continue
        logical_name = evidence_path.name.split("__", 1)[-1]
        if expected_logical_name is not None and logical_name != expected_logical_name:
            continue
        if _sha256_file(evidence_path) == expected_sha256:
            return evidence_id
    return None


def _find_structured_primary_model_source(
    *,
    records: Sequence[Dict[str, Any]],
    run_dir: Path,
    evidence_records: Sequence[Dict[str, Any]] = (),
) -> Optional[Dict[str, Any]]:
    """Locate a completed primary model whose exact code can be replayed.

    Complex source-aware/transformed models must never be approximated by the
    generic one-column estimator adapter.  A usable source therefore requires
    all three pieces of executable evidence: model contracts, a term-level
    coefficient table, and the registered analysis script.
    """

    from ...authority.runtime_artifacts import current_successful_step_records

    run_root = Path(run_dir).resolve()
    steps_root = run_root / "steps"
    evidence_root = run_root / "evidence"
    evidence_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in evidence_records
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
    }

    successful_records = current_successful_step_records(records)
    selected_primary = unique_verified_primary_record(successful_records)
    if selected_primary is None or not isinstance(selected_primary, dict):
        return None

    for record in (selected_primary,):
        step_id = str(record.get("step_id") or "").strip()
        if not _safe_step_id(step_id):
            continue
        step_dir = steps_root / step_id
        script_path = contained_regular_file(step_dir / "analysis.py", run_root)
        outputs_dir = step_dir / "outputs"
        summary_path = contained_regular_file(
            outputs_dir / "step_summary.json",
            run_root,
        )
        if script_path is None or summary_path is None:
            continue
        if outputs_dir.is_symlink() or step_dir.is_symlink():
            continue
        script_sha256 = _sha256_file(script_path)
        if str(record.get("executed_code_sha256") or "").strip() != script_sha256:
            continue
        active_evidence_ids = {
            str(evidence_id)
            for evidence_id in record.get("evidence_ids") or []
            if str(evidence_id).strip()
        }
        code_evidence_id = _matching_active_evidence_id(
            evidence_by_id=evidence_by_id,
            evidence_ids=active_evidence_ids,
            run_root=run_root,
            evidence_root=evidence_root,
            step_id=step_id,
            expected_sha256=script_sha256,
            kind="code",
            expected_logical_name="analysis.py",
        )
        if code_evidence_id is None:
            continue
        summary_sha256 = _sha256_file(summary_path)
        summary_evidence_id = _matching_active_evidence_id(
            evidence_by_id=evidence_by_id,
            evidence_ids=active_evidence_ids,
            run_root=run_root,
            evidence_root=evidence_root,
            step_id=step_id,
            expected_sha256=summary_sha256,
            kind="statistic",
            required_script_evidence_id=code_evidence_id,
            expected_logical_name="step_summary.json",
        )
        if (
            summary_evidence_id is None
            or str(record.get("step_summary_evidence_id") or "") != summary_evidence_id
        ):
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        coefficient_path = _coefficient_path_from_summary(
            summary=summary,
            outputs_dir=outputs_dir,
            containment_root=run_root,
        )
        if coefficient_path is None:
            continue
        primary_contract = _primary_contract_from_summary(summary)
        if not isinstance(primary_contract, dict):
            continue
        coefficient_sha256 = _sha256_file(coefficient_path)
        coefficient_evidence_id = _matching_active_evidence_id(
            evidence_by_id=evidence_by_id,
            evidence_ids=active_evidence_ids,
            run_root=run_root,
            evidence_root=evidence_root,
            step_id=step_id,
            expected_sha256=coefficient_sha256,
            kind="table",
            required_script_evidence_id=code_evidence_id,
            expected_logical_name=coefficient_path.name,
        )
        if coefficient_evidence_id is None:
            continue
        outputs_dir = outputs_dir.resolve()
        return {
            "step_id": step_id,
            "record": record,
            "summary": summary,
            "primary_contract": primary_contract,
            "script_path": script_path,
            "script_sha256": script_sha256,
            "code_evidence_id": code_evidence_id,
            "summary_evidence_id": summary_evidence_id,
            "coefficient_evidence_id": coefficient_evidence_id,
            "outputs_dir": outputs_dir,
            "coefficient_path": coefficient_path,
        }
    return None


def _fit_structured_robustness_rows(
    *,
    specs: Sequence[RobustnessSpec],
    primary_payload: Dict[str, Any],
    source: Dict[str, Any],
    data: Any,
    primary_data: Any,
    context: Any,
    out_dir: Path,
) -> tuple[List[RobustnessPanelRow], List[str], Dict[str, Any]]:
    """Fit variants by replaying the exact registered primary-model code."""

    import pandas as pd  # type: ignore

    rows = [_primary_panel_row(primary_payload)]
    warnings = [
        "robustness variants used exact registered primary-model code replay; "
        "the generic raw-column estimator adapter was not used"
    ]
    replay_index: List[Dict[str, Any]] = []
    variant_contracts: List[Dict[str, Any]] = []
    variant_coefficients: List[Dict[str, Any]] = []

    for spec in specs:
        if spec.axis == "missing":
            strategy = str((spec.missing_override or {}).get("strategy") or "").lower()
            if strategy == "complete_case":
                contract = _matching_primary_contract(
                    source,
                    analysis_set="complete_case",
                )
                if contract is None:
                    (
                        row,
                        coefficient_rows,
                        contract_copy,
                        error,
                    ) = _verified_complete_case_equivalence(
                        spec=spec,
                        source=source,
                        primary_data=primary_data,
                    )
                    rows.append(row)
                    if contract_copy is not None:
                        variant_contracts.append(contract_copy)
                    variant_coefficients.extend(coefficient_rows)
                    replay_index.append(
                        {
                            "spec_id": spec.spec_id,
                            "axis": spec.axis,
                            "mode": "verified_complete_case_equivalence",
                            "source_step_id": source["step_id"],
                            "status": "ok" if error is None else "blocked",
                            "error": error,
                        }
                    )
                    if error:
                        warnings.append(f"{spec.spec_id}: {error}")
                    continue
            elif strategy == "source_aware_categories_no_imputation":
                contract = dict(source["primary_contract"])
            else:
                contract = None
            row, coefficient_rows, contract_copy, error = _structured_model_row(
                spec_id=spec.spec_id,
                axis=spec.axis,
                outputs_dir=source["outputs_dir"],
                coefficient_path=source["coefficient_path"],
                contract=contract,
                evidence_id=str(source["coefficient_evidence_id"]),
                note_prefix=(
                    "Inherited the exact fitted model from the completed primary "
                    f"step for missing-data strategy {strategy or 'unspecified'}."
                ),
            )
            rows.append(row)
            evidence_contracts, evidence_coefficients = _variant_model_evidence(
                summary=source["summary"],
                outputs_dir=source["outputs_dir"],
                coefficient_path=source["coefficient_path"],
                spec_id=spec.spec_id,
                replay_mode="inherited_primary_step_output",
                analysis_set=(
                    "complete_case" if strategy == "complete_case" else "source_aware"
                ),
            )
            variant_contracts.extend(evidence_contracts)
            variant_coefficients.extend(evidence_coefficients or coefficient_rows)
            replay_index.append(
                {
                    "spec_id": spec.spec_id,
                    "axis": spec.axis,
                    "mode": "inherited_primary_step_output",
                    "source_step_id": source["step_id"],
                    "status": "ok" if error is None else "blocked",
                    "error": error,
                }
            )
            if error:
                warnings.append(f"{spec.spec_id}: {error}")
            continue

        if spec.axis == "cohort":
            replay = _replay_primary_model_for_cohort(
                spec=spec,
                source=source,
                data=data,
                context=context,
                out_dir=out_dir,
            )
            rows.append(replay["row"])
            variant_coefficients.extend(replay["coefficient_rows"])
            variant_contracts.extend(replay["contracts"])
            replay_index.append(replay["index"])
            if replay["error"]:
                warnings.append(f"{spec.spec_id}: {replay['error']}")

    coefficient_filename = "robustness_variant_coefficients.csv"
    if variant_coefficients:
        pd.DataFrame(variant_coefficients).to_csv(
            out_dir / coefficient_filename,
            index=False,
        )
    else:
        coefficient_filename = ""
    replay_index_filename = "model_replay_index.json"
    (out_dir / replay_index_filename).write_text(
        json.dumps(
            {
                "mode": "exact_registered_primary_model_code",
                "source_step_id": source["step_id"],
                "source_analysis_script": str(source["script_path"]),
                "source_script_sha256": source["script_sha256"],
                "variants": replay_index,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return (
        rows,
        warnings,
        {
            "variant_contracts": variant_contracts,
            "variant_coefficients": variant_coefficients,
            "replay_index_file": replay_index_filename,
            "variant_coefficients_file": coefficient_filename or None,
        },
    )


def _verified_complete_case_equivalence(
    *,
    spec: RobustnessSpec,
    source: Dict[str, Any],
    primary_data: Any,
) -> tuple[
    RobustnessPanelRow,
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[str],
]:
    """Reuse a fit only after proving the locked complete-case set is identical."""

    override = spec.missing_override or {}
    # One reader, shared with the plan-time requirement: the two used to name
    # the key independently and only the executor's copy was enforced, so a
    # plan the host had accepted died here after every other step had run.
    variables = complete_case_variables(spec)
    if variables is None:
        error = (
            "complete-case equivalence requires explicit locked variables in "
            f"missing_override.{COMPLETE_CASE_VARIABLES_KEY}"
        )
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error
    if len(variables) != len(set(variables)):
        error = "complete-case equivalence variables are not unique"
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error
    data_columns = {str(column) for column in getattr(primary_data, "columns", [])}
    missing_columns = [column for column in variables if column not in data_columns]
    if missing_columns:
        error = (
            "complete-case equivalence variables are absent from the primary "
            "cohort: " + ", ".join(missing_columns)
        )
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error

    # One reader, so the proof and the primary owner cannot mean different
    # things by "the analysis". This used to read ``analysis_definition`` and
    # nothing else -- a key the host's own primary owner has never written, and
    # which a repository-wide search finds in exactly two places: here, and the
    # test fixture added in the same commit. Over 358 recorded step summaries it
    # appeared once, in a Coder summary, so this proof was unreachable in
    # production from the day it was written.
    definition = model_summary_analysis_definition(source.get("summary") or {})
    if definition is None:
        error = (
            "primary analysis definition is unavailable for equivalence proof: "
            "the primary summary states no exposure, outcome and covariate set "
            f"under '{MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY}' or as the flat "
            f"{'/'.join(MODEL_SUMMARY_EXPOSURE_KEYS)}, "
            f"{'/'.join(MODEL_SUMMARY_OUTCOME_KEYS)} and "
            f"{'/'.join(MODEL_SUMMARY_COVARIATE_KEYS)} keys"
        )
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error
    contract = source.get("primary_contract")
    if not isinstance(contract, dict):
        error = "primary model contract is unavailable for equivalence proof"
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error
    exposure = definition["exposure"]
    outcome = definition["outcome"]
    covariates = definition["covariates"]
    required_variables = {exposure, outcome, *covariates}
    required_variables.discard("")
    if not required_variables or not required_variables <= set(variables):
        error = "locked complete-case variables do not cover every primary model input"
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error

    complete_case_n = int(primary_data.dropna(subset=variables).shape[0])
    model_n = int(contract.get("n") or 0)
    if complete_case_n <= 0 or complete_case_n != model_n:
        error = (
            "locked complete-case membership is not identical to the fitted "
            f"primary analysis set (complete_case_n={complete_case_n}, model_n={model_n})"
        )
        return _blocked_panel_row(spec.spec_id, spec.axis, error), [], None, error

    row, coefficient_rows, contract_copy, error = _structured_model_row(
        spec_id=spec.spec_id,
        axis=spec.axis,
        outputs_dir=source["outputs_dir"],
        coefficient_path=source["coefficient_path"],
        contract=contract,
        evidence_id=str(source["coefficient_evidence_id"]),
        note_prefix=(
            "The locked complete-case membership exactly equals the fitted "
            "primary analysis set; no duplicate refit can change the estimate."
        ),
    )
    if error is not None or contract_copy is None:
        return row, coefficient_rows, contract_copy, error
    contract_copy.update(
        {
            "spec_id": spec.spec_id,
            "source_model_id": contract_copy.get("model_id"),
            "source_analysis_role": contract_copy.get("analysis_role"),
            "source_analysis_set": contract_copy.get("analysis_set"),
            "analysis_role": "sensitivity",
            "analysis_set": "complete_case",
            "replay_mode": "verified_complete_case_equivalence",
            "missing_override": dict(override),
            "complete_case_n": complete_case_n,
        }
    )
    for item in coefficient_rows:
        item["replay_mode"] = "verified_complete_case_equivalence"
        item["analysis_set"] = "complete_case"
        item["analysis_role"] = "sensitivity"
    return row, coefficient_rows, contract_copy, None


def _replay_primary_model_for_cohort(
    *,
    spec: RobustnessSpec,
    source: Dict[str, Any],
    data: Any,
    context: Any,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run the completed primary analysis script on one locked cohort override."""

    if spec.cohort_override is None:
        error = "cohort-axis specification has no locked cohort override"
        return _blocked_structured_replay(spec=spec, error=error)

    replay_slug = re.sub(r"[^a-z0-9]+", "_", spec.spec_id.lower()).strip("_")
    replay_root = (out_dir / "model_replays" / (replay_slug or "variant")).resolve()
    if replay_root.exists():
        shutil.rmtree(replay_root)
    replay_outputs = replay_root / "outputs"
    replay_outputs.mkdir(parents=True, exist_ok=True)
    try:
        data_for_filter = _data_with_predicate_aliases(
            data=data,
            cohort_definition=spec.cohort_override,
            exposure=str(source["primary_contract"].get("exposure_source") or ""),
            context=context,
        )
        variant_cohort = build_cohort(spec.cohort_override, data=data_for_filter)
        cohort_path = replay_root / "cohort.parquet"
        variant_cohort.to_parquet(cohort_path, index=False)
        input_cohort_sha256 = _sha256_file(cohort_path)
        identifier = _identifier_column(variant_cohort)
        membership_payload = (
            sorted(str(value) for value in variant_cohort[identifier].dropna())
            if identifier is not None
            else [str(index) for index in variant_cohort.index]
        )
        input_membership_sha256 = hashlib.sha256(
            json.dumps(
                membership_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        return _blocked_structured_replay(
            spec=spec,
            error=f"locked cohort override could not be materialised: {exc}",
        )

    env = os.environ.copy()
    env["COHORT_PARQUET"] = str(cohort_path)
    env["STEP_OUT_DIR"] = str(replay_outputs)
    env["MPLCONFIGDIR"] = str(replay_root / "matplotlib")
    try:
        completed = subprocess.run(
            [sys.executable, str(source["script_path"])],
            cwd=str(source["script_path"].parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        return _blocked_structured_replay(
            spec=spec,
            error=f"registered primary-model replay failed to start: {exc}",
        )
    (replay_root / "run.log").write_text(
        "\n".join(
            [
                f"returncode: {completed.returncode}",
                "---- stdout ----",
                completed.stdout or "",
                "---- stderr ----",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return _blocked_structured_replay(
            spec=spec,
            error=(
                "registered primary-model replay returned "
                f"exit code {completed.returncode}"
            ),
            replay_root=replay_root,
            input_n=int(len(variant_cohort)),
        )

    summary_path = replay_outputs / "step_summary.json"
    try:
        replay_summary = _load_json_object(summary_path)
    except Exception as exc:
        return _blocked_structured_replay(
            spec=spec,
            error=f"replayed primary step has no readable summary: {exc}",
            replay_root=replay_root,
            input_n=int(len(variant_cohort)),
        )
    contract = _primary_contract_from_summary(replay_summary)
    replay_coefficient_path = _coefficient_path_from_summary(
        summary=replay_summary,
        outputs_dir=replay_outputs,
        containment_root=replay_root,
    )
    if not isinstance(contract, dict):
        return _blocked_structured_replay(
            spec=spec,
            error="replayed primary step did not emit its primary model contract",
            replay_root=replay_root,
            input_n=int(len(variant_cohort)),
        )
    if replay_coefficient_path is None:
        return _blocked_structured_replay(
            spec=spec,
            error=(
                "replayed primary step did not emit its declared term-level "
                "coefficient companion"
            ),
            replay_root=replay_root,
            input_n=int(len(variant_cohort)),
        )
    source_contract = source["primary_contract"]
    for field in ("exposure_source", "exposure_expression"):
        if str(contract.get(field) or "") != str(source_contract.get(field) or ""):
            return _blocked_structured_replay(
                spec=spec,
                error=f"replayed primary model changed locked {field}",
                replay_root=replay_root,
                input_n=int(len(variant_cohort)),
            )

    row, coefficient_rows, _contract_copy, error = _structured_model_row(
        spec_id=spec.spec_id,
        axis=spec.axis,
        outputs_dir=replay_outputs,
        coefficient_path=replay_coefficient_path,
        contract=contract,
        evidence_id="model_replay_index",
        note_prefix=(
            "Exact registered primary-model code replay on the locked cohort "
            f"override {spec.cohort_override.name}."
        ),
    )
    input_n = int(len(variant_cohort))
    if error is None and (row.n <= 0 or row.n > input_n):
        return _blocked_structured_replay(
            spec=spec,
            error=(
                "replayed model analytic n is not contained in the locked "
                f"cohort membership (model_n={row.n}, input_n={input_n})"
            ),
            replay_root=replay_root,
            input_n=input_n,
        )
    evidence_contracts, evidence_coefficients = _variant_model_evidence(
        summary=replay_summary,
        outputs_dir=replay_outputs,
        coefficient_path=replay_coefficient_path,
        spec_id=spec.spec_id,
        replay_mode="exact_registered_primary_model_code",
        cohort_override=spec.cohort_override.to_dict(),
    )
    index = {
        "spec_id": spec.spec_id,
        "axis": spec.axis,
        "mode": "exact_registered_primary_model_code",
        "source_step_id": source["step_id"],
        "source_script_sha256": source["script_sha256"],
        "replay_dir": str(replay_root.relative_to(out_dir.resolve())),
        "input_n": input_n,
        "input_cohort_sha256": input_cohort_sha256,
        "input_membership_sha256": input_membership_sha256,
        "modeled_n": row.n,
        "status": "ok" if error is None else "blocked",
        "error": error,
    }
    return {
        "row": row,
        "coefficient_rows": evidence_coefficients or coefficient_rows,
        "contracts": evidence_contracts,
        "index": index,
        "error": error,
    }


def _structured_model_row(
    *,
    spec_id: str,
    axis: str,
    outputs_dir: Path,
    contract: Optional[Dict[str, Any]],
    evidence_id: str,
    note_prefix: str,
    # Required, and deliberately without a default. ``coefficient.py``'s
    # resolver returns None to mean *this summary declares no coefficient
    # companion*, and its docstring calls that a refusal the caller must not
    # paper over -- a guessed name that happens to exist would bind the replay
    # to a table nobody declared. This signature used to default to
    # ``outputs_dir / "coefficients.csv"``, a filename measured to exist zero
    # times across every recorded run, so the guess could only ever produce a
    # missing-file error that blamed the file instead of the declaration. All
    # six call sites already resolve the path and refuse None before arriving
    # here; removing the default keeps the next one from re-opening the hole.
    coefficient_path: Path,
) -> tuple[
    RobustnessPanelRow,
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[str],
]:
    import pandas as pd  # type: ignore

    if not isinstance(contract, dict):
        error = "no compatible primary-exposure model contract is available"
        return _blocked_panel_row(spec_id, axis, error), [], None, error
    try:
        coefficients = pd.read_csv(coefficient_path)
    except Exception as exc:
        error = f"term-level coefficient table is unavailable: {exc}"
        return _blocked_panel_row(spec_id, axis, error), [], dict(contract), error
    required = {"model_id", "term", "term_role", "source_variable"}
    if not required.issubset(coefficients.columns):
        error = "term-level coefficient table lacks the structured model columns"
        return _blocked_panel_row(spec_id, axis, error), [], dict(contract), error
    model_id = str(contract.get("model_id") or "")
    model_rows = coefficients[coefficients["model_id"].astype(str).eq(model_id)].copy()
    exposure_rows = model_rows[
        model_rows["term_role"].astype(str).str.lower().eq("exposure")
    ].copy()
    # A declared gradient fits one term per non-reference level, so counting
    # exposure terms answers "how many contrasts" -- never "which one is the
    # primary result". The producer already answers that: its contract names
    # the primary contrast in ``exposure_expression`` and its estimates table
    # marks the same row ``is_primary_contrast``. Reading that instead of the
    # row count is what lets an ordinal exposure reach the robustness panel;
    # counting alone refused every one of them.
    fitted_exposure_n = len(exposure_rows)
    exposure_expression = str(contract.get("exposure_expression") or "").strip()
    if fitted_exposure_n > 1 and exposure_expression:
        exposure_rows = exposure_rows[
            exposure_rows["term"].astype(str).eq(exposure_expression)
        ].copy()
    if len(exposure_rows) != 1:
        # Still fail closed: no declared primary term, or one that names no
        # fitted coefficient, leaves the headline genuinely unidentified.
        # Report what was fitted AND what was asked for -- "emitted 0" after a
        # failed lookup would send the reader hunting a model that fitted fine.
        error = (
            "robustness matrix requires one scalar primary exposure coefficient; "
            f"model {model_id!r} emitted {fitted_exposure_n}"
        )
        if fitted_exposure_n > 1:
            error += (
                "; the contract names "
                f"{exposure_expression!r} as the primary contrast"
                if exposure_expression
                else "; the contract names no primary contrast"
            )
        return _blocked_panel_row(spec_id, axis, error), [], dict(contract), error
    effect_col = next(
        (
            column
            for column in ("odds_ratio", "adjusted_or", "point_estimate", "estimate")
            if column in exposure_rows.columns
        ),
        None,
    )
    low_col = next(
        (
            column
            for column in ("ci_low", "or_ci_low", "ci_lower")
            if column in exposure_rows.columns
        ),
        None,
    )
    high_col = next(
        (
            column
            for column in ("ci_high", "or_ci_high", "ci_upper")
            if column in exposure_rows.columns
        ),
        None,
    )
    if effect_col is None or low_col is None or high_col is None:
        error = "primary exposure coefficient lacks an effect estimate and confidence interval"
        return _blocked_panel_row(spec_id, axis, error), [], dict(contract), error
    exposure_row = exposure_rows.iloc[0]
    point = _float_or_none(exposure_row.get(effect_col))
    low = _float_or_none(exposure_row.get(low_col))
    high = _float_or_none(exposure_row.get(high_col))
    converged = bool(
        contract.get("converged")
        and str(contract.get("fit_status") or "").lower() == "fitted"
        and _finite(point)
        and _finite(low)
        and _finite(high)
    )
    if not converged:
        error = "structured primary model did not yield a finite fitted estimate"
        return _blocked_panel_row(spec_id, axis, error), [], dict(contract), error
    coefficient_rows = model_rows.to_dict(orient="records")
    for item in coefficient_rows:
        item["spec_id"] = spec_id
        item["source_model_id"] = model_id
    notes = _join_notes(
        note_prefix,
        f"model_id={model_id}; fit_method={contract.get('fit_method')}",
    )
    row = RobustnessPanelRow(
        spec_id=spec_id,
        axis=axis,
        n=int(contract.get("n") or 0),
        point_estimate=point,
        ci_low=low,
        ci_high=high,
        se=_float_or_none(exposure_row.get("std_error")),
        evidence_id=evidence_id,
        converged=True,
        notes=notes,
    )
    return row, coefficient_rows, dict(contract), None


def _matching_primary_contract(
    source: Dict[str, Any],
    *,
    analysis_set: str,
) -> Optional[Dict[str, Any]]:
    primary_source = str(source["primary_contract"].get("exposure_source") or "")
    contracts = source["summary"].get("model_contracts") or []
    return next(
        (
            dict(contract)
            for contract in contracts
            if isinstance(contract, dict)
            and str(contract.get("exposure_source") or "") == primary_source
            and str(contract.get("exposure_role") or "primary").lower() == "primary"
            and str(contract.get("analysis_set") or "").lower() == analysis_set
        ),
        None,
    )


def _matrix_model_trace(
    *,
    spec_id: str,
    spec: Optional[RobustnessSpec],
    structured_source: Optional[Dict[str, Any]],
    structured_replay: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind one scalar sensitivity row to the exact fitted-model contract.

    The full ``spec_id x model_id`` evidence remains in
    ``robustness_model_contracts`` and the coefficient tables.  This helper
    records which one of those models supplied the scalar row plotted in the
    manuscript-facing sensitivity figure, so renderers never have to infer a
    model identity from prose in ``notes``.
    """

    empty = {
        "model_contract_n": None,
        "event_n": None,
        "model_id": None,
        "source_model_id": None,
        "exposure_source": None,
        "exposure_expression": None,
        "exposure_role": None,
        "analysis_role": None,
        "analysis_set": None,
        "baseline_missing_policy": None,
        "fit_status": None,
        "fit_method": None,
        "replay_mode": None,
        "coefficient_source_table": None,
        "coefficient_term": None,
        "model_contract_source": None,
        "source_script_sha256": None,
    }
    if structured_source is None:
        return empty

    primary_contract = structured_source.get("primary_contract")
    if not isinstance(primary_contract, dict):
        return empty
    primary_model_id = str(primary_contract.get("model_id") or "").strip()
    primary_source = str(primary_contract.get("exposure_source") or "").strip()

    contract: Optional[Dict[str, Any]] = None
    # Still resolved, because the rows are READ from the upstream file.
    coefficient_path = structured_source.get("coefficient_path")
    # Name the copy THIS step owns, not the upstream file it was copied from.
    #
    # ``_copy_structured_primary_contract_artifacts`` copies the primary
    # coefficients into this step's outputs under
    # ``_PRIMARY_COEFFICIENT_COPY_NAME``.  Naming the source path instead left
    # the row pointing at a file that exists only in the parent step, and the
    # figure lineage check resolves ``coefficient_source_table`` against the
    # outputs of the step that owns the row -- so it read nothing and reported
    # ``coefficient_source_unreadable``.  Measured over every recorded run: 11
    # matrix rows name a file their own step does not own (all of them the
    # primary row, all naming the parent's filename) against 4 that name one it
    # does.
    coefficient_source = _PRIMARY_COEFFICIENT_COPY_NAME
    contract_source = "step_summary.json:model_contracts"
    replay_mode = "completed_primary_step_output"
    coefficient_rows: List[Dict[str, Any]] = []
    if spec_id == PRIMARY_SPEC_ID:
        contract = dict(primary_contract)
        try:
            import pandas as pd  # type: ignore

            coefficient_rows = pd.read_csv(coefficient_path).to_dict(orient="records")
        except Exception:
            coefficient_rows = []
    else:
        raw_contracts = structured_replay.get("variant_contracts") or []
        candidates = [
            dict(item)
            for item in raw_contracts
            if isinstance(item, dict)
            and str(item.get("spec_id") or "") == spec_id
            and str(item.get("exposure_source") or "") == primary_source
            and str(item.get("exposure_role") or "primary").lower() == "primary"
        ]
        desired_analysis_set = str(primary_contract.get("analysis_set") or "").lower()
        if spec is not None and spec.axis == "missing":
            strategy = str((spec.missing_override or {}).get("strategy") or "").lower()
            desired_analysis_set = (
                "complete_case" if strategy == "complete_case" else "source_aware"
            )
        preferred = [
            item
            for item in candidates
            if str(item.get("analysis_set") or "").lower() == desired_analysis_set
        ]
        if spec is not None and spec.axis == "cohort" and primary_model_id:
            same_model = [
                item
                for item in preferred
                if str(item.get("source_model_id") or item.get("model_id") or "")
                == primary_model_id
            ]
            if same_model:
                preferred = same_model
        if len(preferred) == 1:
            contract = preferred[0]
        elif len(candidates) == 1:
            contract = candidates[0]
        coefficient_source = "robustness_variant_coefficients.csv"
        contract_source = "step_summary.json:robustness_model_contracts"
        replay_mode = str((contract or {}).get("replay_mode") or "") or None
        coefficient_rows = [
            dict(item)
            for item in (structured_replay.get("variant_coefficients") or [])
            if isinstance(item, dict) and str(item.get("spec_id") or "") == spec_id
        ]

    if not isinstance(contract, dict):
        return empty
    model_id = str(contract.get("model_id") or "")
    exposure_terms = [
        item
        for item in coefficient_rows
        if str(item.get("model_id") or "") == model_id
        and str(item.get("term_role") or "").lower() == "exposure"
        and str(item.get("source_variable") or "") == primary_source
    ]
    # Same shape as the headline selection in ``_structured_model_row``: with a
    # declared gradient several exposure terms are fitted, and "exactly one or
    # give up" leaves this trace field empty. The trace check then refuses a row
    # whose coefficient IS identified -- the contract names it right here.
    exposure_expression = str(contract.get("exposure_expression") or "").strip()
    if len(exposure_terms) > 1 and exposure_expression:
        exposure_terms = [
            item
            for item in exposure_terms
            if str(item.get("term") or "") == exposure_expression
        ]
    coefficient_term = (
        exposure_terms[0].get("term") if len(exposure_terms) == 1 else None
    )
    return {
        "model_contract_n": contract.get("n"),
        "event_n": contract.get("event_n"),
        "model_id": contract.get("model_id"),
        "source_model_id": contract.get("source_model_id") or contract.get("model_id"),
        "exposure_source": contract.get("exposure_source"),
        "exposure_expression": contract.get("exposure_expression"),
        "exposure_role": contract.get("exposure_role"),
        "analysis_role": contract.get("analysis_role"),
        "analysis_set": contract.get("analysis_set"),
        "baseline_missing_policy": contract.get("baseline_missing_policy"),
        "fit_status": contract.get("fit_status"),
        "fit_method": contract.get("fit_method"),
        "replay_mode": replay_mode,
        "coefficient_source_table": coefficient_source,
        "coefficient_term": coefficient_term,
        "model_contract_source": contract_source,
        "source_script_sha256": structured_source.get("script_sha256"),
    }


def _variant_model_evidence(
    *,
    summary: Dict[str, Any],
    outputs_dir: Path,
    spec_id: str,
    replay_mode: str,
    analysis_set: Optional[str] = None,
    cohort_override: Optional[Dict[str, Any]] = None,
    # Required for the same reason as ``_structured_model_row``: the guessed
    # ``coefficients.csv`` never existed, and here the read is wrapped in a bare
    # ``except`` that returns empty coefficients -- so a guess would not even
    # surface as an error, it would silently publish contracts with no
    # term-level evidence attached.
    coefficient_path: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect the full spec-by-model contract and term-level evidence."""

    import pandas as pd  # type: ignore

    raw_contracts = summary.get("model_contracts") or []
    contracts = [
        dict(contract)
        for contract in raw_contracts
        if isinstance(contract, dict)
        and (
            analysis_set is None
            or str(contract.get("analysis_set") or "").lower() == analysis_set
        )
    ]
    model_ids = {str(contract.get("model_id") or "") for contract in contracts}
    evidence_contracts: List[Dict[str, Any]] = []
    for contract in contracts:
        contract["spec_id"] = spec_id
        contract["source_model_id"] = contract.get("model_id")
        contract["replay_mode"] = replay_mode
        if cohort_override is not None:
            contract["cohort_override"] = cohort_override
        evidence_contracts.append(contract)

    try:
        coefficients = pd.read_csv(coefficient_path)
    except Exception:
        return evidence_contracts, []
    if "model_id" not in coefficients.columns:
        return evidence_contracts, []
    selected = coefficients[coefficients["model_id"].astype(str).isin(model_ids)].copy()
    selected["spec_id"] = spec_id
    selected["source_model_id"] = selected["model_id"].astype(str)
    selected["replay_mode"] = replay_mode
    return evidence_contracts, selected.to_dict(orient="records")


def _blocked_panel_row(spec_id: str, axis: str, error: str) -> RobustnessPanelRow:
    return RobustnessPanelRow(
        spec_id=spec_id,
        axis=axis,
        n=0,
        point_estimate=None,
        ci_low=None,
        ci_high=None,
        se=None,
        evidence_id="",
        converged=False,
        notes=error,
    )


def _blocked_structured_replay(
    *,
    spec: RobustnessSpec,
    error: str,
    replay_root: Optional[Path] = None,
    input_n: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "row": _blocked_panel_row(spec.spec_id, spec.axis, error),
        "coefficient_rows": [],
        "contracts": [],
        "index": {
            "spec_id": spec.spec_id,
            "axis": spec.axis,
            "mode": "exact_registered_primary_model_code",
            "replay_dir": str(replay_root) if replay_root is not None else None,
            "input_n": input_n,
            "modeled_n": 0,
            "status": "blocked",
            "error": error,
        },
        "error": error,
    }


def _copy_structured_primary_contract_artifacts(
    *,
    source: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, str]:
    import pandas as pd  # type: ignore

    copied: Dict[str, str] = {}
    coefficient_path = source.get("coefficient_path")
    if isinstance(coefficient_path, Path) and coefficient_path.is_file():
        shutil.copy2(coefficient_path, out_dir / _PRIMARY_COEFFICIENT_COPY_NAME)
        copied["coefficients"] = _PRIMARY_COEFFICIENT_COPY_NAME

    # Never copy an unregistered sibling model_summaries.csv.  Re-materialize
    # it from the digest-verified step_summary evidence that authorized the
    # structured source instead.
    raw_contracts = source.get("summary", {}).get("model_contracts")
    if (
        isinstance(raw_contracts, list)
        and raw_contracts
        and all(isinstance(contract, dict) for contract in raw_contracts)
    ):
        pd.DataFrame(raw_contracts).to_csv(
            out_dir / "model_summaries.csv",
            index=False,
        )
        copied["model_summaries"] = "model_summaries.csv"
    return copied


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _canonical_authority_bytes(payload: Dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _load_authority_snapshot(
    *,
    path: Path,
    expected_sha256: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """Read the host-selected current checkpoint receipt exactly once.

    The deterministic runner is intentionally not allowed to inspect
    ``manifest_partial.json`` or ``manifest.json``.  It consumes only the
    snapshot selected by :func:`runtime_artifacts.load_run_artifact_authority`
    on the trusted host and verifies both the receipt bytes and its embedded
    authority digest before parsing any step record.
    """

    digest = str(expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("run artifact authority snapshot digest is invalid")
    candidate = contained_regular_file(Path(path), Path(run_dir))
    if candidate is None:
        raise ValueError(
            "run artifact authority snapshot is not a contained regular file"
        )
    try:
        snapshot_bytes = candidate.read_bytes()
    except OSError as exc:
        raise ValueError("run artifact authority snapshot is unreadable") from exc
    if hashlib.sha256(snapshot_bytes).hexdigest() != digest:
        raise ValueError("run artifact authority snapshot digest mismatch")
    try:
        snapshot = json.loads(snapshot_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("run artifact authority snapshot is invalid JSON") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("run artifact authority snapshot must be a JSON object")
    if snapshot.get("schema_version") != _AUTHORITY_SNAPSHOT_SCHEMA:
        raise ValueError("run artifact authority snapshot schema is unsupported")
    authority = snapshot.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("run artifact authority snapshot has no authority payload")
    authority_sha256 = str(snapshot.get("authority_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", authority_sha256):
        raise ValueError("run artifact authority receipt digest is invalid")
    if hashlib.sha256(_canonical_authority_bytes(authority)).hexdigest() != (
        authority_sha256
    ):
        raise ValueError("run artifact authority receipt digest mismatch")
    checkpoint_sequence = snapshot.get("checkpoint_sequence")
    if (
        isinstance(checkpoint_sequence, bool)
        or not isinstance(checkpoint_sequence, int)
        or checkpoint_sequence < 1
        or authority.get("checkpoint_sequence") != checkpoint_sequence
    ):
        raise ValueError(
            "run artifact authority snapshot is not bound to a current checkpoint"
        )
    if not isinstance(authority.get("per_step_records"), list):
        raise ValueError(
            "run artifact authority checkpoint has no valid per-step ledger"
        )
    return authority


def _load_locked_specs(
    path: Path,
    *,
    run_dir: Optional[Path] = None,
) -> tuple[List[RobustnessSpec], Optional[str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("robustness_specs_locked.json must be a regular file")
    payload = _load_json_object(path)
    raw_specs = payload.get("specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("robustness_specs_locked.json has no specs")
    if any(not isinstance(item, dict) for item in raw_specs):
        raise ValueError("robustness_specs_locked.json has invalid spec entries")
    specs = [RobustnessSpec.from_dict(item) for item in raw_specs]
    validate_robustness_specs(specs)
    expected_sha = str(payload.get("spec_sha256") or "")
    observed_sha = robustness_specs_sha(specs)
    if not expected_sha or expected_sha != observed_sha:
        raise ValueError("robustness specification lock hash mismatch")
    if run_dir is not None:
        _assert_lock_matches_evidence_anchor(
            run_dir=Path(run_dir),
            lock_path=path,
        )
    return specs, str(payload.get("locked_at") or "") or None


def _load_frame(path: Path):
    import pandas as pd  # type: ignore

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _complete_primary_payload(payload: Optional[Dict[str, Any]]) -> bool:
    return _primary_effect_payload_is_complete(payload)


def _primary_panel_row(payload: Dict[str, Any]) -> RobustnessPanelRow:
    return RobustnessPanelRow(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        n=int(payload.get("sample_size") or 0),
        point_estimate=float(payload["primary_or"]),
        ci_low=float(payload["primary_ci_low"]),
        ci_high=float(payload["primary_ci_high"]),
        se=None,
        evidence_id=str(payload.get("evidence_id") or ""),
        converged=True,
        notes="Validated primary estimate from current digest-bound evidence.",
    )


def _structured_primary_effect_payload(
    *,
    source: Dict[str, Any],
    reported_payload: Optional[Dict[str, Any]],
    preferred_predictor: Optional[str],
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    """Bind the headline to one digest-verified primary coefficient row.

    The append-only manifest embeds a convenient step-summary copy, but that
    copy is not value authority.  The registered summary must agree with the
    unique primary-exposure row in the registered coefficient table, and the
    current manifest's reported headline must agree with both.  The returned
    payload is rebuilt from the coefficient row so downstream matrices cannot
    inherit a forged summary scalar or an unregistered evidence id.
    """

    row, _coefficients, _contract, error = _structured_model_row(
        spec_id=PRIMARY_SPEC_ID,
        axis="primary",
        outputs_dir=source["outputs_dir"],
        coefficient_path=source["coefficient_path"],
        contract=source["primary_contract"],
        evidence_id=str(source["coefficient_evidence_id"]),
        note_prefix="Digest-verified primary exposure coefficient.",
    )
    if error is not None or not _panel_row_has_verifiable_estimate(row):
        return None, [
            "Digest-verified primary coefficient could not authorize the "
            f"robustness headline: {error or 'invalid coefficient row'}"
        ]

    # Preserve the original host-owned role binding and immutable
    # ``analysis_request`` snapshot. Only replace the payload coordinates with
    # the digest-verified files selected above; fabricating a role-less record
    # here would bypass (or, after hardening, incorrectly fail) the same
    # authority contract used by the ordinary headline selector.
    authoritative_record = dict(source["record"])
    authoritative_record.update(
        {
            "step_id": source["step_id"],
            "status": "ok",
            "step_summary": source["summary"],
            "step_summary_evidence_id": source["summary_evidence_id"],
            "evidence_ids": [
                source["summary_evidence_id"],
                source["coefficient_evidence_id"],
            ],
        }
    )
    summary_payload = _extract_primary_effect_payload_from_records(
        [authoritative_record],
        preferred_predictor=preferred_predictor,
    )
    errors: List[str] = []

    def _payload_disagrees(payload: Optional[Dict[str, Any]], label: str) -> None:
        if not isinstance(payload, dict):
            return
        claims_effect = bool(
            _float_or_none(payload.get("primary_or")) is not None
            or _float_or_none(payload.get("primary_ci_low")) is not None
            or _float_or_none(payload.get("primary_ci_high")) is not None
            or str(payload.get("effect_measure") or "").strip()
        )
        if not claims_effect:
            # A model step may intentionally register its scientific values
            # only in the digest-bound term table and keep the summary free of
            # duplicate headline scalars.  A denominator copied into summary
            # metadata is still checked when present.
            sample_size = _float_or_none(payload.get("sample_size"))
            if sample_size is not None and int(float(sample_size)) != int(row.n):
                errors.append(
                    f"{label} sample_size disagrees with the primary model contract"
                )
            return
        if not _complete_primary_payload(payload):
            errors.append(f"{label} primary headline is incomplete")
            return
        assert payload is not None
        expected = {
            "primary_or": row.point_estimate,
            "primary_ci_low": row.ci_low,
            "primary_ci_high": row.ci_high,
        }
        for key, expected_value in expected.items():
            observed = _float_or_none(payload.get(key))
            if (
                observed is None
                or expected_value is None
                or not math.isclose(
                    float(observed),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                errors.append(
                    f"{label} {key} disagrees with the digest-verified "
                    "primary coefficient"
                )
        if int(float(payload.get("sample_size") or 0)) != int(row.n):
            errors.append(
                f"{label} sample_size disagrees with the primary model contract"
            )
        if str(payload.get("effect_measure") or "").strip().upper() != "OR":
            errors.append(f"{label} effect measure is not the verified OR scale")

    _payload_disagrees(summary_payload, "Registered step summary")
    _payload_disagrees(reported_payload, "Current manifest")
    if isinstance(reported_payload, dict) and str(
        reported_payload.get("evidence_id") or ""
    ) not in {
        str(source["summary_evidence_id"]),
        str(source["coefficient_evidence_id"]),
    }:
        errors.append(
            "Current manifest primary headline references an evidence id that "
            "is not the registered summary or coefficient source"
        )

    payload = {
        "step_id": source["step_id"],
        "predictor": str(
            source["primary_contract"].get("exposure_source")
            or preferred_predictor
            or ""
        ),
        "primary_or": float(row.point_estimate),
        "primary_ci_low": float(row.ci_low),
        "primary_ci_high": float(row.ci_high),
        "effect_measure": "OR",
        "sample_size": int(row.n),
        "evidence_id": str(source["coefficient_evidence_id"]),
    }
    return payload, errors


def _load_primary_cohort(run_dir: Path):
    path = run_dir / "cohort_locked.json"
    if not path.exists():
        return None
    try:
        payload = _load_json_object(path)
        return coerce_cohort_definition(payload.get("cohort"))
    except Exception:
        return None


def _outcome_executability_audit(
    *,
    specs: Sequence[RobustnessSpec],
    data: Any,
    primary_outcome: str,
    exact_primary_replay_available: bool = False,
) -> List[Dict[str, Any]]:
    outcome_specs = [spec for spec in specs if spec.axis == "outcome"]
    if not outcome_specs:
        return []
    columns = [str(column) for column in getattr(data, "columns", [])]
    lower_columns = {column.lower(): column for column in columns}
    id_col = _identifier_column(data)
    scalar_shape = bool(
        data is not None
        and id_col
        and data[id_col].notna().all()
        and not data[id_col].duplicated().any()
    )
    data_shape = "one_row_per_stay" if scalar_shape else "event_level_or_unknown"
    targets = [_outcome_target(spec, primary_outcome) for spec in outcome_specs]
    target_counts = {target: targets.count(target) for target in set(targets)}
    rows: List[Dict[str, Any]] = []
    for spec, requested_target in zip(outcome_specs, targets):
        target_column = lower_columns.get(requested_target.lower())
        override = spec.outcome_override or {}
        aggregation = str(override.get("aggregation") or "").strip().lower()
        explicit_time_candidates = [
            str(override.get("event_time_column") or "").strip(),
            str(override.get("time_column") or "").strip(),
            f"{requested_target}_time",
        ]
        time_column = next(
            (
                lower_columns[candidate.lower()]
                for candidate in explicit_time_candidates
                if candidate and candidate.lower() in lower_columns
            ),
            None,
        )
        timing_available = time_column is not None
        scalar_compatible = aggregation in {"", "first", "identity", "value"}
        data_executable = bool(
            target_column and (scalar_compatible or timing_available)
        )
        same_scalar_label = bool(
            scalar_shape
            and target_column
            and (
                requested_target.lower() == primary_outcome.lower()
                or target_counts.get(requested_target, 0) > 1
            )
        )
        independent = bool(data_executable and not same_scalar_label)
        executable = bool(
            data_executable and not (exact_primary_replay_available and independent)
        )
        if target_column is None:
            note = "Declared outcome column is absent; the variant was not executed."
        elif same_scalar_label:
            note = (
                "The declared variants resolve to the same one-value-per-stay scalar "
                "outcome. They are not independently executable, and no duplicate "
                "estimate is presented as robustness evidence."
            )
        elif not data_executable:
            note = (
                "The requested aggregation requires explicit event timing, but no "
                "declared or '<outcome>_time' column is available."
            )
        elif exact_primary_replay_available:
            note = (
                "The registered primary-model script fixes its outcome. The "
                "auxiliary runner will not substitute a different endpoint, so "
                "this locked variant is not executable by exact replay."
            )
        else:
            note = "Outcome variant is independently executable from explicit columns."
        rows.append(
            {
                "spec_id": spec.spec_id,
                "target_column": target_column or requested_target,
                "aggregation": aggregation or None,
                "data_shape": data_shape,
                "event_timing_column": time_column,
                "event_timing_available": timing_available,
                "outcome_executable": executable,
                "independent_variant": independent,
                "same_scalar_label": same_scalar_label,
                "notes": note,
            }
        )
    return rows


def _missing_strategy_audit(
    specs: Sequence[RobustnessSpec],
    *,
    structured_source_aware_available: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        if spec.axis != "missing":
            continue
        strategy = str((spec.missing_override or {}).get("strategy") or "").strip()
        normalized = strategy.lower()
        if structured_source_aware_available:
            executable = normalized in _STRUCTURED_MISSING_STRATEGIES
        else:
            executable = normalized in _SUPPORTED_MISSING_STRATEGIES
        rows.append(
            {
                "spec_id": spec.spec_id,
                "strategy": strategy or None,
                "strategy_executable": executable,
                "notes": (
                    "Strategy is supported by exact registered primary-model replay."
                    if executable and structured_source_aware_available
                    else (
                        "Strategy is supported by the deterministic estimator adapter."
                        if executable
                        else (
                            "Strategy is not emitted by the registered primary-model "
                            "script; exact replay refuses to invent an imputation fit."
                            if structured_source_aware_available
                            else "Strategy is not supported by the deterministic estimator adapter."
                        )
                    )
                ),
            }
        )
    return rows


def _executable_specs(
    *,
    specs: Sequence[RobustnessSpec],
    membership_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
    missing_rows: Sequence[Dict[str, Any]],
) -> List[RobustnessSpec]:
    membership = {row["spec_id"]: row for row in membership_rows}
    outcome = {row["spec_id"]: row for row in outcome_rows}
    missing = {row["spec_id"]: row for row in missing_rows}
    selected: List[RobustnessSpec] = []
    for spec in specs:
        if spec.axis == "cohort" and not membership.get(spec.spec_id, {}).get(
            "membership_executable"
        ):
            continue
        if spec.axis == "missing" and not missing.get(spec.spec_id, {}).get(
            "strategy_executable"
        ):
            continue
        if spec.axis == "outcome":
            audit = outcome.get(spec.spec_id, {})
            if not audit.get("outcome_executable") or not audit.get(
                "independent_variant"
            ):
                continue
        selected.append(spec)
    return selected


def _unexecutable_locked_spec_ids(
    *,
    specs: Sequence[RobustnessSpec],
    membership_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
    missing_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    """Locked variants that cannot run, excluding true scalar duplicates."""

    membership = {row["spec_id"]: row for row in membership_rows}
    outcome = {row["spec_id"]: row for row in outcome_rows}
    missing = {row["spec_id"]: row for row in missing_rows}
    blocked: List[str] = []
    for spec in specs:
        if spec.axis == "cohort":
            if (
                membership.get(spec.spec_id, {}).get("membership_executable")
                is not True
            ):
                blocked.append(spec.spec_id)
        elif spec.axis == "missing":
            if missing.get(spec.spec_id, {}).get("strategy_executable") is not True:
                blocked.append(spec.spec_id)
        elif spec.axis == "outcome":
            audit = outcome.get(spec.spec_id, {})
            # Repeating the same one-value-per-stay label is explicitly
            # disclosed as non-independent, not misrepresented as a failed fit.
            if audit.get("same_scalar_label") is True:
                continue
            if audit.get("outcome_executable") is not True:
                blocked.append(spec.spec_id)
    return blocked


def _outcome_target(spec: RobustnessSpec, primary_outcome: str) -> str:
    override = spec.outcome_override or {}
    return str(
        override.get("column")
        or override.get("concept_id")
        or override.get("target")
        or primary_outcome
    ).strip()


def _robustness_summary(matrix: Any):
    import pandas as pd  # type: ignore

    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "axis",
                "total_specs",
                "converged_specs",
                "non_independent_specs",
                "range_low",
                "range_high",
            ]
        )
    rows: List[Dict[str, Any]] = []
    for axis, group in matrix.groupby("axis", dropna=False, sort=False):
        converged = group[group["converged"].astype(bool)]
        independent = group["independent_variant"]
        rows.append(
            {
                "axis": axis,
                "total_specs": int(len(group)),
                "converged_specs": int(len(converged)),
                "non_independent_specs": int(
                    (independent == False).sum()  # noqa: E712
                ),
                "range_low": (
                    float(converged["ci_low"].min()) if not converged.empty else None
                ),
                "range_high": (
                    float(converged["ci_high"].max()) if not converged.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _complete_case_n(
    matrix_rows: Sequence[Dict[str, Any]],
    specs: Sequence[RobustnessSpec],
) -> Optional[int]:
    complete_ids = {
        spec.spec_id
        for spec in specs
        if spec.axis == "missing"
        and str((spec.missing_override or {}).get("strategy") or "") == "complete_case"
    }
    for row in matrix_rows:
        if row["spec_id"] in complete_ids and row["converged"]:
            return int(row["modeled_analytic_n"])
    return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _panel_row_has_verifiable_estimate(
    row: Optional[RobustnessPanelRow],
) -> bool:
    if row is None or not row.converged or int(row.n or 0) <= 0:
        return False
    if not row.evidence_id:
        return False
    if not all(
        _finite(value) for value in (row.point_estimate, row.ci_low, row.ci_high)
    ):
        return False
    assert row.ci_low is not None and row.ci_high is not None
    return float(row.ci_low) <= float(row.ci_high)


def _float_or_none(value: Any) -> Optional[float]:
    return float(value) if _finite(value) else None


def _join_notes(*parts: Any) -> str:
    out: List[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in out:
            out.append(text)
    return "; ".join(out)
