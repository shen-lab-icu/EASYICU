"""Select the exact host-authorized cohort exposed to one step.

The run-level execution cohort is only a fallback.  Once typed-input lineage
has resolved an ``analysis_cohort`` product for a step, the compatibility
``COHORT_PARQUET`` surface and deterministic validators must use that same
digest-bound product.  Otherwise generated code sees two different populations
under two names and is forced either to fail or to guess which one is primary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ..authority.evidence_store import sha256_of_file
from ..contracts.primary_cohort import (
    primary_analysis_cohort_producer_uses_universe,
    reserved_primary_cohort_product,
)
from ..schema import AnalysisPlan, AnalysisStep


class StepExecutionCohortRoutingError(RuntimeError):
    """A typed analysis-cohort binding cannot be exposed without ambiguity."""


def step_execution_cohort_path(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Path,
) -> Path:
    """Choose the run-level data plane before typed inputs are resolved.

    Development sampling is a child of the already locked, post-QC analysis
    cohort.  The step that materialises and reports that parent cohort must
    therefore continue to consume the full universe; only downstream
    scientific steps consume the deterministic development child.  Feeding
    the child back into its own producer destroys the attrition denominator
    and makes host replay impossible.
    """

    del run_dir  # Kept in the stable public signature for routing callers.
    if primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
        return universe_path
    return cohort_path


def bound_step_execution_cohort_path(
    *,
    run_dir: Path,
    fallback_path: Path,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Prefer the unique digest-bound primary-cohort typed input.

    Bindings have already passed lineage resolution, but this final routing
    boundary rechecks path agreement, containment, existence, and digest before
    changing the compatibility runner surface.  It never infers a cohort from
    a variable name or from a file scan.

    The mapping key is the raw declared input the Planner wrote, so the
    reserved primary-cohort identity is decided by its owner rather than by
    matching one spelling of the product name -- the canonical binding view
    has already folded ``cohort:`` into ``dataset:`` and cannot answer it.
    """

    run_root = Path(run_dir).resolve()
    candidates: dict[Path, str] = {}
    for declared_input, binding in resolved_input_bindings.items():
        if reserved_primary_cohort_product(declared_input) is None:
            continue
        relative_path = str(binding.get("relative_path") or "").strip()
        absolute_path = str(binding.get("absolute_path") or "").strip()
        expected_sha256 = str(binding.get("sha256") or "").strip()
        if not relative_path or not absolute_path or len(expected_sha256) != 64:
            raise StepExecutionCohortRoutingError(
                "Typed analysis_cohort binding is incomplete"
            )
        relative_candidate = (run_root / relative_path).resolve()
        absolute_candidate = Path(absolute_path).resolve()
        try:
            relative_candidate.relative_to(run_root)
        except ValueError as exc:
            raise StepExecutionCohortRoutingError(
                "Typed analysis_cohort path escapes the run root"
            ) from exc
        if relative_candidate != absolute_candidate:
            raise StepExecutionCohortRoutingError(
                "Typed analysis_cohort absolute and relative paths disagree"
            )
        if not relative_candidate.is_file():
            raise StepExecutionCohortRoutingError(
                "Typed analysis_cohort file is missing"
            )
        if sha256_of_file(relative_candidate) != expected_sha256:
            raise StepExecutionCohortRoutingError(
                "Typed analysis_cohort digest changed before execution"
            )
        candidates[relative_candidate] = expected_sha256
    if not candidates:
        return Path(fallback_path).resolve()
    if len(candidates) != 1:
        raise StepExecutionCohortRoutingError(
            "Multiple distinct typed analysis_cohort inputs are ambiguous"
        )
    return next(iter(candidates))


def bind_step_execution_cohort(
    run_dir: Path,
    fallback_path: Path,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    step_record: MutableMapping[str, Any],
) -> Path:
    """Select the step cohort and bind its exact bytes into the step record."""

    selected = bound_step_execution_cohort_path(
        run_dir=run_dir,
        fallback_path=fallback_path,
        resolved_input_bindings=resolved_input_bindings,
    )
    if selected != Path(fallback_path).resolve():
        step_record["execution_cohort_role"] = "resolved_typed_analysis_cohort"
    else:
        step_record.setdefault(
            "execution_cohort_role",
            "run_level_execution_cohort",
        )
    step_record["execution_cohort_sha256"] = sha256_of_file(selected)
    return selected


__all__ = [
    "StepExecutionCohortRoutingError",
    "bind_step_execution_cohort",
    "bound_step_execution_cohort_path",
    "step_execution_cohort_path",
]
