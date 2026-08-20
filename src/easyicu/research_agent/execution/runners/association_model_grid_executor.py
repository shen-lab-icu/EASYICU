"""Verified-tool adapter for prespecified adjusted-association model grids.

The executor composes the existing adjusted-association owner; it does not fit
regressions itself.  A run-bound authority declares eligibility filters and
stable nonlinear covariate bases, while the parent primary step supplies the
outcome, exposure, adjustment set, term coding, contrast and estimator.  Every
variant is then passed through ``run_adjusted_association_from_env`` and its
``statsmodels`` adapter.  Failed convergence, separation, rank loss, non-finite
results, or disagreement with the parent reference fit are terminal.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Mapping, Sequence

from ...authority.current_case_scientific_runtime import (
    AssociationModelGridLandmarkFilter,
    AssociationModelGridLevelFilter,
    AssociationModelGridRuntimeAuthority,
    AssociationModelGridVariant,
    load_current_case_scientific_runtime_authority,
)
from ...authority.declared_levels import execution_model_requirement
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.association_execution import sole_primary_model_requirement
from ...contracts.model_terms import ModelTermSpec, level_spelling
from ...schema import AnalysisPlan, AnalysisStep, PlannedModelRequirement
from ...numeric_scalars import coerce_finite_float
from .adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
    run_adjusted_association_from_env,
)
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import load_typed_input

ASSOCIATION_MODEL_GRID_ANALYSIS_KIND = "association_model_grid"
_CORE_COLUMNS = (
    "analysis_id",
    "n_stays",
    "n_events",
    "estimate",
    "ci_low",
    "ci_high",
    "effect_measure",
    "fit_n",
    "fit_events",
    "standard_error",
    "converged",
    "separation_detected",
)


class AssociationModelGridError(RuntimeError):
    """The signed grid could not be executed exactly as declared."""


def _sealed(
    authority: AssociationModelGridRuntimeAuthority | Mapping[str, Any] | None,
) -> AssociationModelGridRuntimeAuthority | None:
    if authority is None:
        return None
    value = load_current_case_scientific_runtime_authority(authority)
    return value if isinstance(value, AssociationModelGridRuntimeAuthority) else None


def association_model_grid_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: AssociationModelGridRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    """Claim only the exact child bound by the run's signed authority."""

    sealed = _sealed(authority)
    if sealed is None:
        return False
    try:
        # Runtime replanning rebuilds Pydantic objects even when the signed
        # step is structurally unchanged.  Ownership is the fully validated
        # typed step, not one particular in-memory instance of it.
        return sealed.governed_step(plan) == step
    except ValueError:
        return False


def _parent_requirement(
    *,
    plan: AnalysisPlan,
    authority: AssociationModelGridRuntimeAuthority,
) -> PlannedModelRequirement:
    parent = next(
        step
        for step in plan.steps
        if authority.parent_product in set(step.expected_outputs)
    )
    requirement = sole_primary_model_requirement(parent)
    if requirement is None:
        raise AssociationModelGridError("model-grid parent has no single requirement")
    return execution_model_requirement(parent, requirement)


def association_model_grid_executor_code(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: AssociationModelGridRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return the fully host-owned sandbox entrypoint for one grid."""

    sealed = _sealed(authority)
    if sealed is None or not association_model_grid_executor_owns_step(
        step, plan=plan, authority=sealed
    ):
        raise ValueError("The step is not owned by the association model-grid executor")
    requirement = _parent_requirement(plan=plan, authority=sealed)
    typed_cohort_input = sealed.cohort_product
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.association_model_grid_executor import (
            run_association_model_grid,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input={typed_cohort_input!r},
        )
        {receipt_code}
        summary = run_association_model_grid(
            frame=frame,
            cohort_path=cohort_path,
            authority={sealed.model_dump(mode="json")!r},
            runtime_projection_sha256={runtime_projection_sha256!r},
            parent_requirement={requirement.model_dump(mode="json")!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
        )
        {"summary['plausibility_audit'] = plausibility_audit" if receipt_code else ""}
        (Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def _binary_outcome(frame: Any, column: str):
    import pandas as pd

    if column not in frame.columns:
        raise AssociationModelGridError(f"outcome column {column!r} is absent")
    source = frame[column]
    numeric = pd.to_numeric(source, errors="coerce")
    if bool((source.notna() & numeric.isna()).any()) or bool(numeric.isna().any()):
        raise AssociationModelGridError("model-grid outcome is missing or non-numeric")
    if not bool(numeric.isin((0.0, 1.0)).all()):
        raise AssociationModelGridError("model-grid outcome is not binary 0/1")
    return numeric.astype(float)


def _eligibility_mask(
    frame: Any,
    *,
    variant: AssociationModelGridVariant,
    outcome_column: str,
):
    import pandas as pd

    mask = pd.Series(True, index=frame.index)
    outcome = _binary_outcome(frame, outcome_column)
    for rule in variant.filters:
        if isinstance(rule, AssociationModelGridLandmarkFilter):
            if rule.outcome_column != outcome_column:
                raise AssociationModelGridError(
                    "landmark filter outcome disagrees with the parent model"
                )
            if rule.event_time_column not in frame.columns:
                raise AssociationModelGridError(
                    f"event-time column {rule.event_time_column!r} is absent"
                )
            source = frame[rule.event_time_column]
            event_time = pd.to_numeric(source, errors="coerce")
            if bool((source.notna() & event_time.isna()).any()):
                raise AssociationModelGridError("landmark event time is non-numeric")
            if bool((outcome.eq(1.0) & event_time.isna()).any()):
                raise AssociationModelGridError(
                    "landmark eligibility cannot time every outcome event"
                )
            alive = outcome.eq(0.0) | event_time.gt(rule.landmark_hours)
            mask &= alive
            continue
        if not isinstance(rule, AssociationModelGridLevelFilter):
            raise AssociationModelGridError("unsupported model-grid filter")
        if rule.column not in frame.columns:
            raise AssociationModelGridError(
                f"level-filter column {rule.column!r} is absent"
            )
        source = frame[rule.column].astype("object").where(frame[rule.column].notna(), None)
        spellings = source.map(level_spelling)
        if bool(spellings.eq("").any()):
            raise AssociationModelGridError(
                f"level-filter column {rule.column!r} contains missing values"
            )
        observed = set(spellings.unique().tolist())
        unexpected = sorted(observed - set(rule.declared_levels))
        if unexpected:
            raise AssociationModelGridError(
                f"level-filter column {rule.column!r} has undeclared levels: "
                + ", ".join(unexpected)
            )
        mask &= spellings.isin(rule.retained_levels)
    return mask


def _natural_cubic_spline_basis(
    source: Any,
    *,
    source_column: str,
    degrees_of_freedom: int,
):
    """Return a centered finite natural-cubic-spline basis with stable names."""

    import numpy as np
    import pandas as pd
    import patsy

    numeric = pd.to_numeric(source, errors="coerce")
    if bool((source.notna() & numeric.isna()).any()):
        raise AssociationModelGridError(
            f"nonlinear source {source_column!r} contains non-numeric values"
        )
    observed = numeric.dropna().astype(float)
    if len(observed) <= degrees_of_freedom + 2:
        raise AssociationModelGridError(
            f"nonlinear source {source_column!r} has too few complete rows"
        )
    if not bool(observed.map(math.isfinite).all()):
        raise AssociationModelGridError(
            f"nonlinear source {source_column!r} contains non-finite values"
        )
    center = float(observed.mean())
    centered = observed - center
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        basis = patsy.dmatrix(
            f"cr(x, df={degrees_of_freedom}, constraints='center') - 1",
            {"x": centered.to_numpy(dtype=float)},
            return_type="dataframe",
        )
    names = [
        f"{source_column}__natural_cubic_spline_{index + 1}"
        for index in range(basis.shape[1])
    ]
    basis.columns = names
    basis.index = observed.index
    if basis.shape[1] != degrees_of_freedom:
        raise AssociationModelGridError("natural spline basis width drifted")
    values = basis.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise AssociationModelGridError("natural spline basis is non-finite")
    if int(np.linalg.matrix_rank(values)) != basis.shape[1]:
        raise AssociationModelGridError("natural spline basis is rank deficient")
    full = pd.DataFrame(float("nan"), index=source.index, columns=names)
    full.loc[basis.index, names] = values
    return full, center


def _variant_model(
    frame: Any,
    *,
    requirement: PlannedModelRequirement,
    variant: AssociationModelGridVariant,
):
    nonlinear = {item.source_column: item for item in variant.nonlinear_terms}
    terms = list(requirement.model_terms or ())
    term_by_name = {item.name: item for item in terms}
    invalid = sorted(set(nonlinear) - set(requirement.covariates or ()))
    if invalid:
        raise AssociationModelGridError(
            "nonlinear model-grid terms are not parent covariates: "
            + ", ".join(invalid)
        )
    derived = frame.copy()
    compiled_terms: list[ModelTermSpec] = []
    covariates: list[str] = []
    basis_receipts: list[dict[str, Any]] = []
    for term in terms:
        transform = nonlinear.get(term.name)
        if transform is None:
            compiled_terms.append(term)
            if term.role == "covariate":
                covariates.append(term.name)
            continue
        if term.role != "covariate" or term.coding != "continuous":
            raise AssociationModelGridError(
                f"nonlinear source {term.name!r} is not a continuous covariate"
            )
        basis, center = _natural_cubic_spline_basis(
            derived[term.name],
            source_column=term.name,
            degrees_of_freedom=transform.degrees_of_freedom,
        )
        derived = derived.join(basis)
        for column in basis.columns:
            covariates.append(str(column))
            compiled_terms.append(
                ModelTermSpec(
                    name=str(column),
                    role="covariate",
                    coding="continuous",
                    transform="identity",
                )
            )
        basis_receipts.append(
            {
                "source_column": term.name,
                "basis": transform.basis,
                "degrees_of_freedom": transform.degrees_of_freedom,
                "center": center,
                "generated_columns": list(basis.columns),
            }
        )
    if set(nonlinear) - set(term_by_name):
        raise AssociationModelGridError("nonlinear source is absent from parent terms")
    return derived, covariates, compiled_terms, basis_receipts


def _model_grid_number(value: Any, *, field: str) -> float:
    try:
        return coerce_finite_float(value, label=field)
    except ValueError as exc:
        raise AssociationModelGridError(str(exc)) from exc


def _primary_parent_row(frame: Any) -> Mapping[str, Any]:
    markers = frame["is_primary_contrast"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    )
    rows = frame.loc[markers]
    if len(rows) != 1:
        raise AssociationModelGridError(
            "adjusted-association parent has no unique primary contrast"
        )
    return rows.iloc[0].to_dict()


def _close(left: Any, right: Any) -> bool:
    return math.isclose(
        _model_grid_number(left, field="reference value"),
        _model_grid_number(right, field="parent value"),
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def run_association_model_grid(
    *,
    frame: Any,
    cohort_path: Any,
    authority: AssociationModelGridRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    parent_requirement: PlannedModelRequirement | Mapping[str, Any],
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Execute every signed variant through the existing verified adapter."""

    import pandas as pd

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("model-grid executor received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise AssociationModelGridError("runtime projection digest is required")
    requirement = (
        parent_requirement
        if isinstance(parent_requirement, PlannedModelRequirement)
        else PlannedModelRequirement.model_validate(parent_requirement)
    )
    if requirement.outcome_type != "binary":
        raise AssociationModelGridError("model-grid v1 supports binary outcomes")
    missing = sorted(set(sealed.required_columns_from_requirement(requirement)) - set(frame.columns))
    if missing:
        raise AssociationModelGridError(
            "model-grid cohort lacks required columns: " + ", ".join(missing)
        )

    parent = load_typed_input(
        input_key=sealed.parent_product,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
        require_consumption_contract=True,
        consumption_mode="all_rows",
        minimum_row_count=1,
    )
    parent_row = _primary_parent_row(parent.frame)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    basis_receipts: dict[str, Sequence[Mapping[str, Any]]] = {}
    for variant in sealed.variants:
        mask = _eligibility_mask(
            frame,
            variant=variant,
            outcome_column=requirement.outcome,
        )
        eligible = frame.loc[mask].copy()
        if eligible.empty:
            raise AssociationModelGridError(
                f"model-grid variant {variant.analysis_id!r} has no eligible rows"
            )
        outcome = _binary_outcome(eligible, requirement.outcome)
        n_stays = int(len(eligible))
        n_events = int(outcome.eq(1.0).sum())
        model_frame, covariates, terms, receipts = _variant_model(
            eligible,
            requirement=requirement,
            variant=variant,
        )
        basis_receipts[variant.analysis_id] = receipts
        with tempfile.TemporaryDirectory(
            prefix=f"model_grid_{variant.analysis_id}_",
            dir=out_dir,
        ) as temporary:
            summary = run_adjusted_association_from_env(
                requirement_id=(
                    f"{requirement.requirement_id}__{variant.analysis_id}"
                ),
                exposure=requirement.exposure_source,
                outcome=requirement.outcome,
                covariates=covariates,
                model_terms=terms,
                estimator_kind="logistic",
                analysis_set=variant.analysis_id,
                analysis_role="sensitivity",
                method_family=requirement.method_family,
                primary_contrast_level=requirement.primary_contrast_level,
                dependence=requirement.dependence,
                typed_cohort_input=sealed.cohort_product,
                frame=model_frame,
                cohort_path=cohort_path,
                emit_step_summary=False,
                output_dir=temporary,
            )
            estimate_records = pd.read_csv(
                Path(temporary) / "adjusted_association_estimates.csv"
            ).to_dict("records")
        contracts = summary.get("model_contracts") or []
        if len(contracts) != 1:
            raise AssociationModelGridError("variant omitted its model contract")
        contract = contracts[0]
        if contract.get("converged") is not True:
            raise AssociationModelGridError("variant did not converge")
        if contract.get("separation_detected") is not False:
            raise AssociationModelGridError("variant shows separation")
        estimate = _model_grid_number(summary.get("primary_or"), field="odds ratio")
        interval = summary.get("primary_or_ci")
        if not isinstance(interval, list) or len(interval) != 2:
            raise AssociationModelGridError("variant omitted its confidence interval")
        low = _model_grid_number(interval[0], field="ci_low")
        high = _model_grid_number(interval[1], field="ci_high")
        if not (0 < low <= estimate <= high):
            raise AssociationModelGridError("variant effect interval is incoherent")
        standard_error = next(
            (
                _model_grid_number(
                    item.get("standard_error"), field="standard_error"
                )
                for item in estimate_records
                if str(item.get("is_primary_contrast")).casefold()
                in {"true", "1", "yes"}
            ),
            None,
        )
        if standard_error is None or standard_error <= 0:
            raise AssociationModelGridError("variant standard error is invalid")
        row = {
            "analysis_id": variant.analysis_id,
            "n_stays": n_stays,
            "n_events": n_events,
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "effect_measure": "odds_ratio",
            "fit_n": int(summary["n_total"]),
            "fit_events": int(summary["n_events"]),
            "standard_error": standard_error,
            "converged": True,
            "separation_detected": False,
            **variant.metadata,
        }
        for source, alias in sealed.output_aliases.items():
            row[alias] = row[source]
        rows.append(row)

    reference = next(
        row for row in rows if row["analysis_id"] == sealed.reference_variant_id
    )
    if not all(
        (
            _close(reference["estimate"], parent_row["estimate"]),
            _close(reference["ci_low"], parent_row["ci_low"]),
            _close(reference["ci_high"], parent_row["ci_high"]),
            int(reference["fit_n"]) == int(parent_row["n"]),
            int(reference["fit_events"]) == int(parent_row["n_events"]),
        )
    ):
        raise AssociationModelGridError(
            "model-grid reference variant disagrees with the bound primary fit"
        )

    product_name = sealed.output_product.partition(":")[2]
    table_path = out_dir / f"{product_name}.csv"
    columns = [
        *_CORE_COLUMNS,
        *sealed.output_aliases.values(),
        *sealed.metadata_columns,
    ]
    pd.DataFrame(rows, columns=list(dict.fromkeys(columns))).to_csv(
        table_path,
        index=False,
    )
    return {
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "association",
        "interpretation_class": "adjusted_association_sensitivity",
        "deterministic_standard_analysis": ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
        "execution_mode": "composed_workflow",
        "typed_cohort_input": sealed.cohort_product,
        "source_cohort": Path(cohort_path).name if cohort_path is not None else None,
        "parent_product": sealed.parent_product,
        "parent_evidence_id": parent.evidence_id,
        "parent_sha256": parent.sha256,
        "analysis_rows": rows,
        "basis_receipts": basis_receipts,
        "output_files": {sealed.output_product: table_path.name},
        "scientific_runtime_receipt": {
            "schema_version": "easyicu.association_model_grid_runtime_receipt/1",
            "execution_contract_sha256": sealed.execution_contract_sha256,
            "runtime_projection_sha256": runtime_projection_sha256,
            "variant_ids": list(sealed.sensitivity_ids),
            "reference_variant_id": sealed.reference_variant_id,
            "adapter": "adjusted_association_executor/statsmodels",
        },
    }


__all__ = [
    "ASSOCIATION_MODEL_GRID_ANALYSIS_KIND",
    "AssociationModelGridError",
    "association_model_grid_executor_code",
    "association_model_grid_executor_owns_step",
    "run_association_model_grid",
]
