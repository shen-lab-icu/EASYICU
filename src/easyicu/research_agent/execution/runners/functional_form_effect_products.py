"""Seal and consume effects of one reviewed covariate-form refit.

The registered curve and contrast tables carry the same typed model receipt.
This module validates their relation to the reviewed parent step, the current
runtime and the actually loaded primary products. It never fits a model or
authorizes a different design.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...contracts.cohort_product_keys import is_closed_cohort_product_key
from ...contracts.functional_form_effects import FunctionalFormEffects
from ...contracts.functional_form_projection import FunctionalFormProjection

_IDENTITY_COLUMN = "functional_form_effects_json"
_PROJECTION_IDENTITY_COLUMN = "functional_form_projection_json"
_WALD_Z = 1.96
_EFFECT_COLUMNS = (
    "exposure_value", "reference_exposure_value",
    "adjusted_odds_ratio", "ci_low", "ci_high",
)


@dataclass(frozen=True)
class ConsumedFunctionalFormEffects:
    """A validated pair returned by the bound native consumer, not a new grant."""

    contract: FunctionalFormEffects
    points: pd.DataFrame
    curve_evidence_id: str
    contrast_evidence_id: str


def _effect_values(frame: pd.DataFrame) -> np.ndarray:
    if not set(_EFFECT_COLUMNS) <= set(frame.columns):
        raise ValueError("functional-form effects require native primary coordinates and intervals")
    values = frame[list(_EFFECT_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("functional-form effects contain non-finite values")
    return values


def _require_close(actual: Any, expected: Any, label: str) -> None:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, rtol=1e-8, atol=1e-10):
        raise ValueError(f"functional-form {label} disagrees with the sealed primary/model")


def _validate_parent_effects(contract, contrasts, linear_sensitivity, authority):
    if len(contrasts) != 2 or not contrasts["exposure"].eq(authority.exposure_column).all():
        raise ValueError("functional-form source primary contrast identity mismatch")
    values = _effect_values(contrasts.sort_values("exposure_value"))
    _require_close(values, contract.primary_contrasts, "primary contrast reproduction")
    if len(linear_sensitivity) != 1:
        raise ValueError("functional-form source diagnostics must contain exactly one row")
    linear = linear_sensitivity.iloc[0]
    if float(linear["n"]) != contract.n or float(linear["events"]) != contract.events:
        raise ValueError("functional-form source population changed")
    _require_close(
        linear["nonlinearity_p_value"], contract.primary_exposure_nonlinearity_p_value,
        "primary exposure nonlinearity p-value",
    )


def _validate_scope(contract, *, step, authority, runtime_projection_sha256):
    form = authority.require_functional_form_spec(step)
    expected = {
        "step_id": step.step_id,
        "spec_id": step.sensitivity_spec_ids[0],
        "form": form,
        "exposure": authority.exposure_column,
        "outcome": authority.outcome_column,
        "outcome_time_column": authority.outcome_time_column,
        "adjustment_columns": tuple(authority.required_adjustment_columns),
        "categorical_adjustment_columns": tuple(authority.categorical_adjustment_columns),
        "landmark_hours": authority.landmark_hours,
        "observation_duration_column": authority.observation_duration_column,
        "observation_duration_unit": authority.observation_duration_unit,
        "runtime_projection_sha256": runtime_projection_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "protocol_content_sha256": authority.protocol_content_sha256,
        "dependence": authority.dependence,
        "curve_points": authority.curve_points,
    }
    if any(getattr(contract, key) != value for key, value in expected.items()):
        raise ValueError("functional-form effect scope drifted from its reviewed parent/runtime")
    cohort_inputs = [key for key in step.inputs if is_closed_cohort_product_key(key)]
    if len(cohort_inputs) != 1:
        raise ValueError("functional-form effects require one exact cohort input")
    cohort_input = cohort_inputs[0]
    expected_inputs = {cohort_input, authority.downstream_parent_product, authority.linear_sensitivity_product}
    sources = {item.input_key: item for item in contract.input_bindings}
    if set(sources) != expected_inputs or sources[cohort_input].row_count < contract.n:
        raise ValueError("functional-form effect lineage lacks the exact primary model inputs")
    if sources[authority.downstream_parent_product].row_count != 2 or sources[authority.linear_sensitivity_product].row_count != 1:
        raise ValueError("functional-form parent product row counts changed")


def _validate_effect_math(frame: pd.DataFrame, contract: FunctionalFormEffects) -> None:
    if not frame["exposure"].eq(contract.exposure).all():
        raise ValueError("functional-form effect exposure changed")
    values = _effect_values(frame)
    _require_close(values[:, 1], np.full(len(frame), contract.reference), "reference")
    if not (0 < values[:, 3]).all() or not (values[:, 3] <= values[:, 2]).all() or not (values[:, 2] <= values[:, 4]).all():
        raise ValueError("functional-form effect intervals are invalid")
    vectors = np.asarray([json.loads(value) for value in frame["contrast_vector"]], dtype=float)
    k = len(contract.parameter_columns)
    if vectors.shape != (len(frame), k) or not np.isfinite(vectors).all():
        raise ValueError("functional-form effect contrast vectors have invalid dimensions")
    exposure_indices = [i for i, name in enumerate(contract.parameter_columns) if name in {"exposure_rcs_1", "exposure_rcs_2"}]
    if len(exposure_indices) != 2 or np.any(vectors[:, [i for i in range(k) if i not in exposure_indices]] != 0):
        raise ValueError("functional-form contrast must hold every adjustment term constant")
    covariance = np.asarray(contract.covariance_matrix)
    if np.linalg.eigvalsh(covariance).min() < -1e-8 * max(1.0, np.linalg.norm(covariance, ord=2)):
        raise ValueError("functional-form covariance is not positive semidefinite")
    eta = vectors @ np.asarray(contract.parameter_values)
    variance = np.einsum("ij,jk,ik->i", vectors, covariance, vectors)
    if (variance < -1e-10).any():
        raise ValueError("functional-form contrast variance is negative")
    se = np.sqrt(np.maximum(variance, 0))
    expected = np.exp(np.column_stack([eta, eta - 1.96 * se, eta + 1.96 * se]))
    _require_close(values[:, 2:], expected, "parameter-derived OR/CI")


def seal_functional_form_effects(
    *, step, authority, runtime_projection_sha256, comparison,
    contrasts, linear_sensitivity, input_bindings,
):
    """Return both validated tables and their receipt before any output write."""

    bundle = comparison["effect_bundle"]
    contract = FunctionalFormEffects.model_validate({
        "step_id": step.step_id, "spec_id": step.sensitivity_spec_ids[0],
        "form": step.functional_form_spec,
        "exposure": authority.exposure_column, "outcome": authority.outcome_column,
        "outcome_time_column": authority.outcome_time_column,
        "adjustment_columns": authority.required_adjustment_columns,
        "categorical_adjustment_columns": authority.categorical_adjustment_columns,
        "landmark_hours": authority.landmark_hours,
        "observation_duration_column": authority.observation_duration_column,
        "observation_duration_unit": authority.observation_duration_unit,
        "runtime_projection_sha256": runtime_projection_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "protocol_content_sha256": authority.protocol_content_sha256,
        "model_rows_sha256": bundle["model_rows_sha256"],
        "n": comparison["n_complete_case"], "events": comparison["event_n"],
        "dependence": authority.dependence, "cluster_count": comparison["cluster_count"],
        "reproduced_primary_cluster_count": bundle["reproduced_primary_cluster_count"],
        "exposure_knots": json.loads(comparison["primary_exposure_knots"]),
        "target_knots": json.loads(comparison["target_knots"]),
        "reference": bundle["curve"][0]["reference_exposure_value"],
        "curve_points": authority.curve_points,
        **{key: bundle[key] for key in ("parameter_columns", "parameter_values", "covariance_matrix")},
        "primary_contrasts": _effect_values(pd.DataFrame(bundle["primary_contrasts"])).tolist(),
        "primary_exposure_nonlinearity_p_value": float(linear_sensitivity.iloc[0]["nonlinearity_p_value"]),
        "input_bindings": input_bindings,
    })
    _validate_scope(contract, step=step, authority=authority, runtime_projection_sha256=runtime_projection_sha256)
    _validate_parent_effects(contract, contrasts, linear_sensitivity, authority)
    curve = pd.DataFrame(bundle["curve"])
    curve[_IDENTITY_COLUMN] = contract.model_dump_json()
    points = curve.iloc[[0, -1]].copy()
    _validate_pair(curve, points, contract)
    return curve, points, contract


def _validate_pair(curve, points, contract):
    if len(curve) != contract.curve_points or len(points) != 2:
        raise ValueError("functional-form effects require the full curve and both point contrasts")
    expected_grid = np.linspace(contract.exposure_knots[0], contract.exposure_knots[-1], contract.curve_points)
    _require_close(curve["exposure_value"], expected_grid, "primary exposure grid")
    _require_close(_effect_values(points), _effect_values(curve.iloc[[0, -1]]), "curve/point agreement")
    if points["contrast_vector"].tolist() != curve.iloc[[0, -1]]["contrast_vector"].tolist():
        raise ValueError("functional-form curve/point model vectors differ")
    _validate_effect_math(curve, contract)
    _validate_effect_math(points, contract)


def consume_functional_form_effects(
    *, step, authority, runtime_projection_sha256, curve, points,
    contrasts, linear_sensitivity, primary_input_bindings,
) -> FunctionalFormEffects:
    """Consume two already digest-verified registered products, never a fit table."""

    contracts = []
    for frame in (curve, points):
        if _IDENTITY_COLUMN not in frame or frame.empty:
            raise ValueError("functional-form fit-only/empty product is not exposure-effect evidence")
        values = frame[_IDENTITY_COLUMN].unique()
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("functional-form effect table contains mixed model identities")
        contracts.append(FunctionalFormEffects.model_validate_json(values[0]))
    if contracts[0] != contracts[1]:
        raise ValueError("functional-form curve and contrasts belong to different model receipts")
    contract = contracts[0]
    _validate_scope(contract, step=step, authority=authority, runtime_projection_sha256=runtime_projection_sha256)
    _validate_parent_effects(contract, contrasts, linear_sensitivity, authority)
    sources = {item.input_key: item for item in contract.input_bindings}
    for key in (authority.downstream_parent_product, authority.linear_sensitivity_product):
        bound = primary_input_bindings[key]
        if (sources[key].evidence_id, sources[key].sha256, sources[key].row_count) != (
            bound.evidence_id, bound.sha256, bound.row_count,
        ):
            raise ValueError("functional-form refit was derived from different primary evidence")
    _validate_pair(curve, points, contract)
    return contract


def _projection_linear_term(linear_sensitivity: pd.DataFrame, authority) -> tuple[float, float]:
    """Invert the sealed per-unit odds ratio into its coefficient and standard error.

    The primary owner published ``exp(beta * increment)`` with a symmetric Wald
    interval on the log-odds scale, so both are recoverable without another fit.
    The roundtrip is asserted here so a row sealed by any other interval method
    fails closed instead of being silently reinterpreted as one.
    """

    if len(linear_sensitivity) != 1:
        raise ValueError("functional-form projection source must contain exactly one row")
    missing = sorted(
        {"exposure_increment", "adjusted_odds_ratio", "ci_low", "ci_high"}
        - set(linear_sensitivity.columns)
    )
    if missing:
        raise ValueError(
            "sealed linear sensitivity lacks projection coordinates: " + ", ".join(missing)
        )
    row = linear_sensitivity.iloc[0]
    increment = float(row["exposure_increment"])
    if increment != authority.linear_sensitivity_per_unit:
        raise ValueError("functional-form projection increment changed from the reviewed contract")
    odds, low, high = float(row["adjusted_odds_ratio"]), float(row["ci_low"]), float(row["ci_high"])
    if increment <= 0 or not 0 < low <= odds <= high:
        raise ValueError("sealed linear sensitivity interval is invalid")
    beta = math.log(odds) / increment
    standard_error = (math.log(high) - math.log(low)) / (2 * _WALD_Z * increment)
    if standard_error < 0:
        raise ValueError("sealed linear sensitivity interval is not a symmetric Wald interval")
    _require_close(
        np.exp(np.array([beta, beta - _WALD_Z * standard_error, beta + _WALD_Z * standard_error]) * increment),
        np.array([odds, low, high]),
        "sealed linear term reproduction",
    )
    return beta, standard_error


def _sealed_primary_coordinates(contrasts: pd.DataFrame, authority):
    """Read the two sealed contrast coordinates and the reference they share."""

    if len(contrasts) != 2 or not contrasts["exposure"].eq(authority.exposure_column).all():
        raise ValueError("functional-form source primary contrast identity mismatch")
    values = _effect_values(contrasts.sort_values("exposure_value"))
    lower, upper, reference = values[0][0], values[1][0], values[0][1]
    if not lower < reference < upper:
        raise ValueError("functional-form primary contrasts do not bracket their reference")
    return (lower, reference, upper), values.tolist()


def _projection_curve_rows(contract: FunctionalFormProjection) -> list[dict[str, Any]]:
    grid = np.linspace(contract.exposure_knots[0], contract.exposure_knots[-1], contract.curve_points)
    beta = contract.parameter_values[0]
    standard_error = math.sqrt(contract.covariance_matrix[0][0])
    rows = []
    for value in grid:
        delta = float(value) - contract.reference
        eta = beta * delta
        se = standard_error * abs(delta)
        interval = np.exp([eta, eta - _WALD_Z * se, eta + _WALD_Z * se])
        if not np.isfinite(interval).all() or (interval <= 0).any():
            raise ValueError("functional-form projection effect is non-finite")
        rows.append({
            "exposure": contract.exposure, "exposure_value": float(value),
            "reference_exposure_value": contract.reference,
            "adjusted_odds_ratio": float(interval[0]),
            "ci_low": float(interval[1]), "ci_high": float(interval[2]),
            "contrast_vector": json.dumps([delta], allow_nan=False),
        })
    return rows


def _validate_projection_scope(contract, *, step, authority, runtime_projection_sha256):
    form = authority.require_functional_form_spec(step)
    if form.target_column != authority.exposure_column:
        raise ValueError("a covariate functional-form target must refit, not project")
    expected = {
        "step_id": step.step_id,
        "spec_id": step.sensitivity_spec_ids[0],
        "form": form,
        "exposure": authority.exposure_column,
        "outcome": authority.outcome_column,
        "outcome_time_column": authority.outcome_time_column,
        "adjustment_columns": tuple(authority.required_adjustment_columns),
        "categorical_adjustment_columns": tuple(authority.categorical_adjustment_columns),
        "landmark_hours": authority.landmark_hours,
        "observation_duration_column": authority.observation_duration_column,
        "observation_duration_unit": authority.observation_duration_unit,
        "runtime_projection_sha256": runtime_projection_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "protocol_content_sha256": authority.protocol_content_sha256,
        "dependence": authority.dependence,
        "curve_points": authority.curve_points,
        "exposure_increment": authority.linear_sensitivity_per_unit,
    }
    if any(getattr(contract, key) != value for key, value in expected.items()):
        raise ValueError("functional-form projection scope drifted from its reviewed parent/runtime")
    if [key for key in step.inputs if is_closed_cohort_product_key(key)]:
        raise ValueError("a projection binds no cohort because it refits no population")
    expected_inputs = {authority.downstream_parent_product, authority.linear_sensitivity_product}
    sources = {item.input_key: item for item in contract.input_bindings}
    if set(sources) != expected_inputs:
        raise ValueError("functional-form projection lineage lacks the exact sealed primary tables")
    if sources[authority.downstream_parent_product].row_count != 2 or sources[authority.linear_sensitivity_product].row_count != 1:
        raise ValueError("functional-form projection source row counts changed")


def _validate_projection_math(frame: pd.DataFrame, contract: FunctionalFormProjection) -> None:
    if not frame["exposure"].eq(contract.exposure).all():
        raise ValueError("functional-form projection exposure changed")
    values = _effect_values(frame)
    _require_close(values[:, 1], np.full(len(frame), contract.reference), "reference")
    if not (0 < values[:, 3]).all() or not (values[:, 3] <= values[:, 2]).all() or not (values[:, 2] <= values[:, 4]).all():
        raise ValueError("functional-form projection intervals are invalid")
    vectors = np.asarray([json.loads(value) for value in frame["contrast_vector"]], dtype=float)
    if vectors.shape != (len(frame), 1) or not np.isfinite(vectors).all():
        raise ValueError("functional-form projection contrast vectors have invalid dimensions")
    # The single coordinate must be the distance from the sealed reference, so a
    # table cannot display one contrast while claiming another.
    _require_close(vectors[:, 0], values[:, 0] - contract.reference, "contrast coordinate")
    variance = contract.covariance_matrix[0][0]
    if variance < 0:
        raise ValueError("functional-form projection variance is negative")
    eta = contract.parameter_values[0] * vectors[:, 0]
    se = math.sqrt(variance) * np.abs(vectors[:, 0])
    expected = np.exp(np.column_stack([eta, eta - _WALD_Z * se, eta + _WALD_Z * se]))
    _require_close(values[:, 2:], expected, "projection-derived OR/CI")


def _validate_projection_pair(curve, points, contract):
    if len(curve) != contract.curve_points or len(points) != 2:
        raise ValueError("functional-form projection requires the full curve and both point contrasts")
    expected_grid = np.linspace(contract.exposure_knots[0], contract.exposure_knots[-1], contract.curve_points)
    _require_close(curve["exposure_value"], expected_grid, "primary exposure grid")
    _require_close(_effect_values(points), _effect_values(curve.iloc[[0, -1]]), "curve/point agreement")
    if points["contrast_vector"].tolist() != curve.iloc[[0, -1]]["contrast_vector"].tolist():
        raise ValueError("functional-form projection curve/point model vectors differ")
    _require_close(
        points["exposure_value"],
        [contract.exposure_knots[0], contract.exposure_knots[-1]],
        "projection contrast coordinates",
    )
    _validate_projection_math(curve, contract)
    _validate_projection_math(points, contract)


def seal_functional_form_projection(
    *, step, authority, runtime_projection_sha256,
    contrasts, linear_sensitivity, input_bindings, nonlinearity_p_value,
):
    """Restate the sealed linear term on the primary grid, then validate it.

    No model is fitted and no coordinate is chosen here: the knots, reference,
    grid length, and increment all come from the reviewed runtime, and the two
    contrast coordinates come from the bound primary product.
    """

    beta, standard_error = _projection_linear_term(linear_sensitivity, authority)
    knots, primary_contrasts = _sealed_primary_coordinates(contrasts, authority)
    source = linear_sensitivity.iloc[0]
    contract = FunctionalFormProjection.model_validate({
        "step_id": step.step_id, "spec_id": step.sensitivity_spec_ids[0],
        "form": step.functional_form_spec,
        "exposure": authority.exposure_column, "outcome": authority.outcome_column,
        "outcome_time_column": authority.outcome_time_column,
        "adjustment_columns": authority.required_adjustment_columns,
        "categorical_adjustment_columns": authority.categorical_adjustment_columns,
        "landmark_hours": authority.landmark_hours,
        "observation_duration_column": authority.observation_duration_column,
        "observation_duration_unit": authority.observation_duration_unit,
        "runtime_projection_sha256": runtime_projection_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "protocol_content_sha256": authority.protocol_content_sha256,
        "n": int(source["n"]), "events": int(source["events"]),
        "dependence": authority.dependence,
        "exposure_knots": knots, "reference": knots[1],
        "curve_points": authority.curve_points,
        "exposure_increment": authority.linear_sensitivity_per_unit,
        "parameter_columns": (authority.exposure_column,),
        "parameter_values": (beta,), "covariance_matrix": ((standard_error ** 2,),),
        "primary_contrasts": primary_contrasts,
        "primary_exposure_nonlinearity_p_value": float(nonlinearity_p_value),
        "input_bindings": input_bindings,
    })
    _validate_projection_scope(contract, step=step, authority=authority, runtime_projection_sha256=runtime_projection_sha256)
    _validate_parent_effects(contract, contrasts, linear_sensitivity, authority)
    curve = pd.DataFrame(_projection_curve_rows(contract))
    curve[_PROJECTION_IDENTITY_COLUMN] = contract.model_dump_json()
    points = curve.iloc[[0, -1]].copy()
    _validate_projection_pair(curve, points, contract)
    return curve, points, contract


def effect_product_summary(*, contract: Any, step: Any) -> dict[str, Any]:
    """Bounded summary of one sealed effect contract, without model internals.

    The full parameters, covariance and lineage stay in the registered tables.
    The key names say which kind of evidence a reader holds: an independent
    refit seals its own model rows, a projection seals the increment it reused.
    """

    digest = hashlib.sha256(contract.model_dump_json().encode("utf-8")).hexdigest()
    if contract.independent_refit:
        return {
            "contract_sha256": digest,
            "spec_id": contract.spec_id,
            "target_column": contract.form.target_column,
            "exposure": contract.exposure,
            "analysis_role": contract.analysis_role,
            "independent_refit": contract.independent_refit,
            "model_rows_sha256": contract.model_rows_sha256,
            "curve_product": step.expected_outputs[1],
            "contrast_product": step.expected_outputs[2],
            "primary_exposure_nonlinearity_p_value": contract.primary_exposure_nonlinearity_p_value,
        }
    return {
        "contract_sha256": digest,
        "spec_id": contract.spec_id,
        "target_column": contract.form.target_column,
        "exposure": contract.exposure,
        "analysis_role": contract.analysis_role,
        "independent_refit": contract.independent_refit,
        "projected_specification": contract.projected_specification,
        "exposure_increment": contract.exposure_increment,
        "curve_product": step.expected_outputs[1],
        "contrast_product": step.expected_outputs[2],
        "primary_exposure_nonlinearity_p_value": contract.primary_exposure_nonlinearity_p_value,
    }
