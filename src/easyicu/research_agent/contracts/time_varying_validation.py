"""Deterministic validation for the analysis-only time-varying Cox owner."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .time_varying_exposure import (
    TIME_VARYING_ANALYSIS_KIND,
    TIME_VARYING_EXPOSURE_CAPABILITY,
    TIME_VARYING_EXPOSURE_METHOD,
    TimeVaryingExposureSpecification,
)


_CONTRACT_REF = re.compile(r"^scientific_runtime_contract:([0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ESTIMATE_COLUMNS = (
    "term",
    "coefficient",
    "standard_error",
    "hazard_ratio",
    "ci_low",
    "ci_high",
    "z_value",
    "p_value",
)


def _primary_step(plan: object) -> Any | None:
    steps = [
        step
        for step in (getattr(plan, "steps", ()) or ())
        if getattr(step, "planned_analysis_role", None) == "primary"
    ]
    return steps[0] if len(steps) == 1 else None


def _safe_output_path(
    *, run_dir: Path, step_id: str, output_files: Mapping[str, Any], product: str
) -> Path | None:
    filename = output_files.get(product)
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    return Path(run_dir) / "steps" / step_id / "outputs" / filename


def _integer_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        counts = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError):
        return None
    return counts if all(item >= 0 for item in counts.values()) else None


def time_varying_runtime_bundle_errors(
    *, plan: object, records: Sequence[Mapping[str, Any]], run_dir: Path
) -> list[str]:
    """Replay the sealed receipt and aggregate tables without widening its claim."""

    step = _primary_step(plan)
    if step is None:
        return ["time-varying validator requires exactly one primary owner"]
    refs = tuple(getattr(step, "icu_rule_refs", ()) or ())
    contract_match = (
        _CONTRACT_REF.fullmatch(str(refs[0])) if len(refs) == 1 else None
    )
    expected_outputs = {
        "table:time_varying_cox_estimates",
        "table:time_varying_input_audit",
        "log:time_varying_runtime_receipt",
    }
    if not (
        getattr(step, "method", None) == TIME_VARYING_EXPOSURE_METHOD
        and getattr(step, "scientific_capability", None)
        == TIME_VARYING_EXPOSURE_CAPABILITY
        and set(getattr(step, "expected_outputs", ()) or ()) == expected_outputs
        and contract_match is not None
    ):
        return ["time-varying plan does not match its signed primary owner"]

    matches = [
        record
        for record in records
        if record.get("step_id") == getattr(step, "step_id", None)
        and record.get("deterministic_standard_analysis")
        == TIME_VARYING_ANALYSIS_KIND
        and isinstance(record.get("step_summary"), Mapping)
    ]
    if len(matches) != 1:
        return ["time-varying validator requires exactly one current owner receipt"]
    record = matches[0]
    summary = record["step_summary"]
    receipt = summary.get("scientific_runtime_receipt")
    if not isinstance(receipt, Mapping):
        return ["time-varying scientific runtime receipt is absent"]

    errors: list[str] = []
    if not (
        record.get("status") == "ok"
        and summary.get("status") == "ok"
        and summary.get("analysis_family") == "association"
        and summary.get("interpretation_class")
        == "descriptive_time_updated_association"
        and summary.get("variance_estimator") == "cluster_robust"
    ):
        errors.append("time-varying owner summary is not a successful descriptive fit")

    construction = receipt.get("construction")
    fit = receipt.get("fit")
    if not isinstance(construction, Mapping) or not isinstance(fit, Mapping):
        return errors + ["time-varying receipt lacks construction or fit authority"]
    try:
        specification = TimeVaryingExposureSpecification.model_validate(
            construction.get("specification")
        )
    except Exception as exc:
        return errors + [f"time-varying specification receipt is invalid: {exc}"]

    if not (
        receipt.get("schema_version") == "easyicu.time_varying_runtime_receipt/1"
        and receipt.get("execution_contract_sha256") == contract_match.group(1)
        and receipt.get("specification_sha256") == specification.sha256
        and construction.get("specification_sha256") == specification.sha256
        and construction.get("schema_version")
        == "easyicu.time_varying_materialization/1"
        and receipt.get("claim_ceiling") == "analysis_only"
        and construction.get("claim_ceiling") == "analysis_only"
        and receipt.get("publication_ready") is False
        and receipt.get("interpretation")
        == "descriptive_time_updated_association_not_causal"
        and _SHA256.fullmatch(str(receipt.get("runtime_projection_sha256") or ""))
        and _SHA256.fullmatch(
            str(receipt.get("counting_process_input_sha256") or "")
        )
    ):
        errors.append("time-varying receipt authority or analysis-only ceiling drifted")

    execution_input = construction.get("execution_input")
    exposure_panel = construction.get("exposure_panel")
    followup = construction.get("followup")
    if not all(
        isinstance(value, Mapping)
        for value in (execution_input, exposure_panel, followup)
    ):
        return errors + ["time-varying construction sub-receipts are incomplete"]
    counts = _integer_counts(execution_input.get("counts"))
    panel_counts = _integer_counts(exposure_panel.get("counts"))
    exclusion_counts = _integer_counts(followup.get("exclusion_counts"))
    if counts is None or panel_counts is None or exclusion_counts is None:
        return errors + ["time-varying construction counts are malformed"]

    required_counts = {
        "cluster_count",
        "event_count",
        "fully_unmeasured_stays",
        "interval_rows",
        "observed_exposure_interval_rows",
        "patient_grouping_mapping_rows",
        "stay_count",
        "unmeasured_exposure_interval_rows",
    }
    if set(counts) != required_counts:
        errors.append("time-varying execution count contract drifted")
    else:
        stay_count = counts["stay_count"]
        event_count = counts["event_count"]
        interval_rows = counts["interval_rows"]
        cluster_count = counts["cluster_count"]
        coherent_counts = (
            stay_count > 0
            and 0 < event_count <= stay_count
            and 0 < cluster_count <= stay_count
            and interval_rows >= stay_count
            and counts["fully_unmeasured_stays"] <= stay_count
            and counts["patient_grouping_mapping_rows"] >= stay_count
            and counts["observed_exposure_interval_rows"]
            + counts["unmeasured_exposure_interval_rows"]
            == interval_rows
            and int(construction.get("analysis_stays") or -1) == stay_count
            and int(fit.get("stay_count") or -1) == stay_count
            and int(fit.get("event_count") or -1) == event_count
            and int(fit.get("interval_rows") or -1) == interval_rows
            and int(fit.get("cluster_count") or -1) == cluster_count
            and int(summary.get("n_total") or -1) == stay_count
            and int(summary.get("n_model_stays") or -1) == stay_count
            and int(summary.get("n_events") or -1) == event_count
            and int(summary.get("cluster_count") or -1) == cluster_count
            and panel_counts.get("panel_stays") == stay_count
            and panel_counts.get("panel_rows") == interval_rows
            and panel_counts.get("panel_hospital_deaths") == event_count
            and panel_counts.get("input_hospital_deaths") == event_count
            and int(followup.get("valid_stays") or -1) == stay_count
            and int(followup.get("event_stays") or -1) == event_count
            and int(followup.get("event_stays") or 0)
            + int(followup.get("censored_stays") or 0)
            == stay_count
            and int(followup.get("input_stays") or 0)
            - int(followup.get("excluded_stays", -1))
            == stay_count
            and sum(exclusion_counts.values())
            == int(followup.get("excluded_stays", -1))
        )
        if not coherent_counts:
            errors.append("time-varying population, event or interval counts disagree")

    if not (
        execution_input.get("schema_version")
        == "easyicu.time_varying_execution_input/1"
        and execution_input.get("local_only") is True
        and tuple(execution_input.get("model_covariates") or ())
        == specification.model_covariates
        and tuple(fit.get("covariates") or ()) == specification.model_covariates
        and fit.get("schema_version") == "easyicu.clustered_time_varying_cox/1"
        and fit.get("engine") == "R_survival"
        and fit.get("method") == "coxph_counting_process"
        and fit.get("ties") == "efron"
        and fit.get("variance_estimator") == "cluster_robust"
        and fit.get("diagnostics") == {"converged": True, "warnings": []}
    ):
        errors.append("time-varying fit method, covariates or diagnostics drifted")

    missingness = execution_input.get("missingness_policy")
    if not (
        isinstance(missingness, Mapping)
        and missingness.get("kind") == "observed_state_indicator"
        and missingness.get("clinical_value_imputed") is False
        and missingness.get("unmeasured_state") == "separate_indicator_term"
        and tuple(missingness.get("model_terms") or ())
        == specification.model_covariates[:2]
        and exposure_panel.get("event_predictability")
        == "measurements_at_or_after_source_event_time_excluded"
        and exposure_panel.get("pre_measurement_state") == "unmeasured"
        and exposure_panel.get("post_window_state")
        == "last_observed_running_max_persists_to_followup"
        and followup.get("analysis_unit") == "icu_stay"
        and followup.get("time_origin") == "icu_admission"
        and followup.get("time_unit") == "hours"
    ):
        errors.append("time-varying missingness, timing or follow-up policy drifted")

    privacy_contracts = (
        (
            execution_input.get("privacy"),
            {
                "identifier_values_returned": False,
                "patient_rows_returned": False,
                "source_paths_returned": False,
                "local_ephemeral_input": True,
            },
        ),
        (
            exposure_panel.get("privacy"),
            {
                "identifier_values_returned": False,
                "patient_rows_returned": False,
                "source_paths_returned": False,
            },
        ),
        (
            followup.get("privacy"),
            {
                "identifier_values_returned": False,
                "raw_rows_returned": False,
                "source_paths_returned": False,
            },
        ),
        (
            fit.get("privacy"),
            {
                "identifier_values_returned": False,
                "patient_rows_returned": False,
                "source_paths_returned": False,
            },
        ),
    )
    for scope, expected in privacy_contracts:
        if not isinstance(scope, Mapping) or any(
            scope.get(key) is not value for key, value in expected.items()
        ):
            errors.append("time-varying privacy receipt is incomplete")
            break

    bindings = summary.get("input_bindings")
    binding = bindings[0] if isinstance(bindings, list) and len(bindings) == 1 else None
    if not (
        isinstance(binding, Mapping)
        and binding.get("loaded") is True
        and int(binding.get("row_count") or -1) == counts.get("stay_count")
        and binding.get("sha256") == construction.get("analysis_cohort_sha256")
    ):
        errors.append("time-varying typed cohort binding disagrees with construction")

    output_files = summary.get("output_files")
    if not isinstance(output_files, Mapping) or set(output_files) != expected_outputs:
        return errors + ["time-varying outputs disagree with the signed plan"]
    step_id = str(getattr(step, "step_id", ""))
    receipt_path = _safe_output_path(
        run_dir=run_dir,
        step_id=step_id,
        output_files=output_files,
        product="log:time_varying_runtime_receipt",
    )
    audit_path = _safe_output_path(
        run_dir=run_dir,
        step_id=step_id,
        output_files=output_files,
        product="table:time_varying_input_audit",
    )
    estimates_path = _safe_output_path(
        run_dir=run_dir,
        step_id=step_id,
        output_files=output_files,
        product="table:time_varying_cox_estimates",
    )
    try:
        persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (AttributeError, OSError, UnicodeDecodeError, ValueError):
        persisted_receipt = None
    if persisted_receipt != receipt:
        errors.append("time-varying persisted receipt disagrees with the step summary")

    try:
        with audit_path.open(newline="", encoding="utf-8") as handle:
            audit_rows = list(csv.DictReader(handle))
        audit_counts = {row["metric"]: int(row["value"]) for row in audit_rows}
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        audit_counts = None
    if audit_counts != counts:
        errors.append("time-varying input audit table disagrees with its receipt")

    estimates_valid = True
    try:
        with estimates_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            estimate_rows = list(reader)
            columns = tuple(reader.fieldnames or ())
        if columns != _ESTIMATE_COLUMNS:
            estimates_valid = False
        terms = [row.get("term") for row in estimate_rows]
        if terms != list(specification.model_covariates):
            estimates_valid = False
        for row in estimate_rows:
            coefficient = float(row["coefficient"])
            standard_error = float(row["standard_error"])
            hazard_ratio = float(row["hazard_ratio"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            z_value = float(row["z_value"])
            p_value = float(row["p_value"])
            if not (
                all(
                    math.isfinite(value)
                    for value in (
                        coefficient,
                        standard_error,
                        hazard_ratio,
                        ci_low,
                        ci_high,
                        z_value,
                        p_value,
                    )
                )
                and standard_error > 0
                and 0 < ci_low <= hazard_ratio <= ci_high
                and 0 <= p_value <= 1
                and math.isclose(hazard_ratio, math.exp(coefficient), rel_tol=1e-10)
                and math.isclose(z_value, coefficient / standard_error, rel_tol=1e-10)
            ):
                estimates_valid = False
                break
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        estimates_valid = False
    if not estimates_valid:
        errors.append("time-varying estimate table is malformed or internally inconsistent")
    return errors


__all__ = ["time_varying_runtime_bundle_errors"]
