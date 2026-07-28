"""Machine acceptance for the case-scoped E1 scientific closure.

This module is deliberately outside the installed research-agent package.
It owns the E1 benchmark contract and validates only durable, structured run
artifacts.  Shared prompts and shared execution gates remain case-neutral.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import pandas as pd

CONTRACT_SCHEMA = "easyicu.e1_scientific_acceptance_contract/1"
RECEIPT_SCHEMA = "easyicu.e1_scientific_acceptance_receipt/1"
TASK_ID = "e1_sepsis3_prevalence_mortality"
SENSITIVITY_PRODUCT = "table:e1_scientific_sensitivity"

_SENSITIVITY_COLUMNS = (
    "analysis_id",
    "n_stays",
    "n_deaths",
    "odds_ratio",
    "ci_low",
    "ci_high",
    "landmark_hours",
    "alive_at_landmark_required",
    "negative_event_times_excluded",
    "readmission_restriction",
    "age_form",
    "charlson_form",
)
_SENSITIVITY_ROWS = (
    "primary_full_cohort",
    "landmark_alive_at_24h",
    "non_readmission_icu_stays",
    "flexible_age_charlson",
)
_DISPLAY_LABELS = {
    "sep3_sofa2_max=0": "Sepsis-3 absent",
    "sep3_sofa2_max=1": "Sepsis-3 present",
}


def e1_scientific_acceptance_contract() -> dict[str, Any]:
    """Return the public E1 benchmark contract embedded in materialized JSONL."""

    return {
        "schema_version": CONTRACT_SCHEMA,
        "task_id": TASK_ID,
        "table_one_product": "table:table_one",
        "missingness_product": "table:missingness_measurement_audit",
        "primary_model_product": "table:adjusted_association_estimates",
        "sensitivity_product": SENSITIVITY_PRODUCT,
        "sensitivity_columns": list(_SENSITIVITY_COLUMNS),
        "sensitivity_analysis_ids": list(_SENSITIVITY_ROWS),
        "required_display_labels": dict(_DISPLAY_LABELS),
        "analysis_cohort_input": "artifact:analysis_cohort",
        "exposure_column": "sep3_sofa2_max",
        "outcome_column": "death",
        "event_time_column": "death_time",
        "readmission_column": "icu_readmission",
        "positive_only_event_column": "susp_inf_max",
        "landmark_hours": 24.0,
    }


def sensitivity_output_instruction() -> str:
    """Return the exact public artifact instruction sent only to the E1 task."""

    columns = ", ".join(_SENSITIVITY_COLUMNS)
    rows = ", ".join(_SENSITIVITY_ROWS)
    return (
        "In a separate analysis step (do not widen the standard "
        "robustness_sensitivity step), emit table:e1_scientific_sensitivity "
        f"with exact columns [{columns}] and one row for each analysis_id "
        f"[{rows}]. The landmark row must require survival to 24 hours and "
        "exclude negative death times; the repeated-stay row must restrict to "
        "non-readmission ICU stays; the flexible row must use non-linear age "
        "and Charlson terms. Every model row must report n_stays, n_deaths, "
        "odds_ratio, ci_low, and ci_high."
    )


def display_label_instruction() -> str:
    """Return the E1 level-label convention consumed by deterministic figures."""

    return (
        "Declare AnalysisPlan.display_labels entries "
        "'sep3_sofa2_max=0': 'Sepsis-3 absent' and "
        "'sep3_sofa2_max=1': 'Sepsis-3 present'; figures must consume these "
        "Planner-owned labels and must not render Category 0/1."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _contained_regular_file(root: Path, path: Path) -> Path | None:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _issue(
    code: str,
    message: str,
    **detail: object,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"reason_code": code, "message": message}
    if detail:
        payload["detail"] = detail
    return payload


def _step_summaries(
    run_dir: Path,
    issues: list[dict[str, Any]],
) -> list[tuple[str, Path, dict[str, Any]]]:
    summaries: list[tuple[str, Path, dict[str, Any]]] = []
    steps_dir = run_dir / "steps"
    if steps_dir.is_symlink() or not steps_dir.is_dir():
        issues.append(
            _issue("e1_steps_missing", "Run does not contain a regular steps directory.")
        )
        return summaries
    for step_dir in sorted(steps_dir.iterdir()):
        path = _contained_regular_file(
            run_dir,
            step_dir / "outputs" / "step_summary.json",
        )
        if path is None:
            continue
        try:
            payload = _read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _issue(
                    "e1_step_summary_invalid",
                    "A step summary is unreadable.",
                    step_id=step_dir.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        summaries.append((step_dir.name, path, payload))
    return summaries


def _summary_for_product(
    summaries: Sequence[tuple[str, Path, dict[str, Any]]],
    product: str,
    issues: list[dict[str, Any]],
) -> tuple[str, Path, dict[str, Any], Path] | None:
    matches: list[tuple[str, Path, dict[str, Any], Path]] = []
    for step_id, summary_path, summary in summaries:
        output_files = summary.get("output_files")
        raw_path = output_files.get(product) if isinstance(output_files, dict) else None
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        artifact = _contained_regular_file(
            summary_path.parents[3],
            summary_path.parent / raw_path,
        )
        if artifact is not None:
            matches.append((step_id, summary_path, summary, artifact))
    if len(matches) != 1:
        issues.append(
            _issue(
                "e1_product_cardinality_invalid",
                "A required E1 product must have exactly one current producer.",
                product=product,
                producer_count=len(matches),
            )
        )
        return None
    return matches[0]


def _artifact_is_registered(
    *,
    manifest: Mapping[str, Any],
    step_id: str,
    artifact: Path,
    issues: list[dict[str, Any]],
) -> bool:
    observed_sha = _sha256(artifact)
    candidates = [
        record
        for record in manifest.get("evidence") or []
        if isinstance(record, dict)
        and str(record.get("produced_by_step") or "") == step_id
        and str(record.get("relative_path") or "").endswith(
            f"__{artifact.name}"
        )
        and str(record.get("sha256") or "") == observed_sha
    ]
    if not candidates:
        issues.append(
            _issue(
                "e1_artifact_not_registered",
                "A scientific acceptance artifact is not digest-bound in EvidenceStore.",
                step_id=step_id,
                artifact=artifact.name,
                sha256=observed_sha,
            )
        )
        return False
    return True


def _require_typed_cohort_receipt(
    *,
    summary: Mapping[str, Any],
    step_id: str,
    input_key: str,
    issues: list[dict[str, Any]],
) -> None:
    rows = summary.get("input_bindings")
    receipt = next(
        (
            row
            for row in rows or []
            if isinstance(row, dict) and row.get("input_key") == input_key
        ),
        None,
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("loaded") is not True
        or not isinstance(receipt.get("row_count"), int)
        or not isinstance(receipt.get("sha256"), str)
        or len(str(receipt.get("sha256"))) != 64
    ):
        issues.append(
            _issue(
                "e1_typed_input_consumption_missing",
                "A model step did not prove consumption of the typed analysis cohort.",
                step_id=step_id,
                input_key=input_key,
            )
        )


def _validate_table_one(
    *,
    match: tuple[str, Path, dict[str, Any], Path] | None,
    manifest: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if match is None:
        return
    step_id, _, _, path = match
    _artifact_is_registered(
        manifest=manifest,
        step_id=step_id,
        artifact=path,
        issues=issues,
    )
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        issues.append(
            _issue(
                "e1_table_one_unreadable",
                "Table 1 could not be read.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return
    required = {
        "schema_version",
        "variable",
        "absolute_standardized_mean_difference",
        "standardized_difference_status",
    }
    if not required.issubset(table.columns):
        issues.append(
            _issue(
                "e1_table_one_smd_columns_missing",
                "Table 1 does not expose the structured SMD contract.",
                missing_columns=sorted(required - set(table.columns)),
            )
        )
        return
    if set(table["schema_version"].dropna().astype(str)) != {
        "easyicu.table_one_result/2"
    }:
        issues.append(
            _issue(
                "e1_table_one_schema_invalid",
                "Table 1 is not the SMD-bearing schema version.",
            )
        )
    numeric = pd.to_numeric(
        table["absolute_standardized_mean_difference"],
        errors="coerce",
    )
    status = table["standardized_difference_status"].astype(str)
    for variable in sorted(table["variable"].dropna().astype(str).unique()):
        mask = table["variable"].astype(str).eq(variable)
        computed = mask & status.eq("computed") & numeric.map(
            lambda value: bool(pd.notna(value) and math.isfinite(float(value)))
        )
        if not bool(computed.any()):
            issues.append(
                _issue(
                    "e1_table_one_smd_incomplete",
                    "A Table 1 variable has no computed finite SMD.",
                    variable=variable,
                )
            )


def _validate_missingness(
    *,
    match: tuple[str, Path, dict[str, Any], Path] | None,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if match is None:
        return
    step_id, _, summary, path = match
    _artifact_is_registered(
        manifest=manifest,
        step_id=step_id,
        artifact=path,
        issues=issues,
    )
    audits = summary.get("observation_semantics_audit")
    if not isinstance(audits, dict):
        issues.append(
            _issue(
                "e1_observation_semantics_missing",
                "Missingness audit omitted typed observation-semantics evidence.",
            )
        )
        return
    positive_column = str(contract["positive_only_event_column"])
    positive = audits.get(positive_column)
    if (
        not isinstance(positive, dict)
        or positive.get("indicator_semantics") != "binary_event_presence"
        or not isinstance(positive.get("event_absent_n"), int)
        or int(positive.get("invalid_pair_n") or 0) != 0
        or int(positive.get("discordant_n") or 0) != 0
    ):
        issues.append(
            _issue(
                "e1_positive_only_event_semantics_invalid",
                "The positive-only suspected-infection event was not reconciled as a complete binary status.",
                column=positive_column,
            )
        )
    event_time_column = str(contract["event_time_column"])
    event_time = audits.get(event_time_column)
    if (
        not isinstance(event_time, dict)
        or event_time.get("observation_semantics") != "conditional_event_time"
        or not isinstance(event_time.get("not_applicable_event_absent_n"), int)
        or not isinstance(event_time.get("before_origin_n"), int)
        or int(event_time.get("contradictory_event_absent_with_time_n") or 0) != 0
    ):
        issues.append(
            _issue(
                "e1_conditional_event_time_semantics_invalid",
                "Death time was not audited as a conditional event time.",
                column=event_time_column,
            )
        )
    temporal = summary.get("temporal_validity_audit")
    if (
        not isinstance(temporal, dict)
        or temporal.get("status")
        not in {"ok", "flagged_requires_downstream_protocol"}
    ):
        issues.append(
            _issue(
                "e1_temporal_validity_status_missing",
                "Missingness audit omitted the structured event-time validity status.",
            )
        )


def _numeric_series(
    table: pd.DataFrame,
    column: str,
    *,
    issues: list[dict[str, Any]],
) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    invalid = values.isna() | ~values.map(
        lambda value: bool(pd.notna(value) and math.isfinite(float(value)))
    )
    if bool(invalid.any()):
        issues.append(
            _issue(
                "e1_sensitivity_numeric_invalid",
                "Sensitivity table contains a missing or non-finite required number.",
                column=column,
                invalid_rows=table.loc[invalid, "analysis_id"].astype(str).tolist(),
            )
        )
    return values


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _load_analysis_cohort(
    *,
    summaries: Sequence[tuple[str, Path, dict[str, Any]]],
    manifest: Mapping[str, Any],
    input_key: str,
    issues: list[dict[str, Any]],
) -> pd.DataFrame | None:
    match = _summary_for_product(summaries, input_key, issues)
    if match is None:
        return None
    step_id, _, _, path = match
    if not _artifact_is_registered(
        manifest=manifest,
        step_id=step_id,
        artifact=path,
        issues=issues,
    ):
        return None
    try:
        if path.suffix.casefold() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.casefold() == ".csv":
            return pd.read_csv(path)
    except Exception as exc:
        issues.append(
            _issue(
                "e1_analysis_cohort_unreadable",
                "The typed analysis cohort could not be read for acceptance checks.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return None
    issues.append(
        _issue(
            "e1_analysis_cohort_format_invalid",
            "The typed analysis cohort uses an unsupported format.",
            suffix=path.suffix,
        )
    )
    return None


def _validate_sensitivity(
    *,
    match: tuple[str, Path, dict[str, Any], Path] | None,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    cohort: pd.DataFrame | None,
    issues: list[dict[str, Any]],
) -> None:
    if match is None:
        return
    step_id, _, summary, path = match
    _artifact_is_registered(
        manifest=manifest,
        step_id=step_id,
        artifact=path,
        issues=issues,
    )
    _require_typed_cohort_receipt(
        summary=summary,
        step_id=step_id,
        input_key=str(contract["analysis_cohort_input"]),
        issues=issues,
    )
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        issues.append(
            _issue(
                "e1_sensitivity_unreadable",
                "The E1 scientific sensitivity table could not be read.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return
    required_columns = set(str(item) for item in contract["sensitivity_columns"])
    if not required_columns.issubset(table.columns):
        issues.append(
            _issue(
                "e1_sensitivity_columns_missing",
                "The E1 sensitivity table is missing required structured columns.",
                missing_columns=sorted(required_columns - set(table.columns)),
            )
        )
        return
    analysis_ids = table["analysis_id"].astype(str)
    expected_ids = [str(item) for item in contract["sensitivity_analysis_ids"]]
    if analysis_ids.duplicated().any() or set(analysis_ids) != set(expected_ids):
        issues.append(
            _issue(
                "e1_sensitivity_rows_invalid",
                "The E1 sensitivity table does not contain exactly the required analyses.",
                observed=analysis_ids.tolist(),
                expected=expected_ids,
            )
        )
        return
    table = table.set_index("analysis_id", drop=False)
    numeric = {
        column: _numeric_series(table, column, issues=issues)
        for column in ("n_stays", "n_deaths", "odds_ratio", "ci_low", "ci_high")
    }
    if bool(
        (numeric["n_stays"] < 1).any()
        or (numeric["n_deaths"] < 0).any()
        or (numeric["n_deaths"] > numeric["n_stays"]).any()
        or (numeric["odds_ratio"] <= 0).any()
        or (numeric["ci_low"] <= 0).any()
        or (numeric["ci_high"] < numeric["ci_low"]).any()
        or (numeric["odds_ratio"] < numeric["ci_low"]).any()
        or (numeric["odds_ratio"] > numeric["ci_high"]).any()
    ):
        issues.append(
            _issue(
                "e1_sensitivity_estimate_contract_invalid",
                "Sensitivity denominators or confidence intervals are incoherent.",
            )
        )

    landmark_id = "landmark_alive_at_24h"
    landmark = table.loc[landmark_id]
    landmark_hours = pd.to_numeric(
        pd.Series([landmark["landmark_hours"]]),
        errors="coerce",
    ).iloc[0]
    if (
        not math.isfinite(float(landmark_hours))
        or not math.isclose(
            float(landmark_hours),
            float(contract["landmark_hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not _truthy(landmark["alive_at_landmark_required"])
        or not _truthy(landmark["negative_event_times_excluded"])
    ):
        issues.append(
            _issue(
                "e1_landmark_protocol_invalid",
                "The landmark sensitivity did not prove the 24-hour alive-at-landmark protocol.",
            )
        )
    if (
        str(
            table.loc["non_readmission_icu_stays", "readmission_restriction"]
        ).strip()
        != "non_readmission_only"
    ):
        issues.append(
            _issue(
                "e1_repeat_stay_sensitivity_invalid",
                "The repeated-stay sensitivity did not restrict to non-readmission ICU stays.",
            )
        )
    flexible = table.loc["flexible_age_charlson"]
    for column in ("age_form", "charlson_form"):
        value = str(flexible[column] or "").strip().casefold()
        if value in {"", "linear"}:
            issues.append(
                _issue(
                    "e1_functional_form_sensitivity_invalid",
                    "The flexible sensitivity retained a linear functional form.",
                    column=column,
                    value=value,
                )
            )

    if cohort is None:
        return
    required_cohort_columns = {
        str(contract["outcome_column"]),
        str(contract["event_time_column"]),
        str(contract["readmission_column"]),
    }
    if not required_cohort_columns.issubset(cohort.columns):
        issues.append(
            _issue(
                "e1_acceptance_cohort_columns_missing",
                "The analysis cohort lacks columns required to verify sensitivity denominators.",
                missing_columns=sorted(required_cohort_columns - set(cohort.columns)),
            )
        )
        return
    outcome = pd.to_numeric(
        cohort[str(contract["outcome_column"])],
        errors="coerce",
    )
    event_time = pd.to_numeric(
        cohort[str(contract["event_time_column"])],
        errors="coerce",
    )
    readmission = pd.to_numeric(
        cohort[str(contract["readmission_column"])],
        errors="coerce",
    )
    if (
        outcome.isna().any()
        or not outcome.isin([0, 1]).all()
        or ((outcome.eq(1)) & event_time.isna()).any()
        or readmission.isna().any()
        or not readmission.isin([0, 1]).all()
    ):
        issues.append(
            _issue(
                "e1_acceptance_cohort_semantics_invalid",
                "The analysis cohort cannot support exact E1 denominator verification.",
            )
        )
        return
    alive_at_landmark = ~(
        outcome.eq(1)
        & event_time.le(float(contract["landmark_hours"]))
    )
    non_readmission = readmission.eq(0)
    expected_n = {
        "primary_full_cohort": int(len(cohort)),
        "landmark_alive_at_24h": int(alive_at_landmark.sum()),
        "non_readmission_icu_stays": int(non_readmission.sum()),
        "flexible_age_charlson": int(len(cohort)),
    }
    for analysis_id, expected in expected_n.items():
        observed = int(round(float(numeric["n_stays"].loc[analysis_id])))
        if observed != expected:
            issues.append(
                _issue(
                    "e1_sensitivity_denominator_mismatch",
                    "A sensitivity denominator disagrees with the sealed analysis cohort.",
                    analysis_id=analysis_id,
                    observed_n=observed,
                    expected_n=expected,
                )
            )


def evaluate_e1_scientific_acceptance(
    *,
    run_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one finalized E1 run without altering pipeline authority."""

    run_dir = Path(run_dir).resolve()
    issues: list[dict[str, Any]] = []
    expected_contract = e1_scientific_acceptance_contract()
    if dict(contract) != expected_contract:
        issues.append(
            _issue(
                "e1_scientific_contract_mismatch",
                "The imported E1 scientific contract differs from the repository contract.",
                expected_sha256=_canonical_sha256(expected_contract),
                observed_sha256=_canonical_sha256(dict(contract)),
            )
        )
        contract = expected_contract

    manifest_path = _contained_regular_file(run_dir, Path("manifest.json"))
    plan_path = _contained_regular_file(run_dir, Path("analysis_plan.json"))
    if manifest_path is None or plan_path is None:
        issues.append(
            _issue(
                "e1_run_authority_missing",
                "Final manifest or analysis plan is missing.",
            )
        )
        return {
            "schema_version": RECEIPT_SCHEMA,
            "task_id": TASK_ID,
            "run_id": run_dir.name,
            "status": "rejected",
            "contract_sha256": _canonical_sha256(dict(contract)),
            "issues": issues,
        }
    try:
        manifest = _read_json_object(manifest_path)
        plan = _read_json_object(plan_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(
            _issue(
                "e1_run_authority_invalid",
                "Final manifest or analysis plan is unreadable.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        manifest = {}
        plan = {}

    labels = plan.get("display_labels")
    for key, expected_label in dict(contract["required_display_labels"]).items():
        observed = labels.get(key) if isinstance(labels, dict) else None
        if observed != expected_label:
            issues.append(
                _issue(
                    "e1_clinical_display_label_missing",
                    "A required Planner-owned clinical level label is absent.",
                    key=key,
                    expected=expected_label,
                    observed=observed,
                )
            )

    summaries = _step_summaries(run_dir, issues)
    table_one = _summary_for_product(
        summaries,
        str(contract["table_one_product"]),
        issues,
    )
    missingness = _summary_for_product(
        summaries,
        str(contract["missingness_product"]),
        issues,
    )
    primary = _summary_for_product(
        summaries,
        str(contract["primary_model_product"]),
        issues,
    )
    sensitivity = _summary_for_product(
        summaries,
        str(contract["sensitivity_product"]),
        issues,
    )

    _validate_table_one(match=table_one, manifest=manifest, issues=issues)
    _validate_missingness(
        match=missingness,
        manifest=manifest,
        contract=contract,
        issues=issues,
    )
    if primary is not None:
        primary_step_id, _, primary_summary, primary_path = primary
        _artifact_is_registered(
            manifest=manifest,
            step_id=primary_step_id,
            artifact=primary_path,
            issues=issues,
        )
        _require_typed_cohort_receipt(
            summary=primary_summary,
            step_id=primary_step_id,
            input_key=str(contract["analysis_cohort_input"]),
            issues=issues,
        )
    cohort = _load_analysis_cohort(
        summaries=summaries,
        manifest=manifest,
        input_key=str(contract["analysis_cohort_input"]),
        issues=issues,
    )
    _validate_sensitivity(
        match=sensitivity,
        manifest=manifest,
        contract=contract,
        cohort=cohort,
        issues=issues,
    )

    return {
        "schema_version": RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "status": "accepted" if not issues else "rejected",
        "contract_sha256": _canonical_sha256(dict(contract)),
        "manifest_sha256": _sha256(manifest_path),
        "analysis_plan_sha256": _sha256(plan_path),
        "issues": issues,
    }


def write_e1_scientific_acceptance_receipt(
    *,
    run_dir: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Write the evaluator receipt beside, never inside, run authority."""

    run_dir = Path(run_dir).resolve()
    payload = evaluate_e1_scientific_acceptance(
        run_dir=run_dir,
        contract=contract,
    )
    destination = (
        run_dir.parent
        / f"e1_scientific_acceptance__{run_dir.name}.json"
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    raw = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while persisting E1 acceptance receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload, destination


__all__ = [
    "CONTRACT_SCHEMA",
    "RECEIPT_SCHEMA",
    "SENSITIVITY_PRODUCT",
    "TASK_ID",
    "display_label_instruction",
    "e1_scientific_acceptance_contract",
    "evaluate_e1_scientific_acceptance",
    "sensitivity_output_instruction",
    "write_e1_scientific_acceptance_receipt",
]
