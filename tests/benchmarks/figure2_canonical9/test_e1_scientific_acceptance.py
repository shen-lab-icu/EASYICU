from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from benchmarks.figure2_canonical9.e1_scientific_acceptance import (
    e1_scientific_acceptance_contract,
    evaluate_e1_scientific_acceptance,
    write_e1_scientific_acceptance_receipt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_summary(
    run_dir: Path,
    *,
    step_id: str,
    output_files: dict[str, str],
    extra: dict[str, object] | None = None,
) -> Path:
    out_dir = run_dir / "steps" / step_id / "outputs"
    out_dir.mkdir(parents=True)
    summary = {"status": "ok", "output_files": output_files, **(extra or {})}
    path = out_dir / "step_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _register(
    evidence: list[dict[str, object]],
    *,
    step_id: str,
    artifact: Path,
) -> None:
    evidence.append(
        {
            "evidence_id": f"{step_id}_{artifact.stem}",
            "kind": "table",
            "relative_path": (
                f"evidence/{step_id}_{artifact.stem}__{artifact.name}"
            ),
            "sha256": _sha256(artifact),
            "produced_by_step": step_id,
        }
    )


def _typed_receipt(*, row_count: int) -> list[dict[str, object]]:
    return [
        {
            "input_key": "artifact:analysis_cohort",
            "loaded": True,
            "row_count": row_count,
            "sha256": "a" * 64,
        }
    ]


def _accepted_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "aware" / "run_e1"
    run_dir.mkdir(parents=True)
    contract = e1_scientific_acceptance_contract()
    (run_dir / "analysis_plan.json").write_text(
        json.dumps({"display_labels": contract["required_display_labels"]}),
        encoding="utf-8",
    )
    evidence: list[dict[str, object]] = []

    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "death": [0, 1, 1, 1, 0],
            "death_time": [None, 10.0, -1.0, 30.0, None],
            "icu_readmission": [0, 0, 1, 0, 1],
        }
    )
    _write_summary(
        run_dir,
        step_id="01_cohort",
        output_files={"artifact:analysis_cohort": "analysis_cohort.parquet"},
    )
    cohort_path = run_dir / "steps" / "01_cohort" / "outputs" / "analysis_cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    _register(evidence, step_id="01_cohort", artifact=cohort_path)

    table_one = pd.DataFrame(
        {
            "schema_version": ["easyicu.table_one_result/2"] * 2,
            "variable": ["age", "charlson_max"],
            "absolute_standardized_mean_difference": [0.12, 0.08],
            "standardized_difference_status": ["computed", "computed"],
        }
    )
    _write_summary(
        run_dir,
        step_id="02_table_one",
        output_files={"table:table_one": "table_one.csv"},
    )
    table_one_path = run_dir / "steps" / "02_table_one" / "outputs" / "table_one.csv"
    table_one.to_csv(table_one_path, index=False)
    _register(evidence, step_id="02_table_one", artifact=table_one_path)

    missingness = pd.DataFrame({"concept": ["susp_inf", "death_time"]})
    _write_summary(
        run_dir,
        step_id="04_missingness",
        output_files={
            "table:missingness_measurement_audit": (
                "missingness_measurement_audit.csv"
            )
        },
        extra={
            "observation_semantics_audit": {
                "susp_inf_max": {
                    "indicator_semantics": "binary_event_presence",
                    "event_absent_n": 2,
                    "invalid_pair_n": 0,
                    "discordant_n": 0,
                },
                "death_time": {
                    "observation_semantics": "conditional_event_time",
                    "not_applicable_event_absent_n": 2,
                    "before_origin_n": 1,
                    "contradictory_event_absent_with_time_n": 0,
                },
            },
            "temporal_validity_audit": {
                "status": "flagged_requires_downstream_protocol"
            },
        },
    )
    missingness_path = (
        run_dir
        / "steps"
        / "04_missingness"
        / "outputs"
        / "missingness_measurement_audit.csv"
    )
    missingness.to_csv(missingness_path, index=False)
    _register(evidence, step_id="04_missingness", artifact=missingness_path)

    primary = pd.DataFrame({"odds_ratio": [1.6]})
    _write_summary(
        run_dir,
        step_id="05_primary",
        output_files={
            "table:adjusted_association_estimates": (
                "adjusted_association_estimates.csv"
            )
        },
        extra={"input_bindings": _typed_receipt(row_count=len(cohort))},
    )
    primary_path = (
        run_dir
        / "steps"
        / "05_primary"
        / "outputs"
        / "adjusted_association_estimates.csv"
    )
    primary.to_csv(primary_path, index=False)
    _register(evidence, step_id="05_primary", artifact=primary_path)

    sensitivity = pd.DataFrame(
        [
            {
                "analysis_id": "primary_full_cohort",
                "n_stays": 5,
                "n_deaths": 3,
                "odds_ratio": 1.60,
                "ci_low": 1.50,
                "ci_high": 1.70,
                "landmark_hours": None,
                "alive_at_landmark_required": False,
                "negative_event_times_excluded": False,
                "readmission_restriction": "all_stays",
                "age_form": "linear",
                "charlson_form": "linear",
            },
            {
                "analysis_id": "landmark_alive_at_24h",
                "n_stays": 3,
                "n_deaths": 1,
                "odds_ratio": 1.58,
                "ci_low": 1.40,
                "ci_high": 1.78,
                "landmark_hours": 24.0,
                "alive_at_landmark_required": True,
                "negative_event_times_excluded": True,
                "readmission_restriction": "all_stays",
                "age_form": "linear",
                "charlson_form": "linear",
            },
            {
                "analysis_id": "non_readmission_icu_stays",
                "n_stays": 3,
                "n_deaths": 2,
                "odds_ratio": 1.70,
                "ci_low": 1.50,
                "ci_high": 1.90,
                "landmark_hours": None,
                "alive_at_landmark_required": False,
                "negative_event_times_excluded": False,
                "readmission_restriction": "non_readmission_only",
                "age_form": "linear",
                "charlson_form": "linear",
            },
            {
                "analysis_id": "flexible_age_charlson",
                "n_stays": 5,
                "n_deaths": 3,
                "odds_ratio": 1.62,
                "ci_low": 1.52,
                "ci_high": 1.73,
                "landmark_hours": None,
                "alive_at_landmark_required": False,
                "negative_event_times_excluded": False,
                "readmission_restriction": "all_stays",
                "age_form": "restricted_cubic_spline",
                "charlson_form": "restricted_cubic_spline",
            },
        ]
    )
    _write_summary(
        run_dir,
        step_id="07_e1_sensitivity",
        output_files={
            "table:e1_scientific_sensitivity": "e1_scientific_sensitivity.csv"
        },
        extra={"input_bindings": _typed_receipt(row_count=len(cohort))},
    )
    sensitivity_path = (
        run_dir
        / "steps"
        / "07_e1_sensitivity"
        / "outputs"
        / "e1_scientific_sensitivity.csv"
    )
    sensitivity.to_csv(sensitivity_path, index=False)
    _register(
        evidence,
        step_id="07_e1_sensitivity",
        artifact=sensitivity_path,
    )

    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_dir.name, "evidence": evidence}),
        encoding="utf-8",
    )
    return run_dir


def _reason_codes(receipt: dict[str, object]) -> set[str]:
    return {
        str(issue["reason_code"])
        for issue in receipt["issues"]
        if isinstance(issue, dict)
    }


def test_e1_scientific_acceptance_accepts_complete_structured_closure(
    tmp_path: Path,
) -> None:
    run_dir = _accepted_run(tmp_path)

    receipt = evaluate_e1_scientific_acceptance(
        run_dir=run_dir,
        contract=e1_scientific_acceptance_contract(),
    )

    assert receipt["status"] == "accepted"
    assert receipt["issues"] == []


def test_e1_scientific_acceptance_rejects_missing_typed_consumption(
    tmp_path: Path,
) -> None:
    run_dir = _accepted_run(tmp_path)
    summary_path = run_dir / "steps" / "05_primary" / "outputs" / "step_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_bindings"] = []
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    receipt = evaluate_e1_scientific_acceptance(
        run_dir=run_dir,
        contract=e1_scientific_acceptance_contract(),
    )

    assert receipt["status"] == "rejected"
    assert "e1_typed_input_consumption_missing" in _reason_codes(receipt)


def test_e1_scientific_acceptance_rejects_wrong_landmark_denominator(
    tmp_path: Path,
) -> None:
    run_dir = _accepted_run(tmp_path)
    path = (
        run_dir
        / "steps"
        / "07_e1_sensitivity"
        / "outputs"
        / "e1_scientific_sensitivity.csv"
    )
    table = pd.read_csv(path)
    table.loc[table["analysis_id"] == "landmark_alive_at_24h", "n_stays"] = 5
    table.to_csv(path, index=False)

    receipt = evaluate_e1_scientific_acceptance(
        run_dir=run_dir,
        contract=e1_scientific_acceptance_contract(),
    )

    codes = _reason_codes(receipt)
    assert "e1_artifact_not_registered" in codes
    assert "e1_sensitivity_denominator_mismatch" in codes


def test_e1_scientific_acceptance_receipt_is_written_beside_run(
    tmp_path: Path,
) -> None:
    run_dir = _accepted_run(tmp_path)

    receipt, path = write_e1_scientific_acceptance_receipt(
        run_dir=run_dir,
        contract=e1_scientific_acceptance_contract(),
    )

    assert receipt["status"] == "accepted"
    assert path.parent == run_dir.parent
    assert path.name == "e1_scientific_acceptance__run_e1.json"
    assert not path.is_symlink()
    assert json.loads(path.read_text(encoding="utf-8")) == receipt
