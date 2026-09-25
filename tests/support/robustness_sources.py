"""A registered primary-model step, as the robustness replay reads it.

Shared by the robustness tests; test modules may not import one another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_structured_source_authority(
    run_dir: Path,
    *,
    coefficient_filename: str = "coefficients.csv",
    diagnostic_reference: object | None = None,
    include_primary_model_id: bool = True,
    include_headline: bool = True,
    model_n: int = 100,
    analysis_covariates: list[str] | None = None,
):
    step_id = "01_primary_model"
    step_dir = run_dir / "steps" / step_id
    outputs_dir = step_dir / "outputs"
    evidence_dir = run_dir / "evidence"
    outputs_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    script_path = step_dir / "analysis.py"
    coefficient_path = outputs_dir / coefficient_filename
    script_path.write_text("print('registered primary model')\n", encoding="utf-8")
    coefficient_path.write_text(
        "model_id,term,term_role,source_variable,odds_ratio,ci_low,ci_high,std_error\n"
        "primary,exposure,exposure,exposure,1.4,1.1,1.8,0.1\n",
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "primary_exposure": "exposure",
        "primary_model_n": model_n,
        "model_contracts": [
            {
                "model_id": "primary",
                "analysis_role": "primary",
                "exposure_role": "primary",
                "exposure_source": "exposure",
                "exposure_expression": "exposure",
                "analysis_set": "source_aware",
                "n": model_n,
                "event_n": min(20, model_n),
                "fit_status": "fitted",
                "converged": True,
                "fit_method": "registered_test_model",
            }
        ],
    }
    if analysis_covariates is not None:
        summary["analysis_definition"] = {
            "exposure": "exposure",
            "outcome": "outcome",
            "covariates": list(analysis_covariates),
        }
    if include_headline:
        summary.update(
            {
                "primary_or": 1.4,
                "primary_ci_low": 1.1,
                "primary_ci_high": 1.8,
            }
        )
    if include_primary_model_id:
        summary["primary_model_id"] = "primary"
    if diagnostic_reference is not None:
        summary["diagnostic_companions"] = {
            "coefficients": diagnostic_reference,
        }
    else:
        summary["diagnostic_companions"] = {
            "coefficients": coefficient_filename,
        }
    # A summary that names its companion badly is refused outright, so this
    # sibling never rescues the malformed case above -- it is here so that the
    # ordinary fixture declares the file the way a producer does, instead of
    # relying on a default filename no run has ever written.
    summary["coefficient_table"] = coefficient_filename
    summary_path = outputs_dir / "step_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    coefficient_sha = hashlib.sha256(coefficient_path.read_bytes()).hexdigest()
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    code_copy = evidence_dir / "code_primary__analysis.py"
    coefficient_copy = evidence_dir / f"table_coefficients__{coefficient_filename}"
    summary_copy = evidence_dir / "stat_primary__step_summary.json"
    code_copy.write_bytes(script_path.read_bytes())
    coefficient_copy.write_bytes(coefficient_path.read_bytes())
    summary_copy.write_bytes(summary_path.read_bytes())
    evidence = [
        {
            "evidence_id": "code_primary",
            "kind": "code",
            "relative_path": str(code_copy.relative_to(run_dir)),
            "sha256": script_sha,
            "produced_by_step": step_id,
        },
        {
            "evidence_id": "table_coefficients",
            "kind": "table",
            "relative_path": str(coefficient_copy.relative_to(run_dir)),
            "sha256": coefficient_sha,
            "produced_by_step": step_id,
            "script_evidence_id": "code_primary",
        },
        {
            "evidence_id": "stat_primary",
            "kind": "statistic",
            "relative_path": str(summary_copy.relative_to(run_dir)),
            "sha256": summary_sha,
            "produced_by_step": step_id,
            "script_evidence_id": "code_primary",
        },
    ]
    record = {
        "step_id": step_id,
        "status": "ok",
        "planned_analysis_role": "primary",
        "analysis_request": {
            "step": {
                "step_id": step_id,
                "planned_analysis_role": "primary",
            }
        },
        "executed_code_sha256": script_sha,
        "evidence_ids": ["code_primary", "table_coefficients", "stat_primary"],
        "step_summary_evidence_id": "stat_primary",
        "step_summary": summary,
    }
    return record, evidence, script_path
