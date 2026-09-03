"""Governed sandbox executor for an opaque counting-process Cox input."""

from __future__ import annotations

import io
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...authority.filesystem import AnchoredDirectory
from ...authority.time_varying_runtime import TimeVaryingRuntimeAuthority
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import canonical_json, sha256_bytes
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.time_varying_exposure import TIME_VARYING_INPUT_METADATA_KEY
from ...methods.time_varying_exposure_cox import fit_cluster_robust_time_varying_cox
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input


def _authority(
    value: TimeVaryingRuntimeAuthority | Mapping[str, Any],
) -> TimeVaryingRuntimeAuthority:
    if isinstance(value, TimeVaryingRuntimeAuthority):
        return value
    return TimeVaryingRuntimeAuthority.model_validate_json(canonical_json(dict(value)))


def _verified_panel(
    path: Path, frame: pd.DataFrame, authority: TimeVaryingRuntimeAuthority
):
    with AnchoredDirectory.open(path.parent) as directory:
        payload = directory.read_bytes(path.name, max_bytes=512 * 1024 * 1024)
    table = pq.read_table(io.BytesIO(payload))
    encoded = (table.schema.metadata or {}).get(
        TIME_VARYING_INPUT_METADATA_KEY.encode()
    )
    if encoded is None:
        raise ValueError("time-varying input lacks its source construction receipt")
    receipt = json.loads(encoded)
    if (
        receipt.get("schema_version") != "easyicu.time_varying_materialization/1"
        or receipt.get("specification_sha256") != authority.specification.sha256
        or receipt.get("specification")
        != authority.specification.model_dump(mode="json")
    ):
        raise ValueError(
            "time-varying input specification differs from the reviewed plan"
        )
    panel = table.to_pandas()
    expected_columns = {
        "analysis_stay_index",
        "analysis_cluster_index",
        "interval_start_hours",
        "interval_stop_hours",
        "hospital_death",
        *authority.specification.model_covariates,
    }
    if set(panel.columns) != expected_columns:
        raise ValueError("time-varying input has missing or undeclared columns")
    last = panel.groupby("analysis_stay_index", sort=False).tail(1).copy()
    last[authority.identity_column] = (
        "p"
        + last["analysis_cluster_index"].astype(str)
        + ":s"
        + last["analysis_stay_index"].astype(str)
    )
    if frame[authority.identity_column].duplicated().any() or set(
        frame[authority.identity_column]
    ) != set(last[authority.identity_column]):
        raise ValueError("time-varying input population differs from the bound cohort")
    cohort = frame.set_index(authority.identity_column).loc[
        last[authority.identity_column]
    ]
    if not np.array_equal(
        cohort[authority.outcome_column].to_numpy(), last["hospital_death"].to_numpy()
    ):
        raise ValueError("time-varying event coding differs from the bound cohort")
    for column in authority.specification.baseline_columns:
        encoding = authority.specification.baseline_categorical_encodings.get(column)
        values = (
            last[column].to_numpy()
            if encoding is None
            else np.where(
                last[encoding.output_column].eq(1),
                encoding.positive_level,
                encoding.negative_level,
            )
        )
        if not np.array_equal(cohort[column].to_numpy(), values):
            raise ValueError(
                "time-varying baseline coding differs from the bound cohort"
            )
    expected_exposure = (
        last["exposure_running_max_when_observed"]
        .where(last["exposure_unmeasured_indicator"].eq(0))
        .to_numpy()
    )
    if not np.array_equal(
        cohort[authority.exposure_column].to_numpy(), expected_exposure, equal_nan=True
    ):
        raise ValueError("time-varying exposure summary differs from the bound cohort")
    if int(receipt["analysis_stays"]) != len(last) or int(
        receipt["execution_input"]["counts"]["interval_rows"]
    ) != len(panel):
        raise ValueError("time-varying input counts differ from their receipt")
    return panel, receipt, sha256_bytes(payload)


def run_time_varying_association(
    *,
    frame: pd.DataFrame,
    trajectory_path: Path,
    authority: TimeVaryingRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    source_cohort: Path | None = None,
) -> dict[str, Any]:
    sealed = _authority(authority)
    panel, construction, input_sha = _verified_panel(trajectory_path, frame, sealed)
    fit = fit_cluster_robust_time_varying_cox(
        panel,
        id_col="analysis_stay_index",
        group_col="analysis_cluster_index",
        start_col="interval_start_hours",
        stop_col="interval_stop_hours",
        event_col="hospital_death",
        covariates=sealed.specification.model_covariates,
    )
    # Nothing is published before the fit owner has rejected invalid input,
    # separation, singularity, non-convergence, or an unavailable R runtime.
    out_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = out_dir / "time_varying_cox_estimates.csv"
    audit_path = out_dir / "time_varying_input_audit.csv"
    receipt_path = out_dir / "time_varying_runtime_receipt.json"
    fit.estimates.to_csv(estimates_path, index=False)
    counts = construction["execution_input"]["counts"]
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in counts.items()]
    ).to_csv(audit_path, index=False)
    receipt = {
        "schema_version": "easyicu.time_varying_runtime_receipt/1",
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "counting_process_input_sha256": input_sha,
        "specification_sha256": sealed.specification.sha256,
        "construction": construction,
        "fit": fit.receipt,
        "claim_ceiling": "analysis_only",
        "publication_ready": False,
        "interpretation": sealed.specification.interpretation,
        "limitations": [
            "Time-updated descriptive association, not a causal effect.",
            "Measured-value and unmeasured-state terms do not eliminate informative measurement or residual confounding.",
            "Proportional-hazards validation and publication review remain outstanding.",
        ],
    }
    receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
    return {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": "descriptive_time_updated_association",
        "scientific_runtime_receipt": receipt,
        "n_total": len(frame),
        "n_model_stays": len(frame),
        "n_events": int(panel["hospital_death"].sum()),
        "variance_estimator": "cluster_robust",
        "cluster_count": int(panel["analysis_cluster_index"].nunique()),
        "source_cohort": source_cohort.name if source_cohort else None,
        "output_files": dict(
            zip(
                sealed.plan_outputs,
                [estimates_path.name, audit_path.name, receipt_path.name],
            )
        ),
    }


def time_varying_executor_code(
    step: AnalysisStep,
    *,
    authority: TimeVaryingRuntimeAuthority,
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    cohort_key = sole_typed_cohort_input(step)
    if not cohort_key:
        raise ValueError("time-varying executor requires one typed cohort input")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    prologue = textwrap.dedent(f"""
        import os, json
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.time_varying_executor import run_time_varying_association
        frame, cohort_path = load_step_cohort_frame(typed_cohort_input={cohort_key!r})
    """).strip()
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    if receipt_code:
        prologue += "\n" + receipt_code
    prologue += (
        "\n"
        + textwrap.dedent(f"""
        summary = run_time_varying_association(
            frame=frame, trajectory_path=Path(os.environ["COHORT_TRAJECTORY_PARQUET"]),
            authority=json.loads({canonical_json(authority.model_dump(mode="json"))!r}),
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]), source_cohort=cohort_path)
    """).strip()
    )
    epilogue = (
        'summary["plausibility_audit"] = plausibility_audit\n' if receipt_code else ""
    )
    epilogue += 'Path(os.environ["STEP_OUT_DIR"], "step_summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")\nprint(json.dumps(summary, ensure_ascii=False, allow_nan=False))'
    return HostScaffoldedScript(
        prologue=prologue, body="", epilogue=epilogue
    ).assembled()


__all__ = ["run_time_varying_association", "time_varying_executor_code"]
