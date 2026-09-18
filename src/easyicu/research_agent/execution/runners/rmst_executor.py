"""Governed sandbox executor for a prespecified two-group RMST contrast."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.rmst_runtime import RmstRuntimeAuthority
from ...canonical_json import canonical_json
from ...contracts.host_scaffold import HostScaffoldedScript
from ...methods.rmst import rmst, rmst_difference
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input


RMST_CONTRAST_ANALYSIS_KIND = "signed_rmst_contrast"


def _authority(
    value: RmstRuntimeAuthority | Mapping[str, Any],
) -> RmstRuntimeAuthority:
    if isinstance(value, RmstRuntimeAuthority):
        return value
    return RmstRuntimeAuthority.model_validate_json(canonical_json(dict(value)))


def _numeric(series: pd.Series, *, column: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if bool(converted.isna().any()) or not bool(
        np.isfinite(converted.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"RMST column '{column}' has missing or non-finite values")
    return converted


def run_rmst_contrast(
    *,
    frame: pd.DataFrame,
    authority: RmstRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    source_cohort: Path | None = None,
) -> dict[str, Any]:
    sealed = _authority(authority)
    spec = sealed.specification
    required = {
        spec.time_column,
        spec.event_column,
        spec.group_column,
        sealed.identity_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"RMST cohort is missing declared columns: {missing}")

    reference, comparator = spec.group_levels
    observed_levels = sorted(
        {str(value) for value in frame[spec.group_column].dropna().tolist()}
    )
    if observed_levels != sorted(spec.group_levels):
        raise ValueError(
            "RMST group levels differ from the reviewed specification: "
            f"observed={observed_levels}"
        )

    times = _numeric(frame[spec.time_column], column=spec.time_column)
    if bool((times < 0).any()):
        raise ValueError("RMST follow-up times must be non-negative")
    if float(times.max()) <= float(spec.tau):
        raise ValueError("RMST horizon must lie inside observed follow-up")
    event_values = _numeric(frame[spec.event_column], column=spec.event_column)
    unexpected_event_values = sorted(
        set(event_values.astype(float).unique().tolist())
        - {0.0, float(spec.event_code)}
    )
    if unexpected_event_values:
        raise ValueError(
            "RMST event column contains values outside censor code 0 and the "
            f"reviewed event code {spec.event_code:g}: {unexpected_event_values}"
        )
    events = event_values.eq(float(spec.event_code))

    groups = frame[spec.group_column].astype(str)
    identity_values = frame[sealed.identity_column].astype("string")
    if bool(identity_values.isna().any()) or bool(
        identity_values.str.strip().eq("").any()
    ):
        raise ValueError("RMST identity column has missing or blank values")
    identities = identity_values.astype(str)
    per_group: dict[str, dict[str, Any]] = {}
    for label in (reference, comparator):
        mask = groups.eq(label)
        n = int(mask.sum())
        event_n = int(events[mask].sum())
        if n < 2:
            raise ValueError(f"RMST group '{label}' has fewer than two rows")
        if event_n < 1:
            raise ValueError(f"RMST group '{label}' records no observed events")
        result = rmst(times[mask], events[mask], float(spec.tau))
        per_group[label] = {
            "n": n,
            "events": event_n,
            "rmst": result.rmst,
            "se": result.se,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }

    kernel_diff = rmst_difference(times, events, groups, float(spec.tau))
    diff_value = float(kernel_diff["diff"])
    ci_low, ci_high = kernel_diff["ci"]
    if str(kernel_diff["group_a"]) != reference:
        diff_value = -diff_value
        ci_low, ci_high = -ci_high, -ci_low

    duplicates = int(identities.duplicated().sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "rmst_summary.csv"
    receipt_path = out_dir / "rmst_runtime_receipt.json"
    rows = [
        {
            "row_type": "group",
            "group": label,
            "n": values["n"],
            "events": values["events"],
            "tau": float(spec.tau),
            "time_unit": spec.time_unit,
            "rmst": values["rmst"],
            "se": values["se"],
            "ci_low": values["ci_low"],
            "ci_high": values["ci_high"],
        }
        for label, values in (
            (reference, per_group[reference]),
            (comparator, per_group[comparator]),
        )
    ]
    rows.append(
        {
            "row_type": "difference",
            "group": f"{reference} - {comparator}",
            "n": int(len(times)),
            "events": int(events.sum()),
            "tau": float(spec.tau),
            "time_unit": spec.time_unit,
            "rmst": diff_value,
            "se": float(kernel_diff["se_diff"]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    )
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    input_sha = hashlib.sha256(
        json.dumps(
            {
                "time": times.tolist(),
                "event": event_values.tolist(),
                "group": groups.tolist(),
                "identity": identities.tolist(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema_version": "easyicu.rmst_runtime_receipt/1",
        "runtime_projection_sha256": runtime_projection_sha256,
        "specification_sha256": hashlib.sha256(
            canonical_json(spec.model_dump(mode="json")).encode("utf-8")
        ).hexdigest(),
        "sensitivity_spec_id": sealed.sensitivity_spec_id,
        "analysis_input_sha256": input_sha,
        "source_cohort": source_cohort.name if source_cohort else None,
        "rows": int(len(frame)),
        "duplicate_identity_rows": duplicates,
        "estimands": {
            "reference": reference,
            "comparator": comparator,
            "reference_rmst": per_group[reference]["rmst"],
            "comparator_rmst": per_group[comparator]["rmst"],
            "difference": diff_value,
            "se_difference": float(kernel_diff["se_diff"]),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(kernel_diff["p_value"]),
            "tau": float(spec.tau),
            "time_unit": spec.time_unit,
        },
        "claim_ceiling": "analysis_only",
        "publication_ready": False,
        "interpretation": "descriptive_rmst_contrast_not_causal",
        "limitations": [
            "Restricted mean survival time contrast is descriptive, not causal.",
            "Uncertainty assumes independent observations; repeated stays from "
            "one patient are not clustered in this estimate.",
            "Proportional-hazards diagnostics, measurement error and publication "
            "review remain outstanding.",
        ],
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "analysis_family": "survival",
        "interpretation_class": "descriptive_rmst_contrast",
        "rmst_runtime_receipt": receipt,
        "n_total": int(len(frame)),
        "n_events": int(events.sum()),
        "source_cohort": source_cohort.name if source_cohort else None,
        "output_files": {
            "table:rmst_summary": summary_path.name,
            "log:rmst_runtime_receipt": receipt_path.name,
        },
    }


def rmst_executor_code(
    step: AnalysisStep,
    *,
    authority: RmstRuntimeAuthority,
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    cohort_key = sole_typed_cohort_input(step)
    if not cohort_key:
        raise ValueError("RMST executor requires one typed cohort input")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    prologue = textwrap.dedent(f"""
        import os, json
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.rmst_executor import run_rmst_contrast
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
        summary = run_rmst_contrast(
            frame=frame,
            authority=json.loads({canonical_json(authority.model_dump(mode="json"))!r}),
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]), source_cohort=cohort_path)
    """).strip()
    )
    epilogue = (
        'summary["plausibility_audit"] = plausibility_audit\n' if receipt_code else ""
    )
    epilogue += (
        'Path(os.environ["STEP_OUT_DIR"], "step_summary.json").write_text('
        'json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")\n'
        "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))"
    )
    return HostScaffoldedScript(
        prologue=prologue, body="", epilogue=epilogue
    ).assembled()


__all__ = [
    "RMST_CONTRAST_ANALYSIS_KIND",
    "rmst_executor_code",
    "run_rmst_contrast",
]
