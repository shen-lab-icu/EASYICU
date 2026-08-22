"""Deterministically publish the sealed run cohort as a typed root product."""

from __future__ import annotations

import textwrap

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code

HOST_BOUND_COHORT_ANALYSIS_KIND = "host_bound_analysis_cohort"
HOST_BOUND_COHORT_METHOD = "host_materialized_locked_cohort"
HOST_BOUND_COHORT_PRODUCT = "table:analysis_cohort"


def host_bound_cohort_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only the exact interpretation-free root-product declaration."""

    return bool(
        step.method == HOST_BOUND_COHORT_METHOD
        and step.planned_analysis_role == "auxiliary"
        and not step.inputs
        and list(step.expected_outputs) == [HOST_BOUND_COHORT_PRODUCT]
        and not step.model_requirements
        and step.table_one_spec is None
        and step.family_primary_result_requirement is None
    )


def host_bound_cohort_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Copy the already selected cohort and bind its exact bytes to the plan."""

    if not host_bound_cohort_executor_owns_step(step):
        raise ValueError("step is not owned by the host-bound cohort executor")
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
    receipt_assignment = (
        'summary["plausibility_audit"] = plausibility_audit'
        if receipt_code
        else ""
    )
    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        import shutil
        from pathlib import Path

        import pandas as pd

        source = Path(os.environ["COHORT_PARQUET"]).resolve()
        out_dir = Path(os.environ["STEP_OUT_DIR"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / "analysis_cohort.parquet"
        frame = pd.read_parquet(source)
        shutil.copyfile(source, destination)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        output_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        if source_sha256 != output_sha256:
            raise ValueError("host-bound cohort bytes changed during publication")

        {receipt_code}

        summary = {{
            "status": "ok",
            "deterministic_standard_analysis": {HOST_BOUND_COHORT_ANALYSIS_KIND!r},
            "n_source": int(len(frame)),
            "n_analysis_cohort": int(len(frame)),
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "cohort_binding": {{
                "source": "COHORT_PARQUET",
                "selection_mode": "sealed_run_cohort",
                "n_full": int(len(frame)),
                "n_complete_case": int(len(frame)),
                "n_dropped": 0,
                "final_cohort_n": int(len(frame)),
                "row_order_preserved": True,
            }},
            "output_files": {{
                {HOST_BOUND_COHORT_PRODUCT!r}: destination.name,
            }},
        }}
        {receipt_assignment}
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        """
    ).strip()


__all__ = [
    "HOST_BOUND_COHORT_ANALYSIS_KIND",
    "HOST_BOUND_COHORT_METHOD",
    "HOST_BOUND_COHORT_PRODUCT",
    "host_bound_cohort_executor_code",
    "host_bound_cohort_executor_owns_step",
]
