"""Deterministically publish the sealed run cohort as a typed root product."""

from __future__ import annotations

import json
import textwrap

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.primary_cohort import (
    HOST_BOUND_COHORT_METHOD,
    HOST_BOUND_COHORT_PRODUCT,
    STUDY_POPULATION_PRODUCTS,
)
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code

HOST_BOUND_COHORT_ANALYSIS_KIND = "host_bound_analysis_cohort"
_PUBLISHED_PRODUCTS = (HOST_BOUND_COHORT_PRODUCT, *STUDY_POPULATION_PRODUCTS)


def _published_filename(product: str) -> str:
    return f"{product.partition(':')[2]}.parquet"


def host_bound_cohort_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only the exact interpretation-free root-product declaration.

    The root publishes one product: the legacy primary cohort, or the study
    population a narrower primary analysis cannot supply.
    """

    outputs = list(step.expected_outputs)
    return bool(
        step.method == HOST_BOUND_COHORT_METHOD
        and step.planned_analysis_role == "auxiliary"
        and not step.inputs
        and len(outputs) == 1
        and outputs[0] in _PUBLISHED_PRODUCTS
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
    product = str(step.expected_outputs[0])
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
        destination = out_dir / {json.dumps(_published_filename(product))}
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
            "analysis_cohort_n": int(len(frame)),
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
                {product!r}: destination.name,
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


def care_unit_landmark_cohort_executor_code(step, *, plan, authority, plausibility_scope=None):
    from ...authority.case_duration_landmark import CaseDurationLandmarkAuthority
    from ...canonical_json import canonical_sha256

    if not isinstance(authority, CaseDurationLandmarkAuthority):
        raise ValueError("signed case-duration authority is required")
    authority.validate_plan(plan)
    if step.method != "signed_care_unit_landmark_cohort" or step not in plan.steps:
        raise ValueError("landmark cohort step differs from the signed plan")
    copy_step = step.model_copy(update={"method": HOST_BOUND_COHORT_METHOD, "inputs": [],
                                        "planned_analysis_role": "auxiliary",
                                        "expected_outputs": [HOST_BOUND_COHORT_PRODUCT]})
    code = host_bound_cohort_executor_code(copy_step, plausibility_scope=plausibility_scope)
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    contract_sha = authority.execution_contract_sha256
    verify = (
        'from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority\n'
        'binding = load_verified_materialized_cohort_authority(source)\n'
        'if binding is None:\n'
        '    raise ValueError("landmark cohort has no sealed authority")\n'
        'proof = binding.authority.semantic_provenance\n'
        f'if proof.get("plan_sha256") != {json.dumps(plan_sha)} or '
        f'proof.get("landmark_contract_sha256") != {json.dumps(contract_sha)}:\n'
        '    raise ValueError("landmark cohort differs from signed plan")\n'
    )
    marker = 'source = Path(os.environ["COHORT_PARQUET"]).resolve()\n'
    if code.count(marker) != 1:
        raise ValueError("host cohort publication template changed")
    code = code.replace(repr(HOST_BOUND_COHORT_PRODUCT), repr("artifact:analysis_cohort"))
    return code.replace(marker, marker + verify, 1)
