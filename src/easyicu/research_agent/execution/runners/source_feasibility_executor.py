"""Deterministic current-source feasibility result for a fail-closed contrast."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...schema import AnalysisPlan, AnalysisStep

SOURCE_FEASIBILITY_ANALYSIS_KIND = "signed_source_feasibility_fail_closed"


def source_feasibility_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: SourceFeasibilityRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None:
        return False
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, SourceFeasibilityRuntimeAuthority):
        return False
    return sealed.governed_step(plan) == step


def source_feasibility_executor_code(
    *,
    authority: SourceFeasibilityRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
) -> str:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, SourceFeasibilityRuntimeAuthority):
        raise TypeError("source feasibility executor received wrong authority kind")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return (
        "import json, os\n"
        "from pathlib import Path\n"
        "from easyicu.research_agent.execution.runners.source_feasibility_executor "
        "import run_source_feasibility_fail_closed\n"
        f"authority = json.loads({json.dumps(authority_json)})\n"
        "summary = run_source_feasibility_fail_closed("
        "authority=authority, "
        f"runtime_projection_sha256={runtime_projection_sha256!r}, "
        "out_dir=Path(os.environ['STEP_OUT_DIR']))\n"
        "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))\n"
    )


def run_source_feasibility_fail_closed(
    *,
    authority: SourceFeasibilityRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Emit the signed non-identifiability result and no effect estimate."""

    import pandas as pd

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, SourceFeasibilityRuntimeAuthority):
        raise TypeError("source feasibility executor received wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "source": sealed.source,
        "window_start_hours": sealed.audited_window_hours[0],
        "window_end_hours": sealed.audited_window_hours[1],
        "verified_non_use_available": sealed.verified_non_use_available,
        "binary_control_arm_authorized": sealed.binary_control_arm_authorized,
        "causal_contrast_authorized": sealed.causal_contrast_authorized,
        "decision": sealed.decision,
        "reason_code": sealed.reason_code,
        "effect_estimate": None,
    }
    table_path = out_dir / "h2_source_feasibility.csv"
    pd.DataFrame([row]).to_csv(table_path, index=False)
    receipt = {
        "schema_version": "easyicu.source_feasibility_runtime_receipt/1",
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        **row,
        "forbidden_actions_enforced": list(sealed.forbidden_plan_tokens),
        "future_design_authorized": sealed.future_design_authorized,
    }
    receipt_path = out_dir / "h2_scientific_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "analysis_family": "causal_feasibility",
        "scientific_decision": "blocked_by_source_authority",
        "reason_code": sealed.reason_code,
        "causal_contrast_authorized": False,
        "effect_estimate": None,
        "scientific_runtime_receipt": receipt,
        "output_files": {
            "table:h2_source_feasibility": table_path.name,
            "log:h2_scientific_runtime_receipt": receipt_path.name,
        },
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "SOURCE_FEASIBILITY_ANALYSIS_KIND",
    "run_source_feasibility_fail_closed",
    "source_feasibility_executor_code",
    "source_feasibility_executor_owns_step",
]
