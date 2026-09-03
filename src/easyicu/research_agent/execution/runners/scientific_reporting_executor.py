"""Deterministically index existing result products for a report step.

The executor is intentionally non-analytic: it verifies and lists the exact
typed evidence already produced by upstream analysis owners.  It does not
recompute estimates, choose claims, or replace the manuscript Writer.  Its
purpose is to keep a completed analysis distinct from a genuinely
non-executable ``feasibility_protocol``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

from ...contracts.declared_product import RUNTIME_BINDABLE_TYPED_INPUT_KINDS
from ...schema import AnalysisStep
from .report_input_authority import verify_report_input_authorities
from .typed_input_binding import sha256_file


SCIENTIFIC_REPORTING_ANALYSIS_KIND = "evidence_bound_scientific_reporting"
_TYPED_KEY = re.compile(r"([a-z][a-z0-9_]*):([a-z][a-z0-9_]*)")


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _report_product(step: AnalysisStep) -> str | None:
    if len(step.expected_outputs or ()) != 1:
        return None
    match = _TYPED_KEY.fullmatch(str(step.expected_outputs[0] or "").strip())
    if match is None or match.group(1) != "report":
        return None
    return match.group(2)


def scientific_reporting_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    keys: list[str] = []
    for value in step.inputs or ():
        token = str(value or "").strip()
        match = _TYPED_KEY.fullmatch(token)
        if match is None or match.group(1) not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
            return ()
        keys.append(token)
    return tuple(keys)


def scientific_reporting_executor_owns_step(step: AnalysisStep) -> bool:
    inputs = scientific_reporting_consumed_input_keys(step)
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "scientific_reporting"
        and _report_product(step) is not None
        and inputs
        and len(inputs) == len(step.inputs or ())
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
        and not step.model_requirements
        and step.family_primary_result_requirement is None
    )


def scientific_reporting_executor_code(step: AnalysisStep) -> str:
    product = _report_product(step)
    if product is None or not scientific_reporting_executor_owns_step(step):
        raise ValueError("step is not an owned evidence-bound scientific report")
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.scientific_reporting_executor import (
            run_scientific_reporting,
        )

        summary = run_scientific_reporting(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            planned_analysis_role={step.planned_analysis_role!r},
            intent={step.intent!r},
            report_product={product!r},
            declared_inputs={list(scientific_reporting_consumed_input_keys(step))!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def run_scientific_reporting(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    planned_analysis_role: str,
    intent: str,
    report_product: str,
    declared_inputs: list[str],
) -> dict[str, Any]:
    """Write a digest-bound result index without adding scientific claims."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(report_product or "")):
        raise ValueError("report product must be one canonical token")
    if not str(intent or "").strip():
        raise ValueError("scientific report intent must be non-empty")
    authorities = [
        authority.to_dict()
        for authority in verify_report_input_authorities(
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            declared_inputs=declared_inputs,
        )
    ]
    if not authorities:
        raise ValueError("scientific report requires at least one bound result")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{report_product}.md"
    lines = [
        "# Evidence-bound scientific report index",
        "",
        "**Scientific status:** registered analysis outputs are available.",
        "",
        "## Reporting intent",
        "",
        str(intent).strip(),
        "",
        "## Bound result authorities",
        "",
        "| Input product | Evidence ID | Producer step | SHA-256 |",
        "|---|---|---|---|",
    ]
    lines.extend(
        "| `{input_key}` | `{evidence_id}` | `{producer}` | `{sha256}` |".format(
            input_key=item["input_key"],
            evidence_id=item["evidence_id"],
            producer=item["produced_by_step"] or "run_input",
            sha256=item["sha256"],
        )
        for item in authorities
    )
    lines.extend(
        [
            "",
            "## Authority boundary",
            "",
            (
                "This index cites existing digest-bound outputs. It does not "
                "recompute an estimate, add a scientific claim, or change the "
                "reviewed analysis plan."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": "easyicu.scientific_reporting_receipt/1",
        "step_id": step_id,
        "planned_analysis_role": planned_analysis_role,
        "scientific_status": "evidence_bound_results_available",
        "effect_estimate": None,
        "declared_intent": str(intent).strip(),
        "bound_input_authorities": authorities,
        "report_sha256": sha256_file(report_path),
    }
    receipt_path = out_dir / f"{report_product}.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "results_available",
        "analysis_family": "scientific_reporting",
        "method": "deterministic_evidence_bound_scientific_reporting",
        "scientific_decision": "index_existing_registered_results",
        "effect_estimate": None,
        "bound_input_count": len(authorities),
        "scientific_reporting_receipt": receipt,
        "output_files": {f"report:{report_product}": report_path.name},
        "contract_files": [receipt_path.name],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "SCIENTIFIC_REPORTING_ANALYSIS_KIND",
    "run_scientific_reporting",
    "scientific_reporting_consumed_input_keys",
    "scientific_reporting_executor_code",
    "scientific_reporting_executor_owns_step",
]
