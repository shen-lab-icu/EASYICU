"""Deterministically materialise a Planner-declared non-executable protocol.

This owner performs no analysis.  A plan step whose exact method is
``feasibility_protocol`` has already decided that the requested sensitivity or
extension is not executable from the current sealed inputs.  Sending that
step to the stochastic Coder invites the script to invent a calculation, while
letting it fail because a Markdown report was not registered turns an honest
scientific limitation into an engineering failure.

The executor therefore does one narrow job: preserve the Planner's exact
intent as a terminal, digest-bound report, list the input authorities that
were available to the step, and record explicitly that it emitted no estimate.
It never infers why data are unavailable, never chooses a replacement field,
and never upgrades the protocol into an executable analysis.
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

FEASIBILITY_PROTOCOL_ANALYSIS_KIND = "planner_declared_feasibility_protocol"

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


def feasibility_protocol_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    """Return every exact typed authority the terminal report will cite."""

    keys: list[str] = []
    for value in step.inputs or ():
        token = str(value or "").strip()
        match = _TYPED_KEY.fullmatch(token)
        if match is not None:
            if match.group(1) not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
                return ()
            keys.append(token)
    return tuple(keys)


def _declared_raw_inputs(step: AnalysisStep) -> tuple[str, ...] | None:
    """Return legacy raw names that the protocol must explicitly ignore."""

    values: list[str] = []
    for value in step.inputs or ():
        token = str(value or "").strip()
        if _TYPED_KEY.fullmatch(token) is not None:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            return None
        values.append(token)
    return tuple(values)


def feasibility_protocol_executor_owns_step(step: AnalysisStep) -> bool:
    """Whether the step is exactly a terminal non-executable protocol."""

    inputs = feasibility_protocol_consumed_input_keys(step)
    raw_inputs = _declared_raw_inputs(step)
    return bool(
        step.planned_analysis_role in {"auxiliary", "secondary", "sensitivity"}
        and _method_head(step.method) == "feasibility_protocol"
        and _report_product(step) is not None
        and raw_inputs is not None
        and len(inputs) + len(raw_inputs) == len(step.inputs or ())
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
        and not step.model_requirements
        and step.family_primary_result_requirement is None
    )


def feasibility_protocol_executor_code(step: AnalysisStep) -> str:
    """Return the sandbox entrypoint for the exact declared terminal report."""

    product = _report_product(step)
    if product is None or not feasibility_protocol_executor_owns_step(step):
        raise ValueError("step is not an owned feasibility protocol")
    raw_inputs = _declared_raw_inputs(step)
    assert raw_inputs is not None
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.feasibility_protocol_executor import (
            run_feasibility_protocol,
        )

        summary = run_feasibility_protocol(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            planned_analysis_role={step.planned_analysis_role!r},
            intent={step.intent!r},
            report_product={product!r},
            declared_inputs={list(feasibility_protocol_consumed_input_keys(step))!r},
            ignored_raw_inputs={list(raw_inputs)!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def run_feasibility_protocol(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    planned_analysis_role: str,
    intent: str,
    report_product: str,
    declared_inputs: list[str],
    ignored_raw_inputs: list[str] | None = None,
) -> dict[str, Any]:
    """Write the declared limitation without computing or inventing a result."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(report_product or "")):
        raise ValueError("report product must be one canonical token")
    if not str(intent or "").strip():
        raise ValueError("feasibility protocol intent must be non-empty")
    authorities = [
        authority.to_dict()
        for authority in verify_report_input_authorities(
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            declared_inputs=declared_inputs,
        )
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{report_product}.md"
    lines = [
        "# Non-executable feasibility protocol",
        "",
        "**Scientific status:** not executable from the current sealed inputs.",
        "",
        "## Planned sensitivity or extension",
        "",
        str(intent).strip(),
        "",
        "## Execution boundary",
        "",
        (
            "No estimate, comparison, cohort substitution, or synthetic value was "
            "produced. This terminal report preserves the reviewed Plan decision; "
            "execution requires a new input authority and a newly reviewed Plan."
        ),
        "",
        "## Bound input authorities",
        "",
    ]
    if authorities:
        lines.extend(
            f"- `{item['input_key']}` — SHA-256 `{item['sha256']}`"
            for item in authorities
        )
    else:
        lines.append("- None declared by this protocol step.")
    ignored = list(ignored_raw_inputs or [])
    if ignored:
        lines.extend(
            [
                "",
                "## Raw inputs intentionally not consumed",
                "",
                *[f"- `{name}`" for name in ignored],
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": "easyicu.feasibility_protocol_receipt/1",
        "step_id": step_id,
        "planned_analysis_role": planned_analysis_role,
        "scientific_status": "not_executable",
        "effect_estimate": None,
        "declared_intent": str(intent).strip(),
        "bound_input_authorities": authorities,
        "ignored_raw_inputs": ignored,
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
        "analysis_status": "not_executable",
        "analysis_family": "feasibility_protocol",
        "method": "deterministic_feasibility_protocol",
        "scientific_decision": "not_executable_from_current_sealed_inputs",
        "effect_estimate": None,
        "bound_input_count": len(authorities),
        "feasibility_protocol_receipt": receipt,
        "output_files": {f"report:{report_product}": report_path.name},
        "contract_files": [receipt_path.name],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "FEASIBILITY_PROTOCOL_ANALYSIS_KIND",
    "feasibility_protocol_consumed_input_keys",
    "feasibility_protocol_executor_code",
    "feasibility_protocol_executor_owns_step",
    "run_feasibility_protocol",
]
