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

import hashlib
import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

from ...contracts.declared_product import RUNTIME_BINDABLE_TYPED_INPUT_KINDS
from ...schema import AnalysisStep

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
        if match is None or match.group(1) not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
            return ()
        keys.append(token)
    return tuple(keys)


def feasibility_protocol_executor_owns_step(step: AnalysisStep) -> bool:
    """Whether the step is exactly a terminal non-executable protocol."""

    inputs = feasibility_protocol_consumed_input_keys(step)
    return bool(
        step.planned_analysis_role in {"auxiliary", "secondary", "sensitivity"}
        and _method_head(step.method) == "feasibility_protocol"
        and _report_product(step) is not None
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


def feasibility_protocol_executor_code(step: AnalysisStep) -> str:
    """Return the sandbox entrypoint for the exact declared terminal report."""

    product = _report_product(step)
    if product is None or not feasibility_protocol_executor_owns_step(step):
        raise ValueError("step is not an owned feasibility protocol")
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
            declared_inputs={list(step.inputs or ())!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> dict[str, Any]:
    """Write the declared limitation without computing or inventing a result."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(report_product or "")):
        raise ValueError("report product must be one canonical token")
    if not str(intent or "").strip():
        raise ValueError("feasibility protocol intent must be non-empty")
    payload = (
        dict(resolved_inputs)
        if isinstance(resolved_inputs, Mapping)
        else json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    )
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    bindings = payload.get("inputs")
    if not isinstance(bindings, dict):
        raise ValueError("resolved-input manifest carries no binding map")

    authorities: list[dict[str, Any]] = []
    for key in declared_inputs:
        match = _TYPED_KEY.fullmatch(str(key or "").strip())
        binding = bindings.get(key)
        if (
            match is None
            or match.group(1) not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS
            or not isinstance(binding, dict)
        ):
            raise ValueError("feasibility protocol input authority is incomplete")
        digest = str(binding.get("sha256") or "")
        relative_path = str(binding.get("relative_path") or "")
        relative_input_path = Path(relative_path)
        resolved_run_dir = Path(run_dir).resolve()
        bound_path = (resolved_run_dir / relative_input_path).resolve()
        try:
            bound_path.relative_to(resolved_run_dir)
        except ValueError as error:
            raise ValueError(
                "feasibility protocol input escapes EASYICU_RUN_DIR"
            ) from error
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not relative_path
            or relative_input_path.is_absolute()
            or not bound_path.is_file()
            or _sha256(bound_path) != digest
            or str(binding.get("declared_kind") or "") != match.group(1)
            or str(binding.get("identity_row", {}).get("input_key") or "") != key
        ):
            raise ValueError("feasibility protocol input lacks a digest binding")
        authorities.append(
            {
                "input_key": key,
                "evidence_id": str(binding.get("evidence_id") or ""),
                "sha256": digest,
                "produced_by_step": binding.get("produced_by_step"),
            }
        )

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
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": "easyicu.feasibility_protocol_receipt/1",
        "step_id": step_id,
        "planned_analysis_role": planned_analysis_role,
        "scientific_status": "not_executable",
        "effect_estimate": None,
        "declared_intent": str(intent).strip(),
        "bound_input_authorities": authorities,
        "report_sha256": _sha256(report_path),
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
