#!/usr/bin/env python3
"""Replay one signed landmark model with the current reporting contract.

This is a provider-free development migration. It verifies the frozen cohort
binding, executes the benchmark protocol's current typed authority, and checks
that legacy relative-association products are numerically unchanged before
accepting the new absolute-risk and population-accounting products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    load_default_case_protocol,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.landmark_spline_executor import (
    run_landmark_spline_association,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_binding(run_dir: Path, step_id: str) -> tuple[pd.DataFrame, Path]:
    manifest_path = run_dir / "resolved_inputs" / f"{step_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or len(inputs) != 1:
        raise ValueError("landmark replay requires exactly one resolved cohort input")
    binding = next(iter(inputs.values()))
    relative_path = binding.get("relative_path")
    expected_sha256 = binding.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise ValueError("resolved cohort binding is incomplete")
    cohort_path = (run_dir / relative_path).resolve()
    cohort_path.relative_to(run_dir.resolve())
    if _sha256(cohort_path) != expected_sha256:
        raise ValueError("resolved cohort digest mismatch")
    frame = pd.read_parquet(cohort_path)
    contract = binding.get("product_contract") or {}
    if len(frame) != contract.get("row_count"):
        raise ValueError("resolved cohort row count mismatch")
    if list(frame.columns) != contract.get("columns"):
        raise ValueError("resolved cohort columns mismatch")
    return frame, cohort_path


def _legacy_table(run_dir: Path, step_id: str, basename: str) -> Path:
    rows = json.loads((run_dir / "evidence/evidence_index.json").read_text())
    matches = [
        row
        for row in rows
        if row.get("kind") == "table"
        and row.get("produced_by_step") == step_id
        and Path(str(row.get("relative_path") or "")).name.endswith(
            f"__{basename}"
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one legacy {step_id}/{basename}")
    path = run_dir / matches[0]["relative_path"]
    if _sha256(path) != matches[0].get("sha256"):
        raise ValueError(f"legacy evidence digest mismatch: {path}")
    return path


def _assert_numeric_identity(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    shared = [column for column in old.columns if column in new.columns]
    if not shared or len(old) != len(new):
        raise ValueError(f"legacy product shape changed: {old_path.name}")
    for column in shared:
        old_numeric = pd.to_numeric(old[column], errors="coerce")
        new_numeric = pd.to_numeric(new[column], errors="coerce")
        if old_numeric.notna().any() or new_numeric.notna().any():
            if not np.allclose(
                old_numeric.to_numpy(dtype=float),
                new_numeric.to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"legacy numeric result changed: {old_path.name}/{column}"
                )
        elif old[column].fillna("").astype(str).tolist() != new[column].fillna("").astype(str).tolist():
            raise ValueError(f"legacy labels changed: {old_path.name}/{column}")
    return {
        "legacy_path": str(old_path),
        "legacy_sha256": _sha256(old_path),
        "replay_path": str(new_path),
        "replay_sha256": _sha256(new_path),
        "shared_columns": shared,
        "numeric_identity": True,
    }


def replay(*, task_id: str, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    protocol = load_default_case_protocol(task_id)
    projection = build_runtime_scientific_projection(protocol)
    authority = load_current_case_scientific_runtime_authority(
        projection.deterministic_execution_contract or {}
    )
    if not isinstance(authority, LandmarkSplineRuntimeAuthority):
        raise TypeError("task does not compile to a landmark-spline authority")
    plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    primary_steps = [
        step
        for step in plan.get("steps") or []
        if step.get("method") == authority.plan_method
    ]
    if len(primary_steps) != 1:
        raise ValueError("frozen plan does not contain one signed landmark step")
    step_id = str(primary_steps[0]["step_id"])
    frame, cohort_path = _resolved_binding(run_dir, step_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_landmark_spline_association(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=output_dir,
        source_cohort=cohort_path,
    )
    comparisons = []
    legacy_products = (
        authority.curve_product,
        authority.downstream_parent_product,
        authority.linear_sensitivity_product,
        authority.exposure_definition_sensitivity_product,
    )
    for product in legacy_products:
        if product is None:
            continue
        basename = Path(summary["output_files"][product]).name
        comparisons.append(
            _assert_numeric_identity(
                _legacy_table(run_dir, step_id, basename), output_dir / basename
            )
        )
    outputs = {
        product: {
            "path": filename,
            "sha256": _sha256(output_dir / filename),
        }
        for product, filename in summary["output_files"].items()
    }
    manifest = {
        "schema_version": "easyicu.landmark_reporting_contract_replay/1",
        "task_id": task_id,
        "source_run": str(run_dir),
        "source_cohort": str(cohort_path),
        "source_cohort_sha256": _sha256(cohort_path),
        "source_step_id": step_id,
        "protocol_content_sha256": authority.protocol_content_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "runtime_projection_sha256": projection.runtime_projection_sha256,
        "provider_calls": 0,
        "scientific_recomputation": True,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "legacy_result_identity": comparisons,
        "outputs": outputs,
        "summary": summary,
    }
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay(
        task_id=args.task_id,
        run_dir=args.run_dir.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
