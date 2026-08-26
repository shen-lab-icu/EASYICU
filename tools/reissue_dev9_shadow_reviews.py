#!/usr/bin/env python3
"""Issue exact-run Dev9 shadow reviews from a post-run assessment packet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.figure2_canonical9.comparator_shadow_review import (  # noqa: E402
    AnchorSourcePack,
    REVIEW_DIMENSIONS,
    ReviewDimension,
    RunBoundShadowReview,
    canonical_json_sha256,
    load_shadow_review_protocol,
    protocol_content_sha256,
    validate_run_bound_review,
)


_EVIDENCE_PATHS = {
    "study_population": ("research_context.json", "run_status.json"),
    "time_zero_and_windows": ("research_context.json", "analysis_plan.json"),
    "variable_operationalization": ("research_context.json", "analysis_plan.json"),
    "missingness_and_censoring": (
        "scientific_maturity_audit.json",
        "display_suite_audit.json",
    ),
    "primary_model_and_sensitivities": (
        "scientific_maturity_audit.json",
        "run_status.json",
    ),
    "table_and_figure_completeness": ("display_suite_audit.json",),
    "conclusion_boundaries": (
        "scientific_maturity_audit.json",
        "manuscript_scaffold_bound.md",
    ),
}

_OWNERS = {
    "study_population": "data_foundation",
    "time_zero_and_windows": "scientific_design",
    "variable_operationalization": "clinical_definition_registry",
    "missingness_and_censoring": "measurement_and_censoring_audit",
    "primary_model_and_sensitivities": "method_adapter",
    "table_and_figure_completeness": "figure_contract",
    "conclusion_boundaries": "evidence_bound_writer",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _find_run(run_root: Path, task_id: str) -> Path:
    candidates = tuple(run_root.glob(f"*/{task_id}/aware/run_*/run_status.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one exact run for {task_id}, found {len(candidates)}"
        )
    return candidates[0].parent


def _next_action(state: str, owner: str) -> str:
    if state in {"meets_anchor", "stronger_than_anchor"}:
        return "Retain the exact run contract and evidence lineage."
    if state in {"fail_closed_appropriate", "not_applicable"}:
        return "Retain the fail-closed boundary; do not synthesize a result."
    return f"Close this general contract through the {owner} owner before paper review."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor-pack", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_shadow_review_protocol(args.protocol)
    anchor_pack = AnchorSourcePack.model_validate_json(
        args.anchor_pack.read_text(encoding="utf-8")
    )
    assessments = _load_json(args.assessments)
    task_assessments = assessments.get("tasks")
    if not isinstance(task_assessments, dict):
        raise ValueError("assessment packet has no task mapping")
    supplement_review = assessments.get("supplement_review")
    if not isinstance(supplement_review, dict):
        raise ValueError("assessment packet has no supplement-review receipt")
    supplement_task_findings = supplement_review.get("task_findings")
    if not isinstance(supplement_task_findings, dict):
        raise ValueError("supplement-review receipt has no task findings")
    expected_task_ids = tuple(task.task_id for task in protocol.tasks)
    if set(task_assessments) != set(expected_task_ids):
        raise ValueError("assessment task ids differ from the comparator protocol")
    if set(supplement_task_findings) != set(expected_task_ids):
        raise ValueError("supplement-review task ids differ from the protocol")
    supplement_review_sha256 = canonical_json_sha256(supplement_review)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews: list[RunBoundShadowReview] = []
    state_counts: Counter[str] = Counter()
    for task in protocol.tasks:
        run_path = _find_run(args.run_root, task.task_id)
        status = _load_json(run_path / "run_status.json")
        code_version = status.get("code_version")
        if not isinstance(code_version, dict) or not code_version.get("git_sha"):
            raise ValueError(f"run has no exact code version: {run_path}")
        rows = task_assessments[task.task_id]
        if not isinstance(rows, dict) or tuple(rows) != REVIEW_DIMENSIONS:
            raise ValueError(f"assessment dimensions differ for {task.task_id}")
        anchor_ids = tuple(anchor.citation_id for anchor in task.anchors)
        dimensions = []
        for dimension in REVIEW_DIMENSIONS:
            assessment = rows[dimension]
            if not isinstance(assessment, list) or len(assessment) != 2:
                raise ValueError(
                    f"invalid {task.task_id}/{dimension} assessment tuple"
                )
            state, rationale = map(str, assessment)
            state_counts[state] += 1
            owner = _OWNERS[dimension]
            dimensions.append(
                ReviewDimension(
                    dimension=dimension,
                    state=state,
                    anchor_source_refs=anchor_ids,
                    run_evidence_paths=_EVIDENCE_PATHS[dimension],
                    gap_or_rationale=rationale,
                    owner_module=owner,
                    next_action=_next_action(state, owner),
                    supports=(
                        "Method and presentation comparison for this exact "
                        "development run."
                    ),
                    cannot_prove=(
                        "Published-effect agreement, external validity, causal "
                        "effect, human approval, or publication readiness."
                    ),
                )
            )
        review = RunBoundShadowReview(
            protocol_ref=protocol.protocol_ref,
            protocol_sha256=protocol_content_sha256(protocol),
            anchor_pack_sha256=canonical_json_sha256(
                anchor_pack.model_dump(mode="json")
            ),
            supplement_review_sha256=supplement_review_sha256,
            task_id=task.task_id,
            run_head=str(code_version["git_sha"]),
            run_image=str(args.run_image),
            run_path=str(run_path.resolve()),
            anchors=anchor_ids,
            dimensions=tuple(dimensions),
            overall_status=(
                "changes_required"
                if any(row.state == "actionable_gap" for row in dimensions)
                else "accepted"
            ),
            claim_boundary=str(assessments.get("claim_boundary") or "").strip(),
        )
        validate_run_bound_review(
            review,
            protocol=protocol,
            anchor_pack=anchor_pack,
        )
        path = output_dir / f"{task.task_id}.json"
        path.write_text(review.model_dump_json(indent=2) + "\n", encoding="utf-8")
        reviews.append(review)

    summary = {
        "schema_version": "easyicu.dev9_exact_shadow_review_summary/1",
        "protocol_ref": protocol.protocol_ref,
        "protocol_sha256": protocol_content_sha256(protocol),
        "anchor_pack_sha256": canonical_json_sha256(
            anchor_pack.model_dump(mode="json")
        ),
        "supplement_review_sha256": supplement_review_sha256,
        "run_root": str(args.run_root.resolve()),
        "run_head": sorted({review.run_head for review in reviews}),
        "run_image": str(args.run_image),
        "task_count": len(reviews),
        "dimension_count": sum(len(review.dimensions) for review in reviews),
        "state_counts": dict(sorted(state_counts.items())),
        "accepted_task_count": sum(
            review.overall_status == "accepted" for review in reviews
        ),
        "changes_required_task_count": sum(
            review.overall_status == "changes_required" for review in reviews
        ),
        "claim_boundary": str(assessments.get("claim_boundary") or "").strip(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
