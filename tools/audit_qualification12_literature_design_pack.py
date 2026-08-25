#!/usr/bin/env python3
"""Zero-Provider audit for the Qualification12 reviewed literature seed pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    validate_preplan_literature_design_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pack(
    pack_path: Path,
    task_bank_path: Path,
    *,
    source_pack_root: Path | None = None,
) -> dict[str, Any]:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    tasks = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in task_bank_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    source_metadata: dict[str, dict[str, Any]] = {}
    source_reviews: dict[str, dict[str, Any]] = {}
    manifest_errors: list[str] = []
    if source_pack_root is not None:
        source_manifest_path = source_pack_root / "source_manifest.json"
        review_manifest_path = source_pack_root / "review_manifest.json"
        if _sha256(source_manifest_path) != pack["source_manifest_sha256"]:
            manifest_errors.append("source_manifest_sha256_mismatch")
        if _sha256(review_manifest_path) != pack["review_manifest_sha256"]:
            manifest_errors.append("review_manifest_sha256_mismatch")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
        source_metadata = {row["pmcid"]: row for row in source_manifest["sources"]}
        source_reviews = {row["pmcid"]: row for row in review_manifest["sources"]}

    rows = []
    seen_task_ids: set[str] = set()
    unique_pmids: set[str] = set()
    total_cards = 0
    total_evidence = 0
    for item in pack["items"]:
        task_id = item["task_id"]
        errors: list[str] = []
        seen_task_ids.add(task_id)
        task = tasks.get(task_id)
        bundle = LiteratureBundle.model_validate(item["bound_preplan_literature"])
        if task is None:
            errors.append("unknown_task_id")
        elif bundle.research_question != task["question"]:
            errors.append("research_question_mismatch")
        if len(bundle.citations) != 2 or len(bundle.design_evidence_cards) != 2:
            errors.append("expected_two_sources_and_cards")
        try:
            validate_preplan_literature_design_authority(bundle)
        except Exception as exc:  # owner validator carries the stable failure
            errors.append(f"preplan_authority:{type(exc).__name__}:{exc}")
        dimensions = {
            evidence.dimension
            for card in bundle.design_evidence_cards
            for evidence in card.evidence
        }
        if dimensions != set(LITERATURE_DESIGN_DIMENSIONS):
            errors.append("seven_dimensions_incomplete_or_extra")
        if pack["selection_policy"]["published_effects_are_expected_answers"]:
            errors.append("published_effects_must_not_be_expected_answers")
        total_cards += len(bundle.design_evidence_cards)
        total_evidence += sum(len(card.evidence) for card in bundle.design_evidence_cards)
        unique_pmids.update(citation.pmid for citation in bundle.citations if citation.pmid)

        if source_pack_root is not None:
            for citation, card in zip(
                bundle.citations, bundle.design_evidence_cards, strict=True
            ):
                pmcid = citation.key.split("_", 1)[0].upper()
                metadata = source_metadata.get(pmcid)
                review = source_reviews.get(pmcid)
                if metadata is None or review is None:
                    errors.append(f"{pmcid}:source_receipt_missing")
                    continue
                fulltext_path = source_pack_root / pmcid / "fulltext.xml"
                if not fulltext_path.is_file() or _sha256(fulltext_path) != card.full_text_sha256:
                    errors.append(f"{pmcid}:fulltext_sha256_mismatch")
                if card.supplement_status != review["supplement_status"]:
                    errors.append(f"{pmcid}:supplement_status_mismatch")
                if card.supplement_sha256 != review.get("supplement_sha256"):
                    errors.append(f"{pmcid}:supplement_sha256_mismatch")
                for supplement in review.get("supplement_files", []):
                    path = source_pack_root / pmcid / "review_extract" / supplement["name"]
                    if not path.is_file() or _sha256(path) != supplement["sha256"]:
                        errors.append(f"{pmcid}:supplement_file_mismatch:{supplement['name']}")

        rows.append(
            {
                "task_id": task_id,
                "status": "pass" if not errors else "fail",
                "card_count": len(bundle.design_evidence_cards),
                "dimension_count": len(dimensions),
                "errors": errors,
            }
        )

    missing_tasks = sorted(set(tasks) - seen_task_ids)
    extra_tasks = sorted(seen_task_ids - set(tasks))
    passed = not manifest_errors and not missing_tasks and not extra_tasks and all(
        row["status"] == "pass" for row in rows
    )
    return {
        "schema_version": "easyicu.qualification12_literature_design_audit/1",
        "status": "pass" if passed else "fail",
        "provider_calls": 0,
        "task_count": len(rows),
        "pass_count": sum(row["status"] == "pass" for row in rows),
        "card_instance_count": total_cards,
        "unique_article_count": len(unique_pmids),
        "evidence_fact_count": total_evidence,
        "missing_tasks": missing_tasks,
        "extra_tasks": extra_tasks,
        "manifest_errors": manifest_errors,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument(
        "--task-bank",
        type=Path,
        default=Path("benchmarks/meta_generalization/meta_benchmark.jsonl"),
    )
    parser.add_argument("--source-pack-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit_pack(
        args.pack,
        args.task_bank,
        source_pack_root=args.source_pack_root,
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
