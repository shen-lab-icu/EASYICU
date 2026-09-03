#!/usr/bin/env python3
"""Zero-Provider audit of saved plans against literature-to-design authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.design_selection import ResearchDesignSelection
from easyicu.research_agent.planning.literature_design_authority import (
    LiteratureDesignAuthorityError,
    validate_preplan_literature_design_authority,
    validate_selected_design_against_literature,
)


def audit_run(plan_path: Path) -> dict[str, Any]:
    bundle_path = plan_path.parent / "preplan_literature_bundle.json"
    bundle = LiteratureBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_selection = raw_plan.get("design_selection")
    selection = (
        ResearchDesignSelection.model_validate(raw_selection)
        if raw_selection is not None
        else None
    )
    result: dict[str, Any] = {
        "run_dir": str(plan_path.parent),
        "preplan_gate": "pass",
        "selected_design_gate": "not_run",
        "candidate_count": len(selection.candidates) if selection else 0,
        "selected_dimension_decision_count": (
            len(selection.selected.literature_design_decisions) if selection else 0
        ),
        "design_evidence_card_count": len(bundle.design_evidence_cards),
    }
    try:
        validate_preplan_literature_design_authority(bundle)
    except LiteratureDesignAuthorityError as exc:
        result["preplan_gate"] = exc.reason_code
        return result
    comparison_keys = [
        decision.citation_key
        for decision in bundle.screening_decisions
        if decision.disposition == "include"
        and decision.evidence_role in {"direct_comparator", "design_analogue"}
    ]
    try:
        validate_selected_design_against_literature(
            selection,
            design_evidence_cards=bundle.design_evidence_cards,
            comparison_keys=comparison_keys,
        )
        result["selected_design_gate"] = "pass"
    except LiteratureDesignAuthorityError as exc:
        result["selected_design_gate"] = exc.reason_code
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_paths = sorted(args.root.rglob("analysis_plan.json"))
    rows = [
        audit_run(path)
        for path in plan_paths
        if (path.parent / "preplan_literature_bundle.json").is_file()
    ]
    payload = {
        "schema_version": "easyicu.literature_design_shadow_audit/1",
        "root": str(args.root),
        "provider_calls": 0,
        "run_count": len(rows),
        "pass_count": sum(
            row["preplan_gate"] == "pass"
            and row["selected_design_gate"] == "pass"
            for row in rows
        ),
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
