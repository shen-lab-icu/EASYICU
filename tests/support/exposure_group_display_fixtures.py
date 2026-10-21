"""Synthetic digest-bound context records shared by figure and host tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import easyicu.research_agent as ra


def context_manifest(run_dir: Path) -> dict:
    context = ra.ResearchContext(
        research_question="Is the lactate tertile associated with ICU readmission?",
        cohort=ra.CohortDescriptor(
            cohort_name="Adult ICU stays", database="mimiciv", n_patients=380, n_stays=412,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="icu_readmission", description="ICU readmission", role="outcome",
                dtype="bool", source_concept="readmission",
            ),
            ra.ConceptDescriptor(
                name="readmission_time_hours", description="ICU readmission", role="time",
                dtype="float64", unit="h", source_concept="readmission",
            ),
            ra.ConceptDescriptor(
                name="lactate_tertile", description="lactate tertile", role="lab",
                dtype="int64", source_concept="lact",
            ),
        ],
        primary_exposure="lactate_tertile",
        target_outcome="icu_readmission",
    )
    path = run_dir / "research_context.json"
    path.write_text(context.model_dump_json(), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"context": {"relative_path": path.name, "sha256": digest}}


def grouped_context_manifest(run_dir: Path) -> dict:
    from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY

    manifest = context_manifest(run_dir)
    path = run_dir / "research_context.json"
    context = json.loads(path.read_text())
    context["variables"][-1]["description"] = "乳酸分组"
    context["cohort"]["provenance"][EXPOSURE_GROUP_LABELS_KEY] = [{
        "variable": "lactate_tertile", "concept": "lact", "scale": "ordinal",
        "levels": [{"code": 1, "group": "g1", "label": "Lower band"},
                   {"code": 2, "group": "g2", "label": "Upper band"}],
    }]
    path.write_text(json.dumps(context))
    manifest["context"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest

