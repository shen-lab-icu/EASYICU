from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.discovery_handoff import (
    build_handoff_from_row,
    select_discovery_row,
    write_handoff_packet,
)
from easyicu.research_agent.discovery_package import (
    validate_discovery_manuscript_package,
)
from easyicu.research_agent.discovery_story_figure import render_discovery_story_figure


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_select_discovery_row_prioritizes_hold_over_db_cannot_do() -> None:
    rows = [
        {
            "literature_idea_id": "a",
            "candidate_topic": "blocked",
            "go_no_go": "db-cannot-do",
            "go_no_go_reason": "missing concept",
            "novelty_label": "apparently_gap",
        },
        {
            "literature_idea_id": "b",
            "candidate_topic": "definition sensitivity",
            "go_no_go": "hold",
            "go_no_go_reason": "needs human differentiation",
            "novelty_label": "crowded_but_differentiable",
            "differentiators": ["cohort_definition_sensitivity"],
        },
    ]

    selected = select_discovery_row(rows)

    assert selected["literature_idea_id"] == "b"


def test_discovery_package_accepts_agent_handoff_and_multi_panel_story(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_ok",
        "executable_candidate_id": "execidea_ok",
        "candidate_topic": "cohort definition sensitivity in adult ICU AKI",
        "go_no_go": "hold",
        "go_no_go_reason": "needs differentiation",
        "novelty_label": "crowded_but_differentiable",
        "literature_source": "Critical Care example",
        "gap_evidence_quote": "Definition choice remains uncertain.",
    }
    handoff = build_handoff_from_row(row, triage_report_path=tmp_path / "triage.json")
    write_handoff_packet(handoff, tmp_path / "discovery_handoff.json")
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "publication_ready",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": True,
                "publication_ready": True,
            },
        },
    )
    (tmp_path / "manuscript_ready.md").write_text(
        "The discovery provenance, cohort evaluability, primary result, and "
        "audit trail are reported without blocked outcome leakage.",
        encoding="utf-8",
    )
    _write_json(
        tmp_path
        / "publication_figures"
        / "easyicu_discovery_story.figure_contract.json",
        {
            "figure_id": "easyicu_discovery_story",
            "core_claim": (
                "Discovery source, cohort evaluability, primary result, and "
                "audit evidence form one manuscript story."
            ),
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Literature source and idea funnel",
                    "role": "overview",
                    "claim": "The idea came from a critical-care gap.",
                    "evidence_ids": ["discovery_handoff"],
                },
                {
                    "panel_id": "B",
                    "title": "Cohort evaluability and missingness",
                    "role": "audit",
                    "claim": "Cohort and definition evaluability are explicit.",
                    "evidence_ids": ["cohort_attrition"],
                },
                {
                    "panel_id": "C",
                    "title": "Primary result",
                    "role": "relationship",
                    "claim": "The primary result is plotted from source data.",
                    "evidence_ids": ["primary_result"],
                },
                {
                    "panel_id": "D",
                    "title": "Evidence audit and reproducibility gate",
                    "role": "validation",
                    "claim": "Claims are bound to hashed evidence.",
                    "evidence_ids": ["claim_ledger"],
                },
            ],
            "source_data": [
                "discovery_handoff",
                "cohort_attrition",
                "primary_result",
                "claim_ledger",
            ],
        },
    )

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.package_ready is True
    assert assessment.status == "article_ready"
    assert assessment.figure_panel_count == 4
    assert assessment.missing_story_roles == []


def test_discovery_package_blocks_single_panel_publication_figure(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_bad",
        "candidate_topic": "single panel outcome robustness",
        "go_no_go": "hold",
        "go_no_go_reason": "needs differentiation",
    }
    handoff = build_handoff_from_row(row, triage_report_path=tmp_path / "triage.json")
    write_handoff_packet(handoff, tmp_path / "discovery_handoff.json")
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "publication_ready",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": True,
                "publication_ready": True,
            },
        },
    )
    (tmp_path / "manuscript_ready.md").write_text(
        "The exploratory association with death was near the null.",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "steps" / "04_outcome_gate" / "outputs" / "step_summary.json",
        {
            "step_id": "04_outcome_gate",
            "primary_analysis_authorized": False,
            "grouped_death_analysis_executed": False,
        },
    )
    _write_json(
        tmp_path / "publication_figures" / "easyicu_publication_figure.figure_contract.json",
        {
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Primary effect and robustness range.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Primary effect and robustness variants",
                    "role": "robustness",
                    "claim": "Primary row and variants are shown.",
                    "evidence_ids": ["robustness_panel"],
                }
            ],
            "source_data": ["robustness_panel"],
        },
    )

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.package_ready is False
    assert assessment.status == "manuscript_only"
    assert "discovery_provenance" in assessment.missing_story_roles
    assert assessment.blocked_outcome_steps == ["04_outcome_gate"]
    assert assessment.manuscript_outcome_leak_terms


def test_discovery_story_figure_writes_four_panel_contract(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_story",
        "candidate_topic": "cohort definition sensitivity in adult ICU AKI",
        "go_no_go": "hold",
        "go_no_go_reason": "needs differentiation",
        "literature_source": "Annals of Intensive Care example",
    }
    handoff = build_handoff_from_row(row, triage_report_path=tmp_path / "triage.json")
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "analysis_only",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": False,
                "publication_ready": False,
                "missing_evidence_count": 0,
            },
        },
    )
    _write_json(
        tmp_path / "evidence_audit.json",
        {"evidence_count": 12},
    )
    _write_json(
        tmp_path / "numeric_audit.json",
        {"numeric_error_count": 0},
    )

    paths = render_discovery_story_figure(run_dir=tmp_path, handoff=handoff)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    assert {"svg", "pdf", "png", "tiff", "contract"} <= set(paths)
    assert len(contract["panels"]) == 4
    assessment = validate_discovery_manuscript_package(
        run_dir=tmp_path, require_handoff=False
    )
    assert assessment.figure_panel_count == 4
    assert assessment.missing_story_roles == []
