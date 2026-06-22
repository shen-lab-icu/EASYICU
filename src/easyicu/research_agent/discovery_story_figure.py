"""Discovery-story composite figure generation.

This figure is for manuscript packaging, not for inventing scientific results.
It visualizes the audited chain from mined idea -> data/evaluability ->
result or blocked gate -> evidence audit, using only already-written run
artifacts.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .discovery_handoff import DiscoveryHandoffPacket
from .publication_figures import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)


def render_discovery_story_figure(
    *,
    run_dir: str | Path,
    handoff: DiscoveryHandoffPacket,
    stem: str = "easyicu_discovery_story",
) -> Dict[str, Path]:
    """Render a four-panel article-story figure from run artifacts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    out_dir = root / "publication_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _load_json(root / "run_status.json")
    evidence_audit = _load_json(root / "evidence_audit.json")
    numeric_audit = _load_json(root / "numeric_audit.json")
    blocked = _has_blocked_outcome_gate(root)

    palette = apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 132 / 25.4))
    for ax in axes.ravel():
        ax.axis("off")

    panels = [
        (
            "A",
            axes[0, 0],
            "Literature source and mined idea",
            _panel_a_text(handoff),
            palette["blue"],
        ),
        (
            "B",
            axes[0, 1],
            "Cohort and evaluability contract",
            _panel_b_text(root),
            palette["teal"],
        ),
        (
            "C",
            axes[1, 0],
            "Result or fail-closed gate",
            _panel_c_text(status, blocked),
            palette["orange"] if not blocked else palette["red"],
        ),
        (
            "D",
            axes[1, 1],
            "Evidence audit",
            _panel_d_text(status, evidence_audit, numeric_audit),
            palette["neutral"],
        ),
    ]
    for label, ax, title, body, color in panels:
        _draw_text_panel(ax, title=title, body=body, color=color)
        add_panel_label(ax, label)

    contract = make_figure_contract(
        figure_id=stem,
        core_claim=(
            "The discovery manuscript package preserves the path from mined "
            "literature idea to cohort/evaluability evidence, result or "
            "fail-closed gate, and audited manuscript readiness."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Literature source and idea funnel",
                "role": "overview",
                "claim": "The manuscript story starts from a frozen mined idea.",
                "evidence_ids": ["discovery_handoff"],
            },
            {
                "panel_id": "B",
                "title": "Cohort evaluability and missingness",
                "role": "audit",
                "claim": "Cohort construction and evaluability are explicit.",
                "evidence_ids": ["cohort_attrition", "evidence_audit"],
            },
            {
                "panel_id": "C",
                "title": "Primary result or blocked outcome gate",
                "role": "relationship",
                "claim": "Outcome claims are reported only when the gate authorizes them.",
                "evidence_ids": ["run_status"],
            },
            {
                "panel_id": "D",
                "title": "Evidence audit and reproducibility gate",
                "role": "validation",
                "claim": "Manuscript readiness is tied to evidence and numeric audits.",
                "evidence_ids": ["run_status", "evidence_audit", "numeric_audit"],
            },
        ],
        source_data=[
            "discovery_handoff.json",
            "run_status.json",
            "evidence_audit.json",
            "numeric_audit.json",
        ],
    )
    paths = save_publication_figure(
        fig,
        out_dir / stem,
        contract=contract,
        formats=["svg", "pdf", "png", "tiff"],
        dpi=300,
    )
    plt.close(fig)
    return paths


def _draw_text_panel(ax, *, title: str, body: str, color: str) -> None:
    ax.add_patch(
        plt_rectangle(
            (0, 0),
            1,
            1,
            facecolor="#FFFFFF",
            edgecolor="#D8D8D8",
            linewidth=0.8,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.06,
        0.88,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
    )
    ax.text(
        0.06,
        0.76,
        _wrap(body, width=46),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        linespacing=1.28,
        color="#272727",
    )


def plt_rectangle(*args, **kwargs):
    import matplotlib.patches as patches

    return patches.Rectangle(*args, **kwargs)


def _panel_a_text(handoff: DiscoveryHandoffPacket) -> str:
    return "\n".join(
        [
            f"Source: {_short(handoff.literature_source or 'source recorded', 92)}",
            f"Idea: {_short(handoff.candidate_topic, 110)}",
            f"Gate: {handoff.go_no_go} ({_short(handoff.go_no_go_reason, 80)})",
        ]
    )


def _panel_b_text(root: Path) -> str:
    cohort = _first_existing(
        root,
        [
            "steps/01_define_cohort_and_attrition/outputs/cohort_attrition.csv",
            "steps/00_probe/outputs/cohort_summary.csv",
        ],
    )
    evaluability = _first_existing(
        root,
        [
            "steps/02_derive_aki_component_flags_and_evaluability/outputs/definition_component_evaluability.csv",
            "steps/00_probe/outputs/aki_definition_evaluability.csv",
        ],
    )
    parts = []
    parts.append(f"Cohort evidence: {cohort.name if cohort else 'not found'}")
    parts.append(
        f"Evaluability evidence: {evaluability.name if evaluability else 'not found'}"
    )
    parts.append("Missingness and component availability must remain explicit.")
    return "\n".join(parts)


def _panel_c_text(status: Mapping[str, Any], blocked: bool) -> str:
    gates = status.get("gates") if isinstance(status.get("gates"), Mapping) else {}
    if blocked:
        return (
            "Outcome-by-group linkage is blocked by the recorded gate. "
            "No near-null, protective, harmful, or equivalence outcome claim "
            "is licensed unless a later certified status-carry-forward run passes."
        )
    return (
        f"Run status: {status.get('status', 'unknown')}. "
        f"Execution complete: {bool(gates.get('execution_complete'))}. "
        f"Manuscript ready: {bool(gates.get('manuscript_ready'))}."
    )


def _panel_d_text(
    status: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    numeric_audit: Mapping[str, Any],
) -> str:
    gates = status.get("gates") if isinstance(status.get("gates"), Mapping) else {}
    return "\n".join(
        [
            f"Evidence records: {evidence_audit.get('evidence_count', 'NA')}",
            f"Missing evidence: {gates.get('missing_evidence_count', 'NA')}",
            f"Numeric errors: {numeric_audit.get('numeric_error_count', 'NA')}",
            f"Publication ready: {bool(gates.get('publication_ready'))}",
        ]
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _has_blocked_outcome_gate(root: Path) -> bool:
    for path in root.glob("steps/*/outputs/step_summary.json"):
        payload = _load_json(path)
        if (
            payload.get("primary_analysis_authorized") is False
            or payload.get("grouped_death_analysis_executed") is False
            or (
                payload.get("analysis_executed") is False
                and "blocked" in json.dumps(payload, ensure_ascii=False).lower()
            )
        ):
            return True
    for path in root.glob("steps/*/outputs/*feasibility_gate.csv"):
        try:
            if "blocked" in path.read_text(encoding="utf-8").lower():
                return True
        except Exception:
            continue
    return False


def _first_existing(root: Path, rels: list[str]) -> Optional[Path]:
    for rel in rels:
        path = root / rel
        if path.exists():
            return path
    return None


def _wrap(text: str, *, width: int) -> str:
    lines: list[str] = []
    for line in str(text).splitlines():
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(lines)


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


__all__ = ["render_discovery_story_figure"]
