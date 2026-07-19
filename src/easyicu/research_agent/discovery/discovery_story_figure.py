"""Discovery-story composite figure generation.

This figure is for manuscript packaging, not for inventing scientific results.
It visualizes the audited chain from mined idea -> data/evaluability ->
result or blocked gate -> evidence audit, using only already-written run
artifacts.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .discovery_handoff import DiscoveryHandoffPacket
from ..authority.evidence_store import EvidenceStore
from ..schema import EvidenceRecord
from ..figures.publication import (
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
    evidence = EvidenceStore(root)
    cohort_record = _find_semantic_record(
        evidence,
        terms=("cohort_attrition", "cohort", "denominator", "sample_flow"),
        kinds=("table", "statistic"),
    )
    evaluability_record = _find_semantic_record(
        evidence,
        terms=(
            "evaluability",
            "missingness",
            "completeness",
            "coverage",
            "availability",
        ),
        kinds=("table", "statistic", "log"),
        excluded_ids=(cohort_record.evidence_id,) if cohort_record else (),
    )
    handoff_record = evidence.get("discovery_handoff")
    run_status_record = evidence.get("run_status")
    evidence_audit_record = evidence.get("evidence_audit")
    numeric_audit_record = evidence.get("numeric_audit")

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
            _panel_b_text(cohort_record, evaluability_record),
            palette["teal"],
        ),
        (
            "C",
            axes[1, 0],
            "Analysis authorization status",
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

    panel_a_evidence = _record_ids(handoff_record)
    panel_b_evidence = _record_ids(cohort_record, evaluability_record)
    panel_c_evidence = _record_ids(run_status_record)
    panel_d_evidence = _record_ids(
        run_status_record,
        evidence_audit_record,
        numeric_audit_record,
    )
    source_data = list(
        dict.fromkeys(
            panel_a_evidence + panel_b_evidence + panel_c_evidence + panel_d_evidence
        )
    )
    contract = make_figure_contract(
        figure_id=stem,
        core_claim=(
            "The discovery manuscript package preserves the path from mined "
            "literature idea to cohort/evaluability evidence, analysis "
            "authorization, and audited manuscript readiness."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Literature source and idea funnel",
                "role": "overview",
                "claim": "The manuscript story starts from a frozen mined idea.",
                "evidence_ids": panel_a_evidence,
                "metadata": {"story_role": "discovery_provenance"},
            },
            {
                "panel_id": "B",
                "title": "Cohort evaluability and missingness",
                "role": "audit",
                "claim": "Cohort construction and evaluability are explicit.",
                "evidence_ids": panel_b_evidence,
                "metadata": {"story_role": "cohort_evaluability"},
            },
            {
                "panel_id": "C",
                "title": "Analysis authorization gate",
                "role": "workflow",
                "claim": "The recorded gate determines whether analysis may proceed.",
                "evidence_ids": panel_c_evidence,
            },
            {
                "panel_id": "D",
                "title": "Evidence audit and reproducibility gate",
                "role": "validation",
                "claim": "Manuscript readiness is tied to evidence and numeric audits.",
                "evidence_ids": panel_d_evidence,
                "metadata": {"story_role": "audit_reproducibility"},
            },
        ],
        source_data=source_data,
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


def _panel_b_text(
    cohort: Optional[EvidenceRecord],
    evaluability: Optional[EvidenceRecord],
) -> str:
    return "\n".join(
        [
            f"Cohort evidence: {cohort.evidence_id if cohort else 'not found'}",
            (
                "Evaluability evidence: "
                f"{evaluability.evidence_id if evaluability else 'not found'}"
            ),
            "Missingness and component availability must remain explicit.",
        ]
    )


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


def _record_ids(*records: Optional[EvidenceRecord]) -> list[str]:
    return list(
        dict.fromkeys(record.evidence_id for record in records if record is not None)
    )


def _find_semantic_record(
    evidence: EvidenceStore,
    *,
    terms: Sequence[str],
    kinds: Sequence[str],
    excluded_ids: Sequence[str] = (),
) -> Optional[EvidenceRecord]:
    """Select by EvidenceStore kind plus structured metadata/explicit aliases."""

    aliases_by_id: Dict[str, list[str]] = {}
    for alias, evidence_id in evidence.aliases().items():
        aliases_by_id.setdefault(evidence_id, []).append(alias)
    excluded = set(excluded_ids)
    normalised_terms = [_semantic_token(term) for term in terms]
    scored: list[tuple[int, str, EvidenceRecord]] = []
    for record in evidence.records():
        if record.evidence_id in excluded or record.kind not in set(kinds):
            continue
        aliases = [
            _semantic_token(alias)
            for alias in aliases_by_id.get(record.evidence_id, [])
        ]
        metadata_blob = _semantic_token(
            json.dumps(record.metadata or {}, sort_keys=True, default=str)
        )
        score = 0
        for term in normalised_terms:
            if any(alias == term for alias in aliases):
                score += 20
            elif any(term in alias for alias in aliases):
                score += 8
            if term in metadata_blob:
                score += 12
        if score:
            scored.append((score, record.evidence_id, record))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _semantic_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _wrap(text: str, *, width: int) -> str:
    lines: list[str] = []
    for line in str(text).splitlines():
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(lines)


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


__all__ = ["render_discovery_story_figure"]
