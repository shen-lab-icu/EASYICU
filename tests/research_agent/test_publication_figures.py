from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.publication_figures import (
    PanelSpec,
    apply_publication_style,
    audit_figure_contract,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)


def test_figure_contract_enforces_unique_panel_ids():
    with pytest.raises(ValueError):
        make_figure_contract(
            figure_id="Figure2",
            core_claim="SOFA2 zero is an audit target.",
            panels=[
                {"panel_id": "a", "title": "one", "role": "overview", "claim": "x"},
                {"panel_id": "a", "title": "two", "role": "audit", "claim": "y"},
            ],
        )


def test_contract_audit_flags_missing_evidence_and_duplicate_roles():
    contract = make_figure_contract(
        figure_id="Figure2",
        core_claim="SOFA2 zero is an audit target.",
        panels=[
            PanelSpec(panel_id="a", title="Overview", role="overview", claim="x"),
            PanelSpec(panel_id="b", title="Another overview", role="overview", claim="y"),
        ],
    )

    findings = audit_figure_contract(contract)
    messages = " ".join(f.message for f in findings)
    assert "repeats panel role" in messages
    assert "without evidence ids" in messages


def test_publication_export_keeps_svg_text_editable(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("SOFA-2")
    ax.set_ylabel("Mortality")
    ax.set_title("Editable text")

    contract = make_figure_contract(
        figure_id="FigureTest",
        core_claim="A simple line can be exported with editable SVG text.",
        panels=[
            {
                "panel_id": "a",
                "title": "Line",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["statistic_step_summary"],
            }
        ],
        export_formats=["svg", "pdf", "png"],
    )
    paths = save_publication_figure(fig, tmp_path / "figure_test", contract=contract, dpi=150)
    plt.close(fig)

    assert {"svg", "pdf", "png", "contract"} <= set(paths)
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert paths["contract"].exists()
    assert audit_publication_exports(paths) == []


def test_svg_audit_flags_pathified_text(tmp_path: Path):
    svg = tmp_path / "bad.svg"
    svg.write_text("<svg><path d='M0 0L1 1'/></svg>", encoding="utf-8")
    findings = audit_publication_exports([svg], min_bytes=1)
    assert any("editable <text>" in f.message for f in findings)
