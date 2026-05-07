from __future__ import annotations

from pathlib import Path
import zipfile

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


def test_make_figure_contract_accepts_agent_style_aliases():
    contract = make_figure_contract(
        {
            "figure_id": "STEP07_claim_first_multipanel",
            "title": "Claim-first EasyICU publication figure",
            "claim": "Mortality rises across SOFA-2 while missingness must remain explicit.",
            "panels": [
                {
                    "panel": "a",
                    "title": "Outcome",
                    "claim": "Mortality is quantified with confidence intervals.",
                    "source_evidence": "figure_source_cohort_summary.csv",
                },
                {
                    "panel": "b",
                    "title": "Missingness audit",
                    "claim": "Missingness is reported instead of silently imputed.",
                    "source_evidence": "figure_source_missingness.csv",
                },
            ],
            "source_evidence": {
                "cohort_summary": "figure_source_cohort_summary.csv",
                "missingness": "figure_source_missingness.csv",
            },
            "statistical_notes": [
                "Ordinal scores are not averaged.",
                "Missingness is explicit.",
            ],
            "target_outcome": "death",
            "cohort": "miiv_crossdb_cohort",
        }
    )

    assert contract.figure_id == "STEP07_claim_first_multipanel"
    assert contract.core_claim.startswith("Mortality rises")
    assert [p.panel_id for p in contract.panels] == ["a", "b"]
    assert contract.panels[0].evidence_ids == ["figure_source_cohort_summary.csv"]
    assert contract.panels[1].role == "audit"
    assert contract.source_data == [
        "figure_source_cohort_summary.csv",
        "figure_source_missingness.csv",
    ]
    assert "Ordinal scores are not averaged." in (contract.statistics_note or "")


def test_make_figure_contract_accepts_agent_role_aliases_and_source_dicts():
    contract = make_figure_contract(
        figure_id="FigureAgent",
        core_claim="Severity and physiology both matter.",
        panels=[
            {
                "panel_id": "A",
                "title": "Anchor",
                "role": "cohort_anchor",
                "claim": "Outcome is defined.",
                "evidence_ids": ["src_incidence"],
            },
            {
                "panel_id": "B",
                "title": "Association forest",
                "role": "association_forest",
                "claim": "Associations are complete-case only.",
                "evidence_ids": ["src_assoc"],
            },
        ],
        source_data=[
            {"evidence_id": "src_incidence", "path": "incidence.csv"},
            {"evidence_id": "src_assoc", "path": "assoc.csv"},
        ],
    )

    assert [p.role for p in contract.panels] == ["overview", "robustness"]
    assert contract.source_data == ["incidence.csv", "assoc.csv"]


def test_figure_contract_defaults_include_tiff():
    contract = make_figure_contract(
        figure_id="FigureDefault",
        core_claim="Default export bundle should include TIFF.",
        panels=[
            {
                "panel_id": "a",
                "title": "Overview",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["src"],
            }
        ],
    )

    assert contract.export_formats == ["svg", "pdf", "png", "tiff"]


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


def test_publication_export_caps_and_compresses_tiff(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot([0, 1, 2, 3], [0.1, 0.4, 0.2, 0.7], linewidth=1.5)
    ax.set_xlabel("SOFA-2")
    ax.set_ylabel("Death risk")
    ax.set_title("Compressed TIFF export")

    paths = save_publication_figure(fig, tmp_path / "figure_tiff", formats=["tiff"], dpi=600)
    plt.close(fig)

    assert paths["tiff"].exists()
    assert paths["tiff"].stat().st_size < 8_000_000


def test_svg_audit_flags_pathified_text(tmp_path: Path):
    svg = tmp_path / "bad.svg"
    svg.write_text("<svg><path d='M0 0L1 1'/></svg>", encoding="utf-8")
    findings = audit_publication_exports([svg], min_bytes=1)
    assert any("editable <text>" in f.message for f in findings)


def test_publication_export_audit_flags_svg_text_overlap(tmp_path: Path):
    svg = tmp_path / "overlap.svg"
    svg.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="title_a">
            <text x="80" y="50" style="font-size: 15px; text-anchor: middle">Primary association</text>
          </g>
          <g id="title_b">
            <text x="84" y="52" style="font-size: 15px; text-anchor: middle">Ascertainment audit</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    findings = audit_publication_exports([svg], min_bytes=1)

    assert any("overlapping text" in f.message for f in findings)


def test_publication_export_writes_pptx(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    paths = save_publication_figure(fig, tmp_path / "figure_pptx", formats=["pptx"])
    plt.close(fig)

    pptx = paths["pptx"]
    assert pptx.exists()
    with zipfile.ZipFile(pptx) as z:
        names = set(z.namelist())
    assert "ppt/slides/slide1.xml" in names
    assert "ppt/media/image1.png" in names


def test_publication_export_audit_accepts_output_dir_and_stem(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Audit path")
    paths = save_publication_figure(fig, tmp_path / "figure_audit", formats=["svg", "png"])
    plt.close(fig)

    findings = audit_publication_exports(output_dir=tmp_path, stem="figure_audit", min_bytes=1)
    assert findings == []
    assert paths["svg"].exists()


def test_save_publication_figure_accepts_legacy_contract_and_output_dir_call(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [1, 0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contract = make_figure_contract(
        figure_id="FigureLegacy",
        core_claim="Legacy call path still exports correctly.",
        panels=[
            {
                "panel_id": "a",
                "title": "Panel",
                "role": "overview",
                "claim": "Line exists.",
                "evidence_ids": ["src"],
            }
        ],
    )

    paths = save_publication_figure(
        fig,
        contract,
        tmp_path,
        "legacy_figure",
        formats=["svg", "png"],
        dpi=150,
    )
    plt.close(fig)

    assert {"svg", "png", "contract"} <= set(paths)
    assert paths["svg"].name == "legacy_figure.svg"
