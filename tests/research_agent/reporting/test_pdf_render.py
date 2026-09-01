from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting.latex import scaffold_to_latex
from easyicu.research_agent.reporting.pdf_render import render_pdf_for_run
from easyicu.research_agent.reporting.write_phase import _latex_figure_paths


def test_scaffold_draft_watermark_is_explicit_and_opt_in() -> None:
    plain = scaffold_to_latex(markdown="# Methods\n\nA bounded analysis.")
    draft = scaffold_to_latex(
        markdown="# Methods\n\nA bounded analysis.",
        draft_watermark=True,
    )

    assert "DRAFT -- NOT FOR SUBMISSION" not in plain
    assert "DRAFT -- NOT FOR SUBMISSION" in draft
    assert "Human scientific and authorship review required" in draft


def test_scaffold_uses_run_relative_figure_path_without_duplicate_evidence_prefix() -> (
    None
):
    tex = scaffold_to_latex(
        markdown="# Results\n\nEvidence-bound result.",
        figure_paths=[
            (
                "primary_figure",
                "evidence/figure_primary__primary_association.pdf",
            )
        ],
    )

    assert (
        r"\includegraphics[width=\textwidth]{evidence/figure\_primary\_\_primary\_association.pdf}"
        in tex
    )
    assert "evidence/evidence/" not in tex

    with pytest.raises(ValueError, match="one safe run-relative evidence path"):
        scaffold_to_latex(
            markdown="# Results\n\nEvidence-bound result.",
            figure_paths=[
                (
                    "primary_figure",
                    "evidence/evidence/figure_primary.pdf",
                )
            ],
        )


def test_scaffold_separates_main_and_supplementary_figures() -> None:
    tex = scaffold_to_latex(
        markdown="# Study\n\n## Results\n\nBound result.",
        figure_paths=[("Main Figure 1", "figures/main/figure1.png")],
        supplementary_figure_paths=[
            ("Supplementary Figure S1", "figures/supplementary/figure_s1.png")
        ],
    )

    assert r"\section*{Main figures}" in tex
    assert r"\section*{Supplementary figures}" in tex
    assert "figures/main/figure1.png" in tex
    assert "figures/supplementary/figure\\_s1.png" in tex


def test_latex_figure_selection_uses_one_compile_safe_export_per_figure() -> None:
    records = [
        SimpleNamespace(
            kind="figure",
            evidence_id="plot_png",
            relative_path="evidence/plot_png__lactate_curve.png",
        ),
        SimpleNamespace(
            kind="figure",
            evidence_id="plot_tiff",
            relative_path="evidence/plot_tiff__lactate_curve.tiff",
        ),
        SimpleNamespace(
            kind="figure",
            evidence_id="plot_pdf",
            relative_path="evidence/plot_pdf__lactate_curve.pdf",
        ),
        SimpleNamespace(
            kind="figure",
            evidence_id="flow_png",
            relative_path="evidence/flow_png__cohort_flow.png",
        ),
        SimpleNamespace(
            kind="table",
            evidence_id="table_pdf",
            relative_path="evidence/table_pdf__cohort_flow.pdf",
        ),
    ]

    selected, omitted = _latex_figure_paths(records)

    assert selected == [
        ("plot_pdf", "evidence/plot_pdf__lactate_curve.pdf"),
        ("flow_png", "evidence/flow_png__cohort_flow.png"),
    ]
    assert omitted == ()


def test_latex_figure_selection_groups_ids_that_end_in_underscore() -> None:
    # ``<evidence_id>__<filename>`` doubles the separator when the id ends in
    # ``_``. Splitting on the first ``__`` keys the PDF as ``_lactate_curve``
    # and the PNG as ``lactate_curve``, so both exports of one figure would be
    # embedded again.
    records = [
        SimpleNamespace(
            kind="figure",
            evidence_id="figure_step_artifact_",
            relative_path="evidence/figure_step_artifact___lactate_curve.pdf",
        ),
        SimpleNamespace(
            kind="figure",
            evidence_id="figure_step_artifact_b1",
            relative_path="evidence/figure_step_artifact_b1__lactate_curve.png",
        ),
    ]

    selected, omitted = _latex_figure_paths(records)

    assert selected == [
        (
            "figure_step_artifact_",
            "evidence/figure_step_artifact___lactate_curve.pdf",
        )
    ]
    assert omitted == ()


def test_latex_figure_selection_reports_figures_without_a_safe_export() -> None:
    records = [
        SimpleNamespace(
            kind="figure",
            evidence_id="vector_only_svg",
            relative_path="evidence/vector_only_svg__forest_plot.svg",
        ),
        SimpleNamespace(
            kind="figure",
            evidence_id="vector_only_tiff",
            relative_path="evidence/vector_only_tiff__forest_plot.tiff",
        ),
    ]

    selected, omitted = _latex_figure_paths(records)

    assert selected == []
    assert omitted == ("vector_only_svg",)


@pytest.mark.skipif(
    not any(
        shutil.which(name) for name in ("tectonic", "latexmk", "xelatex", "pdflatex")
    ),
    reason="a local LaTeX engine is required for the render contract",
)
def test_pdf_render_emits_sandbox_and_digest_receipt(tmp_path) -> None:
    tex_path = tmp_path / "manuscript_scaffold.tex"
    tex_path.write_text(
        scaffold_to_latex(
            markdown="# Methods\n\nThis is a governed preview.",
            draft_watermark=True,
        ),
        encoding="utf-8",
    )

    result = render_pdf_for_run(
        tex_path=tex_path,
        output_dir=tmp_path,
        draft_watermark=True,
        timeout=90,
    )

    assert result.success, result.notes
    assert result.pdf_path is not None
    assert result.pdf_path.read_bytes().startswith(b"%PDF")
    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "easyicu.manuscript_pdf_receipt.v1"
    assert receipt["draft_watermark"] is True
    assert receipt["security"] == {
        "network_allowed": False,
        "shell_escape_allowed": False,
        "untrusted_input_mode": True,
        "working_directory_restricted": True,
    }
    assert receipt["source"]["name"] == "manuscript_scaffold.tex"
    assert receipt["pdf"]["name"] == "manuscript_scaffold.pdf"
    assert receipt["pdf"]["bytes"] == result.pdf_path.stat().st_size
    assert len(receipt["pdf"]["sha256"]) == 64
