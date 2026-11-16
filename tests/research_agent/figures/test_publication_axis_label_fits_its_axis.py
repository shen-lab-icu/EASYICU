"""A y-axis label never reaches above its axis into the panel letter (dH1, 11-16)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from easyicu.research_agent.figures.publication import (  # noqa: E402
    add_panel_label, apply_publication_style, finish_publication_layout, save_publication_figure,
)
from easyicu.research_agent.gates.visual_qa import audit_svg_text_layout  # noqa: E402

LONG = "Model-standardised in-hospital mortality (%)"


def _two_panel_figure():
    apply_publication_style()
    # The landmark figure's fixed grid: a short result row over a density strip.
    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(5.2, 0.72), hspace=0.12, wspace=0.32,
                            left=0.08, right=0.985, bottom=0.19, top=0.90)
    axes = []
    for column, (label, ylabel) in enumerate((("a", "Adjusted odds ratio (95% CI)"), ("b", LONG))):
        ax = fig.add_subplot(grid[0, column])
        ax.plot([3.5, 5.0], [1.0, 1.5])
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel(ylabel)
        ax.set_title("Absolute risk", loc="left", pad=4, fontsize=7.0, fontweight="semibold")
        add_panel_label(ax, label, x=-0.08, y=1.04)
        fig.add_subplot(grid[1, column], sharex=ax).set_yticks([])
        axes.append(ax)
    return fig, axes


def test_a_label_taller_than_its_axis_is_wrapped_to_the_axis_height():
    fig, axes = _two_panel_figure()
    finish_publication_layout(fig)
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        label = ax.yaxis.label.get_window_extent(renderer)
        frame = ax.get_window_extent(renderer)
        assert label.height <= frame.height + 1
        letter = next(text for text in ax.texts if text.get_gid() == "publication-panel-label")
        assert not letter.get_window_extent(renderer).overlaps(label)
    assert "\n" in axes[1].yaxis.label.get_text()
    assert axes[1].yaxis.label.get_text().replace("\n", " ") == LONG
    plt.close(fig)


def test_the_exported_svg_passes_the_text_collision_audit(tmp_path):
    fig, _ = _two_panel_figure()
    outputs = save_publication_figure(fig, tmp_path / "figure", formats=("svg",))
    plt.close(fig)
    errors = [f for f in audit_svg_text_layout(outputs["svg"]) if f.severity == "error"]
    assert not errors, [f.detail for f in errors]
