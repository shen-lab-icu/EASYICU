"""Apply reviewed figure display settings without changing scientific values."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..contracts.figure_plan import FigurePresentationSpec, PlannedFigurePanelSpec
from .publication import PALETTE_CLINICAL, apply_publication_style


def presentation_from_panels(
    panels: Sequence[PlannedFigurePanelSpec],
) -> FigurePresentationSpec | None:
    choices = [panel.presentation for panel in panels if panel.presentation is not None]
    if not choices:
        return None
    if any(choice != choices[0] for choice in choices[1:]):
        raise ValueError(
            "conflicting_figure_presentation: one output must have one display specification"
        )
    return choices[0]


def create_presented_axes(panel_count: int, spec: FigurePresentationSpec):
    """Use a supported grid, bounded physical dimensions and editable fonts."""
    import matplotlib.pyplot as plt

    colors = dict(PALETTE_CLINICAL)
    if spec.palette == "colorblind":
        colors.update(
            blue="#0072B2",
            blue_soft="#B3D7EB",
            green="#009E73",
            orange="#E69F00",
            red="#D55E00",
            neutral="#777777",
        )
    elif spec.palette == "grayscale":
        colors.update(
            blue="#303030",
            blue_soft="#BBBBBB",
            green="#707070",
            orange="#999999",
            red="#555555",
            neutral="#999999",
        )
    apply_publication_style(font_size=spec.font_size, palette=colors)
    plt.rcParams["font.family"] = spec.font_family
    columns = (
        panel_count
        if spec.layout == "row"
        else 1
        if spec.layout == "column"
        else min(2, panel_count)
    )
    rows = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4),
        layout="constrained",
        squeeze=False,
    )
    flat = list(axes.flat)
    for axis in flat[panel_count:]:
        axis.remove()
    return fig, flat[:panel_count], colors


def apply_presented_legend(axis: Any, spec: FigurePresentationSpec) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    kwargs = {
        "loc": spec.legend_location,
        "fontsize": spec.font_size * 0.85,
        "frameon": False,
    }
    if spec.legend_location == "outside bottom":
        kwargs.update(
            loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(2, len(handles))
        )
    axis.legend(handles, labels, **kwargs)


def finish_presented_figure(
    fig: Any, spec: FigurePresentationSpec, *, base_font_size: float = 7.5
) -> None:
    """Scale explicitly sized annotations too; retain their relative hierarchy."""
    from matplotlib.text import Text

    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            # Artist defaults already use the selected font. Only renderer
            # constants below the base size need lifting for large delivery.
            size = text.get_fontsize()
            if size <= base_font_size + 2 and spec.font_size > base_font_size:
                text.set_fontsize(size * spec.font_size / base_font_size)
            text.set_fontfamily(spec.font_family)
    for axis in fig.axes:
        apply_presented_legend(axis, spec)
    fig.canvas.draw()
