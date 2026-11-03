"""The shared interval and curve marks draw exactly the values they are given.

``figures/publication.py`` owns how every deterministic renderer draws an
interval row, a curve, a value column and a number-at-risk row. These tests pin
the geometry to the inputs (so an argument swapped inside the owner fails) and
the house style (so a renderer cannot drift back to capped grey whiskers).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from easyicu.research_agent.figures.display_labels import effect_measure_label  # noqa: E402
from easyicu.research_agent.figures.publication import (  # noqa: E402
    PALETTE_CLINICAL,
    add_panel_label,
    curve_kwargs,
    draw_interval_rows,
    draw_number_at_risk,
    interval_text,
    km_risk_times,
    label_rows,
    save_publication_figure,
    style_category_axis,
    style_value_axis,
)

# Asymmetric intervals with distinct estimate, low and high in every row, so
# any permutation of the three arguments changes the drawn geometry.
ESTIMATES = [1.5, 2.0, 0.8]
LOWS = [1.0, 1.2, 0.5]
HIGHS = [3.0, 2.2, 0.9]
POSITIONS = [0.0, 1.0, 2.0]


@pytest.fixture(autouse=True)
def close_figures():
    with plt.rc_context():
        yield
        plt.close("all")


def _segments(container):
    return [segment.tolist() for segment in container.lines[2][0].get_segments()]


def test_horizontal_rows_put_each_estimate_and_bound_where_it_belongs():
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(
        ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL,
    )

    marker = drawn.container.lines[0]
    np.testing.assert_array_equal(marker.get_xdata(), ESTIMATES)
    np.testing.assert_array_equal(marker.get_ydata(), POSITIONS)
    segments = _segments(drawn.container)
    assert [[point[0] for point in segment] for segment in segments] == [
        [low, high] for low, high in zip(LOWS, HIGHS)
    ]
    assert [{point[1] for point in segment} for segment in segments] == [
        {position} for position in POSITIONS
    ]


def test_vertical_rows_draw_the_estimate_on_y_at_each_position():
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(
        ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL,
        orientation="vertical",
    )

    marker = drawn.container.lines[0]
    np.testing.assert_array_equal(marker.get_xdata(), POSITIONS)
    np.testing.assert_array_equal(marker.get_ydata(), ESTIMATES)
    assert [[point[1] for point in segment] for segment in _segments(drawn.container)] == [
        [low, high] for low, high in zip(LOWS, HIGHS)
    ]


@pytest.mark.parametrize(("orientation", "null"), [("horizontal", 1.0), ("vertical", 0.0)])
def test_the_null_line_sits_at_the_null_on_the_value_axis_beneath_the_marks(orientation, null):
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(
        ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL,
        null=null, orientation=orientation,
    )

    line = drawn.null_line
    value_data = line.get_xdata() if orientation == "horizontal" else line.get_ydata()
    span_data = line.get_ydata() if orientation == "horizontal" else line.get_xdata()
    assert list(value_data) == [null, null]
    assert list(span_data) == [0, 1]  # spans the whole axes across the rows
    assert to_hex(line.get_color()) == PALETTE_CLINICAL["ink_3"].lower()
    assert line.get_linewidth() == pytest.approx(0.6)
    assert line.get_linestyle() == "--"
    assert line.get_zorder() < drawn.container.lines[0].get_zorder()


def test_no_null_draws_no_reference_line():
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL)
    assert drawn.null_line is None
    assert len(ax.lines) == 1  # the marker line only


def test_rows_use_the_house_marker_whisker_and_no_caps():
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(
        ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL, color="orange",
    )

    marker, caps, (whiskers,) = drawn.container.lines
    assert marker.get_marker() == "s"
    assert marker.get_markersize() == pytest.approx(4.0)
    assert marker.get_linestyle() == "None"
    assert to_hex(marker.get_markerfacecolor()) == PALETTE_CLINICAL["orange"].lower()
    assert caps == ()
    assert to_hex(whiskers.get_color()[0]) == PALETTE_CLINICAL["ink_2"].lower()
    assert whiskers.get_linewidth()[0] == pytest.approx(0.9)


def test_rows_accept_a_literal_colour_and_reject_ragged_input():
    _fig, ax = plt.subplots()
    drawn = draw_interval_rows(
        ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL, color="#123456",
    )
    assert to_hex(drawn.container.lines[0].get_markerfacecolor()) == "#123456"
    with pytest.raises(ValueError, match="one estimate"):
        draw_interval_rows(ax, ESTIMATES, LOWS[:2], HIGHS, POSITIONS, palette=PALETTE_CLINICAL)


def test_rounding_clip_is_opt_in_and_only_shortens_inverted_whiskers():
    _fig, ax = plt.subplots()
    inverted_by_an_ulp = [np.nextafter(2.0, 3.0)]
    with pytest.raises(ValueError):
        draw_interval_rows(ax, [2.0], inverted_by_an_ulp, [2.5], [0], palette=PALETTE_CLINICAL)
    drawn = draw_interval_rows(
        ax, [2.0], inverted_by_an_ulp, [2.5], [0], palette=PALETTE_CLINICAL,
        clip_rounding=True,
    )
    assert [[point[0] for point in segment] for segment in _segments(drawn.container)] == [
        [2.0, 2.5]
    ]


def test_log_rows_put_the_value_axis_on_a_log_scale():
    _fig, ax = plt.subplots()
    draw_interval_rows(ax, ESTIMATES, LOWS, HIGHS, POSITIONS, palette=PALETTE_CLINICAL, log=True)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"


def test_axis_styles_hide_the_category_axis_and_grid_only_the_values():
    _fig, ax = plt.subplots()
    ax.set_yticks(POSITIONS, ["a", "b", "c"])
    style_category_axis(ax)
    style_value_axis(ax, grid=True)

    assert not ax.spines["left"].get_visible()
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b", "c"]
    assert all(tick.tick1line.get_markersize() == 0 for tick in ax.yaxis.get_major_ticks())
    assert any(line.get_visible() for line in ax.get_xgridlines())
    assert not any(line.get_visible() for line in ax.get_ygridlines())
    assert ax.get_axisbelow() is True


def test_curves_are_one_point_two_points_in_their_series_colour():
    assert curve_kwargs(PALETTE_CLINICAL, "red") == {
        "color": PALETTE_CLINICAL["red"], "linewidth": 1.2, "solid_capstyle": "butt",
    }
    assert curve_kwargs(PALETTE_CLINICAL, "#2C7F86")["color"] == "#2C7F86"


def test_the_value_column_sits_outside_the_axes_on_each_row():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_ylim(2.5, -0.5)
    texts = [interval_text(e, low, high) for e, low, high in zip(ESTIMATES, LOWS, HIGHS)]
    drawn = label_rows(ax, texts, POSITIONS)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    right = ax.get_window_extent(renderer).x1
    for text, position in zip(drawn, POSITIONS):
        box = text.get_window_extent(renderer)
        assert box.x0 > right
        row_y = ax.transData.transform((0, position))[1]
        assert box.y0 < row_y < box.y1
        assert text.get_fontsize() == pytest.approx(6.5)
        assert to_hex(text.get_color()) == PALETTE_CLINICAL["ink_2"].lower()
    assert [text.get_text() for text in drawn] == [
        "1.50 (1.00–3.00)", "2.00 (1.20–2.20)", "0.80 (0.50–0.90)",
    ]
    assert interval_text(-0.012, -0.03, 0.006, ratio=False) == "-0.012 (-0.03 to 0.006)"


def test_number_at_risk_columns_stand_at_the_km_ticks():
    fig, (ax_km, ax_risk) = plt.subplots(2, 1, sharex=True)
    ax_km.set_xlim(0.0, 27.0)
    times = km_risk_times(ax_km, 27.0)
    assert times == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert list(ax_km.get_xticks()) == times
    rows = [[str(100 - index) for index in range(len(times))], ["9"] * len(times)]
    draw_number_at_risk(ax_risk, times, rows, ["Group A", "Group B"])
    fig.canvas.draw()
    counts = [text for text in ax_risk.texts if text.get_text() not in {"Group A", "Group B", "Number at risk"}]
    assert sorted({text.get_position()[0] for text in counts}) == times
    with pytest.raises(ValueError):
        draw_number_at_risk(ax_risk, times, [["1"]], ["Group A"])


def test_effect_measure_names_read_the_abbreviations_producers_write():
    assert effect_measure_label("hr") == "Hazard ratio"
    assert effect_measure_label("OR") == "Odds ratio"
    assert effect_measure_label("odds_ratio_per_1_mmol_per_l") == "Odds ratio per 1 mmol per l"
    assert effect_measure_label("risk_difference") == "Risk difference"
    assert effect_measure_label("custom_scale") == "Custom scale"


@pytest.mark.parametrize("title", ["Weighted risk", "A long panel title that wraps over two lines here"])
def test_the_panel_letter_shares_the_titles_first_baseline(tmp_path, title):
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_title(title, loc="left")
    ax.plot([0, 1], [0, 1])
    add_panel_label(ax, "b")
    save_publication_figure(fig, tmp_path / "letter", formats=("png",), dpi=100)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letter = ax.texts[0]
    title_artist = ax._left_title
    assert letter.get_verticalalignment() == "baseline"
    assert letter.get_fontsize() == pytest.approx(9.0)
    assert letter.get_fontweight() == "bold"
    letter_box = letter.get_window_extent(renderer)
    title_box = title_artist.get_window_extent(renderer)
    font = title_artist.get_fontproperties()
    first = title_artist.get_text().split("\n")[0]
    _, height, descent = renderer.get_text_width_height_descent(first, font, False)
    _, lp_height, lp_descent = renderer.get_text_width_height_descent("lp", font, False)
    baseline = title_box.y1 - (max(height, lp_height) - max(descent, lp_descent))
    letter_baseline = letter.get_transform().transform(letter.get_position())[1]
    assert letter_baseline == pytest.approx(baseline, abs=1.0)
    assert letter_box.x1 <= ax.get_window_extent(renderer).x0
