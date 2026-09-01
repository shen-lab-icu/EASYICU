from __future__ import annotations

import pytest

from easyicu.cohort_visualization import HAS_PLOTLY, MultiDatabaseDistribution


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly is not installed")
def test_distribution_grid_keeps_dense_feature_layout_readable() -> None:
    concepts = [
        "hr",
        "sbp",
        "dbp",
        "map",
        "temp",
        "resp",
        "o2sat",
        "glu",
        "na",
        "k",
        "crea",
        "bili",
        "lact",
        "hgb",
        "plt",
        "wbc",
    ]

    fig = MultiDatabaseDistribution().create_distribution_grid({}, concepts, cols=4)

    assert fig.layout.height == 1120
    assert fig.layout.legend.font.size == 13
    assert fig.layout.annotations[0].font.size == 12
    assert fig.layout.xaxis.tickfont.size == 11
    assert fig.layout.yaxis.tickfont.size == 10

    first_row_domain = fig.layout.yaxis.domain
    second_row_domain = fig.layout.yaxis5.domain
    first_row_height = first_row_domain[1] - first_row_domain[0]
    row_gap = first_row_domain[0] - second_row_domain[1]

    assert first_row_height > 0.2
    assert row_gap < 0.07
