"""Equivalent font matches cannot depend on fresh cache discovery order."""

from io import BytesIO
from pathlib import Path

import pytest

from easyicu.research_agent.figures.publication import apply_publication_style


@pytest.mark.parametrize("warm_cache", [False, True])
def test_publication_font_and_export_bytes_ignore_discovery_order(monkeypatch, warm_cache):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    # Two real, visibly different bundled fonts with identical match scores
    # simulate duplicate installed versions without relying on system fonts.
    entries = [
        font_manager.FontEntry(fname=str(font_dir / name), name="DejaVu Sans")
        for name in ("DejaVuSans.ttf", "DejaVuSansMono.ttf")
    ]
    manager = font_manager.fontManager
    properties = font_manager.FontProperties(family=["DejaVu Sans"], size=7.5)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    choices, products = [], []
    try:
        with plt.rc_context({"svg.hashsalt": "synthetic-font-order"}):
            for order in (entries, entries[::-1]):
                monkeypatch.setattr(manager, "ttflist", list(order))
                manager._findfont_cached.cache_clear()
                if warm_cache:
                    manager.findfont(properties)
                apply_publication_style()
                choices.append(manager.findfont(properties))
                figure = plt.figure(figsize=(2, 1))
                try:
                    figure.text(0.1, 0.5, "Synthetic observations 123", fontproperties=properties)
                    exports = {}
                    for format_name in ("pdf", "png", "svg", "tiff"):
                        output = BytesIO()
                        figure.savefig(output, format=format_name)
                        exports[format_name] = output.getvalue()
                    products.append(exports)
                finally:
                    plt.close(figure)
    finally:
        # Do not leave synthetic font resolutions in Matplotlib's shared cache.
        manager._findfont_cached.cache_clear()

    assert choices[0] == choices[1]
    assert products[0] == products[1]
