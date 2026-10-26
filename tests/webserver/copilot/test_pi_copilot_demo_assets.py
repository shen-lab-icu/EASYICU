"""The workflow demo keeps its registered run's own figure and labels a redraw."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static"


def test_demo_keeps_the_registered_dossier_figure_one_click_away() -> None:
    """The run's own rendering stays byte-identical to the first figure the
    sealed reviewer dossier embeds, and the answer links to it."""

    dossier = (STATIC / "assets/demo/system-validation-report.html").read_text(encoding="utf-8")
    figures = re.findall(r'<figure><img src="data:image/png;base64,([A-Za-z0-9+/=]+)"', dossier)
    assert figures, "The dossier embeds its figures"
    figure = (STATIC / "assets/demo/sofa2-phenotype-mortality.png").read_bytes()
    assert figure == base64.b64decode(figures[0])
    demo = (STATIC / "js/screens-guided-pi-demo.js").read_text(encoding="utf-8")
    assert "ORIGINAL_FIGURE_URL = 'assets/demo/sofa2-phenotype-mortality.png?v=" in demo
    assert "${originalFigureLink(esc)}" in demo


def test_the_shown_figure_is_the_registered_table_redrawn_and_says_so() -> None:
    """The image the answer shows is the redraw the trace record names: drawn
    from the registered result table, never a generated or edited image."""

    trace = json.loads((STATIC / "assets/demo/sofa2-run-trace.json").read_text(encoding="utf-8"))
    redrawn = trace["evidence"]["rerender"]
    table_id = trace["evidence"]["chain"][0]
    assert redrawn["source_table"] == table_id
    assert redrawn["source_sha256"] == trace["evidence"]["records"][table_id]["sha256"]
    image = (STATIC / "assets/demo" / redrawn["png"]).read_bytes()
    assert hashlib.sha256(image).hexdigest() == redrawn["png_sha256"]
    demo = (STATIC / "js/screens-guided-pi-demo.js").read_text(encoding="utf-8")
    assert f"FIGURE_URL = 'assets/demo/{redrawn['png']}?v=" in demo
    assert "redrawn from the registered result table" in demo


def test_the_supporting_figures_are_their_own_registered_tables_redrawn() -> None:
    trace = json.loads((STATIC / "assets/demo/sofa2-run-trace.json").read_text(encoding="utf-8"))
    supporting = trace["evidence"]["rerender_supporting"]
    assert set(supporting) == {"cohort_flow", "data_quality"}
    for record in supporting.values():
        image = (STATIC / "assets/demo" / record["png"]).read_bytes()
        assert hashlib.sha256(image).hexdigest() == record["png_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
    assert supporting["cohort_flow"]["step_id"] == "05_visualize_cohort_accounting"
    assert supporting["data_quality"]["step_id"] == "06_visualize_data_quality"
