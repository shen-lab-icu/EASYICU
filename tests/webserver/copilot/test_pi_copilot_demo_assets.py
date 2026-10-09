"""The workflow demo shows its registered run's own figure, not a redrawn one."""

from __future__ import annotations

import base64
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static"


def test_demo_main_figure_is_the_registered_dossier_figure() -> None:
    """The demo answer quotes the registered run's numbers, so its main figure
    must be that run's own rendering: byte-identical to the first figure the
    sealed reviewer dossier embeds, never a re-plot or a generated image."""

    dossier = (STATIC / "assets/demo/system-validation-report.html").read_text(encoding="utf-8")
    figures = re.findall(r'<figure><img src="data:image/png;base64,([A-Za-z0-9+/=]+)"', dossier)
    assert figures, "The dossier embeds its figures"
    figure = (STATIC / "assets/demo/sofa2-phenotype-mortality.png").read_bytes()
    assert figure == base64.b64decode(figures[0])
    demo = (STATIC / "js/screens-guided-pi-demo.js").read_text(encoding="utf-8")
    assert 'src="assets/demo/sofa2-phenotype-mortality.png?v=' in demo
