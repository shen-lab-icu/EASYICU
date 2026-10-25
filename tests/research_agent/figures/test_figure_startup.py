"""Figure helpers stay cheap in each fresh desktop executor interpreter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_style_import_does_not_initialize_unused_family_renderers(tmp_path):
    code = """import sys
from easyicu.research_agent import figures
assert not hasattr(figures, 'unknown_renderer')
from easyicu.research_agent.figures.publication import apply_publication_style
assert callable(apply_publication_style)
for name in ('association', 'causal', 'phenotype', 'prediction', 'survival', 'base'):
    assert 'easyicu.research_agent.figures.' + name not in sys.modules, name
assert 'easyicu.research_agent.authority.evidence_store' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert figures.render_family_figure(
    'descriptive', context=None, plan=None, evidence=None, run_dir=None
) is None
assert figures.FAMILY_RENDERERS
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_figure_exports_keep_the_same_callables_and_dispatch(monkeypatch, tmp_path):
    from easyicu.research_agent import figures
    from easyicu.research_agent.figures.association import (
        render_continuous_association_figure,
    )
    from easyicu.research_agent.figures.base import RenderedFigure
    from easyicu.research_agent.figures.causal import render_causal_figure
    from easyicu.research_agent.figures.phenotype import render_phenotype_figure
    from easyicu.research_agent.figures.prediction import render_prediction_figure
    from easyicu.research_agent.figures.survival import render_survival_figure

    expected = {
        "association": render_continuous_association_figure,
        "causal_emulation": render_causal_figure,
        "phenotyping": render_phenotype_figure,
        "prediction": render_prediction_figure,
        "time_to_event": render_survival_figure,
    }
    assert figures.FAMILY_RENDERERS == expected
    assert figures.FAMILY_RENDERERS is figures.FAMILY_RENDERERS
    assert figures.RenderedFigure is RenderedFigure
    for renderer in expected.values():
        assert getattr(figures, renderer.__name__) is renderer
    exported = {}
    exec("from easyicu.research_agent.figures import *", exported)
    for name in figures.__all__:
        assert exported[name] is getattr(figures, name)
    received = []
    sentinel = object()

    def render(**kwargs):
        received.append(kwargs)
        return sentinel

    monkeypatch.setattr(figures, "FAMILY_RENDERERS", {"prediction": render})
    kwargs = dict(context=None, plan=None, evidence=None, run_dir=tmp_path)
    assert figures.render_family_figure("prediction", **kwargs) is sentinel
    assert received == [kwargs]
    assert figures.render_family_figure("descriptive", **kwargs) is None


_FIGURE_SCRIPT = """import logging, os
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from easyicu.research_agent.figures.publication import apply_publication_style
import matplotlib.pyplot as plt
apply_publication_style()
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot([1, 2, 3], [2, 1, 4])
ax.set_xlabel('Synthetic measurement')
fig.savefig(Path(os.environ['STEP_OUT_DIR']) / 'synthetic.png', dpi=72)
plt.close(fig)
"""


def test_host_figure_bytes_match_with_cold_warm_and_stale_font_cache(tmp_path):
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    cohort = tmp_path / "synthetic.parquet"
    pd.DataFrame({"measurement": [1, 2, 3]}).to_parquet(cohort)
    runner = HostRunner(workdir=tmp_path, cohort_parquet=cohort)

    def render(step_id):
        result = runner.run(
            step_id=step_id, code=_FIGURE_SCRIPT,
            code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR,
        )
        assert result.succeeded, result.stderr
        return (result.out_dir / "synthetic.png").read_bytes(), result.stderr

    cold, cold_log = render("cold")
    cache, = (tmp_path / ".matplotlib").glob("fontlist-v*.json")
    original = json.loads(cache.read_text())
    warm, warm_log = render("warm")
    assert "generated new fontManager" in cold_log
    assert "generated new fontManager" not in warm_log
    assert cold == warm

    # A cache from a different FontManager format must be rebuilt by Matplotlib.
    cache.write_text(json.dumps({**original, "_version": -1}), encoding="utf-8")
    stale, stale_log = render("stale")
    assert "generated new fontManager" in stale_log
    assert json.loads(cache.read_text())["_version"] == original["_version"]
    assert stale == cold
