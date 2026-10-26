"""The demo's result report lays out the registered result and adds no finding.

``tools/build_demo_study_report.py`` writes one page per language from the
trace record and the figures redrawn from the run's registered tables. These
checks hold each page to that record: its numbers, its figures' digests, the
plan's claim boundary, and the aggregate-only boundary of the demo assets.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "src/easyicu/webserver/static/assets/demo"
TOOL = ROOT / "tools/build_demo_study_report.py"


def _tool():
    spec = importlib.util.spec_from_file_location("build_demo_study_report", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _trace() -> dict:
    return json.loads((ASSETS / "sofa2-run-trace.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_committed_page_is_what_the_trace_record_builds(lang: str) -> None:
    tool = _tool()
    assert (ASSETS / f"sofa2-study-report.{lang}.html").read_text(encoding="utf-8") == tool.page(_trace(), lang)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_number_and_figure_is_the_registered_one(lang: str) -> None:
    trace = _trace()
    html = (ASSETS / f"sofa2-study-report.{lang}.html").read_text(encoding="utf-8")
    for row in trace["evidence"]["result_rows"]:
        cells = [f"{row['stays']:,}", f"{row['share_pct']:.1f}", f"{row['deaths']:,}", f"{row['rate_pct']:.1f}"]
        assert "".join(f"<td>{cell}</td>" for cell in cells) in html
    figures = [trace["evidence"]["rerender"], *trace["evidence"]["rerender_supporting"].values()]
    shown = re.findall(r'<img src="([^"?]+)\?v=([0-9a-f]{8})"', html)
    assert sorted(shown) == sorted((record["png"], record["png_sha256"][:8]) for record in figures)
    # The page names the result table and the run's own figure by their digests.
    for evidence_id in trace["evidence"]["chain"][:3]:
        assert trace["evidence"]["records"][evidence_id]["sha256"][:12] in html


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_page_keeps_the_plans_boundary_and_says_it_is_not_a_manuscript(lang: str) -> None:
    html = (ASSETS / f"sofa2-study-report.{lang}.html").read_text(encoding="utf-8")
    assert 'content="descriptive_result_report"' in html
    assert ("本页不是论文" if lang == "zh" else "This page is not a manuscript") in html
    assert ("仅描述性" if lang == "zh" else "Descriptive only") in html
    assert ("暴露时间晚于随访起点" if lang == "zh" else "Exposure assessed after the start of follow-up") in html
    for forbidden in ("p &lt;", "p <", "95% CI", "odds ratio", "hazard ratio"):
        assert forbidden not in html
    assert not re.search(r"/(Users|Volumes|home|private)/", html)


def test_the_builder_refuses_a_result_it_was_not_written_for() -> None:
    tool = _tool()
    trace = _trace()
    trace["evidence"]["result_rows"][0]["interval_method"] = "wilson"
    with pytest.raises(SystemExit, match="counts-only"):
        tool.page(trace, "en")
