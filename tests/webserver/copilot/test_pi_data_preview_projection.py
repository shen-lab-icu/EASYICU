"""Executable checks of the result-blind, route-owned preview projection."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def _render(payload: dict) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    owner = (Path(__file__).parents[3]
             / "src/easyicu/webserver/static/js/screens-guided-pi-workbench-preview.js").read_text()
    script = f"""
      let api;
      global.window = {{ EU_LANG: 'zh', EU_HTML: {{ esc: value => String(value)
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') }},
        EasyICU: {{ guidedPi: {{ declare(_name, value) {{ api = value; }} }} }} }};
      eval({json.dumps(owner)});
      const host = {{ innerHTML: '', querySelector() {{ return null; }} }};
      api.mount(host, {json.dumps(payload)});
      console.log(host.innerHTML);
    """
    return subprocess.run([node, "--eval", script], check=True, capture_output=True,
                          text=True).stdout


@pytest.mark.parametrize("value", [None, "", False, []])
def test_unknown_coverage_does_not_turn_into_zero(value):
    html = _render({"review_stage": "post_plan", "concepts": [{
        "study_role": "plan_input", "concept_id": "source_time",
        "availability_status": "semantic_review_required",
        "evaluable_count": value, "denominator_count": 100,
        "physical_coverage_pct": value,
    }]})

    assert "0.0%" not in html
    assert "不展示" in html


def test_summary_and_chart_do_not_count_other_materialized_columns():
    html = _render({"review_stage": "post_plan", "plan_input_count": 2, "concepts": [
        {"study_role": "plan_input", "concept_id": "age", "availability_status": "ready",
         "evaluable_count": 100, "denominator_count": 100},
        {"study_role": "outcome", "concept_id": "outcome", "availability_status": "ready",
         "evaluable_count": 100, "denominator_count": 100},
        {"study_role": "other_materialized_column", "concept_id": "unused_raw",
         "availability_status": "ready", "evaluable_count": 1, "denominator_count": 100},
    ]})

    assert "<strong>2/2</strong>" in html
    chart = html.split('class="gpi-wb-coverage-chart"', 1)[1].split(
        'class="gpi-wb-quality"', 1,
    )[0].split('class="gpi-wb-modules"', 1)[0].split('class="gpi-wb-section"', 1)[0]
    assert "unused_raw" not in chart
    assert "非空覆盖不等于临床定义已验证" in html
    assert "拟合模型" not in html


def test_tiny_missingness_does_not_show_complete_coverage():
    html = _render({"review_stage": "post_plan", "concepts": [{
        "study_role": "plan_input", "concept_id": "clinical_value",
        "availability_status": "partial", "evaluable_count": 9999,
        "denominator_count": 10000,
    }]})

    assert "&gt;99.9%" in html
    assert ">100.0%</" not in html


def test_source_prohibition_is_visible_even_if_other_data_are_ready():
    html = _render({"review_stage": "post_plan", "concepts": [{
        "study_role": "other_materialized_column", "concept_id": "source_flag",
        "availability_status": "structurally_unavailable",
        "reason_code": "plan_bound_source_structurally_unavailable",
        "evaluable_count": None, "denominator_count": None,
    }]})

    assert "来源不支持" in html
    assert "列非空不代表可用" in html
    assert "0.0%" not in html
