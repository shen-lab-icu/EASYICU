"""Entry starters ask for a question, not for a questionnaire.

Design choices belong to the research plan the researcher reviews once; a
starter must not tell the conversation to collect them first.  The official
demo card fills the composer with one runnable question that names the demo,
so the conversation can offer that exact source.
"""

import json
import shutil
import subprocess

import pytest

from easyicu.research_agent.method_skills import method_skill_catalog
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def test_every_method_starter_leaves_design_choices_to_the_plan():
    catalog = method_skill_catalog(enabled=True)
    rows = catalog["items"] + catalog["components"]
    assert rows
    for row in rows:
        assert "先确认" not in row["prompt_zh"] and "First confirm" not in row["prompt"], row["id"]
        assert row["prompt_zh"].endswith("我的研究问题是："), row["id"]
        assert row["prompt"].endswith("My research question: "), row["id"]


def _render(lang):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = {{EU_LANG: {json.dumps(lang)}}};
      eval({json.dumps(_read('js/html-escape.js'))});
      eval({json.dumps(_read('js/screens-guided-pi-starters.js'))});
      const tr = (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en;
      process.stdout.write(window.EU_GUIDED_PI_STARTERS.render({{tr, composer: '<textarea></textarea>'}}));
    """
    return subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout


def test_the_official_demo_card_offers_one_runnable_named_demo_question():
    html = _render("zh")

    assert html.count('class="gpi-starter-demo"') == 1
    assert "用官方 Demo 数据试一个问题" in html
    assert 'data-gpi-starter-method=""' in html
    assert "用 eICU 官方 Demo 数据（eICU Collaborative Research Database Demo v2.0.1）研究" in html
    # It sits outside the workflow grid that search and shuffle re-render.
    demo_at = html.index('class="gpi-starter-demo"')
    grid_at = html.index('class="gpi-starter-actions"')
    assert demo_at < grid_at
    assert "Using the official eICU demo" in _render("en")
