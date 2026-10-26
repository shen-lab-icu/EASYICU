"""Entry starters ask for a question, not for a questionnaire.

Design choices belong to the research plan the researcher reviews once; a
starter must not tell the conversation to collect them first.  The official
demo card fills the composer with one runnable question that names the demo,
so the conversation can offer that exact source.
"""

import json
import shutil

import pytest

from easyicu.research_agent.method_skills import method_skill_catalog
from tests.support.node import run_node
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
    return run_node(node, script, check=True).stdout


def test_the_official_demo_card_offers_one_runnable_named_demo_question():
    html = _render("zh")

    assert html.count('class="gpi-starter-demo"') == 1
    assert "用官方 Demo 数据试一个问题" in html
    assert 'data-gpi-starter-method=""' in html
    assert "用 eICU 官方 Demo 数据（eICU Collaborative Research Database Demo v2.0.1）研究" in html
    # It sits outside the workflow grid that search and shuffle re-render.
    demo_at = html.index('class="gpi-starter-demo"')
    grid_at = html.index('class="gpi-starter-actions')
    assert demo_at < grid_at
    assert "Using the official eICU demo" in _render("en")


def _render_with(lang, capabilities):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = {{EU_LANG: {json.dumps(lang)}, EU_CAPABILITIES: {json.dumps(capabilities)}}};
      eval({json.dumps(_read('js/html-escape.js'))});
      eval({json.dumps(_read('js/screens-guided-pi-starters.js'))});
      const tr = (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en;
      process.stdout.write(window.EU_GUIDED_PI_STARTERS.render({{tr, composer: ''}}));
    """
    return run_node(node, script, check=True).stdout


def test_the_home_offers_clinician_questions_inside_v1_scope():
    html = _render("zh")
    assert html.count('data-gpi-starter-example=') == 5
    # Each card names its database and what a run of it produces.
    assert html.count('class="gpi-starter-gets"') == 5
    assert html.count(">eICU<") == 3 and html.count(">MIMIC-IV<") == 2
    assert "在 eICU 中，按入 ICU 第一天的血糖分为低血糖、仅高血糖和血糖正常三组" in html
    # eICU questions stay within eICU's v1 types: no survival or causal card names eICU.
    cards = html.split("data-gpi-starter-example=")[1:]
    eicu_cards = [card for card in cards if ">eICU<" in card]
    assert len(eicu_cards) == 3
    assert not any("生存" in card or "降低" in card for card in eicu_cards)
    # A card fills the composer; it neither sends nor names a method template.
    assert html.count('data-gpi-starter-method=""') == 6
    assert "data-gpi-starter-search" not in html and "data-gpi-starter-shuffle" not in html
    # The notes name all six databases and claim no real-data acceptance yet.
    assert "使用前须知" in html and "AmsterdamUMCdb、SICdb 上的" in html
    assert "六个库都还没有通过正常入口的真实数据运行验收" in html and "已验收" not in html
    en = _render("en")
    assert "Before you start" in en and "Use this question" in en


def test_the_demo_card_is_offered_only_without_a_registered_full_source():
    assert 'class="gpi-starter-demo"' in _render_with("zh", {})
    assert 'class="gpi-starter-demo"' in _render_with("zh", {"registered_sources": {"by_database": {"eicu": 0}}})
    assert 'class="gpi-starter-demo"' not in _render_with("zh", {"registered_sources": {"by_database": {"eicu": 1}}})
