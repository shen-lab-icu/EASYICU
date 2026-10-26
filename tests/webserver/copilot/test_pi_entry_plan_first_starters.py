"""Entry starters ask for a question, not for a questionnaire.

Design choices belong to the research plan the researcher reviews once; a
starter must not tell the conversation to collect them first.  The official
demo card fills the composer with one runnable question that names the demo,
so the conversation can offer that exact source.
"""

import json
import re
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


# What each example asks of its database, as the capability owner reads it.
_EXAMPLE_REQUIREMENTS = {
    "example-quick-lactate": ("hirid", "descriptive_epidemiology", {"concepts": ("lact", "sex")}),
    "example-ventilation-table-one": ("mimic", "descriptive_epidemiology", {"concepts": ("mech_vent", "age", "sex", "lact", "map", "crea")}),
    "example-measurement-audit": ("hirid", "data_quality_audit", {"concepts": ("lact", "crea", "po2", "map", "hr")}),
    "example-lactate-death": ("eicu", "association_study", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex", "sofa")}),
    "example-sofa-trend": ("eicu", "ordinal_dose_response", {"outcome": "hospital_death", "concepts": ("sofa", "age", "sex")}),
    "example-lactate-dose-response": ("sic", "association_study", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex")}),
    "example-landmark-lactate": ("miiv", "association_study", {"outcome": "mort_28d", "concepts": ("lact", "age", "sex", "map", "crea")}),
    "example-aki-survival": ("miiv", "survival", {"outcome": "mort_28d", "concepts": ("aki_stage_reference", "age", "sex"),
                                                  "censoring_unit": "hospitalization"}),
    "example-ventilation-rmst": ("sic", "survival", {"outcome": "mort_28d", "concepts": ("mech_vent", "age", "sex"),
                                                     "censoring_unit": "hospitalization"}),
    "example-ventilation-time-varying": ("mimic", "survival", {"outcome": "icu_death", "concepts": ("mech_vent", "age", "sex"),
                                                               "censoring_unit": "icu_stay"}),
    "example-creatinine-competing": ("sic", "survival", {"outcome": "icu_death", "concepts": ("crea", "age", "sex"),
                                                         "censoring_unit": "icu_stay"}),
    "example-predict-death": ("sic", "prediction_model", {"outcome": "hospital_death", "concepts": ("age", "sex", "map", "hr", "lact", "crea")}),
    "example-dynamic-prediction": ("aumc", "dynamic_prediction", {"outcome": "icu_death", "concepts": ("map", "hr", "lact", "crea", "age")}),
    "example-early-norepinephrine": ("miiv", "causal_inference", {"outcome": "mort_28d", "concepts": ("age", "sex", "lact", "map"),
                                                                  "treatment_concepts": ("norepi_rate",)}),
    "example-first-day-phenotypes": ("aumc", "trajectory_clustering", {"outcome": "icu_death",
                                                                       "concepts": ("map", "hr", "lact", "crea", "glu", "plt")}),
    "example-pressure-trajectories": ("eicu", "trajectory_clustering", {"concepts": ("map", "age", "sex", "lact")}),
    "example-norepinephrine-use": ("hirid", "descriptive_epidemiology", {"concepts": ("norepi_rate", "mech_vent")}),
}
# A replication names its first database and the ones it repeats on; each
# must answer the same study exactly.
_REPLICATION_EXAMPLES = {
    "example-replicate-lactate": tuple(
        (database, "cross_database_replication", {"outcome": "hospital_death", "concepts": ("lact", "age", "sex")})
        for database in ("miiv", "eicu", "sic")
    ),
}
_DATABASE_KEYS = {"MIMIC-IV": "miiv", "MIMIC-III": "mimic", "eICU": "eicu", "HiRID": "hirid",
                  "AmsterdamUMCdb": "aumc", "SICdb": "sic"}


def _example_cards(html):
    return {
        card.split('"', 2)[1]: re.search(r'class="gpi-starter-category">([^<]+)</span>', card).group(1)
        for card in html.split("data-gpi-starter-example=")[1:]
    }


def test_the_home_offers_one_question_per_method_across_the_six_databases():
    html = _render("zh")
    cards = _example_cards(html)
    assert set(cards) == set(_EXAMPLE_REQUIREMENTS) | set(_REPLICATION_EXAMPLES)
    replication = {example: cards.pop(example) for example in _REPLICATION_EXAMPLES}
    assert replication["example-replicate-lactate"] == "MIMIC-IV、eICU、SICdb · 多库验证"
    # Each card names its database and study kind, and what a run of it produces.
    assert html.count('class="gpi-starter-gets"') == len(cards) + len(replication)
    databases = [label.split(" · ")[0] for label in cards.values()]
    kinds = [label.split(" · ")[1] for label in cards.values()]
    assert set(databases) == set(_DATABASE_KEYS)
    # Every database offers at least two questions, and no method repeats.
    assert min(databases.count(name) for name in _DATABASE_KEYS) >= 2
    assert len(set(kinds)) == len(kinds)
    for example, label in cards.items():
        assert _DATABASE_KEYS[label.split(" · ")[0]] == _EXAMPLE_REQUIREMENTS[example][0], example
    assert "在 SICdb 中，入 ICU 24 小时内的最高乳酸与院内死亡之间" in html
    # A card fills the composer; it neither sends nor names a method template.
    assert html.count('data-gpi-starter-method=""') == len(cards) + len(replication) + 1
    assert "data-gpi-starter-search" not in html and "data-gpi-starter-shuffle" not in html
    # No before-you-start block: per-database status lives on the Data map.
    assert "gpi-entry-notes" not in html and "使用前须知" not in html
    assert "尚未验收" not in html and "口径不同" not in html
    en = _render("en")
    assert "Before you start" not in en and "Use this question" in en


def test_every_example_asks_only_what_its_database_records():
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study

    rows = [(example, *row) for example, row in _EXAMPLE_REQUIREMENTS.items()]
    rows += [(example, *row) for example, targets in _REPLICATION_EXAMPLES.items() for row in targets]
    for example, database, family, fields in rows:
        profile = research_capability_profile(database)
        decision = assess_study(profile, StudyRequirements(family=family, **fields))
        # Exact: no substituted endpoint, unit, timing or threshold to approve.
        assert decision.status == "exact", (example, decision.substitutions, decision.missing)
        for concept in (*fields.get("concepts", ()), *fields.get("treatment_concepts", ())):
            availability = profile.concept(concept)
            assert availability.mapped and availability.derivation_status in (None, "full"), (example, concept)


def test_the_demo_card_is_offered_only_without_a_registered_full_source():
    assert 'class="gpi-starter-demo"' in _render_with("zh", {})
    assert 'class="gpi-starter-demo"' in _render_with("zh", {"registered_sources": {"by_database": {"eicu": 0}}})
    assert 'class="gpi-starter-demo"' not in _render_with("zh", {"registered_sources": {"by_database": {"eicu": 1}}})
