from __future__ import annotations

import json
from pathlib import Path

from tests.support.node import run_node


def test_literature_reader_deduplicates_sources_across_semantic_decisions() -> None:
    renderer = Path(
        "src/easyicu/webserver/static/js/screens-guided-pi-literature.js"
    ).resolve()
    modules = Path(
        "src/easyicu/webserver/static/js/screens-guided-pi-modules.js"
    ).resolve()
    payload = {
        "research_question": "Does exposure relate to mortality?",
        "direct_comparator_count": 1,
        "direct_comparator_keys": [],
        "citations": [],
        "search": {"search_conducted": True, "prisma": {"identified": 2, "screened": 2}},
        "step_citation_map": [
            {
                "intent": "adjusted model",
                "citation_bindings": [
                    {
                        "key": "model-paper",
                        "title": "Unique model paper",
                        "design_elements": ["model"],
                    },
                    {
                        "key": "strobe",
                        "title": "Unique reporting guide",
                        "design_elements": ["reporting"],
                    },
                ],
            },
            {
                "intent": "covariate adjustment model",
                "citation_bindings": [
                    {
                        "key": "model-paper",
                        "title": "Unique model paper",
                        "design_elements": ["model"],
                    },
                    {
                        "key": "strobe",
                        "title": "Unique reporting guide",
                        "design_elements": ["reporting"],
                    },
                ],
            },
        ],
    }
    script = f"""
global.window = {{
  EU_HTML: {{ esc: value => String(value ?? '').replace(/[&<>\"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}}[ch])) }},
  EU_LANG: 'en'
}};
require({json.dumps(str(modules))});
require({json.dumps(str(renderer))});
const html = window.EasyICU.guidedPi.require('literature').renderArtifact({json.dumps(payload)}, {{}});
process.stdout.write(html);
"""
    html = run_node("node", script, check=True).stdout

    assert html.count("Unique model paper") == 1
    assert html.count("Unique reporting guide") == 1
    assert "2 source(s)" in html
    assert "1 decision(s)" not in html


def test_one_card_keeps_distinct_binding_applications_and_differences() -> None:
    first = {"key": "paper", "title": "One methodological source",
             "design_elements": ["time_zero"], "application": "Anchor follow-up at entry.",
             "divergence": "Source follow-up starts later."}
    second = {**first, "application": "Check the alternate entry definition.",
              "divergence": "Source uses a different entry rule."}
    payload = {"step_citation_map": [
        {"intent": intent, "citation_bindings": [binding]}
        for intent, binding in [("Describe the cohort", first), ("Fit a model", first),
                                ("Sensitivity analysis", second)]
    ]}
    js_root = Path("src/easyicu/webserver/static/js").resolve()
    script = f"""
global.window = {{ EU_LANG: 'zh' }};
const root = {json.dumps(str(js_root))};
require(root + '/html-escape.js');
require(root + '/screens-guided-pi-modules.js');
require(root + '/screens-guided-pi-literature.js');
process.stdout.write(window.EasyICU.guidedPi.require('literature').renderArtifact({json.dumps(payload)}));
"""
    html = run_node("node", script, check=True).stdout
    assert html.count("One methodological source") == 1
    assert html.count("Anchor follow-up at entry.") == 1
    assert html.count("Check the alternate entry definition.") == 1
    assert html.count("Source follow-up starts later.") == 1
    assert html.count("Source uses a different entry rule.") == 1
