from __future__ import annotations

import json
from pathlib import Path
import subprocess


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
    html = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert html.count("Unique model paper") == 1
    assert html.count("Unique reporting guide") == 1
    assert "2 source(s)" in html
    assert "1 decision(s)" not in html
