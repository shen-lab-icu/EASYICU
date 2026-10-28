"""Workflow script order, route ownership and metadata-only UI contracts."""

from pathlib import Path
import re


STATIC = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static"


def test_workflow_dependencies_and_changed_cache_tokens_are_explicit():
    html = (STATIC / "index.html").read_text()
    before = ["html-escape.js", "screens-agent-reader-vocab.js", "screens-skills-workflows.js", "screens-skills.js"]
    assert [html.index(f"js/{file}?") for file in before] == sorted(html.index(f"js/{file}?") for file in before)
    assert html.index("screens-guided-pi-skill-attach.js?") < html.index("screens-guided-pi-study-workspace.js?")
    assert html.index("screens-guided-pi-starters.js?") < html.index("screens-guided-pi.js?")
    assert 'js/screens-skills-workflows.js?v=20261028-cust1"' in html
    assert 'js/desktop-module-shell.js?v=20261028-cust1"' in html
    assert 'js/screens-agent-reader-vocab.js?v=20261028-voc1"' in html
    assert 'js/screens-guided-pi-study-workspace.js?v=20261028-task1"' in html
    for file in ["screens-guided-pi-starters.js", "screens-guided-pi-session-view.js"]:
        assert f'js/{file}?v=20261028-hub2"' in html
    assert 'js/screens-guided-pi.js?v=20261028-task1"' in html
    assert 'js/screens-guided-pi-skill-attach.js?v=20261028-chip1"' in html
    assert 'js/screens-skills.js?v=20261028-cust1"' in html
    assert 'css/skills-workflows.css?v=20261028-cust1"' in html


def test_workflow_navigation_uses_the_skills_route_and_existing_attachment_intent():
    shell = (STATIC / "js/desktop-module-shell.js").read_text()
    # Workflows open as a tab inside Skills; there is no separate route or rail item.
    assert 'data-sk-mode="workflows"' in shell
    assert 'data-nav="workflows"' not in shell
    assert 'data-nav="skills" data-sk-mode="workflows"' not in shell
    owner = (STATIC / "js/screens-skills-workflows.js").read_text()
    assert '/api/capabilities/research-workflows' in owner
    assert 'easyicu.skillHub.method' in owner
    assert "location.hash = '#guided'" in owner
    assert not re.search(r"(?:execute|launch|runPi|startRun)\s*\(", owner)
    hub = (STATIC / "js/screens-skills.js").read_text()
    assert "workflows().start(skill, workflows().find(skill.name))" in hub
    assert "overviewMarkdown(main.content)" in hub


def test_the_request_survives_until_the_composer_consumes_it():
    study = (STATIC / "js/screens-guided-pi-study-workspace.js").read_text()
    assert "removeItem('easyicu.skillHub.question')" not in study
    assert "source: 'builtin', id: String(methodIntent.id)" in study
    composer = (STATIC / "js/screens-guided-pi.js").read_text()
    assert "await attachOwner.ensure(expectedProjectId, session, hubAttachment.source, hubAttachment.id)" in composer
    assert "state.draft = suggestion" in composer
    attach = (STATIC / "js/screens-guided-pi-skill-attach.js").read_text()
    assert "attached_skills: rows" in attach


def test_graph_scroll_stays_inside_its_owner_without_a_library():
    css = (STATIC / "css/skills-workflows.css").read_text()
    assert ".eusk-workflow-graph{max-width:100%;overflow-x:auto" in css
    assert "minmax(0,1fr)" in css and "@media(max-width:700px)" in css
    assert ".eusk-workflow-database:focus .eusk-workflow-reason{display:block}" in css
    owner = (STATIC / "js/screens-skills-workflows.js").read_text()
    assert '<svg width=' in owner and 'viewBox=' in owner
    assert 'tabindex="0"' in owner
    assert not re.search(r"https?://|d3\.|mermaid|ReactFlow", owner)
