"""Static contracts for the native WebApp home and navigation flow."""

from pathlib import Path


STATIC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easyicu"
    / "webserver"
    / "static"
)


def _asset(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


def _home_owner() -> str:
    source = _asset("js", "screens-entry.js")
    return source.split("S.entry =", 1)[1]


def test_entry_owner_is_loaded_before_extraction_without_duplicate_route_owner() -> None:
    index = _asset("index.html")
    entry = _asset("js", "screens-entry.js")
    extraction = _asset("js", "screens-extraction.js")

    assert index.index("js/screens-entry.js?") < index.index(
        "js/screens-extraction.js?"
    )
    assert "S.entry =" in entry
    assert "S.entry =" not in extraction


def test_home_leads_with_three_user_intents_and_keeps_demo_secondary() -> None:
    home = _home_owner()

    assert "I have a paper or topic" in home
    assert "I have a clear research question" in home
    assert "I have local ICU data" in home
    assert "Mine a feasible question in Idea Mining" in home
    assert "Validate and extract analysis-ready tables" in home
    assert "data-home-new-study" in home

    # Demo remains available, but it follows the real starting choices.
    assert home.index('<div class="home-split">') < home.index(
        '<div class="entry-firsttime" id="firstTimeNudge"'
    )


def test_browser_title_uses_the_product_name_not_an_internal_polish_label() -> None:
    index = _asset("index.html")

    assert "<title>EasyICU — ICU Research Workspace</title>" in index
    assert "Polished Workspace" not in index


def test_home_classic_entry_names_the_real_patient_destination() -> None:
    home = _home_owner()

    assert "t('Patient Review', '患者审阅')" in home
    assert "Review patients, tables, and trends from an export" in home
    assert "t('Data Workspace', '数据工作台')" not in home


def test_home_question_survives_language_rerender() -> None:
    source = _asset("js", "screens-entry.js")
    home = _home_owner()

    assert "let homeQuestionDraft = '';" in source
    assert "${escHtml(homeQuestionDraft)}</textarea>" in home
    assert "input.addEventListener('input', () => { homeQuestionDraft = input.value; });" in home


def test_new_home_starts_create_fresh_study_contexts() -> None:
    source = _asset("js", "screens-entry.js")

    assert "typeof store.startNew !== 'function'" in source
    assert "store.startNew(Object.assign({" in source
    assert "last_route: route" in source
    assert "current_stage: 'study_setup'" in source
    assert "startHomeStudy('guided', { question, analysis_goal: analysisGoal })" in source
    assert "data-home-new-study=\"${newStudy}\"" in source
    assert "target === 'ideas' ? { purpose: 'idea_mining' } : {}" in source
    assert "target === 'extraction' && window.setDataMode" in source
    assert "window.setDataMode('real', { force: true })" in source


def test_resume_prefers_allowlisted_study_context_then_maps_legacy_branches() -> None:
    source = _asset("js", "screens-entry.js")
    resume = source.split("// Resume banner", 1)[1].split("setTimeout", 1)[0]

    assert (
        "new Set(['guided', 'ideas', 'extraction', 'patient', 'cohort', "
        "'crossdb', 'agent'])"
    ) in source
    assert (
        "Object.freeze({ predict: 'patient', crossdb: 'crossdb', "
        "quality: 'cohort' })"
    ) in source
    assert "RESUME_ROUTE_ALLOWLIST.has(activeContext.last_route)" in source
    assert "const resumeRoute = contextRoute || legacyRoute || null" in source
    assert "location.hash = '#' + resumeRoute" in source
    assert "startHomeStudy(" not in resume


def test_sidebar_follows_research_lifecycle_and_agent_comes_last() -> None:
    source = _asset("js", "app.js")
    sidebar = source.split("function sidebar()", 1)[1].split("function topbar()", 1)[0]

    discovery = sidebar.index("Discovery & Plan")
    guided = sidebar.index('data-nav="guided"')
    ideas = sidebar.index('data-nav="ideas"')
    data_review = sidebar.index("Data & Review")
    workspace = sidebar.index('class="wsnav"')
    analysis = sidebar.index("Analysis & Evidence")
    agent = sidebar.index('class="cp-entry agent-entry')

    assert discovery < guided < ideas < data_review < workspace < analysis < agent


def test_data_workspace_breadcrumb_is_a_non_clickable_group_label() -> None:
    source = _asset("js", "app.js")

    assert "'Data Workspace': 'patient'" not in source
    assert "CRUMB_NAV" not in source
    assert 'else node = `<span class="mid">${label}</span>`;' in source
