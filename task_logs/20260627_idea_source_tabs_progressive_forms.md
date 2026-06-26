# 2026-06-27 Idea source tabs progressive forms

## Scope

User reported that Guided Copilot idea-mining source tabs reused the same large form: `手动想法`, `文章链接`, `PDF 文件`, `文献库文件夹`, and `前沿主题` should each show a source-specific interaction. The same contract must hold in the classic `#ideas` view.

## Changes

- Guided Copilot `#guided`
  - `src/easyicu/webserver/static/js/screens-guided.js`
  - Replaced the generic idea source editor with per-source forms:
    - manual: candidate research question + motivation/rationale only
    - article URL: URL, DOI/PMID, title, journal, year, article insight, source opt-in
    - PDF: local PDF picker, reading note, title/DOI, bounded excerpt override
    - literature folder: local folder picker/scan flow and review scope
    - frontier topic: topic/journal/year scope, opt-in, discovery panel
  - Source switching now resets network opt-in so opt-in does not leak across source types.

- Classic Idea Mining `#ideas`
  - `src/easyicu/webserver/static/js/screens-ideas.js`
  - Added the same per-source form model and wired PDF picker / literature-folder scan controls.
  - `src/easyicu/webserver/static/css/ideas.css`
  - Added route-owned styles for source picker/folder rows.

- Cache bust / tests
  - `src/easyicu/webserver/static/index.html`
  - `tests/test_webserver_static_routes.py`
  - Bumped Guided/Ideas assets and replaced old "single generic form" assertions with source-specific tab assertions.

## Verification

- `node --check src/easyicu/webserver/static/js/screens-guided.js`
- `node --check src/easyicu/webserver/static/js/screens-ideas.js`
- `python -m pytest tests/test_webserver_static_routes.py::test_native_guided_copilot_runs_extraction_inline_and_answers_catalog_questions tests/test_webserver_static_routes.py::test_native_idea_mining_is_first_class_route_and_backend_wired -q`
  - 2 passed
- `python -m pytest tests/test_webserver_idea_sources.py -q`
  - 9 passed
- `git diff --check`
  - clean
- Browser QA on `http://127.0.0.1:8765/?_v=20260627-source-tabs#guided`
  - `手动想法`: shows candidate question + why worth testing
  - `文章链接`: shows URL, DOI/PMID, title, journal, year, article insight, opt-in
  - `PDF 文件`: shows PDF picker, reading note, title/DOI, bounded excerpt
  - `文献库文件夹`: shows folder browse/scan and review scope
  - `前沿主题`: shows topic/journal/year scope and discovery entry
- Browser QA on `http://127.0.0.1:8765/?_v=20260627-source-tabs#ideas`
  - The same five source modes render source-specific content in the classic view.
- Console logs only showed normal EasyICU hydration messages.
