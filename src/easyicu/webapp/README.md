# `easyicu.webapp`

Deprecated legacy Streamlit WebApp package.

The maintained no-code Web UI is now the native FastAPI implementation under
`easyicu.webserver`, launched with `easyicu-webapp` or the root
`start_easyicu.*` launchers. This package remains only for archive forensics,
compatibility shims, and tests that have not yet been retired.

If the legacy UI must be inspected, install `easyicu[webapp-legacy]` and run
`easyicu-webapp-legacy`. It boots `app.py` under Streamlit on port 8501, but
visual parity and route-specific CSS polish are no longer maintained.

## Legacy invariants this layer must preserve while it exists

These are load-bearing — breaking them changes the safety posture of the
whole tool, so they are enforced in code and called out here:

- **AI features are opt-in and start disabled.** The sidebar toggle is
  the canonical gate. **Any** code path that may issue an *external* LLM
  call (real OpenAI / OpenRouter / Anthropic / …) MUST go through
  `easyicu.webapp.ai_optin.enforce_external_llm_opt_in(llm_choice)`
  before instantiating its client. `MockLLMClient` (offline,
  deterministic) is intentionally exempt — nothing leaves the machine.
  Do **not** add a per-page API-key shortcut that bypasses this gate.
- **Human confirmation stays in the loop.** Cohort, feature, conversion,
  and export actions require an explicit user click. Do not introduce
  silent auto-runs.
- **Prepared data is the shared contract.** The web app converts raw
  CSV / CSV.GZ / tar.gz to the prepared Parquet layout via the single
  converter engine (`DataConverter.convert_all()`), then reads the
  prepared directory — the same contract the Python API uses. It never
  extracts features straight from raw files.

## Top-level tabs

The main tab bar is defined in `page_registry.build_main_page_registry()`:

| Tab | Key | Backing module(s) |
|-----|-----|-------------------|
| Tutorial / Workflow help | `tutorial` | `home_page.py`, `workflow_figure.py` |
| Data Extraction & Quick Visualization | `quick_viz` | `quick_visualization_page.py`, `home_extract_page.py`, `patient_page.py`, `timeseries_page.py`, `quality_page.py` |
| Cohort Statistics | `cohort` | `cohort_dashboard_page.py`, `cohort_group_page.py`, `cohort_severity_page.py`, `sofa_reclassification.py` |
| Cross-DB Benchmark | `cross_db` | `cohort_multidb_page.py` |
| Research Agent | `research_agent` | `research_agent.py`, `agent_workbench.py` |

Data Visualization (Patient Review / Cohort Statistics / Cross-DB) is
also grouped in the sidebar; the registry keeps those pages routable as
top-level keys.

## How the modules are organized

55 modules, but they fall into a few clear groups:

- **Bootstrap & shell** — `__main__.py` (CLI: `run` / `stop` / `status`),
  `app.py` (Streamlit entry), `bootstrap.py` (page config + runtime env),
  `compat.py` (Streamlit version shims), `sidebar.py`, `entry_page.py`,
  `session_state.py`, `i18n.py` (EN/ZH text via `get_text`).
- **Pages** — anything `*_page.py`, plus `research_agent.py` /
  `agent_workbench.py`. One page per major view; `page_registry.py`
  orders the top-level tabs.
- **Data & conversion workflows** — `data_workflows.py`,
  `conversion_workflow.py`, `data_paths.py`, `concept_catalog.py`
  (the loadable-concept catalog shown in the UI), `demo_data.py` /
  `mock_data.py` (Demo Mode fixtures), `subprocess_workers.py`.
- **Cohort engine** — `cohort_config.py`, `cohort_filters.py`,
  `cohort_workspace.py`, `cohort_charts.py`, `cohort_redesign.py`.
- **Export** — `export_page.py`, `export_workflow.py`, `export_reports.py`.
- **AI / LLM** — `ai_optin.py` (the opt-in gate above), `llm_config.py`,
  `llm_chat.py`.
- **Presentation** — `styles.py`, `shell_styles.py`, `design_primitives.py`,
  `page_header.py`, `ui_helpers.py`, `quality_metrics.py`,
  `paper_figures.py`, `workflow_figure.py`.

## Working modes

On launch the user picks a mode:

- **Demo Mode** — a guided tour over simulated ICU data (`demo_data.py` /
  `mock_data.py`), no tokens or local dataset required.
- **Real Data Mode** — points at a local prepared dataset (or a supported
  public database) and runs the full extraction-and-review workflow,
  including the Validate Data Path → Convert & Setup preparation step.

First launch creates a local `.easyicu-runtime/` and installs the web
dependencies automatically.

## Where to make changes

- A new view → add a `*_page.py`, register it in `page_registry.py`, and
  reuse `page_header.py` / `design_primitives.py` for chrome.
- A new concept surfaced in the UI → it flows from the concept layer
  (`src/easyicu/data/concept-dict.json` + callbacks), surfaced here through
  `concept_catalog.py`. Do not hard-code concept lists per page.
- Anything that talks to a hosted LLM → call
  `enforce_external_llm_opt_in()` first. See `ai_optin.py`'s own
  docstring for the rule.

For the package-wide architecture (concept layer, converter, scores,
public API) see [`../README.md`](../README.md); for the agent layer see
[`../research_agent/README.md`](../research_agent/README.md).
