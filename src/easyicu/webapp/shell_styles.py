"""Shell A · EasyICU design-system layer.

This module injects the EasyICU shell-A design tokens (``tokens.css``)
plus all the Streamlit-specific overrides needed to land the redesign on
top of the existing app:

* IBM Plex font stack
* Override the older :root tokens from ``styles.py`` so accent / surface
  references resolve to the new restrained-teal palette
* Hide Streamlit's native chrome (deploy header, toolbar, footer, hamburger)
* Reskin native widgets (button, selectbox, text_input, slider, radio,
  tabs) so they blend into the EasyICU shell
* Restyle the legacy ``.main-header`` / ``.sub-header`` / ``.main-nav``
  blocks so the redesign reaches the existing page chrome without
  invasive edits

The module exposes a single entry point, :func:`render_shell_styles`,
which should be called **after** :func:`easyicu.webapp.styles.render_global_styles`
so it wins the CSS cascade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TOKENS_PATH = Path(__file__).with_name("tokens.css")


def _load_tokens_css() -> str:
    try:
        return _TOKENS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


_STREAMLIT_OVERRIDES = """
/* ============================================================
   EasyICU shell-A · Streamlit re-skin
   ============================================================ */

/* 1. Map Streamlit's CSS vars + the older styles.py palette to the new
      shell-A tokens so legacy classes (.main-header, .sub-header, etc.)
      still pick up the new surfaces / accents without per-file edits. */
:root, .stApp {
  --primary-color: var(--ink) !important;
  --primary-dark: var(--ink) !important;
  --primary-light: var(--accent) !important;
  --secondary-color: var(--accent) !important;
  --accent-color: var(--accent) !important;

  --bg-primary: var(--bg) !important;
  --bg-secondary: var(--surface) !important;
  --bg-tertiary: var(--surface-2) !important;
  --card-bg-light: var(--surface) !important;
  --text-primary-light: var(--ink) !important;
  --text-secondary-light: var(--ink-3) !important;
  --text-tertiary-light: var(--ink-4) !important;
  --border-light: var(--hair) !important;
  --border-subtle: var(--hair) !important;

  --success-color: var(--ok) !important;
  --warning-color: var(--warn) !important;
  --danger-color: var(--bad) !important;
  --info-color: var(--info) !important;

  --gradient-primary: var(--ink) !important;
  --gradient-hero: var(--ink) !important;

  --shadow-soft: var(--sh-1) !important;
  --shadow-card: var(--sh-1) !important;
  --shadow-hover: var(--sh-2) !important;
  --shadow-elevated: var(--sh-2) !important;
  --shadow-glow: var(--sh-2) !important;

  --radius-sm: var(--r-2) !important;
  --radius-md: var(--r-3) !important;
  --radius-lg: var(--r-3) !important;
  --radius-xl: var(--r-4) !important;
  --radius-2xl: var(--r-4) !important;

  --font: 'IBM Plex Sans', 'IBM Plex Sans SC', 'PingFang SC',
          'Hiragino Sans GB', 'Microsoft YaHei', system-ui, sans-serif;
  --font-sans: var(--font) !important;
  --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--ink);
  font-family: var(--font);
  font-feature-settings: "ss01", "cv11", "tnum";
  font-size: 14px;
  line-height: 1.5;
  letter-spacing: -0.005em;
  -webkit-font-smoothing: antialiased;
}

/* 2. Hide Streamlit's own chrome. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
footer { display: none !important; }

/* Tighten main padding so the redesign top bar can sit close to the
   first card row. */
.main .block-container,
[data-testid="stMain"] .block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1280px;
}

/* 3. Sidebar surface. */
[data-testid="stSidebar"] {
  background: var(--rail) !important;
  border-right: 1px solid var(--hair) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding: 0.75rem 0.75rem 1.25rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--hair) !important;
  margin: 0.75rem 0 !important;
}

/* 4. Re-skin native widgets so they blend into the shell.
      The selectors are scoped under .stApp and use higher specificity
      than the legacy styles.py block. */
.stApp .stButton > button,
.stApp [data-testid="stBaseButton-secondary"],
.stApp [data-testid="stBaseButton-tertiary"],
.stApp [data-testid="stFormSubmitButton"] > button {
  border: 1px solid var(--hair-2) !important;
  background: var(--surface) !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 12.5px !important;
  min-height: 30px !important;
  border-radius: var(--r-2) !important;
  box-shadow: none !important;
  letter-spacing: -0.005em;
  transition: background .12s ease, border-color .12s ease;
}
.stApp .stButton > button:hover,
.stApp [data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stBaseButton-tertiary"]:hover,
.stApp [data-testid="stFormSubmitButton"] > button:hover {
  background: var(--surface-2) !important;
  border-color: var(--hair-3) !important;
  color: var(--ink) !important;
}
.stApp .stButton > button[kind="primary"],
.stApp [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important;
  color: #fff !important;
  border-color: var(--ink) !important;
}
.stApp .stButton > button[kind="primary"]:hover,
.stApp [data-testid="stBaseButton-primary"]:hover {
  background: #000 !important;
  border-color: #000 !important;
}

/* Selectbox + text + number inputs */
.stApp .stSelectbox div[data-baseweb="select"] > div,
.stApp .stMultiSelect div[data-baseweb="select"] > div,
.stApp .stTextInput input,
.stApp .stNumberInput input,
.stApp .stDateInput input,
.stApp .stTimeInput input,
.stApp [data-baseweb="input"],
.stApp [data-baseweb="textarea"] {
  background: var(--surface) !important;
  border-color: var(--hair-2) !important;
  border-radius: var(--r-2) !important;
  font-family: var(--font) !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
.stApp .stSelectbox div[data-baseweb="select"] > div:hover,
.stApp .stTextInput input:focus,
.stApp .stNumberInput input:focus {
  border-color: var(--accent) !important;
}
.stApp .stSelectbox label,
.stApp .stMultiSelect label,
.stApp .stTextInput label,
.stApp .stNumberInput label,
.stApp .stRadio label,
.stApp .stCheckbox label,
.stApp .stSlider label,
.stApp .stDateInput label,
.stApp .stFileUploader label {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--ink-3) !important;
  letter-spacing: 0.02em;
}

/* Slider — use ink instead of red. */
.stApp .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 4px rgba(14,17,22,0.06) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[role="progressbar"] {
  background: var(--ink) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[style*="background"] {
  background: var(--ink) !important;
}

/* Radio + checkbox accents */
.stApp .stRadio input:checked + div,
.stApp .stCheckbox input:checked + div {
  background-color: var(--ink) !important;
  border-color: var(--ink) !important;
}

/* Tabs — used inside cohort subtabs. */
.stApp [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--hair) !important;
  gap: 4px !important;
}
.stApp [data-baseweb="tab"] {
  height: 34px !important;
  padding: 0 12px !important;
  font-family: var(--font) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  color: var(--ink-3) !important;
  background: transparent !important;
  border-radius: var(--r-2) var(--r-2) 0 0 !important;
}
.stApp [data-baseweb="tab"][aria-selected="true"] {
  color: var(--ink) !important;
  background: var(--surface-2) !important;
}
.stApp [data-baseweb="tab-highlight"] {
  background: var(--ink) !important;
  height: 2px !important;
}

/* Expander */
.stApp [data-testid="stExpander"] {
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  background: var(--surface) !important;
}
.stApp [data-testid="stExpander"] summary {
  font-family: var(--font) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}

/* Metric */
.stApp [data-testid="stMetricLabel"] {
  color: var(--ink-4) !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.stApp [data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}
.stApp [data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  color: var(--ink-3) !important;
}

/* Alerts — flatten Streamlit's coloured callouts. */
.stApp [data-testid="stAlert"] {
  border: 1px solid var(--hair-2) !important;
  border-radius: var(--r-3) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
  color: var(--ink) !important;
  font-size: 13px;
}
.stApp [data-testid="stAlert"][data-baseweb="notification"] [data-testid="stAlertContentInfo"]    { color: var(--info); }
.stApp [data-testid="stAlert"][data-baseweb="notification"] [data-testid="stAlertContentSuccess"] { color: var(--ok); }
.stApp [data-testid="stAlert"][data-baseweb="notification"] [data-testid="stAlertContentWarning"] { color: var(--warn); }
.stApp [data-testid="stAlert"][data-baseweb="notification"] [data-testid="stAlertContentError"]   { color: var(--bad); }

/* Markdown copy */
.stApp [data-testid="stMarkdownContainer"] {
  color: var(--ink-2);
  font-family: var(--font);
  font-size: 13.5px;
  line-height: 1.55;
}
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4 {
  color: var(--ink) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em;
}
.stApp [data-testid="stMarkdownContainer"] code {
  font-family: var(--font-mono) !important;
  font-size: 12px;
  background: var(--surface-2);
  border-radius: var(--r-1);
  padding: 1px 5px;
  color: var(--ink);
}

/* Data tables — keep the underlying widget, neutralise the chrome. */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  overflow: hidden;
}

/* Legacy hero / header blocks from styles.py — keep them but flatten
   so the redesign doesn't have a heavy gradient running across the top
   of every page. */
.stApp .main-header {
  background: var(--surface) !important;
  background-clip: initial !important;
  -webkit-background-clip: initial !important;
  -webkit-text-fill-color: var(--ink) !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 22px !important;
  letter-spacing: -0.01em !important;
  padding: 0 !important;
  margin: 0 0 4px !important;
  text-align: left !important;
  box-shadow: none !important;
}
.stApp .sub-header {
  color: var(--ink-3) !important;
  font-family: var(--font) !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  margin: 0 0 18px !important;
  text-align: left !important;
}

/* Soften the legacy "main-nav" radio container if styles.py wrapped it. */
.stApp .main-nav-wrap,
.stApp [data-testid="stHorizontalBlock"][data-testid="main_nav_bar"] {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  padding: 6px 8px !important;
  box-shadow: none !important;
}

/* === Shell-A primitives === */
.eu-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px; font-weight: 500;
  border: 1px solid var(--hair-2);
  background: var(--surface);
  color: var(--ink-2);
  white-space: nowrap;
  height: 22px;
  font-family: var(--font);
}
.eu-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ink-3); }
.eu-pill.demo { color: var(--accent-ink); background: var(--accent-soft); border-color: var(--accent-border); }
.eu-pill.demo .dot { background: var(--accent); }
.eu-pill.real { color: oklch(40% 0.10 30); background: oklch(96% 0.03 30); border-color: oklch(86% 0.06 30); }
.eu-pill.real .dot { background: oklch(58% 0.13 30); }
.eu-pill.ok   { color: var(--ok); background: var(--ok-soft); border-color: oklch(86% 0.05 150); }
.eu-pill.ok .dot { background: var(--ok); }
.eu-pill.warn { color: var(--warn); background: var(--warn-soft); border-color: oklch(86% 0.06 75); }
.eu-pill.warn .dot { background: var(--warn); }
.eu-pill.bad  { color: var(--bad);  background: var(--bad-soft);  border-color: oklch(86% 0.06 25); }
.eu-pill.bad .dot  { background: var(--bad); }
.eu-pill.info { color: var(--info); background: var(--info-soft); border-color: oklch(86% 0.05 240); }
.eu-pill.info .dot { background: var(--info); }

.eu-card {
  background: var(--surface);
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  padding: 14px 16px;
}
.eu-card.sunken { background: var(--surface-2); }
.eu-card.flush { padding: 0; }

.eu-stat { padding: 14px 16px; }
.eu-stat .label {
  font-size: 10.5px; font-weight: 500;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-4);
}
.eu-stat .val {
  font-family: var(--font-mono);
  font-size: 24px; line-height: 1.1; font-weight: 500;
  letter-spacing: -0.01em;
  margin-top: 4px;
  color: var(--ink);
}
.eu-stat .delta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 4px;
}
.eu-stat .delta.up  { color: var(--ok); }
.eu-stat .delta.dn  { color: var(--bad); }
.eu-stat .val.bad   { color: var(--bad); }
.eu-stat .val.warn  { color: var(--warn); }
.eu-stat .val.ok    { color: var(--ok); }
.eu-stat .val.info  { color: var(--info); }

.eu-chip {
  display: inline-flex; align-items: center; gap: 6px;
  height: 22px; padding: 0 8px;
  border-radius: 999px;
  font-size: 11.5px;
  background: var(--surface-2);
  color: var(--ink-2);
  border: 1px solid var(--hair);
  font-family: var(--font-mono);
  letter-spacing: -0.005em;
}
.eu-chip .x { margin-left: 2px; opacity: 0.6; cursor: pointer; }

.eu-section-label {
  font-size: 10.5px; font-weight: 500;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-4);
  margin: 16px 0 8px;
  display: flex; align-items: center; justify-content: space-between;
}
.eu-section-label .num {
  font-family: var(--font-mono);
  color: var(--ink-4);
  text-transform: none;
  letter-spacing: 0.04em;
}

/* Sidebar brand + nav */
.eu-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 6px 14px;
}
.eu-brand .logo {
  width: 32px; height: 32px;
  border-radius: 8px;
  background: var(--ink);
  color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-weight: 600; font-size: 14px;
  letter-spacing: -0.02em;
}
.eu-brand .text .name {
  font-size: 14px; font-weight: 600; color: var(--ink);
  letter-spacing: -0.01em;
  display: block;
}
.eu-brand .text .sub {
  font-size: 11px; color: var(--ink-4);
  font-family: var(--font-mono);
  letter-spacing: 0.04em;
}

.eu-nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 10px;
  border-radius: var(--r-2);
  font-size: 13px;
  color: var(--ink-2);
  margin: 1px 0;
}
.eu-nav-item .ico { width: 16px; height: 16px; flex: none; color: var(--ink-3); display: inline-flex; }
.eu-nav-item .label { flex: 1; min-width: 0; }
.eu-nav-item .count {
  font-family: var(--font-mono); font-size: 11px;
  color: var(--ink-4);
}
.eu-nav-item.active {
  background: var(--ink); color: #fff;
}
.eu-nav-item.active .ico { color: #fff; }
.eu-nav-item.active .count { color: rgba(255,255,255,0.6); }

/* Pipeline step */
.eu-pipe-step {
  display: flex; gap: 10px;
  padding: 6px 8px;
  border-radius: var(--r-2);
  margin: 1px 0;
  align-items: flex-start;
}
.eu-pipe-step.active { background: var(--surface-2); }
.eu-pipe-step .dot {
  margin-top: 2px;
  width: 14px; height: 14px;
  border-radius: 999px;
  border: 1.5px solid var(--hair-3);
  flex: none;
  display: flex; align-items: center; justify-content: center;
}
.eu-pipe-step.done .dot {
  background: var(--ink);
  border-color: var(--ink);
  color: #fff;
}
.eu-pipe-step.done .dot svg { display: block; }
.eu-pipe-step.active .dot {
  background: var(--accent-soft);
  border-color: var(--accent);
  border-width: 1.5px;
}
.eu-pipe-step.active .dot::after {
  content: "";
  width: 5px; height: 5px; border-radius: 999px; background: var(--accent);
}
.eu-pipe-step.todo .dot { border-style: dashed; }
.eu-pipe-step .body { min-width: 0; }
.eu-pipe-step .title { font-size: 12.5px; color: var(--ink-2); }
.eu-pipe-step.active .title { color: var(--ink); font-weight: 500; }
.eu-pipe-step.todo  .title { color: var(--ink-3); }
.eu-pipe-step .meta { font-size: 11px; color: var(--ink-4); font-family: var(--font-mono); }

/* Top bar (breadcrumb + actions) */
.eu-topbar {
  display: flex; align-items: center;
  padding: 8px 4px;
  border-bottom: 1px solid var(--hair);
  margin: 0 0 16px;
  font-size: 13px;
  color: var(--ink-2);
}
.eu-topbar .bc { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; }
.eu-topbar .bc .sep { color: var(--ink-4); padding: 0 2px; }
.eu-topbar .bc .crumb { color: var(--ink-3); }
.eu-topbar .bc .crumb.current { color: var(--ink); font-weight: 500; }
.eu-topbar .right { display: flex; align-items: center; gap: 6px; }

/* Hide hacks for "ghost" routing buttons we sit beneath custom HTML. */
.stApp .eu-hidden-button,
.stApp .eu-hidden-button > div,
.stApp .eu-hidden-button [data-testid="stVerticalBlock"] { margin: 0 !important; }
.stApp .eu-hidden-button button {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  opacity: 0 !important;
  cursor: pointer !important;
  border: 0 !important;
  padding: 0 !important;
}

/* Density tweak — used by the workspace tweak slider. */
.stApp[data-eu-density="compact"] { font-size: 13px; }
.stApp[data-eu-density="compact"] .main .block-container { padding-top: 0.75rem !important; }
.stApp[data-eu-density="compact"] .eu-card { padding: 10px 12px; }
.stApp[data-eu-density="comfy"] { font-size: 14.5px; }
.stApp[data-eu-density="comfy"] .eu-card { padding: 18px 20px; }

/* Hide bilingual subtitles when the user opts out. */
.stApp[data-eu-bilingual="off"] .eu-cn { display: none !important; }

/* ------------------------------------------------------------------
   Page header (render_page_header used by every top-level tab).
   We flatten the legacy gradient surface and align it to the
   shell-A typographic hierarchy.
   ------------------------------------------------------------------ */
.stApp .app-page-header {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  box-shadow: none !important;
  padding: 16px 18px !important;
  margin: 0 0 14px !important;
}
.stApp .app-page-header::before,
.stApp .app-page-header::after { display: none !important; }
.stApp .app-page-kicker {
  color: var(--ink-4) !important;
  background: transparent !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 0 !important;
  margin: 0 0 4px !important;
  border: 0 !important;
}
.stApp .app-page-title-row {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}
.stApp .app-page-icon {
  font-size: 16px !important;
  background: transparent !important;
  padding: 0 !important;
  width: auto !important;
  height: auto !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
}
.stApp .app-page-title {
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  background: transparent !important;
  background-clip: initial !important;
  -webkit-background-clip: initial !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 18px !important;
  letter-spacing: -0.01em !important;
  line-height: 1.2 !important;
}
.stApp .app-page-subtitle {
  color: var(--ink-3) !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  margin-top: 4px !important;
  letter-spacing: -0.005em !important;
}

/* Status banners + guide cards from ui_helpers: keep them but trim
   shadows and align radii. */
.stApp .app-status-banner,
.stApp .app-guide-card,
.stApp .app-option-card,
.stApp .app-feature-card,
.stApp .app-stat-card,
.stApp .app-mini-card,
.stApp .app-note {
  border-radius: var(--r-3) !important;
  box-shadow: none !important;
}
.stApp .app-feature-card,
.stApp .app-stat-card,
.stApp .app-mini-card,
.stApp .app-option-card {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
}
.stApp .app-stat-card__label,
.stApp .app-mini-card__title,
.stApp .app-option-card__title {
  color: var(--ink-3) !important;
}
.stApp .app-stat-card__value,
.stApp .app-feature-card__title {
  color: var(--ink) !important;
  font-family: var(--font) !important;
}
.stApp .app-stat-card__value {
  font-family: var(--font-mono) !important;
  letter-spacing: -0.01em;
}

/* Sidebar widgets — give buttons and selects in the rail a tighter feel. */
[data-testid="stSidebar"] .stButton > button {
  font-size: 12px !important;
  min-height: 28px !important;
  padding: 4px 10px !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
  font-size: 11px !important;
  letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--hair) !important;
  background: var(--surface) !important;
  border-radius: var(--r-2) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size: 12.5px !important;
  font-weight: 500 !important;
}

/* Shell-A sidebar primary nav — buttons keyed `euonav_*` are styled
   like flat nav rows: left-aligned label, accent on active, hover
   raises the surface. */
[data-testid="stSidebar"] [class*="st-key-euonav_"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-euonav_"] button {
  justify-content: flex-start !important;
  text-align: left !important;
  padding: 6px 10px !important;
  min-height: 32px !important;
  border-radius: var(--r-2) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: -0.005em !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: var(--ink-2) !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_"] .stButton > button:hover,
[data-testid="stSidebar"] [class*="st-key-euonav_"] button:hover {
  background: var(--surface-2) !important;
  color: var(--ink) !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_"] button[kind="primary"],
[data-testid="stSidebar"] [class*="st-key-euonav_"] [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important;
  color: #fff !important;
  border-color: var(--ink) !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_"] button[kind="primary"]:hover,
[data-testid="stSidebar"] [class*="st-key-euonav_"] [data-testid="stBaseButton-primary"]:hover {
  background: #000 !important;
}

/* Compact sidebar layout — drop the gap between consecutive nav buttons. */
[data-testid="stSidebar"] [class*="st-key-euonav_"] {
  margin-top: 1px !important;
  margin-bottom: 1px !important;
}

/* Workflow Help / Back to Mode Selection / language: keep them readable
   but visually de-emphasised so they don't compete with the primary nav. */
[data-testid="stSidebar"] [class*="st-key-back_to_entry"] button,
[data-testid="stSidebar"] [class*="st-key-open_tutorial"] button {
  font-size: 12px !important;
  color: var(--ink-3) !important;
}
"""


def render_shell_styles(st: Any) -> None:
    """Inject the shell-A token layer + Streamlit re-skin.

    Must be called after :func:`easyicu.webapp.styles.render_global_styles`
    so the cascade resolves to the new tokens.
    """
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Sans:wght@300;400;500;600&'
        'family=IBM+Plex+Sans+SC:wght@300;400;500;600&'
        'family=IBM+Plex+Mono:wght@400;500&display=swap">'
    )
    st.markdown(fonts, unsafe_allow_html=True)
    tokens = _load_tokens_css()
    if tokens:
        st.markdown(f"<style>{tokens}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{_STREAMLIT_OVERRIDES}</style>", unsafe_allow_html=True)
