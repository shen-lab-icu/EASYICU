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

from functools import lru_cache
from pathlib import Path
from typing import Any

_TOKENS_PATH = Path(__file__).with_name("tokens.css")


@lru_cache(maxsize=1)
def _load_tokens_css() -> str:
    """Read tokens.css once per process (cached across reruns).

    Streamlit re-runs the whole script on every interaction; reading
    this file from disk each time was a needless per-rerun cost.
    """
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
  letter-spacing: 0;
  -webkit-font-smoothing: antialiased;
}
.stApp, .stApp * {
  letter-spacing: 0 !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
section.main {
  background: var(--bg) !important;
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
  padding-top: 0 !important;
  padding-left: 32px !important;
  padding-right: 32px !important;
  padding-bottom: 2.5rem !important;
  max-width: 1460px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]:has(.eu-topbar) {
  gap: 0 !important;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(style),
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(link[rel="stylesheet"]),
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(style),
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(link[rel="stylesheet"]) {
  display: none !important;
}
.stApp [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(style),
.stApp [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(link[rel="stylesheet"]) {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* 3. Sidebar surface. */
[data-testid="stSidebar"] {
  background: var(--rail) !important;
  border-right: 1px solid var(--hair) !important;
  min-width: 240px !important;
  max-width: 240px !important;
  width: 240px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding: 0 !important;
  min-height: 100vh;
  width: 100% !important;
  max-width: none !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: var(--rail) !important;
  border-right: 1px solid var(--hair) !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.28rem !important;
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

/* Slider — full coverage so the BaseWeb thumb / tick / filled track
   no longer renders in Streamlit's default red. Targets every painted
   surface the slider can use: the thumb role node, any direct child
   div the BaseWeb component fills with the brand color, and the
   inline-style background overrides BaseWeb applies. */
.stApp .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 4px rgba(14,17,22,0.06) !important;
}
/* The numeric value label floats above the thumb on the page surface,
   so it must be ink text on no background (not white-on-ink, and not
   the Streamlit-red default). */
.stApp .stSlider [data-testid="stThumbValue"],
.stApp .stSlider [data-testid="stSliderThumbValue"] {
  display: none !important;
  background: transparent !important;
  color: #fff !important;
  font-family: var(--font-mono) !important;
}
.stApp .stSlider [data-baseweb="slider"] > div > div {
  filter: grayscale(1) brightness(0.42) contrast(1.8) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[role="progressbar"],
.stApp .stSlider [data-baseweb="slider"] div[style*="background"],
.stApp .stSlider [data-baseweb="slider"] > div > div > div > div,
.stApp .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
  background: var(--ink) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[style*="linear-gradient"] {
  filter: grayscale(1) brightness(0.42) contrast(1.8) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[style*="height: 0.25rem"] {
  filter: grayscale(1) brightness(0.42) contrast(1.8) !important;
}
.stApp .stSlider [data-testid="stTickBarMin"],
.stApp .stSlider [data-testid="stTickBarMax"] {
  color: var(--ink-4) !important;
  font-family: var(--font-mono) !important;
}

/* Radio + checkbox accents.
   IMPORTANT: only the radio/checkbox *control* (the small circle/box,
   which BaseWeb renders as the first child <div> right after the
   <input>) gets the ink fill. We must NOT paint the label wrapper —
   doing so produced black-on-black, unreadable option text. */
.stApp .stRadio [data-baseweb="radio"] input:checked ~ div:first-of-type,
.stApp .stCheckbox [data-baseweb="checkbox"] input:checked ~ div:first-of-type {
  background-color: var(--ink) !important;
  border-color: var(--ink) !important;
}
/* Radio option label text — always readable, never inverted. */
.stApp .stRadio [data-baseweb="radio"],
.stApp .stRadio [data-baseweb="radio"] div,
.stApp .stCheckbox [data-baseweb="checkbox"] {
  color: var(--ink-2) !important;
}
/* The inner selected dot BaseWeb draws inside the control. */
.stApp .stRadio [role="radio"][aria-checked="true"],
.stApp .stRadio [data-baseweb="radio"] [aria-checked="true"] > div {
  background-color: var(--ink) !important;
  border-color: var(--ink) !important;
}
/* Horizontal radio: kill any inherited dark pill background on the
   option container so the text stays on the page surface. */
.stApp .stRadio [role="radiogroup"] label {
  background: transparent !important;
  color: var(--ink-2) !important;
}
.stApp label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
}
.stApp label[data-baseweb="checkbox"] p {
  white-space: nowrap !important;
}

/* Tabs — flat underline (design SubTabs). Fully neutralize the legacy
   gradient-pill container (background / radius / padding / shadow /
   backdrop) so the row reads as a calm underline strip, not a capsule. */
.stApp [data-baseweb="tab-list"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--hair) !important;
  border-radius: 0 !important;
  padding: 0 !important;
  gap: 2px !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
}
.stApp [data-baseweb="tab"] {
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 14px !important;
  margin-bottom: -1px !important;
  font-family: var(--font) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  color: var(--ink-3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
.stApp [data-baseweb="tab"] p {
  font-size: 12.5px !important;
  font-weight: 500 !important;
}
.stApp [data-baseweb="tab"]:hover {
  background: transparent !important;
  color: var(--ink-2) !important;
}
.stApp [data-baseweb="tab"][aria-selected="true"] {
  color: var(--ink) !important;
  background: transparent !important;
  border-bottom: 2px solid var(--ink) !important;
  box-shadow: none !important;
}
.stApp [data-baseweb="tab"][aria-selected="true"] * {
  color: var(--ink) !important;
}
.stApp [data-baseweb="tab-highlight"] { display: none !important; }
.stApp [data-baseweb="tab-border"] { display: none !important; }

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
  margin: 18px 10px 8px;
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
  padding: 14px 14px 10px;
}
.eu-brand .logo {
  width: 20px; height: 20px;
  border-radius: 5px;
  background: var(--surface);
  border: 1.5px solid var(--ink);
  color: var(--ink);
  display: flex; align-items: center; justify-content: center;
  font-weight: 600; font-size: 0;
  letter-spacing: 0;
  position: relative;
}
.eu-brand .logo::before {
  content: "";
  width: 12px;
  height: 7px;
  border-left: 1.5px solid currentColor;
  border-bottom: 1.5px solid currentColor;
  transform: skewX(-20deg);
}
.eu-brand .text {
  min-width: 0;
}
.eu-brand .text .name {
  font-size: 14px; font-weight: 500; color: var(--ink);
  letter-spacing: 0;
  display: block;
  line-height: 1.2;
}
.eu-brand .text .sub {
  font-size: 10.5px; color: var(--ink-4);
  font-family: var(--font);
  letter-spacing: 0;
  line-height: 1.2;
}

.eu-workspace-field,
.eu-search-field {
  margin: 0 10px;
  height: 28px;
  display: flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--hair-2);
  background: var(--surface);
  border-radius: var(--r-2);
  color: var(--ink);
  font-size: 12.5px;
}
.eu-workspace-field {
  justify-content: space-between;
  padding: 0 8px 0 10px;
}
.eu-workspace-field .inner {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.eu-workspace-field .cohort {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  color: var(--ink-2);
}
.eu-workspace-field .eu-pill {
  height: 18px;
  padding: 0 7px;
  font-size: 10px;
  gap: 5px;
}
.eu-workspace-field .eu-pill .dot {
  width: 5px;
  height: 5px;
}
.eu-workspace-field .chev {
  color: var(--ink-4);
  display: inline-flex;
}
.eu-search-field {
  margin-top: 8px;
  margin-bottom: 12px;
  padding: 0 8px 0 10px;
  color: var(--ink-4);
}
.eu-search-field svg {
  flex: none;
  color: var(--ink-3);
}
.eu-search-field > span:not(.keys) {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eu-search-field .keys {
  margin-left: auto;
  display: flex;
  gap: 3px;
}
.eu-kbd {
  font-family: var(--font-mono);
  font-size: 10.5px;
  padding: 1px 5px;
  border: 1px solid var(--hair-2);
  border-bottom-width: 2px;
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink-3);
  line-height: 1.2;
}

.eu-nav-item {
  display: flex; align-items: center; gap: 10px;
  height: 28px;
  padding: 0 10px;
  border-radius: var(--r-2);
  font-size: 13px;
  color: var(--ink-2);
  margin: 1px 0;
  box-sizing: border-box;
}
.eu-nav-item .ico { width: 16px; height: 16px; flex: none; color: var(--ink-3); display: inline-flex; }
.eu-nav-item .ico svg { color: var(--ink-3); }
.eu-nav-item .label { flex: 1; min-width: 0; color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.eu-nav-item .count {
  font-family: var(--font-mono); font-size: 11px;
  color: var(--ink-4);
}
.eu-nav-item.active {
  background: var(--ink); color: #fff !important;
}
.eu-nav-item.active .ico,
.eu-nav-item.active .ico svg,
.eu-nav-item.active .label,
.eu-nav-item.active span { color: #fff !important; }
.eu-nav-item.active .count { color: rgba(255,255,255,0.6) !important; }

/* Pipeline step */
.eu-pipe-step {
  display: flex; gap: 10px;
  padding: 6px 8px;
  border-radius: var(--r-2);
  margin: 1px 0;
  align-items: flex-start;
  position: relative;
}
.eu-pipe-step::after {
  content: "";
  position: absolute;
  left: 15px;
  top: 22px;
  bottom: -3px;
  width: 1px;
  background: var(--hair-2);
}
.eu-pipe-step:last-child::after { display: none; }
.eu-section-label + .eu-pipe-step,
.eu-pipe-step {
  margin-left: 8px;
  margin-right: 8px;
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

.eu-sidebar-footer-rule {
  border-top: 1px solid var(--hair);
  margin: 0 10px 8px;
  height: 1px;
}
.stApp [class*="st-key-eu_sidebar_footer"] {
  position: fixed !important;
  left: 0 !important;
  bottom: 0 !important;
  width: 240px !important;
  z-index: 50 !important;
  padding: 10px !important;
  background: var(--rail) !important;
  border-right: 1px solid var(--hair) !important;
}

/* Top bar (breadcrumb + actions) */
.eu-topbar {
  display: flex; align-items: center;
  min-height: 52px;
  padding: 14px 4px 12px;
  border-bottom: 0;
  margin: 0;
  font-size: 13px;
  color: var(--ink-2);
}
.eu-topbar .bc { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; }
.eu-topbar .bc .sep { color: var(--ink-4); padding: 0 2px; }
.eu-topbar .bc .crumb { color: var(--ink-3); }
.eu-topbar .bc .crumb.current { color: var(--ink); font-weight: 500; }
.eu-topbar .right { display: flex; align-items: center; gap: 6px; }
.stApp [data-testid="stHorizontalBlock"]:has(.eu-topbar) {
  border-bottom: 1px solid var(--hair);
  margin-bottom: 18px !important;
  align-items: center !important;
  min-height: 52px !important;
  position: relative !important;
  width: calc(100% + 64px) !important;
  margin-left: -32px !important;
  padding-left: 32px !important;
  padding-right: 32px !important;
  box-sizing: border-box !important;
}
.eu-topbar-stage {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  min-height: 32px;
  position: fixed;
  right: 390px;
  top: 21px;
  z-index: 3;
}
.eu-topbar-stage .eu-pill {
  height: 22px;
  font-family: var(--font) !important;
  letter-spacing: 0;
}

.eu-status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  padding: 8px 10px;
  margin: 0 0 16px;
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: color-mix(in oklab, var(--surface) 86%, var(--bg));
}
.eu-status-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
  padding: 2px 8px;
  border: 1px solid var(--hair);
  border-radius: 999px;
  background: var(--surface);
  white-space: nowrap;
}
.eu-status-item .label {
  font-size: 10px;
  color: var(--ink-4);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.eu-status-item .value {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-2);
}
.eu-status-item .value.demo { color: var(--accent-ink); }
.eu-status-item .value.real { color: oklch(40% 0.10 30); }
.eu-status-item .value.ok { color: var(--ok); }
.eu-status-item .value.warn { color: var(--warn); }
.eu-status-item .value.bad { color: var(--bad); }
.eu-status-item .value.info { color: var(--info); }
.eu-status-strip { display: none !important; }

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

/* Agent workbench animations. */
@keyframes eu-pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
@keyframes eu-blink { 0%,49% { opacity: 1 } 50%,100% { opacity: 0 } }
.stApp .eu-pulse { animation: eu-pulse 1.6s ease-in-out infinite; }
.stApp .eu-blink { animation: eu-blink 1s steps(2) infinite; }

/* Research Agent: EasyICU workbench with planagent-inspired density. */
.eu-agent-gate {
  margin: 0 0 14px;
  padding: 10px 12px;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
}
.eu-agent-gate-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-agent-gate-head .muted {
  opacity: 0.72;
}
.eu-agent-gate-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}
.eu-gate-card {
  border: 1px solid var(--hair);
  border-left: 2px solid var(--ink-3);
  border-radius: 2px;
  background: var(--surface-2);
  padding: 8px 10px;
  min-width: 0;
}
.eu-gate-card.ok {
  border-left-color: var(--ok);
}
.eu-gate-card.warn {
  border-left-color: var(--warn);
}
.eu-gate-card.bad {
  border-left-color: var(--bad);
}
.eu-gate-card .k {
  font-size: 10px;
  color: var(--ink-4);
  text-transform: uppercase;
  font-weight: 600;
}
.eu-gate-card .v {
  margin-top: 2px;
  font-size: 12px;
  color: var(--ink-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eu-agent-contract {
  margin-top: 10px;
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
  gap: 10px;
}
.eu-contract-col {
  min-width: 0;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface-2);
  overflow: hidden;
}
.eu-contract-title {
  padding: 7px 10px;
  border-bottom: 1px solid var(--hair);
  color: var(--ink-4);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.eu-manifest-list,
.eu-review-rule-list {
  padding: 4px 10px;
}
.eu-manifest-row {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  min-height: 25px;
  border-top: 1px dashed var(--hair);
}
.eu-manifest-row:first-child {
  border-top: 0;
}
.eu-manifest-row .op {
  color: var(--ink-4);
  font-size: 9.5px;
}
.eu-manifest-row .path {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink-2);
  font-size: 10.5px;
}
.eu-manifest-row .note {
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-review-rule {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 6px 0;
  border-top: 1px dashed var(--hair);
}
.eu-review-rule:first-child {
  border-top: 0;
}
.eu-review-rule > span {
  width: 6px;
  height: 6px;
  margin-top: 6px;
  flex: none;
  border-radius: 999px;
  background: var(--accent);
}
.eu-review-rule p {
  margin: 0;
  color: var(--ink-3);
  font-size: 11.5px;
  line-height: 1.35;
}
.eu-config-note {
  margin: 0 0 12px;
  padding: 8px 10px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface-2);
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.45;
}
.eu-agent-command {
  margin: 12px 0 18px;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
  overflow: hidden;
}
.eu-agent-command-run {
  min-height: 36px;
  padding: 0 10px;
  border-bottom: 1px solid var(--hair);
  background: var(--surface-2);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.58fr);
  gap: 14px;
  align-items: center;
}
.eu-agent-command-run .left,
.eu-agent-command-run .mid {
  min-width: 0;
  display: flex;
  align-items: center;
}
.eu-agent-command-run .left {
  gap: 9px;
}
.eu-agent-command-run .idx {
  color: var(--ink-4);
  font-size: 10px;
}
.eu-agent-command-run b {
  min-width: 0;
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-agent-command-run .mid {
  justify-content: flex-end;
  gap: 8px;
  color: var(--ink-3);
  font-size: 10px;
}
.eu-agent-command-run .bar {
  position: relative;
  flex: 1;
  max-width: 230px;
  min-width: 90px;
  height: 5px;
  background: var(--hair-2);
  border-radius: 1px;
  overflow: hidden;
}
.eu-agent-command-run .bar i {
  display: block;
  height: 100%;
  background: var(--accent);
}
.eu-agent-command-line {
  min-height: 48px;
  padding: 7px 10px;
  display: grid;
  grid-template-columns: minmax(180px, 0.24fr) minmax(380px, 1fr) minmax(260px, 0.34fr);
  gap: 14px;
  align-items: center;
}
.eu-agent-command-now {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.eu-agent-command-now span {
  color: var(--ink-4);
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.eu-agent-command-now b {
  color: var(--ink);
  font-size: 12px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-cmd-stage-row {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
}
.eu-cmd-node {
  min-width: 0;
  height: 30px;
  flex: 1 1 0;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface-2);
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 7px;
}
.eu-cmd-node i {
  width: 16px;
  height: 16px;
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid var(--hair-2);
  color: var(--ink-4);
  font-size: 7.5px;
  font-style: normal;
}
.eu-cmd-node.ok i {
  color: #fff;
  border-color: var(--ink);
  background: var(--ink);
}
.eu-cmd-node.running {
  border-color: color-mix(in srgb, var(--accent) 58%, var(--hair));
  background: var(--accent-soft);
}
.eu-cmd-node.running i {
  border-color: var(--accent);
  color: var(--accent-ink);
}
.eu-cmd-node b {
  min-width: 0;
  color: var(--ink);
  font-size: 10.5px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-cmd-ledger-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  min-width: 0;
}
.eu-cmd-metric {
  min-width: 0;
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  padding-left: 8px;
  border-left: 2px solid var(--ink-3);
  white-space: nowrap;
}
.eu-cmd-metric.ok {
  border-left-color: var(--ok);
}
.eu-cmd-metric.warn {
  border-left-color: var(--warn);
}
.eu-cmd-metric em {
  color: var(--ink-4);
  font-size: 9px;
  text-transform: uppercase;
  font-style: normal;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eu-cmd-metric b {
  color: var(--ink);
  font-size: 11px;
  font-weight: 500;
}
.eu-agent-contract-details {
  border-top: 1px solid var(--hair);
  background: var(--surface-2);
}
.eu-agent-contract-details summary {
  min-height: 28px;
  padding: 0 10px;
  display: flex;
  align-items: center;
  color: var(--ink-3);
  cursor: pointer;
  font-size: 10.5px;
  user-select: none;
}
.eu-agent-contract-details summary::marker {
  color: var(--ink-4);
}
.eu-agent-contract.compact {
  margin: 0;
  padding: 0 10px 10px;
}
.eu-agent-panel-spacer {
  height: 6px;
}
.eu-agent-arch {
  margin: 0 0 10px;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
  padding: 9px 10px 10px;
}
.eu-agent-arch-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  color: var(--ink-4);
  font-size: 10.5px;
  margin-bottom: 8px;
}
.eu-agent-arch-head .muted {
  opacity: 0.72;
}
.eu-agent-arch-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
}
.eu-agent-arch-card {
  position: relative;
  min-width: 0;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface-2);
  padding: 8px 9px;
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}
.eu-agent-arch-card::after {
  content: "";
  position: absolute;
  right: -8px;
  top: 50%;
  width: 8px;
  height: 1px;
  background: var(--hair-3);
}
.eu-agent-arch-card:last-child::after {
  display: none;
}
.eu-agent-arch-card .idx {
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid var(--hair-2);
  color: var(--ink-4);
  font-size: 9.5px;
}
.eu-agent-arch-card.ok .idx {
  background: var(--ink);
  border-color: var(--ink);
  color: #fff;
}
.eu-agent-arch-card.running {
  border-color: color-mix(in srgb, var(--accent) 55%, var(--hair));
  background: var(--accent-soft);
}
.eu-agent-arch-card.running .idx {
  border-color: var(--accent);
  color: var(--accent-ink);
}
.eu-agent-arch-card .label {
  color: var(--ink);
  font-weight: 600;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-agent-arch-card .sub {
  margin-top: 2px;
  color: var(--ink-4);
  font-size: 9.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-agent-panel {
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
  box-shadow: none;
}
.eu-agent-panel .eu-card {
  border-radius: 2px !important;
}
.eu-agent-queue {
  position: relative;
  padding-left: 13px;
}
.eu-agent-queue::before {
  content: "";
  position: absolute;
  left: 6px;
  top: 20px;
  bottom: 8px;
  width: 1px;
  background: var(--hair-2);
}
.eu-agent-start {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 23px;
  min-width: 70px;
  margin: 0 0 9px 12px;
  padding: 0 12px;
  border-radius: 999px;
  background: var(--ink);
  color: #fff;
  font-size: 10px;
  letter-spacing: 0.04em;
}
.eu-agent-start::before,
.eu-agent-step::before {
  content: "";
  position: absolute;
  left: -12px;
  top: 50%;
  width: 12px;
  height: 1px;
  background: var(--hair-2);
}
.eu-agent-step {
  position: relative;
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  gap: 9px;
  align-items: center;
  min-height: 45px;
  margin: 0 0 7px 12px;
  padding: 7px 9px;
  border: 1px solid var(--hair-2);
  border-radius: 5px;
  background: var(--surface);
  color: var(--ink);
}
.eu-agent-step.branch {
  margin-left: 30px;
  border-style: dashed;
}
.eu-agent-step.ok {
  background: var(--ink);
  border-color: var(--ink);
  color: #fff;
}
.eu-agent-step.running {
  border-color: var(--accent);
  background: var(--accent-soft);
  box-shadow: inset 3px 0 0 var(--accent);
}
.eu-agent-step.fail {
  border-color: color-mix(in srgb, var(--bad) 60%, var(--hair));
  background: color-mix(in srgb, var(--bad) 8%, var(--surface));
}
.eu-agent-step.retry {
  border-color: color-mix(in srgb, var(--warn) 65%, var(--hair));
  background: color-mix(in srgb, var(--warn) 10%, var(--surface));
}
.eu-agent-step.pending {
  color: var(--ink-3);
  border-style: dashed;
}
.eu-agent-step-num {
  color: var(--ink-4);
  font-size: 10px;
}
.eu-agent-step.ok .eu-agent-step-num,
.eu-agent-step.ok .eu-agent-step-status,
.eu-agent-step.ok .eu-agent-step-sub {
  color: rgba(255, 255, 255, 0.68);
}
.eu-agent-step-label {
  font-size: 11.5px;
  font-weight: 600;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-agent-step-sub {
  margin-top: 2px;
  font-size: 9.5px;
  color: var(--ink-4);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-agent-step-status {
  align-self: start;
  color: var(--ink-4);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.eu-agent-process-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin: 2px 0 10px;
}
.eu-agent-process-head b {
  font-size: 12.5px;
  font-weight: 650;
}
.eu-agent-process-head span {
  font-size: 10.5px;
  color: var(--ink-4);
}
.eu-agent-step-readout {
  margin-top: 12px;
  padding: 10px 11px;
  border: 1px solid var(--hair);
  border-radius: 6px;
  background: var(--surface-2);
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 3px 8px;
  align-items: baseline;
}
.eu-agent-step-readout span {
  color: var(--ink-4);
  font-size: 10px;
}
.eu-agent-step-readout b {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}
.eu-agent-step-readout small {
  grid-column: 2;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink-4);
  font-size: 10.5px;
}
.stApp [class*="st-key-_eu_wb_step_select_"] [role="radiogroup"] {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.stApp [class*="st-key-_eu_wb_step_select_"] label {
  width: 100%;
  min-height: 44px;
  margin: 0 !important;
  padding: 8px 10px !important;
  border: 1px solid var(--hair-2);
  border-radius: 6px;
  background: var(--surface);
  transition: border-color .12s ease, background .12s ease, color .12s ease;
}
.stApp [class*="st-key-_eu_wb_step_select_"] label:hover {
  border-color: var(--ink-3);
  background: var(--surface-2);
}
.stApp [class*="st-key-_eu_wb_step_select_"] label:has(input:checked) {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--ink);
}
.stApp [class*="st-key-_eu_wb_step_select_"] label:has(input:checked) p,
.stApp [class*="st-key-_eu_wb_step_select_"] label:has(input:checked) span {
  color: var(--ink) !important;
}
.eu-agent-timeline > div {
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
}
.eu-agent-state-track {
  overflow: hidden;
}
.eu-agent-state-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--hair);
}
.eu-agent-state-head > div {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.eu-agent-state-head b {
  color: var(--ink);
  font-size: 12.5px;
}
.eu-agent-state-head .mono {
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-agent-state-head .muted {
  opacity: 0.72;
}
.eu-state-grid {
  display: grid;
  grid-template-columns: 190px minmax(0, 1fr);
  min-height: 194px;
}
.eu-state-labels {
  border-right: 1px solid var(--hair);
  background: var(--surface-2);
  padding-top: 24px;
}
.eu-state-lane-label {
  min-height: 34px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 12px;
  border-top: 1px dashed var(--hair);
}
.eu-state-lane-label b {
  display: block;
  color: var(--ink-2);
  font-size: 11.5px;
  line-height: 1.1;
}
.eu-state-lane-label small {
  display: block;
  margin-top: 2px;
  color: var(--ink-4);
  font-size: 10px;
  line-height: 1.1;
}
.eu-state-dot {
  width: 7px;
  height: 7px;
  border-radius: 2px;
  flex: none;
  background: var(--ink-4);
}
.eu-state-dot.staging { background: var(--accent); }
.eu-state-dot.running { background: var(--accent); }
.eu-state-dot.issue { background: var(--warn); }
.eu-state-dot.review { background: #7c6cc5; }
.eu-state-dot.approved { background: var(--ok); }
.eu-state-canvas {
  position: relative;
  min-width: 0;
  background: var(--surface);
}
.eu-state-axis {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  height: 24px;
}
.eu-state-axis span {
  position: absolute;
  top: 6px;
  transform: translateX(-1px);
  color: var(--ink-4);
  font-family: var(--font-mono);
  font-size: 9px;
}
.eu-state-segment {
  position: absolute;
  height: 18px;
  min-width: 44px;
  border-radius: 1px;
  color: #fff;
  font-family: var(--font-mono);
  font-size: 9px;
  line-height: 18px;
  padding: 0 5px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.eu-state-segment.staging,
.eu-state-segment.running {
  background: var(--accent);
}
.eu-state-segment.issue {
  background: var(--warn);
  color: var(--ink);
}
.eu-state-segment.review {
  background: #7c6cc5;
}
.eu-state-segment.approved {
  background: var(--ok);
}
.eu-state-playhead {
  position: absolute;
  top: 18px;
  bottom: 8px;
  width: 1px;
  background: var(--accent);
}
.eu-agent-audit {
  margin-top: 18px;
  border: 1px solid var(--hair);
  border-radius: 2px;
  background: var(--surface);
  overflow: hidden;
}
.eu-audit-head {
  min-height: 38px;
  padding: 0 12px;
  border-bottom: 1px solid var(--hair);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.eu-audit-head b {
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 600;
}
.eu-audit-head span {
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-audit-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  padding: 10px 12px;
}
.eu-audit-metrics > div {
  border: 1px solid var(--hair);
  border-left: 2px solid var(--ink-3);
  background: var(--surface-2);
  padding: 8px 10px;
}
.eu-audit-metrics > div.err { border-left-color: var(--bad); }
.eu-audit-metrics > div.warn { border-left-color: var(--warn); }
.eu-audit-metrics span {
  display: block;
  color: var(--ink-4);
  font-size: 9.5px;
  text-transform: uppercase;
  font-weight: 600;
}
.eu-audit-metrics b {
  display: block;
  margin-top: 2px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 16px;
  font-weight: 500;
}
.eu-audit-gates {
  padding: 0 12px 10px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}
.eu-audit-gate {
  border: 1px solid var(--hair);
  background: var(--surface-2);
  min-height: 36px;
  padding: 6px 8px;
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
}
.eu-audit-gate > span {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--bad);
}
.eu-audit-gate > span.ok { background: var(--ok); }
.eu-audit-gate b {
  min-width: 0;
  color: var(--ink-2);
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eu-audit-gate small {
  color: var(--ink-4);
  font-size: 10px;
  white-space: nowrap;
}
.eu-audit-repro {
  margin: 0 12px 10px;
  padding: 7px 9px;
  border: 1px solid var(--hair);
  background: var(--surface-2);
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-audit-findings {
  border-top: 1px solid var(--hair);
}
.eu-audit-finding {
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr);
  gap: 8px;
  padding: 8px 12px;
  border-top: 1px dashed var(--hair);
}
.eu-audit-finding:first-child { border-top: 0; }
.eu-audit-finding > span {
  width: 7px;
  height: 7px;
  margin-top: 5px;
  border-radius: 999px;
  background: var(--ink-4);
}
.eu-audit-finding.error > span { background: var(--bad); }
.eu-audit-finding.warning > span { background: var(--warn); }
.eu-audit-finding b {
  display: block;
  color: var(--ink-2);
  font-size: 10.5px;
}
.eu-audit-finding p,
.eu-agent-audit .muted {
  margin: 0;
  color: var(--ink-3);
  font-size: 11.5px;
  line-height: 1.35;
}
.eu-agent-audit .muted {
  padding: 8px 12px;
}
.stApp [class*="st-key-_eu_ra_view_"] button {
  height: 32px !important;
  min-height: 32px !important;
  padding: 0 14px !important;
  border-radius: 8px !important;
}
.stApp [class*="st-key-_eu_wb_"] button {
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 11px !important;
  border-radius: 4px !important;
  font-size: 11.5px !important;
}
.eu-wb-action-panel {
  margin-top: 10px;
  padding: 12px 14px;
  border: 1px solid var(--hair);
  border-radius: 6px;
  background: var(--surface);
}
.eu-wb-action-panel > b {
  display: inline-flex;
  margin-right: 10px;
  font-size: 12.5px;
}
.eu-wb-action-panel > span {
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-wb-action-panel p {
  margin: 7px 0 0;
  color: var(--ink-3);
  font-size: 11.5px;
}
.eu-wb-action-panel ul {
  margin: 7px 0 0;
  padding-left: 16px;
  color: var(--ink-3);
  font-size: 11.5px;
}
.eu-wb-action-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}
.eu-wb-action-grid div,
.eu-wb-manifest-mini div {
  min-width: 0;
  padding: 8px 9px;
  border: 1px solid var(--hair);
  border-radius: 5px;
  background: var(--surface-2);
}
.eu-wb-action-grid span,
.eu-wb-manifest-mini span,
.eu-wb-manifest-mini small {
  display: block;
  color: var(--ink-4);
  font-size: 10px;
}
.eu-wb-action-grid b,
.eu-wb-manifest-mini b {
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink);
  font-size: 11.5px;
}
.eu-wb-manifest-mini {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}

@media (max-width: 1100px) {
  .eu-agent-command-run {
    grid-template-columns: 1fr;
    gap: 6px;
    padding: 7px 10px;
  }
  .eu-agent-command-run .mid {
    justify-content: flex-start;
  }
  .eu-agent-command-line {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  .eu-cmd-stage-row,
  .eu-cmd-ledger-row {
    justify-content: flex-start;
  }
}

/* Language switch (not mixing): the primary copy is already rendered
   in the active language via the Python _T() helper. The .eu-cn spans
   are the redundant opposite-language duplicates that produced the
   EN/ZH-mixed look, so we hide them globally. */
.stApp .eu-cn { display: none !important; }
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

/* Data source page — close to easyicu design/page-data-source.jsx. */
.eu-source-header {
  margin: 12px 0 34px;
}
.eu-source-header h1 {
  margin: 0 !important;
  padding: 0 !important;
  color: var(--ink) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  line-height: 1.18 !important;
}
.eu-source-header h1 span {
  margin-left: 10px;
  color: var(--ink-3);
  font-weight: 400;
}
.eu-source-header p {
  margin: 6px 0 0 !important;
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp [class*="st-key-_eu_source_mode_"] button {
  margin-top: 18px !important;
  min-height: 32px !important;
  height: 32px !important;
  border-radius: var(--r-2) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  box-shadow: none !important;
  white-space: nowrap !important;
}
.stApp [class*="st-key-_eu_source_mode_"] button[kind="primary"],
.stApp [class*="st-key-_eu_source_mode_"] [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important;
  color: #fff !important;
  border-color: var(--ink) !important;
}
.stApp [class*="st-key-_eu_source_mode_"] button:disabled {
  opacity: 0.58 !important;
}
.eu-source-banner {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 14px 18px;
  margin: 0 0 20px;
  border: 1px solid var(--accent-border);
  border-radius: var(--r-3);
  background: var(--accent-soft);
  color: var(--accent-ink);
}
.eu-source-banner .banner-icon {
  margin-top: 1px;
  color: var(--accent-ink);
  font-size: 16px;
}
.eu-source-banner .banner-copy {
  flex: 1;
  min-width: 0;
}
.eu-source-banner .title {
  font-size: 13.5px;
  font-weight: 500;
  color: var(--accent-ink);
}
.eu-source-banner .title span {
  margin-left: 6px;
  font-weight: 400;
}
.eu-source-banner .sub {
  margin-top: 2px;
  font-size: 12.5px;
  color: var(--accent-ink);
  opacity: 0.86;
  line-height: 1.45;
}
.eu-source-banner .learn {
  height: 24px;
  display: inline-flex;
  align-items: center;
  padding: 0 8px;
  border-radius: var(--r-2);
  color: var(--accent-ink);
  font-size: 12px;
  white-space: nowrap;
}
.stApp [class*="st-key-eu_generation_card"] {
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 24px 24px 24px;
  margin-top: 20px;
  margin-bottom: 20px;
}
.stApp [class*="st-key-eu_generation_card"] [data-testid="stVerticalBlock"] {
  gap: 1rem !important;
}
.stApp [class*="st-key-eu_generation_card"] .stSlider label p {
  color: var(--ink-2) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}
.eu-source-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 8px;
}
.eu-source-metric {
  min-width: 0;
  min-height: 88px;
  padding: 12px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface);
}
.eu-source-metric .label {
  font-size: 10.5px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-4);
}
.eu-source-metric .value {
  margin-top: 3px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 18px;
  font-weight: 500;
  line-height: 1.15;
}
.eu-source-metric .sub {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: 11px;
}
.eu-source-table-card {
  overflow: hidden;
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  margin-bottom: 20px;
}
.eu-source-table-card .table-head {
  padding: 12px 18px;
  border-bottom: 1px solid var(--hair);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.eu-source-table-card .title {
  color: var(--ink);
  font-size: 13px;
  font-weight: 500;
}
.eu-source-table-card .sub {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: 11.5px;
}
.eu-mini-button {
  height: 24px;
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--hair-2);
  border-radius: var(--r-2);
  padding: 0 8px;
  color: var(--ink);
  background: var(--surface);
  font-size: 12px;
  white-space: nowrap;
}
.eu-source-table-card table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 12px;
}
.eu-source-table-card th {
  text-align: left;
  padding: 6px 14px;
  background: var(--surface-2);
  color: var(--ink-4);
  font-size: 10.5px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.eu-source-table-card td {
  padding: 5px 14px;
  border-top: 1px solid var(--hair);
  color: var(--ink);
}
.eu-source-table-card td.muted {
  color: var(--ink-3);
}
.eu-source-footer-note {
  color: var(--ink-3);
  font-size: 12px;
  padding-top: 6px;
}
.eu-step-divider {
  height: 1px;
  background: var(--hair);
  margin: 20px 0;
}

/* Cohort builder page — visual implementation of page-cohort-builder.jsx. */
.eu-cohort-header {
  margin: 12px 0 18px;
}
.eu-cohort-header h1 {
  margin: 0 !important;
  padding: 0 !important;
  color: var(--ink) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  line-height: 1.18 !important;
}
.eu-cohort-header p {
  margin: 6px 0 0 !important;
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp [class*="st-key-eu_cohort_demographics_card"],
.stApp [class*="st-key-eu_cohort_clinical_card"],
.stApp [class*="st-key-eu_cohort_icd_card"] {
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 14px 18px 16px;
  margin-bottom: 14px;
}
.stApp [class*="st-key-eu_cohort_demographics_card"] label p,
.stApp [class*="st-key-eu_cohort_clinical_card"] label p,
.stApp [class*="st-key-eu_cohort_icd_card"] label p {
  font-size: 11.5px !important;
  color: var(--ink-3) !important;
  letter-spacing: 0 !important;
}
.eu-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.eu-card-head span {
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 500;
}
.eu-card-head small {
  color: var(--ink-4);
  font-size: 11px;
  font-weight: 400;
}
.eu-card-head em {
  color: var(--ink-4);
  font-style: normal;
  font-size: 10.5px;
}
.stApp [class*="st-key-cohort_disease_card_"] button {
  min-height: 35px !important;
  height: 35px !important;
  border-radius: var(--r-2) !important;
  justify-content: flex-start !important;
  padding: 0 10px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-align: left !important;
  box-shadow: none !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.stApp [class*="st-key-step2_confirm_design"] button,
.stApp [class*="st-key-cohort_builder_save_preset"] button,
.stApp [class*="st-key-cohort_builder_reset"] button {
  white-space: nowrap !important;
  min-height: 34px !important;
  height: 34px !important;
}
.stApp [class*="st-key-cohort_disease_card_"] button[kind="primary"],
.stApp [class*="st-key-cohort_disease_card_"] [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: #fff !important;
}
.stApp [class*="st-key-cohort_disease_card_"] + [data-testid="stCaptionContainer"],
.stApp [class*="st-key-eu_cohort_clinical_card"] [data-testid="stCaptionContainer"] {
  margin-top: -8px !important;
  padding-left: 2px !important;
  color: var(--ink-4) !important;
  font-family: var(--font-mono);
  font-size: 10.5px !important;
}
.eu-cohort-preview-stack {
  position: sticky;
  top: 68px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.eu-cohort-preview-card,
.eu-cohort-chip-card,
.eu-cohort-sample-card {
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  overflow: hidden;
}
.eu-cohort-preview-card {
  padding: 18px;
}
.eu-cohort-preview-card .preview-head,
.eu-cohort-chip-card .chip-head,
.eu-cohort-sample-card .table-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.eu-cohort-preview-card .title {
  color: var(--ink);
  font-size: 13px;
  font-weight: 500;
}
.eu-cohort-preview-card .sub {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: 11.5px;
}
.eu-cohort-preview-card .preview-big {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-top: 18px;
}
.eu-cohort-preview-card .preview-big span {
  color: var(--ink);
  font-size: 32px;
  font-weight: 500;
  line-height: 1;
}
.eu-cohort-preview-card .preview-big em {
  color: var(--ink-3);
  font-style: normal;
  font-size: 12px;
}
.eu-cohort-funnel {
  display: block;
  width: 100%;
  height: 34px;
  margin-top: 8px;
}
.eu-cohort-funnel rect {
  fill: var(--hair-2);
}
.eu-cohort-funnel rect.ink {
  fill: var(--ink);
}
.eu-cohort-funnel rect.accent {
  fill: var(--accent);
  opacity: 0.55;
}
.eu-cohort-preview-card .funnel-labels {
  display: flex;
  justify-content: space-between;
  color: var(--ink-4);
  font-size: 10.5px;
  margin-top: 2px;
}
.eu-cohort-preview-card .hist-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-top: 18px;
  color: var(--ink-3);
  font-size: 11.5px;
}
.eu-cohort-preview-card .hist-head .mono {
  color: var(--ink-4);
  font-size: 10.5px;
}
.eu-cohort-hist {
  display: block;
  width: 100%;
  height: 60px;
  margin-top: 4px;
}
.eu-cohort-hist rect {
  fill: var(--ink);
  opacity: 0.85;
}
.eu-cohort-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 14px;
}
.eu-cohort-metrics div {
  min-width: 0;
  padding: 10px;
  border-radius: var(--r-2);
  background: var(--surface-2);
}
.eu-cohort-metrics b {
  display: block;
  color: var(--ink-4);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.06em;
}
.eu-cohort-metrics span {
  display: block;
  margin-top: 2px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 500;
}
.eu-cohort-chip-card {
  padding: 14px;
}
.eu-cohort-chip-card .chip-head span,
.eu-cohort-sample-card .table-head span {
  color: var(--ink);
  font-size: 12px;
  font-weight: 500;
}
.eu-cohort-chip-card .chip-head em,
.eu-cohort-sample-card .table-head em {
  color: var(--ink-4);
  font-style: normal;
  font-size: 10.5px;
}
.eu-cohort-chip-card .chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
}
.eu-cohort-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 8px;
  border-radius: 4px;
  background: var(--ink);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 11px;
}
.eu-cohort-chip span {
  opacity: 0.62;
}
.eu-cohort-empty {
  color: var(--ink-4);
  font-size: 11.5px;
}
.eu-cohort-sample-card .table-head {
  padding: 10px 14px;
  border-bottom: 1px solid var(--hair);
}
.eu-cohort-sample-card table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11.5px;
}
.eu-cohort-sample-card th {
  padding: 6px 14px;
  text-align: left;
  color: var(--ink-4);
  background: var(--surface-2);
  font-size: 10.5px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.eu-cohort-sample-card td {
  padding: 6px 14px;
  border-top: 1px solid var(--hair);
  color: var(--ink-2);
}

/* Concept selection page — module cards plus live selection summary. */
.eu-concept-header {
  margin: 12px 0 18px;
}
.eu-concept-header h1 {
  margin: 0 !important;
  padding: 0 !important;
  color: var(--ink) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  line-height: 1.18 !important;
}
.eu-concept-header p {
  margin: 6px 0 0 !important;
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp [class*="st-key-concept_defaults_design"] button,
.stApp [class*="st-key-concept_clear_design"] button,
.stApp [class*="st-key-concept_reset_design"] button,
.stApp [class*="st-key-step3_confirm_design"] button {
  min-height: 32px !important;
  height: 32px !important;
  white-space: nowrap !important;
}
.stApp [class*="st-key-concept_module_card_"] button {
  min-height: 42px !important;
  height: 42px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  padding: 0 12px !important;
  border-radius: var(--r-2) !important;
  box-shadow: none !important;
  font-size: 12.2px !important;
  font-weight: 500 !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
.stApp [class*="st-key-concept_module_card_"] button[kind="primary"],
.stApp [class*="st-key-concept_module_card_"] [data-testid="stBaseButton-primary"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: #fff !important;
}
.stApp [class*="st-key-concept_module_card_"] + [data-testid="stCaptionContainer"] {
  margin-top: -7px !important;
  padding-left: 2px !important;
  color: var(--ink-4) !important;
  font-family: var(--font-mono);
  font-size: 10.5px !important;
}
.eu-concept-summary-card {
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 16px;
  margin-bottom: 14px;
}
.eu-concept-summary-card .label {
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 500;
}
.eu-concept-summary-card .big {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-top: 10px;
}
.eu-concept-summary-card .big span {
  color: var(--ink);
  font-size: 34px;
  font-weight: 500;
  line-height: 1;
}
.eu-concept-summary-card .big em {
  color: var(--ink-3);
  font-style: normal;
  font-size: 12px;
}
.eu-concept-module-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 16px;
}
.eu-concept-module-grid div {
  min-width: 0;
  padding: 9px 10px;
  border-radius: var(--r-2);
  background: var(--surface-2);
}
.eu-concept-module-grid b {
  display: block;
  color: var(--ink-4);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.06em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-concept-module-grid span {
  display: block;
  margin-top: 2px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 500;
}
.eu-concept-chip-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 10px;
}
.eu-concept-chip,
.eu-concept-more {
  display: inline-flex;
  align-items: center;
  min-height: 20px;
  padding: 2px 7px;
  border-radius: 4px;
  background: var(--ink);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 11px;
}
.eu-concept-more {
  background: var(--surface-2);
  color: var(--ink-3);
  border: 1px solid var(--hair);
}

/* Export page — real controls with a design-level review surface. */
.eu-export-header {
  margin: 12px 0 18px;
}
.eu-export-header h1 {
  margin: 0 !important;
  padding: 0 !important;
  color: var(--ink) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  line-height: 1.18 !important;
}
.eu-export-header p {
  margin: 6px 0 0 !important;
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp [class*="st-key-eu_export_settings_card"] {
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 16px 18px 18px;
  min-height: 418px;
}
.stApp [class*="st-key-eu_export_settings_card"] [data-testid="stVerticalBlock"] {
  gap: 0.72rem !important;
}
.stApp [class*="st-key-eu_export_settings_card"] label p {
  color: var(--ink-3) !important;
  font-size: 11.5px !important;
  font-weight: 500 !important;
  text-transform: none !important;
}
.eu-export-card-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 2px;
}
.eu-export-card-head span {
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 500;
}
.eu-export-card-head small {
  color: var(--ink-4);
  font-family: var(--font-mono);
  font-size: 10.5px;
}
.eu-export-status {
  min-height: 34px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: var(--r-2);
  font-size: 12.5px;
}
.eu-export-status span {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: none;
  font-size: 10px;
  font-weight: 600;
}
.eu-export-status.ok {
  background: var(--ok-soft);
  border: 1px solid oklch(88% 0.05 160);
  color: var(--ok);
}
.eu-export-status.ok span {
  background: var(--ok);
  color: #fff;
}
.eu-export-status.warn {
  background: var(--warn-soft);
  border: 1px solid oklch(86% 0.05 75);
  color: var(--warn);
}
.eu-export-status.warn span {
  background: var(--warn);
  color: #fff;
}
.eu-export-control-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  margin: 12px 0 2px;
  color: var(--ink);
  font-size: 12px;
  font-weight: 500;
}
.eu-export-control-label small {
  color: var(--ink-4);
  font-size: 10.5px;
  font-weight: 400;
}
.stApp [class*="st-key-eu_export_format_"] button,
.stApp [class*="st-key-eu_export_limit_"] button,
.stApp [class*="st-key-create_export_dir"] button,
.stApp [class*="st-key-final_export_btn"] button,
.stApp [class*="st-key-sanity_back_btn"] button {
  min-height: 32px !important;
  height: 32px !important;
  white-space: nowrap !important;
  border-radius: var(--r-2) !important;
  box-shadow: none !important;
}
.stApp [class*="st-key-eu_export_format_"] button[kind="primary"],
.stApp [class*="st-key-eu_export_limit_"] button[kind="primary"],
.stApp [class*="st-key-final_export_btn"] button[kind="primary"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: #fff !important;
}
.eu-export-review-card {
  min-height: 418px;
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 16px;
}
.eu-export-summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.eu-export-summary-grid > div {
  min-height: 70px;
  padding: 11px 12px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface-2);
}
.eu-export-summary-grid small,
.eu-export-path small,
.eu-export-module-strip small {
  display: block;
  color: var(--ink-4);
  font-size: 10.5px;
  font-weight: 500;
  text-transform: uppercase;
}
.eu-export-summary-grid strong {
  display: block;
  margin-top: 3px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 18px;
  font-weight: 500;
}
.eu-export-path {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--bg);
}
.eu-export-path code {
  display: block;
  margin-top: 4px;
  color: var(--ink-2);
  font-family: var(--font-mono);
  font-size: 11.2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eu-export-checklist {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}
.eu-export-check {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  column-gap: 7px;
  row-gap: 1px;
  align-items: center;
  padding: 8px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface);
}
.eu-export-check span {
  grid-row: span 2;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 10px;
  font-weight: 600;
}
.eu-export-check span.ok { background: var(--ink); }
.eu-export-check span.warn { background: var(--warn); }
.eu-export-check b {
  min-width: 0;
  color: var(--ink);
  font-size: 11.5px;
  font-weight: 500;
}
.eu-export-check em {
  min-width: 0;
  color: var(--ink-4);
  font-family: var(--font-mono);
  font-style: normal;
  font-size: 10.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.eu-export-module-strip {
  margin-top: 12px;
}
.eu-export-module-strip > div {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 6px;
}
.eu-export-module-strip span,
.eu-export-module-strip em {
  min-height: 20px;
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border-radius: 4px;
  background: var(--surface-2);
  color: var(--ink-2);
  border: 1px solid var(--hair);
  font-family: var(--font-mono);
  font-size: 10.5px;
  font-style: normal;
}
.eu-export-module-strip em {
  color: var(--ink-3);
}
.eu-performance-strip {
  margin-top: 14px;
  border: 1px solid var(--hair);
  border-radius: var(--r-3);
  background: var(--surface);
  padding: 12px 14px;
}
.eu-performance-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-top: 8px;
}
.eu-performance-grid > div {
  min-width: 0;
  padding: 8px 10px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface-2);
}
.eu-performance-grid small {
  display: block;
  color: var(--ink-4);
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
}
.eu-performance-grid strong {
  display: block;
  margin-top: 1px;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.stApp .compact-summary-card {
  border: 1px solid var(--hair) !important;
  border-left: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
}
.stApp .compact-summary-card::before {
  display: none !important;
}
.stApp .compact-summary-card .summary-label {
  color: var(--ink-4) !important;
  font-size: 10.5px !important;
  letter-spacing: 0.06em !important;
}
.stApp .compact-summary-card .summary-value {
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  font-weight: 500 !important;
}
.stApp .sidebar-export-detail {
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-2) !important;
  background: var(--surface-2) !important;
  color: var(--ink-2) !important;
}
@media (max-width: 1100px) {
  .eu-source-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .eu-cohort-preview-stack {
    position: static;
  }
}
@media (max-width: 760px) {
  .eu-source-metrics {
    grid-template-columns: 1fr;
  }
  .eu-source-table-card {
    overflow-x: auto;
  }
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
  padding: 0 10px !important;
  min-height: 30px !important;
  height: 30px !important;
  border-radius: var(--r-2) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: var(--ink-2) !important;
  box-shadow: none !important;
  white-space: pre !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_"] button span[data-testid="stIconMaterial"] {
  color: var(--ink-3) !important;
  font-size: 15px !important;
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
[data-testid="stSidebar"] [class*="st-key-euonav_"] button[kind="primary"] span[data-testid="stIconMaterial"],
[data-testid="stSidebar"] [class*="st-key-euonav_"] [data-testid="stBaseButton-primary"] span[data-testid="stIconMaterial"] {
  color: #fff !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_"] button[kind="primary"]:hover,
[data-testid="stSidebar"] [class*="st-key-euonav_"] [data-testid="stBaseButton-primary"]:hover {
  background: #000 !important;
}

/* Compact sidebar layout — drop the gap between consecutive nav buttons. */
[data-testid="stSidebar"] [class*="st-key-euonav_"] {
  margin-top: 1px !important;
  margin-bottom: 1px !important;
  padding: 0 8px !important;
}
[data-testid="stSidebar"] [class*="st-key-euonav_extract"] {
  display: none !important;
}

/* Shell-A redesign: the Quick Visualization screenshot-mode toggle
   used to sit beside the page title in a heavy panel. The shell-A
   topbar already covers status pills, so we shrink the toggle to a
   muted right-side caption + Streamlit-native toggle. */
.stApp [data-testid="stToggle"] label {
  font-size: 12px !important;
  color: var(--ink-3) !important;
}

/* Shell-A redesign: hide the legacy bottom status strip
   ("No Data | 0 Concepts | 0 Patients"). The shell already surfaces
   mode + pipeline state in the sidebar and topbar. */
.stApp .app-footer-status { display: none !important; }
.stApp .divider { display: none !important; }

/* Shell-A topbar action group: keep labels on one line, compact height,
   align with the breadcrumb visual baseline. */
.stApp [class*="st-key-_eu_topbar_"] .stButton > button,
.stApp [class*="st-key-_eu_topbar_"] button {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  min-height: 28px !important;
  height: 28px !important;
  padding: 3px 10px !important;
  font-size: 12px !important;
}

.stApp [class*="st-key-_eu_topbar_cancel"] button {
  border-color: transparent !important;
  background: transparent !important;
  box-shadow: none !important;
}

.stApp [class*="st-key-_eu_topbar_cancel"],
.stApp [class*="st-key-_eu_topbar_confirm_"] {
  position: absolute !important;
  top: 12px !important;
  z-index: 4 !important;
}
.stApp [class*="st-key-_eu_topbar_cancel"] {
  right: 170px !important;
  width: 64px !important;
}
.stApp [class*="st-key-_eu_topbar_confirm_"] {
  right: 0 !important;
  width: 155px !important;
}

/* Shell-A topbar wrapper: subtle bottom border under the whole row. */
.stApp [class*="st-key-_eu_topbar_run"] {
  text-align: right;
}

/* Workflow Help / Back to Mode Selection / language: keep them readable
   but visually de-emphasised so they don't compete with the primary nav. */
[data-testid="stSidebar"] [class*="st-key-back_to_entry"] button,
[data-testid="stSidebar"] [class*="st-key-open_tutorial"] button {
  font-size: 12px !important;
  color: var(--ink-3) !important;
}

/* Tutorial page - workbench layout instead of crowded horizontal strips. */
.stApp .eu-tutorial-hero {
  padding: 0 4px 16px;
  max-width: 940px;
}
.stApp .eu-tutorial-hero h1 {
  margin: 6px 0;
  font-size: 28px;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.16;
}
.stApp .eu-tutorial-hero p {
  margin: 0;
  max-width: 820px;
  color: var(--ink-3);
  font-size: 13.5px;
  line-height: 1.55;
}
.stApp .eu-start-card,
.stApp .eu-rail-card {
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface);
  box-shadow: none;
}
.stApp .eu-start-card {
  min-height: 184px;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.stApp .eu-start-card.primary {
  min-height: 206px;
  padding: 22px;
  border-color: var(--accent-border);
  background: linear-gradient(180deg, var(--accent-soft), var(--surface));
}
.stApp .eu-start-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.stApp .eu-start-kicker {
  color: var(--ink-4);
  font-size: 10.5px;
  font-weight: 500;
  text-transform: uppercase;
}
.stApp .eu-start-head h3 {
  margin: 2px 0 0;
  color: var(--ink);
  font-size: 17px;
  font-weight: 500;
  line-height: 1.2;
}
.stApp .eu-start-card.primary .eu-start-head h3 {
  color: var(--accent-ink);
  font-size: 20px;
}
.stApp .eu-start-desc {
  margin: 0;
  color: var(--ink-2);
  font-size: 12.8px;
  line-height: 1.52;
}
.stApp .eu-start-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.stApp .eu-start-card.primary .eu-start-list {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.stApp .eu-start-list li {
  display: flex;
  gap: 7px;
  min-width: 0;
  color: var(--ink-2);
  font-size: 12px;
  line-height: 1.4;
}
.stApp .eu-start-list li > span {
  width: 5px;
  height: 5px;
  margin-top: 6px;
  flex: 0 0 auto;
  border-radius: 99px;
  background: var(--accent);
}
.stApp .eu-start-list p {
  margin: 0;
  min-width: 0;
}
.stApp .eu-rail-card {
  padding: 14px;
  margin-bottom: 12px;
}
.stApp .eu-rail-title {
  margin-bottom: 10px;
  color: var(--ink);
  font-size: 12.5px;
  font-weight: 600;
}
.stApp .eu-flow-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.stApp .eu-flow-step {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: start;
  padding: 10px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface-2);
}
.stApp .eu-flow-num {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: var(--ink);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 12px;
}
.stApp .eu-flow-title {
  color: var(--ink);
  font-size: 12.8px;
  font-weight: 600;
}
.stApp .eu-flow-desc {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: 11.7px;
  line-height: 1.38;
}
.stApp .eu-flow-tag {
  color: var(--ink-4);
  font-family: var(--font-mono);
  font-size: 10px;
  white-space: nowrap;
}
.stApp .eu-agent-mini p {
  margin: 0 0 10px;
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.45;
}
.stApp .eu-agent-mini-row {
  display: flex;
  gap: 8px;
  padding: 7px 0;
  border-top: 1px solid var(--hair);
}
.stApp .eu-agent-mini-row > span {
  width: 7px;
  height: 7px;
  margin-top: 6px;
  border-radius: 99px;
  background: var(--accent);
}
.stApp .eu-agent-mini-row b,
.stApp .eu-agent-mini-row small {
  display: block;
}
.stApp .eu-agent-mini-row b {
  color: var(--ink);
  font-size: 12px;
  font-weight: 600;
}
.stApp .eu-agent-mini-row small {
  color: var(--ink-4);
  font-size: 11px;
}
.stApp .eu-resource-list {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.stApp .eu-resource-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 10px;
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  background: var(--surface-2);
  color: var(--ink-2);
  font-size: 12px;
}
.stApp .eu-resource-row i {
  width: 6px;
  height: 6px;
  border-top: 1px solid var(--ink-4);
  border-right: 1px solid var(--ink-4);
  transform: rotate(45deg);
  flex: 0 0 auto;
}
.stApp [class*="st-key-_eu_tutorial_demo"] button,
.stApp [class*="st-key-_eu_tutorial_real"] button,
.stApp [class*="st-key-_eu_tutorial_nodata"] button {
  margin-top: 8px !important;
  margin-bottom: 14px !important;
}
@media (max-width: 960px) {
  .stApp .eu-start-card.primary .eu-start-list {
    grid-template-columns: 1fr;
  }
}

/* ====================================================================
   Quick Visualization (A1) — shell-A reskin of the four review subtabs.
   Retargets the legacy "compact"/"paper-figure" blue+gradient classes
   (defined in styles.py) onto the warm-neutral + teal token system so
   Data Tables / Time Series / Patient / Data Quality share one flat,
   hairline-card visual language. Pure visual layer — no markup changes.
   ==================================================================== */

/* Empty/loading state: match the PDF's calm page body instead of the
   previous large native expander. */
.stApp [class*="st-key-eu_qv_loader"] {
  margin: 14px 0 18px !important;
  padding: 14px 16px !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-2) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
}
.stApp [class*="st-key-eu_qv_loader"] [data-testid="stVerticalBlock"] {
  gap: 12px !important;
}
.stApp .eu-qv-loader-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--hair);
}
.stApp .eu-qv-loader-head .k {
  color: var(--ink-4);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 600;
}
.stApp .eu-qv-loader-head .t {
  margin-top: 2px;
  color: var(--ink);
  font-size: 16px;
  font-weight: 600;
}
.stApp .eu-qv-loader-head .s {
  margin-top: 3px;
  max-width: 680px;
  color: var(--ink-3);
  font-size: 12.5px;
  line-height: 1.45;
}
.stApp .eu-qv-loader-badge {
  flex: none;
  min-height: 24px;
  padding: 3px 9px;
  border: 1px solid var(--hair);
  border-radius: var(--r-pill);
  background: var(--surface-2);
  color: var(--ink-3);
  font-family: var(--font-mono);
  font-size: 10.5px;
}
.stApp .viz-demo-load-card {
  margin: 0 !important;
  padding: 12px 14px !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-2) !important;
  background: var(--surface-2) !important;
  box-shadow: none !important;
}
.stApp .viz-demo-load-kicker {
  color: var(--ink-4) !important;
  font-size: 9.5px !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
}
.stApp .viz-demo-load-title {
  color: var(--ink) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
}
.stApp .viz-demo-load-subtitle {
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp .viz-empty-state {
  margin: 18px 0 0 !important;
  padding: 34px 18px !important;
  border: 1px dashed var(--hair-2) !important;
  border-radius: var(--r-2) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
}
.stApp .viz-empty-icon {
  width: 38px !important;
  height: 38px !important;
  margin: 0 auto 10px !important;
  border-radius: var(--r-2) !important;
  background: var(--ink) !important;
  box-shadow: none !important;
  color: #fff !important;
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.04em !important;
}
.stApp .viz-empty-title {
  color: var(--ink) !important;
  font-size: 15px !important;
  font-weight: 600 !important;
}
.stApp .viz-empty-subtitle {
  color: var(--ink-3) !important;
  font-size: 12.5px !important;
}
.stApp .stSlider * {
  accent-color: var(--ink) !important;
}
.stApp .stSlider [data-baseweb="slider"] div[style*="rgb(255"],
.stApp .stSlider [data-baseweb="slider"] div[style*="#ff"],
.stApp .stSlider [data-baseweb="slider"] div[style*="linear-gradient"] {
  background: var(--ink) !important;
}

/* Section heads (used at the top of each subtab) */
.stApp .compact-section-title {
  font-family: var(--font-sans), var(--font-cn) !important;
  font-size: 16.5px !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  line-height: 1.3 !important;
  margin: 0 0 2px 0 !important;
}
.stApp .compact-section-desc {
  font-family: var(--font-sans), var(--font-cn) !important;
  font-size: 12.5px !important;
  color: var(--ink-3) !important;
  line-height: 1.45 !important;
  margin-bottom: 10px !important;
}
.stApp .preview-hint-line {
  font-family: var(--font-mono) !important;
  font-size: 11.5px !important;
  color: var(--ink-4) !important;
  margin: 2px 0 10px !important;
}

/* Inline notices — flat tonal strips instead of saturated blue */
.stApp .compact-inline-notice {
  font-size: 12px !important;
  border-radius: var(--r-2) !important;
}
.stApp .compact-inline-notice.info {
  background: var(--accent-soft) !important;
  border: 1px solid var(--accent-border) !important;
  color: var(--accent-ink) !important;
}
.stApp .compact-inline-notice.warning {
  background: var(--warn-soft) !important;
  border: 1px solid oklch(86% 0.06 80) !important;
  color: oklch(45% 0.10 70) !important;
}

/* Module preview card — flat surface, hairline border, teal eyebrow */
.stApp .module-preview-card {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  box-shadow: none !important;
  padding: 14px 16px !important;
  min-height: 0 !important;
}
.stApp .module-preview-card .eyebrow {
  font-size: 10px !important;
  letter-spacing: 0.07em !important;
  font-weight: 600 !important;
  color: var(--accent-ink) !important;
  text-transform: uppercase !important;
  margin-bottom: 4px !important;
}
.stApp .module-preview-card .title {
  font-size: 15px !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  margin-bottom: 4px !important;
}
.stApp .module-preview-card .summary {
  font-size: 12.5px !important;
  color: var(--ink-3) !important;
  line-height: 1.45 !important;
  margin-bottom: 10px !important;
}

/* Feature chips — mono surface-2 pills */
.stApp .module-feature-chip {
  font-family: var(--font-mono) !important;
  background: var(--surface-2) !important;
  border: 1px solid var(--hair) !important;
  color: var(--ink-2) !important;
  border-radius: var(--r-pill) !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  padding: 2px 8px !important;
}
.stApp .module-feature-chip.muted {
  background: var(--surface-3) !important;
  border-color: var(--hair-2) !important;
  color: var(--ink-4) !important;
}

/* Mini / tiny stat cards — flat, accent left-rule, mono value */
.stApp .mini-stat-card,
.stApp .tiny-stat-card {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-left: 2px solid var(--accent) !important;
  border-radius: var(--r-2) !important;
  box-shadow: none !important;
}
.stApp .mini-stat-card .mini-label,
.stApp .tiny-stat-card .tiny-label {
  font-size: 9.5px !important;
  letter-spacing: 0.06em !important;
  font-weight: 600 !important;
  color: var(--ink-4) !important;
  text-transform: uppercase !important;
}
.stApp .mini-stat-card .mini-value,
.stApp .tiny-stat-card .tiny-value {
  font-family: var(--font-mono) !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
}

/* Preview toolbar + badges */
.stApp .preview-toolbar-title {
  font-size: 12.5px !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
}
.stApp .preview-toolbar-note {
  font-size: 11.5px !important;
  color: var(--ink-4) !important;
}
.stApp .preview-toolbar-note code {
  background: var(--surface-2) !important;
  color: var(--accent-ink) !important;
  border-radius: var(--r-1) !important;
  font-family: var(--font-mono) !important;
  font-weight: 500 !important;
}
.stApp .inline-control-label {
  font-size: 10px !important;
  letter-spacing: 0.06em !important;
  font-weight: 600 !important;
  color: var(--ink-4) !important;
}
.stApp .subtle-preview-note {
  font-size: 11.5px !important;
  color: var(--ink-4) !important;
}
.stApp .preview-badge {
  font-family: var(--font-mono) !important;
  background: var(--surface-2) !important;
  border: 1px solid var(--hair-2) !important;
  color: var(--ink-2) !important;
  border-radius: var(--r-pill) !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
}
.stApp .preview-badge.warning {
  background: var(--warn-soft) !important;
  border-color: oklch(86% 0.06 80) !important;
  color: oklch(45% 0.10 70) !important;
}

/* Data Quality KPI cards — shell-A stat treatment */
.stApp .quality-summary-card {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-3) !important;
  box-shadow: none !important;
}
.stApp .quality-summary-label {
  font-size: 10px !important;
  letter-spacing: 0.06em !important;
  font-weight: 600 !important;
  color: var(--ink-4) !important;
  text-transform: uppercase !important;
}
.stApp .quality-summary-value {
  font-family: var(--font-mono) !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  color: var(--ink) !important;
}

/* Patient overview metric tiles */
.stApp .metric-card {
  background: var(--surface) !important;
  border: 1px solid var(--hair) !important;
  border-radius: var(--r-2) !important;
  box-shadow: none !important;
}
.stApp .metric-card .stat-label,
.stApp .metric-card .stat-number {
  font-family: var(--font-mono) !important;
}
.stApp .metric-card .stat-label {
  font-size: 9.5px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--ink-4) !important;
}
.stApp .metric-card .stat-number {
  font-weight: 500 !important;
  color: var(--ink) !important;
}

/* Subtab page headers (Time Series / Patient / Data Quality) — the legacy
   inline 1.4rem/800 heading is replaced in-page by these classes. */
.stApp .eu-subhead {
  margin-bottom: 14px;
}
.stApp .eu-subhead .t {
  font-family: var(--font-sans), var(--font-cn);
  font-size: 16.5px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.01em;
  line-height: 1.3;
}
.stApp .eu-subhead .s {
  font-family: var(--font-sans), var(--font-cn);
  font-size: 12.5px;
  color: var(--ink-3);
  margin-top: 2px;
  line-height: 1.45;
}

/* Data Tables module rail (left list) + in-card stat tiles */
.stApp .eu-rail-label {
  font-size: 10px;
  letter-spacing: 0.06em;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--ink-4);
  margin: 2px 0 6px;
}
.stApp [class*="st-key-dt_mod_"] { margin-bottom: 2px !important; }
.stApp [class*="st-key-dt_mod_"] button {
  justify-content: flex-start !important;
  text-align: left !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 10px !important;
}
.stApp .eu-mod-tiles { display: flex; gap: 8px; margin-top: 10px; }
.stApp .eu-mod-tile {
  background: var(--surface-2);
  border-radius: var(--r-2);
  padding: 6px 12px;
  min-width: 88px;
}
.stApp .eu-mod-tile .k {
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-4);
  font-weight: 600;
}
.stApp .eu-mod-tile .v {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 500;
  color: var(--ink);
  margin-top: 1px;
}

/* Research-Agent handoff — slim one-line strip instead of a blue banner */
.stApp .eu-handoff-note {
  font-family: var(--font-sans), var(--font-cn);
  font-size: 12px;
  color: var(--ink-3);
  background: var(--surface-2);
  border: 1px solid var(--hair);
  border-radius: var(--r-2);
  padding: 7px 12px;
  line-height: 1.4;
  display: flex;
  align-items: center;
  min-height: 32px;
}

/* Sidebar primary nav — render the design's eu-nav-item HTML row (icon +
   label + flush-right count, dark fill when active) and overlay an invisible
   full-row st.button for routing. This is what makes the count column align
   exactly like the design instead of floating after each label. */
[data-testid="stSidebar"] [class*="st-key-eunavrow_"] {
  position: relative !important;
  margin: 0 !important;
  min-height: 30px !important;
  height: 30px !important;
}
[data-testid="stSidebar"] [class*="st-key-eunavrow_"] [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-eunavrow_"] .eu-nav-item {
  margin: 1px 8px !important;
  height: 28px !important;
}
[data-testid="stSidebar"] [class*="st-key-eunavrow_"]:hover .eu-nav-item:not(.active) {
  background: var(--surface-2);
}
[data-testid="stSidebar"] [class*="st-key-eunavrow_"] [class*="st-key-euonav_"] {
  position: absolute !important;
  inset: 0 !important;
  z-index: 3 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-eunavrow_"] [class*="st-key-euonav_"] button {
  opacity: 0 !important;
  width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}
"""


_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@300;400;500;600&'
    'family=IBM+Plex+Sans+SC:wght@300;400;500;600&'
    'family=IBM+Plex+Mono:wght@400;500&display=swap">'
)


def render_shell_styles(st: Any) -> None:
    """Inject the shell-A token layer + Streamlit re-skin.

    Must be called after :func:`easyicu.webapp.styles.render_global_styles`
    so the cascade resolves to the new tokens.

    Kept as separate ``st.markdown`` calls (font <link> tags, the
    tokens <style>, and the overrides <style>) — combining them into a
    single markdown string made Streamlit's markdown/directive parser
    throw "Cannot set properties of undefined (directiveAttributes)"
    and drop ALL the styles. The per-rerun cost is just the cached
    token read + three small emits.
    """
    st.markdown(_FONTS_LINK, unsafe_allow_html=True)
    tokens = _load_tokens_css()
    if tokens:
        st.markdown(f"<style>{tokens}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{_STREAMLIT_OVERRIDES}</style>", unsafe_allow_html=True)
