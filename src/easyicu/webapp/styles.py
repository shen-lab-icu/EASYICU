"""Global CSS injection for the EasyICU Streamlit web app."""

from __future__ import annotations

from typing import Any


def render_global_styles(st: Any) -> None:
    """Inject all global Streamlit CSS blocks used by the web app."""
    st.markdown("""
    <style>
        /* ============================================================
           EasyICU Design System v1 — Premium Medical Analytics UI
           ============================================================ */

        /* ============ 谷歌字体导入（必须在所有规则之前） ============ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ============ 全局设计令牌 ============ */
        :root {
            /* 品牌主色 — 医疗蓝青 */
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #60a5fa;
            --secondary-color: #0891b2;
            --accent-color: #06b6d4;

            /* 渐变系统 */
            --gradient-primary: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            --gradient-info: linear-gradient(135deg, #06b6d4 0%, #22d3ee 100%);
            --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
            --gradient-hero: #385d90;
            --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));

            /* 语义色 */
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #06b6d4;

            /* 阴影系统 — 分层深度 */
            --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
            --shadow-soft: 0 4px 16px rgba(0,0,0,0.06);
            --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
            --shadow-hover: 0 8px 30px rgba(37,99,235,0.12), 0 4px 12px rgba(0,0,0,0.05);
            --shadow-glow: 0 0 20px rgba(56,93,144,0.18), 0 4px 16px rgba(56,93,144,0.10);
            --shadow-elevated: 0 12px 40px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.03);

            /* 圆角 */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-2xl: 32px;

            /* 动画 */
            --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
            --transition-smooth: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            --transition-fast: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            --transition-spring: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);

            /* 排版 */
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;

            /* 浅色主题 */
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f1f5f9;
            --card-bg-light: #ffffff;
            --text-primary-light: #0f172a;
            --text-secondary-light: #64748b;
            --text-tertiary-light: #94a3b8;
            --border-light: rgba(37,99,235,0.08);
            --border-subtle: #e2e8f0;
            --fluid-body: clamp(0.98rem, 0.12vw + 0.94rem, 1.08rem);
            --fluid-small: clamp(0.88rem, 0.08vw + 0.84rem, 0.98rem);
            --space-1: 0.25rem;
            --space-2: 0.5rem;
            --space-3: 0.75rem;
            --space-4: 1rem;
            --space-5: 1.25rem;
            --space-6: 1.5rem;
            --space-7: 2rem;
        }

        /* ============ Shared semantic UI helpers ============ */
        .app-status-banner {
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: var(--space-6) var(--space-7);
            margin-bottom: var(--space-6);
            display: flex;
            align-items: center;
            gap: var(--space-4);
            background: var(--card-bg-light);
            box-shadow: var(--shadow-xs);
        }
        .app-status-banner--success {
            border-color: #a7f3d0;
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        }
        .app-status-banner--warning {
            border-color: #fcd34d;
            background: #fffbeb;
        }
        .app-status-banner__icon {
            width: 3rem;
            height: 3rem;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 900;
            background: var(--success-color);
        }
        .app-status-banner__title {
            font-weight: 800;
            font-size: 1.15rem;
            color: #065f46;
            letter-spacing: 0;
        }
        .app-status-banner__subtitle {
            color: #047857;
            font-size: var(--fluid-small);
            margin-top: 0.125rem;
        }
        .app-stat-grid,
        .app-feature-grid,
        .app-step-grid {
            display: grid;
            gap: clamp(0.625rem, 0.5rem + 0.5vw, 1.25rem);
            margin-bottom: var(--space-7);
        }
        .app-stat-grid--4,
        .app-feature-grid--4 {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .app-stat-grid--5 {
            grid-template-columns: repeat(5, minmax(0, 1fr));
        }
        .app-stat-grid--2 {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .app-feature-grid--3 {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .app-stat-card,
        .app-feature-card {
            background: var(--card-bg-light);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-xs);
        }
        .app-stat-card {
            padding: var(--space-5) var(--space-4);
            text-align: center;
        }
        .app-stat-grid--compact {
            margin-bottom: var(--space-2);
            gap: var(--space-3);
        }
        .app-stat-grid--compact .app-stat-card {
            min-height: 4.5rem;
            padding: 0.65rem 0.85rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .app-stat-card__label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-tertiary-light);
            margin-bottom: var(--space-2);
        }
        .app-stat-card__value {
            font-size: 1.45rem;
            font-weight: 850;
            color: var(--text-primary-light);
            letter-spacing: 0;
        }
        .app-stat-card--primary .app-stat-card__value {
            color: var(--primary-color);
        }
        .app-stat-card--success .app-stat-card__value {
            color: var(--success-color);
        }
        .app-stat-card--purple .app-stat-card__value {
            color: #6d28d9;
        }
        .app-feature-card {
            padding: var(--space-6) var(--space-5);
            transition: var(--transition-fast);
        }
        .app-feature-grid--muted .app-feature-card {
            background: #f9fafb;
            text-align: center;
        }
        .app-feature-card__icon {
            font-size: 1.8rem;
            margin-bottom: var(--space-3);
        }
        .app-feature-card__title {
            font-size: 1rem;
            font-weight: 800;
            color: var(--text-primary-light);
            margin-bottom: var(--space-2);
        }
        .app-feature-card__description {
            color: var(--text-secondary-light);
            font-size: 0.86rem;
            line-height: 1.55;
        }
        .app-inline-heading {
            margin-bottom: var(--space-3);
        }
        .app-inline-heading__title {
            font-size: 1.08rem;
            font-weight: 800;
            color: var(--text-primary-light);
        }
        .app-inline-heading__subtitle {
            color: var(--text-tertiary-light);
            font-size: 0.88rem;
            margin-left: var(--space-2);
        }
        .app-step-grid {
            display: flex;
            gap: var(--space-7);
        }
        .app-step-row {
            display: flex;
            align-items: center;
            gap: var(--space-3);
        }
        .app-kicker {
            font-size: 0.78rem;
            font-weight: 800;
            color: var(--text-tertiary-light);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: var(--space-3);
        }
        .app-footer-status {
            color: var(--text-secondary-light);
            font-size: 0.82rem;
        }
        .app-anchor {
            height: 0;
            scroll-margin-top: 1rem;
        }
        .app-anchor--spaced {
            height: var(--space-4);
        }
        .app-guide-card {
            background: var(--card-bg-light);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: var(--space-6);
            margin-bottom: var(--space-3);
            box-shadow: var(--shadow-xs);
        }
        .app-guide-card--warning {
            background: #fffbeb;
            border-color: #fcd34d;
        }
        .app-guide-card__title {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-primary-light);
            letter-spacing: 0;
            margin-bottom: var(--space-4);
        }
        .app-guide-panel-grid,
        .app-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: var(--space-3);
            margin-bottom: var(--space-3);
        }
        .app-guide-panel,
        .app-mini-card {
            border-radius: var(--radius-md);
            padding: var(--space-4);
            border: 1px solid var(--border-subtle);
        }
        .app-guide-panel--success,
        .app-mini-card--success {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }
        .app-guide-panel--info,
        .app-mini-card--info {
            background: #eef2ff;
            border-color: #c7d2fe;
        }
        .app-mini-card--primary {
            background: #eff6ff;
            border-color: #bfdbfe;
        }
        .app-mini-card--warning {
            background: #fffbeb;
            border-color: #fde68a;
        }
        .app-mini-card--purple,
        .app-guide-panel--purple {
            background: #f5f3ff;
            border-color: #ddd6fe;
        }
        .app-guide-panel__title,
        .app-mini-card__title {
            font-weight: 800;
            color: var(--text-primary-light);
            margin-bottom: var(--space-2);
            font-size: 1.02rem;
        }
        .app-mini-card__description {
            color: var(--text-secondary-light);
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .app-guide-list,
        .app-option-card__list,
        .app-file-list {
            color: #374151;
            font-size: var(--fluid-small);
            line-height: 1.75;
            margin: 0;
            padding-left: 1.2rem;
        }
        .app-guide-list--ordered {
            line-height: 2;
        }
        .app-guide-tip {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: var(--radius-sm);
            padding: var(--space-3) var(--space-4);
            color: #92400e;
            font-size: 0.86rem;
            margin-top: var(--space-3);
        }
        .app-option-card {
            background: var(--card-bg-light);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: var(--space-4) var(--space-5);
            min-height: 11.75rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            margin-bottom: var(--space-3);
            box-shadow: var(--shadow-xs);
        }
        .app-option-card__title {
            font-weight: 800;
            margin-bottom: var(--space-2);
            line-height: 1.25;
            color: var(--primary-color);
        }
        .app-option-card--purple .app-option-card__title {
            color: #6d28d9;
        }
        .app-note {
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
            padding: var(--space-4) var(--space-5);
            margin-bottom: var(--space-4);
            color: var(--text-primary-light);
            font-size: var(--fluid-small);
            line-height: 1.65;
            background: #f8fbff;
        }
        .app-note--info {
            background: rgba(102, 126, 234, 0.15);
            border-left: 4px solid #667eea;
        }
        .app-note--warning {
            background: #fffbeb;
            border-color: #fcd34d;
            color: #92400e;
        }
        .app-file-list {
            color: var(--text-primary-light);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: var(--space-2);
        }
        .app-file-list__more {
            color: var(--text-secondary-light);
            font-size: 0.9rem;
        }
        .app-footer {
            text-align: center;
            color: var(--text-tertiary-light);
            font-size: 0.85rem;
        }
        .app-footer p {
            color: inherit !important;
            margin: 0.2rem 0 !important;
        }
        .app-dictionary-heading {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.625rem;
            margin-top: var(--space-3);
            font-size: 1.6rem;
        }
        .viz-demo-load-card {
            border: 1px solid #cfe0f3;
            border-radius: var(--radius-lg);
            background:
                radial-gradient(circle at 100% 0%, rgba(37, 99, 235, 0.08), transparent 36%),
                linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
            box-shadow: var(--shadow-soft);
            padding: var(--space-4) var(--space-5);
            margin: 0.55rem 0 0.75rem;
        }
        .viz-demo-load-kicker {
            color: var(--primary-color);
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: var(--space-1);
        }
        .viz-demo-load-title {
            color: #0b1f44;
            font-size: 1.08rem;
            font-weight: 900;
            letter-spacing: 0;
            margin-bottom: var(--space-1);
        }
        .viz-demo-load-subtitle {
            color: #60718a;
            font-size: 0.84rem;
            line-height: 1.55;
        }
        .viz-empty-state {
            text-align: center;
            padding: 2.35rem 1.4rem;
            background:
                radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.08), transparent 34%),
                linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            margin: var(--space-4) 0;
            box-shadow: var(--shadow-soft);
        }
        .viz-empty-icon {
            width: 3.2rem;
            height: 3.2rem;
            margin: 0 auto 0.8rem;
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 1.55rem;
            background: var(--gradient-primary);
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.22);
        }
        .viz-empty-title {
            color: #0b1f44;
            font-size: 1.22rem;
            font-weight: 900;
            letter-spacing: 0;
            margin-bottom: 0.3rem;
        }
        .viz-empty-subtitle {
            color: #60718a;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .figure-table {
            border: 1px solid #dbeafe;
            border-radius: var(--radius-md);
            overflow: hidden;
            background: #ffffff;
            box-shadow: var(--shadow-soft);
        }
        .figure-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
            color: var(--text-primary-light);
        }
        .figure-table th {
            background: #f8fafc;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 9px 10px;
            border-bottom: 1px solid #dbeafe;
        }
        .figure-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #eef2f7;
            vertical-align: middle;
        }
        .figure-table tr:last-child td {
            border-bottom: 0;
        }
        .figure-table td:first-child,
        .figure-table th:first-child {
            color: var(--primary-color);
            font-weight: 700;
        }
        @media (max-width: 1280px) {
            /* Group Contrast's Comparison Mode horizontal radio packs 6
               options into one row; under ~1280 px the labels collide.
               Force flex-wrap so they re-flow to 2 rows cleanly
               (2026-05 Phase D polish). */
            div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] {
                flex-wrap: wrap !important;
                row-gap: 8px !important;
            }
        }
        @media (max-width: 1024px) {
            .app-stat-grid--4,
            .app-stat-grid--5,
            .app-stat-grid--2,
            .app-feature-grid--4,
            .app-feature-grid--3,
            .app-guide-panel-grid,
            .app-mini-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .app-step-grid {
                flex-wrap: wrap;
                gap: var(--space-4);
            }
        }
        @media (max-width: 680px) {
            .app-status-banner {
                padding: var(--space-5);
                align-items: flex-start;
            }
            .app-stat-grid--4,
            .app-stat-grid--5,
            .app-stat-grid--2,
            .app-feature-grid--4,
            .app-feature-grid--3,
            .app-guide-panel-grid,
            .app-mini-grid {
                grid-template-columns: 1fr;
            }
        }
        /* Narrow-viewport notice. The data-dense Plotly grids, multi-column
           Cohort Analysis panels, and side-by-side metric cards in this app
           are designed for ≥1024 px screens — Streamlit's default narrow
           layout overlaps the sidebar onto main content and clips charts.
           Inject a soft banner instead of pretending it'll work. */
        body::before {
            content: "📱 EasyICU is optimised for screens ≥ 1024 px. Cohort Analysis charts, multi-column metric cards, and the sidebar workflow may clip on narrower viewports. Open in a wider window for the full experience.";
            display: none;
            position: sticky;
            top: 0;
            z-index: 9999;
            padding: 10px 16px;
            background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
            color: #7c2d12;
            font-size: 0.82rem;
            font-weight: 600;
            border-bottom: 1px solid #fdba74;
            line-height: 1.45;
            text-align: center;
        }
        @media (max-width: 1024px) {
            body::before { display: block; }
        }
        /* Tame ~6-10 px horizontal overflow from BaseWeb shadows / gradient
           borders extending past the sidebar+main containers. Cosmetic
           rounding from Streamlit's emotion CSS — clip the outermost
           shells to their own box without affecting interactive widgets. */
        section[data-testid="stSidebar"],
        section.stMain,
        section[data-testid="stMain"] {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        section[data-testid="stSidebar"] {
            overflow-x: hidden !important;
        }

        /* ============ 全局排版 ============ */
        html, body, .stApp, *, *::before, *::after {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            color-scheme: light !important;
        }

        div[data-testid="stMarkdownContainer"] hr {
            margin: 0.45rem 0 0.8rem 0 !important;
            border-top: 1px solid #dbe4f0 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            margin-bottom: 0.38rem !important;
        }

        /* 强制浅色背景 — 覆盖系统/浏览器深色模式 */
        html, body {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"],
        .main {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, rgba(248,250,252,0.97), rgba(241,245,249,0.97)) !important;
            color: #0f172a !important;
        }
        /* Streamlit 内部组件强制浅色 */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: inherit !important;
        }
        /* 全局标签、文本强制深色 — 覆盖 Streamlit 暗色 variables */
        .stApp label,
        .stApp .stMarkdown p,
        .stApp .stMarkdown li,
        .stApp .stMarkdown span,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
        }
        /* Streamlit secondary button 强制浅色 */
        div[data-testid="stButton"] > button[kind="secondary"],
        div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            border-color: #e2e8f0 !important;
        }
        /* Streamlit selectbox / input / text_input 强制浅色 */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        /* Radio / Checkbox / NumberInput / Slider 强制浅色 */
        [data-testid="stRadio"],
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] div[role="radiogroup"],
        [data-testid="stRadio"] div[role="radiogroup"] label {
            color: #0f172a !important;
            background-color: transparent !important;
        }
        [data-testid="stCheckbox"],
        [data-testid="stCheckbox"] label {
            color: #0f172a !important;
        }
        [data-testid="stNumberInput"],
        [data-testid="stNumberInput"] label {
            color: #0f172a !important;
        }
        [data-testid="stNumberInput"] input {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        [data-testid="stSlider"],
        [data-testid="stSlider"] label {
            color: #0f172a !important;
        }
        [data-testid="stMultiSelect"],
        [data-testid="stMultiSelect"] label {
            color: #0f172a !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"],
        [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg *,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] path {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: var(--gradient-primary) !important;
            border: none !important;
            box-shadow: 0 6px 16px rgba(37,99,235,0.18) !important;
            max-width: none !important;
        }
        /* Show full label inside multiselect chips instead of clipping
           "Demographics" to "Demograph..." — BaseWeb's default styling
           applies max-width + text-overflow:ellipsis on the inner span. */
        [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] [class*="Text"],
        [data-testid="stMultiSelect"] [data-baseweb="tag"] [title] {
            max-width: none !important;
            text-overflow: clip !important;
            overflow: visible !important;
            white-space: normal !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        /* Tab list 强制浅色 */
        div[data-baseweb="tab-list"] {
            background: rgba(241,245,249,0.8) !important;
        }
        div[data-baseweb="tab-list"] button {
            color: #64748b !important;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            color: white !important;
        }
        /* Expander 强制浅色 — 包括 summary bar */
        details[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border-color: #e2e8f0 !important;
        }
        details[data-testid="stExpander"] summary {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        details[data-testid="stExpander"] summary span {
            color: #0f172a !important;
        }
        details[data-testid="stExpander"] > div {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        /* 对内容元素应用字体 */
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        .stAlert, div[data-testid="stMetric"],
        div[data-baseweb="select"], div[data-baseweb="input"],
        div[data-baseweb="textarea"], div[data-baseweb="tab-list"],
        h1, h2, h3, h4, h5, h6, label,
        input, textarea, select, option, td, th {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* ============ 页面头部 ============ */
        .block-container {
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
            max-width: clamp(1040px, 92vw, 1960px) !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        header[data-testid="stHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            visibility: hidden !important;
        }

        /* ============ 现代化标签页 — Pill 风格 ============ */
        div[data-baseweb="tab-list"] {
            gap: 6px !important;
            margin-top: 0 !important;
            padding: 6px !important;
            background: rgba(241,245,249,0.8) !important;
            border-radius: var(--radius-xl) !important;
            border: 1px solid var(--border-subtle);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            scrollbar-width: thin;
        }

        div[data-baseweb="tab-list"] button {
            font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
            font-weight: 700 !important;
            padding: clamp(8px, 0.12vw + 7px, 10px) clamp(8px, 0.24vw + 7px, 14px) !important;
            border-radius: var(--radius-lg) !important;
            transition: var(--transition-fast) !important;
            border: none !important;
            background: transparent !important;
            color: var(--text-secondary-light) !important;
            letter-spacing: 0.01em;
            white-space: nowrap !important;
            flex: 0 0 auto !important;
        }

        div[data-baseweb="tab-list"] button:hover {
            background: rgba(37,99,235,0.08) !important;
            color: var(--primary-color) !important;
        }

        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
            box-shadow: 0 2px 12px rgba(37,99,235,0.24) !important;
            border: none !important;
        }

        div[data-baseweb="tab-list"] button p {
            font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
            font-weight: 700 !important;
            white-space: nowrap !important;
        }

        /* Tab 下划线隐藏 */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* ============ 主导航 — 连体分段栏 ============ */
        /* The main page nav is an st.radio (programmatically steerable),
           restyled here into a connected segmented bar. Scoped via the
           st.container(key="main_nav_bar") wrapper so other radios are
           untouched. If a Streamlit upgrade changes the radio DOM this
           degrades to a plain radio — it never breaks navigation. */
        .st-key-main_nav_bar div[role="radiogroup"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            gap: 0 !important;
            padding: 4px !important;
            margin-bottom: 0.4rem !important;
            background: rgba(241,245,249,0.8) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-xl) !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            overflow-x: auto !important;
        }
        .st-key-main_nav_bar div[role="radiogroup"] > label {
            flex: 1 1 0 !important;
            margin: 0 !important;
            padding: 9px 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.35rem !important;
            border-radius: var(--radius-lg) !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            font-weight: 700 !important;
            font-size: clamp(0.8rem, 0.08vw + 0.78rem, 0.94rem) !important;
            color: var(--text-secondary-light) !important;
            background: transparent !important;
            transition: var(--transition-fast) !important;
        }
        /* hide the radio circle so each option reads as a segment */
        .st-key-main_nav_bar div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        .st-key-main_nav_bar div[role="radiogroup"] > label:hover {
            background: rgba(37,99,235,0.08) !important;
            color: var(--primary-color) !important;
        }
        .st-key-main_nav_bar div[role="radiogroup"] > label:has(input:checked) {
            background: var(--gradient-primary) !important;
            box-shadow: 0 2px 12px rgba(37,99,235,0.24) !important;
        }
        .st-key-main_nav_bar div[role="radiogroup"] > label:has(input:checked),
        .st-key-main_nav_bar div[role="radiogroup"] > label:has(input:checked) * {
            color: #ffffff !important;
        }


        /* ============ Metric 卡片 — 毛玻璃风格 ============ */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(37,99,235,0.06);
            border-radius: var(--radius-lg);
            padding: 1.2rem 1.5rem;
            box-shadow: var(--shadow-xs);
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 3px 0 0 3px;
            opacity: 0.8;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
            border-color: rgba(37,99,235,0.15);
        }

        div[data-testid="stMetric"] label {
            font-weight: 600 !important;
            color: var(--text-secondary-light) !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            color: var(--text-primary-light) !important;
            background: none !important;
            -webkit-text-fill-color: unset !important;
        }


        /* ============ 主标题 — 精致排版 ============ */
        .main-header {
            font-size: clamp(2rem, 1.55rem + 0.9vw, 3rem);
            font-weight: 800;
            color: var(--text-primary-light);
            margin-top: 0;
            margin-bottom: 0.2rem;
            text-align: center;
            letter-spacing: 0;
            line-height: 1.2;
        }

        .sub-header {
            font-size: clamp(1.04rem, 0.94rem + 0.28vw, 1.34rem);
            color: var(--text-tertiary-light);
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        .app-page-header {
            margin: 0.72rem 0 0.9rem 0;
            padding: 0 0 0.72rem 0;
            border-bottom: 1px solid rgba(205, 219, 235, 0.72);
        }

        .app-page-kicker {
            margin-bottom: 0.28rem;
            font-size: 0.72rem;
            line-height: 1.1;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--primary-color);
        }

        .app-page-title-row {
            display: flex;
            align-items: center;
            gap: 0.56rem;
            min-height: 2rem;
        }

        .app-page-icon {
            width: 2rem;
            height: 2rem;
            border-radius: var(--radius-sm);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(37, 99, 235, 0.09);
            border: 1px solid rgba(37, 99, 235, 0.13);
            font-size: 1.08rem;
            line-height: 1;
            flex: 0 0 auto;
        }

        .app-page-title {
            color: var(--text-primary-light);
            font-size: 1.32rem;
            font-weight: 850;
            line-height: 1.22;
            letter-spacing: 0;
        }

        .app-page-subtitle {
            margin-top: 0.24rem;
            color: var(--text-secondary-light);
            font-size: 0.92rem;
            font-weight: 560;
            line-height: 1.45;
            letter-spacing: 0;
            max-width: 78rem;
        }


        /* ============ Research Agent demo and output preview ============ */
        .ra-demo-hero {
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            background: linear-gradient(180deg, rgba(248,251,255,0.92) 0%, var(--card-bg-light) 100%);
            padding: 1rem 1.1rem;
            margin: 0.65rem 0 1rem;
        }

        .ra-demo-flow {
            display: grid;
            grid-template-columns: repeat(4, minmax(140px, 1fr));
            gap: 0.5rem;
            align-items: stretch;
            margin-top: 0.8rem;
        }

        .ra-demo-intro {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(220px, 0.34fr);
            gap: 1rem;
            align-items: start;
        }

        .ra-demo-kicker {
            color: var(--primary-color);
            font-size: 0.7rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .ra-demo-heading {
            color: var(--text-primary-light);
            font-size: 1.08rem;
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 0.28rem;
        }

        .ra-demo-copy {
            color: var(--text-secondary-light);
            font-size: 0.84rem;
            font-weight: 620;
            line-height: 1.5;
            max-width: 58rem;
        }

        .ra-demo-note {
            border: 1px solid rgba(245,158,11,0.42);
            border-radius: var(--radius-sm);
            background: rgba(254,243,199,0.72);
            color: #7c2d12;
            padding: 0.6rem 0.72rem;
            font-size: 0.76rem;
            font-weight: 800;
            line-height: 1.42;
        }

        .ra-value-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.58rem;
            margin-top: 0.9rem;
        }

        .ra-value-card {
            border: 1px solid rgba(37,99,235,0.13);
            border-radius: var(--radius-sm);
            background: rgba(248,251,255,0.78);
            padding: 0.68rem 0.72rem;
        }

        .ra-value-card-title {
            color: var(--text-primary-light);
            font-size: 0.86rem;
            font-weight: 900;
            line-height: 1.25;
            margin-bottom: 0.22rem;
        }

        .ra-value-card-body {
            color: var(--text-secondary-light);
            font-size: 0.73rem;
            font-weight: 650;
            line-height: 1.42;
        }

        .ra-demo-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.44rem;
        }

        .ra-demo-chip {
            border: 1px solid rgba(37,99,235,0.14);
            border-radius: 999px;
            background: rgba(37,99,235,0.06);
            color: var(--text-primary-light);
            padding: 0.32rem 0.56rem;
            font-size: 0.76rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .ra-demo-node,
        .ra-output-card {
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            background: var(--card-bg-light);
            box-shadow: var(--shadow-card);
        }

        .ra-demo-node {
            padding: 0.78rem 0.82rem;
            min-height: 110px;
        }

        .ra-demo-node.review {
            background: rgba(16,185,129,0.06);
            border-color: rgba(16,185,129,0.28);
        }

        .ra-demo-node-label {
            color: var(--primary-color);
            font-size: 0.66rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.24rem;
        }

        .ra-demo-node-title,
        .ra-output-title {
            color: var(--text-primary-light);
            font-weight: 900;
            line-height: 1.2;
        }

        .ra-demo-node-title {
            font-size: 0.96rem;
            margin-bottom: 0.3rem;
        }

        .ra-demo-node-body,
        .ra-output-note {
            color: var(--text-secondary-light);
            font-weight: 650;
            line-height: 1.46;
        }

        .ra-demo-node-body { font-size: 0.78rem; }

        .ra-output-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr 0.9fr;
            gap: 0.72rem;
            align-items: stretch;
            margin: 0.75rem 0 1rem;
        }

        .ra-output-card {
            padding: 0.78rem 0.86rem;
            min-height: auto;
        }

        .ra-output-card.wide { grid-column: span 2; }

        .ra-output-title {
            font-size: 0.95rem;
            margin-bottom: 0.42rem;
        }

        .ra-output-note {
            font-size: 0.72rem;
            margin-bottom: 0.5rem;
        }

        .ra-mini-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.72rem;
            color: var(--text-primary-light);
        }

        .ra-mini-table th {
            color: var(--text-secondary-light);
            font-weight: 900;
            border-bottom: 1px solid var(--border-subtle);
            padding: 0.32rem 0.24rem;
            text-align: left;
        }

        .ra-mini-table td {
            border-bottom: 1px solid rgba(226,232,240,0.65);
            padding: 0.3rem 0.24rem;
        }

        .ra-finding {
            border: 1px solid rgba(16,185,129,0.22);
            border-radius: var(--radius-sm);
            background: rgba(16,185,129,0.06);
            color: #14532d;
            padding: 0.46rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 750;
            line-height: 1.45;
            margin-top: 0.42rem;
        }

        .ra-manuscript-preview {
            border-left: 4px solid var(--primary-color);
            background: rgba(248,251,255,0.9);
            border-radius: var(--radius-sm);
            padding: 0.62rem 0.72rem;
            color: var(--text-primary-light);
            font-size: 0.75rem;
            line-height: 1.55;
        }

        @media (max-width: 1100px) {
            .ra-demo-intro,
            .ra-value-grid,
            .ra-demo-flow,
            .ra-output-grid { grid-template-columns: 1fr; }
            .ra-output-card.wide { grid-column: auto; }
        }


        /* ============ 功能卡片 — 精致玻璃 ============ */
        .metric-card, .feature-card {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: var(--radius-lg);
            padding: 1.4rem;
            margin: 0.5rem 0;
            box-shadow: var(--shadow-card);
            border: 1px solid rgba(37,99,235,0.06);
            transition: var(--transition-smooth);
            color: var(--text-primary-light);
        }

        .metric-card:hover, .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-hover);
            border-color: rgba(37,99,235,0.15);
        }


        .feature-card h4 {
            color: var(--primary-color);
            margin-bottom: 0.6rem;
            font-weight: 700;
            font-size: 1rem;
        }

        /* ============ 按钮 — 精致渐变 ============ */
        .stButton > button[kind="primary"] {
            background: var(--gradient-primary) !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.65rem 1.8rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.01em;
            box-shadow: 0 2px 8px rgba(37,99,235,0.22) !important;
            transition: var(--transition-smooth) !important;
        }

        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] .stButton button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-glow) !important;
        }

        .stButton > button[kind="primary"]:active {
            transform: translateY(0) !important;
        }

        /* 侧边栏按钮 */
        [data-testid="stSidebar"] .stButton button {
            background: var(--gradient-primary) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: var(--radius-md) !important;
            letter-spacing: 0.01em;
            transition: var(--transition-smooth) !important;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            box-shadow: var(--shadow-glow) !important;
            transform: translateY(-1px) !important;
        }

        /* ============ 状态提示框 — 更现代 ============ */
        .success-box {
            background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(52,211,153,0.04));
            border-left: 3px solid var(--success-color);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: 14px 18px;
            margin: 12px 0;
            color: #065f46;
            font-size: 0.9rem;
        }

        .warning-box {
            background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(251,191,36,0.04));
            border-left: 3px solid var(--warning-color);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: 14px 18px;
            margin: 12px 0;
            color: #92400e;
            font-size: 0.9rem;
        }

        .info-box {
            background: linear-gradient(135deg, rgba(6,182,212,0.08), rgba(34,211,238,0.04));
            border-left: 3px solid var(--info-color);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: 14px 18px;
            margin: 12px 0;
            color: #0e7490;
            font-size: 0.9rem;
        }


        /* ============ 分隔线 — 微妙 ============ */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
            margin: 0.9rem 0;
            border: none;
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
            margin: 0.9rem 0;
        }

        /* ============ 统计数字 ============ */
        .stat-number {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--primary-color);
            letter-spacing: -0.02em;
        }

        .stat-label {
            font-size: 0.78rem;
            color: var(--text-tertiary-light);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }


        /* ============ 患者卡片 ============ */
        .patient-card {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(8px);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            border: 1px solid var(--border-subtle);
            margin-bottom: 1rem;
            color: var(--text-primary-light);
            transition: var(--transition-smooth);
        }

        .patient-card:hover {
            border-color: rgba(99,102,241,0.2);
            box-shadow: var(--shadow-soft);
            transform: translateY(-1px);
        }


        .patient-card.critical { border-color: var(--danger-color); border-width: 2px; }
        .patient-card.warning { border-color: var(--warning-color); border-width: 2px; }
        .patient-card.stable { border-color: var(--success-color); border-width: 2px; }

        /* ============ 侧边栏 — 精致 ============ */
        /* 注意: 侧边栏宽度由顶部动态 CSS 控制 (sidebar_expanded 状态) */

        [data-testid="stSidebar"] > div:first-child {
            background:
                radial-gradient(circle at 20% -6%, rgba(59, 130, 246, 0.09), transparent 32%),
                linear-gradient(180deg, #f7fbff 0%, #eef5fb 100%) !important;
            border-right: 1px solid #d4e2f0 !important;
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.85);
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.68rem;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #0f172a !important;
            letter-spacing: -0.025em;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 1.22rem !important;
            margin-bottom: 0.18rem !important;
        }

        [data-testid="stSidebar"] h3 {
            font-size: 1.02rem !important;
            margin-top: 0.42rem !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #334155;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.22);
            margin: 0.7rem 0;
        }

        [data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid #cfe0f3;
            border-radius: 15px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
            overflow: hidden;
        }

        [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,249,255,0.98));
            color: #0f172a !important;
            font-weight: 800;
            letter-spacing: -0.015em;
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: 12px !important;
            border: 1px solid #bfd4ed !important;
            background: linear-gradient(180deg, #ffffff 0%, #f4f8fd 100%) !important;
            color: #1f3b63 !important;
            box-shadow: 0 6px 15px rgba(15, 23, 42, 0.055);
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #1d7ef2 0%, #0b8fc7 100%) !important;
            border-color: #166fd0 !important;
            color: #ffffff !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            border-radius: 12px !important;
            border-color: #c7d9ee !important;
            background-color: rgba(255, 255, 255, 0.96) !important;
        }

        .sidebar-header {
            background: var(--gradient-primary);
            border-radius: var(--radius-lg);
            padding: 1rem 1.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 2px 12px rgba(99,102,241,0.2);
        }

        .sidebar-header h3 {
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        /* ============ SOFA2 徽章 ============ */
        .sofa2-badge {
            background: linear-gradient(135deg, #ef4444, #f97316);
            color: white;
            padding: 3px 10px;
            border-radius: 100px;
            font-size: 0.72rem;
            font-weight: 700;
            display: inline-block;
            margin-left: 6px;
            letter-spacing: 0.02em;
            box-shadow: 0 2px 6px rgba(239,68,68,0.25);
        }

        /* ============ 数据表格 ============ */
        .dataframe {
            border-radius: var(--radius-md) !important;
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] {
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
            overflow: hidden;
        }

        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td {
            color: #000000 !important;
        }

        [data-testid="stDataFrame"] thead th {
            color: #000000 !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        div[data-testid="stDataFrame"] * {
            color: #000000 !important;
        }

        div[data-testid="stDataFrame"] div[role="columnheader"] {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        .dvn-scroller div[class*="header"],
        [class*="headerCell"] {
            color: #000000 !important;
            font-weight: 700 !important;
        }


        /* ============ 进度条 ============ */
        .progress-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 100px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 100px;
            transition: width 0.5s var(--ease-out-expo);
        }

        /* ============ 高亮卡片 ============ */
        .highlight-card {
            background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(139,92,246,0.02));
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: var(--radius-lg);
            padding: 1.4rem 1.6rem;
            margin: 1rem 0;
            color: #312e81;
        }

        .highlight-card h4 { color: var(--primary-color); margin-bottom: 0.8rem; font-weight: 700; }
        .highlight-card p, .highlight-card li { color: #4338ca; }
        .highlight-card b { color: var(--primary-dark); }


        /* ============ 动画 ============ */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
            animation: fadeInUp 0.5s var(--ease-out-expo);
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .animate-pulse { animation: pulse 2.5s infinite; }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }

        /* ============ 输入控件 ============ */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: var(--radius-md) !important;
            border-color: var(--border-subtle) !important;
            transition: var(--transition-fast) !important;
        }

        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
        }

        /* ============ Figure-aligned native Streamlit controls ============ */
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #cfe0f3 !important;
            border-left: 4px solid #1d7ef2 !important;
            border-radius: 14px !important;
            padding: 0.82rem 0.95rem !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055) !important;
            min-height: 74px;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.68rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0b1f44 !important;
            font-size: clamp(1.12rem, 0.34vw + 1rem, 1.42rem) !important;
            font-weight: 900 !important;
            letter-spacing: -0.035em !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-secondary"] {
            border-radius: 12px !important;
            border: 1px solid #c7d9ee !important;
            background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%) !important;
            color: #102a4c !important;
            font-weight: 760 !important;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.055);
        }

        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: #ffffff !important;
            border-color: #0b63ce !important;
            background: linear-gradient(135deg, #2f7cf6 0%, #0b65d8 100%) !important;
            box-shadow: 0 11px 24px rgba(37, 99, 235, 0.24) !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: #2f7cf6 !important;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.16) !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 13px !important;
            border: 1px solid #dbeafe !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035) !important;
        }

        /* ============ Expander ============ */
        details[data-testid="stExpander"] {
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-lg) !important;
            overflow: hidden;
        }

        details[data-testid="stExpander"] summary {
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        /* ============ Tooltip ============ */
        [data-baseweb="tooltip"] {
            border-radius: var(--radius-md) !important;
            box-shadow: var(--shadow-elevated) !important;
        }

        /* ============ Streamlit Alert 美化 ============ */
        div[data-testid="stAlert"] {
            border-radius: var(--radius-md) !important;
            border: none !important;
            font-size: 0.88rem !important;
        }

        /* ============ 入口页面 Hero ============ */
        .hero-container {
            background: var(--gradient-hero);
            border-radius: var(--radius-2xl);
            padding: clamp(2.5rem, 2rem + 1.5vw, 4rem) clamp(1.5rem, 1rem + 1vw, 3rem) clamp(2rem, 1.5rem + 1.5vw, 3.5rem);
            margin: 0 auto 2rem;
            max-width: min(900px, 85%);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: none;
            animation: none;
        }

        .hero-title {
            font-size: clamp(2rem, 1.5rem + 1.5vw, 3rem);
            font-weight: 900;
            color: rgba(255,255,255,0.98);
            letter-spacing: -0.04em;
            line-height: 1.15;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
            text-shadow: 0 1px 0 rgba(0,0,0,0.06);
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: rgba(255,255,255,0.82);
            font-weight: 500;
            letter-spacing: 0.02em;
            position: relative;
            z-index: 1;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.30);
            border-radius: 100px;
            padding: 6px 16px;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.92);
            font-weight: 600;
            margin-bottom: 1.2rem;
            letter-spacing: 0.04em;
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 30px rgba(15,34,48,0.12);
            position: relative;
            z-index: 1;
        }

        /* ============ 入口模式卡片 — Glass ============ */
        .mode-card {
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: var(--radius-xl);
            padding: 2.5rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
        }

        .mode-card:hover {
            transform: translateY(-4px);
            border-color: rgba(255,255,255,0.2);
            box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        }

        .mode-card-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
        }

        .mode-card-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.6rem;
        }

        .mode-card-desc {
            font-size: 0.88rem;
            color: rgba(255,255,255,0.6);
            line-height: 1.6;
            margin-bottom: 1.2rem;
        }

        .mode-card-tag {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 100px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .mode-card-tag.green {
            background: rgba(16,185,129,0.2);
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.3);
        }

        .mode-card-tag.blue {
            background: rgba(99,102,241,0.2);
            color: #a5b4fc;
            border: 1px solid rgba(99,102,241,0.3);
        }

        /* ============ 入口页紧凑总览 ============ */
        .entry-overview {
            max-width: min(1200px, 92vw);
            margin: 1rem auto 0;
            padding: 0;
            background: transparent;
            border: none;
            border-radius: 0;
            box-shadow: none;
        }

        .entry-overview-lead {
            text-align: center;
            color: var(--text-secondary-light);
            font-size: clamp(0.92rem, 0.12vw + 0.88rem, 1.03rem);
            line-height: 1.7;
            max-width: 920px;
            margin: 0.85rem auto 0;
        }

        .entry-task-launcher {
            max-width: min(1220px, 92vw);
            margin: 0.4rem auto 0;
        }

        .entry-task-launcher-label {
            text-align: center;
            color: #7388a5;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .entry-task-wrap div[data-testid="stButton"] > button {
            min-height: 62px !important;
            padding: 0.9rem 1rem !important;
            border-radius: 999px !important;
            background: rgba(255,255,255,0.78) !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            color: #17304c !important;
            font-size: 0.94rem !important;
            font-weight: 700 !important;
            line-height: 1.35 !important;
            box-shadow: 0 10px 28px rgba(15,23,42,0.04) !important;
            transition: var(--transition-smooth) !important;
            backdrop-filter: blur(10px) !important;
        }

        .entry-task-wrap div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px) !important;
            border-color: rgba(37,99,235,0.26) !important;
            background: rgba(255,255,255,0.94) !important;
            box-shadow: 0 14px 32px rgba(37,99,235,0.08) !important;
        }

        .entry-overview-panel {
            margin-top: 0.55rem;
            padding: 0.4rem 0 0.2rem;
            border-radius: 0;
            background: transparent;
            border: none;
        }

        .entry-overview-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.8rem;
        }

        .entry-overview-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #14263d;
            line-height: 1.3;
            margin-bottom: 0.18rem;
        }

        .entry-overview-subtitle {
            font-size: 0.84rem;
            line-height: 1.55;
            color: #6c8099;
            max-width: 760px;
        }

        .entry-overview-kicker {
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #7185a2;
            white-space: nowrap;
        }

        .entry-overview-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: clamp(0.7rem, 0.35rem + 0.5vw, 1rem);
        }

        .entry-overview-item {
            display: flex;
            gap: 0.72rem;
            align-items: flex-start;
            padding: 0.15rem 0.1rem;
            border-radius: 0;
            background: transparent;
            border: none;
        }

        .entry-overview-item.ai {
            background: transparent;
            border: none;
        }

        .entry-overview-icon {
            font-size: 1rem;
            line-height: 1.2;
            margin-top: 0.02rem;
            flex-shrink: 0;
        }

        .entry-overview-item-title {
            font-size: 0.92rem;
            font-weight: 800;
            color: #14263d;
            line-height: 1.28;
            margin-bottom: 0.16rem;
        }

        .entry-overview-item-desc {
            font-size: 0.8rem;
            color: #6f829b;
            line-height: 1.48;
        }

        .entry-db-inline {
            margin-top: 0.8rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(148,163,184,0.18);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .entry-db-inline-label {
            font-size: 0.75rem;
            color: var(--text-tertiary-light);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .entry-db-inline-list {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            font-size: 0.86rem;
            color: var(--text-secondary-light);
            font-weight: 600;
        }


        @media (max-width: 768px) {
            .entry-overview-head { flex-direction: column; }
            .entry-overview-grid { grid-template-columns: 1fr; }
            .entry-db-inline { flex-direction: column; align-items: flex-start; }
            .hero-title { font-size: 2rem; }
            .entry-task-wrap div[data-testid="stButton"] > button { min-height: 58px !important; font-size: 0.9rem !important; }
        }

        /* ============ 步骤指示器 — 精致 ============ */
        .step-indicator {
            display: flex;
            align-items: center;
            gap: clamp(10px, 0.3vw + 8px, 16px);
            padding: clamp(12px, 0.32vw + 10px, 18px) clamp(14px, 0.45vw + 11px, 24px);
            border-radius: var(--radius-md);
            margin-bottom: 8px;
            transition: var(--transition-fast);
            border: 1px solid transparent;
        }

        .step-indicator.active {
            background: rgba(99,102,241,0.06);
            border-color: rgba(99,102,241,0.12);
        }

        .step-indicator.done {
            background: rgba(16,185,129,0.06);
            border-color: rgba(16,185,129,0.12);
        }

        .step-dot {
            width: clamp(28px, 0.45vw + 24px, 40px);
            height: clamp(28px, 0.45vw + 24px, 40px);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: clamp(0.78rem, 0.12vw + 0.74rem, 1rem);
            font-weight: 700;
            flex-shrink: 0;
        }

        .step-dot.pending {
            background: var(--bg-tertiary);
            color: var(--text-tertiary-light);
            border: 2px solid var(--border-subtle);
        }

        .step-dot.active {
            background: var(--gradient-primary);
            color: white;
            border: none;
            box-shadow: 0 2px 8px rgba(99,102,241,0.3);
        }

        .step-dot.done {
            background: var(--success-color);
            color: white;
            border: none;
        }

        .step-text {
            font-size: clamp(0.98rem, 0.16vw + 0.92rem, 1.18rem);
            font-weight: 600;
            color: var(--text-primary-light);
        }

        .step-text small {
            display: block;
            font-size: clamp(0.88rem, 0.10vw + 0.84rem, 1rem);
            font-weight: 500;
            color: var(--text-tertiary-light);
            margin-top: 4px;
        }


        /* ============ 响应式适配 — 多分辨率 ============ */

        /* 小屏 (≤1366px, 13-14" 笔记本) */
        @media (max-width: 1366px) {
            .block-container { max-width: 97.5% !important; }
            .main-header { font-size: 1.95rem; }
            .sub-header { font-size: 1.02rem; }
            .step-indicator { padding: 12px 14px; gap: 10px; }
            .step-dot { width: 28px; height: 28px; font-size: 0.78rem; }
            .step-text { font-size: 0.96rem; }
            .step-text small { font-size: 0.86rem; }
            div[data-testid="stMetric"] { padding: 1rem 1.2rem; }
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.4rem; }
            .highlight-card { padding: 1.1rem 1.3rem; }
            .mode-card { padding: 2rem 1.5rem; }
            .mode-card-title { font-size: 1.15rem; }
            div[data-baseweb="tab-list"] button { font-size: 1rem !important; padding: 11px 18px !important; }
            div[data-baseweb="tab-list"] button p { font-size: 1rem !important; }
        }

        /* 大屏 (≥1920px, 24-27" 显示器) */
        @media (min-width: 1920px) {
            .block-container { max-width: min(94vw, 2140px) !important; }
            .hero-container { max-width: min(900px, 65%); }
            .main-header { font-size: 2.45rem; }
            .sub-header { font-size: 1.18rem; }
            .step-indicator { padding: 16px 24px; gap: 14px; }
            .step-dot { width: 36px; height: 36px; font-size: 0.92rem; }
            .step-text { font-size: 1.12rem; }
            .step-text small { font-size: 0.96rem; }
            div[data-testid="stMetric"] { padding: 1.4rem 1.8rem; }
            div[data-baseweb="tab-list"] button { padding: 14px 30px !important; font-size: 1.14rem !important; }
            div[data-baseweb="tab-list"] button p { font-size: 1.14rem !important; }
        }

        /* 超大屏 (≥2560px, 27"+ 2K/4K) */
        @media (min-width: 2560px) {
            .block-container { max-width: min(95vw, 2460px) !important; }
            .hero-container { max-width: min(1000px, 55%); }
            .main-header { font-size: 2.9rem; }
            .sub-header { font-size: 1.32rem; }
            .step-indicator { padding: 18px 28px; gap: 16px; }
            .step-dot { width: 40px; height: 40px; font-size: 1rem; }
            .step-text { font-size: 1.22rem; }
            .step-text small { font-size: 1.02rem; }
            div[data-testid="stMetric"] { padding: 1.6rem 2rem; }
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.8rem; }
            div[data-baseweb="tab-list"] button { padding: 16px 34px !important; font-size: 1.2rem !important; }
            div[data-baseweb="tab-list"] button p { font-size: 1.2rem !important; }
            .highlight-card { padding: 1.6rem 2rem; }
            .features-grid { gap: 1.4rem; }
            .feature-item { padding: 1.8rem 1.5rem; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .compact-summary-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-left: 3px solid #60a5fa;
            border-radius: 12px;
            padding: 0.68rem 0.82rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            min-height: 64px;
        }
        .compact-summary-card .summary-label {
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }
        .compact-summary-card .summary-value {
            font-size: 1.22rem;
            line-height: 1.1;
            color: #0f172a;
            font-weight: 800;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sidebar-export-detail {
            font-size: 0.86rem;
            line-height: 1.45;
            margin: 0.42rem 0;
            overflow-wrap: anywhere;
        }
        .sidebar-export-detail code {
            white-space: normal;
            word-break: break-word;
        }
        .compact-inline-notice {
            border-radius: 10px;
            padding: 0.48rem 0.72rem;
            margin: 0.22rem 0 0.42rem;
            font-size: 0.84rem;
            line-height: 1.45;
            border: 1px solid transparent;
        }
        .compact-inline-notice.info {
            background: rgba(59,130,246,0.08);
            border-color: rgba(59,130,246,0.14);
            color: #1d4ed8;
        }
        .compact-inline-notice.success {
            background: rgba(16,185,129,0.09);
            border-color: rgba(16,185,129,0.15);
            color: #047857;
        }
        .compact-inline-notice.warning {
            background: rgba(245,158,11,0.10);
            border-color: rgba(245,158,11,0.18);
            color: #b45309;
        }
        .compact-section-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #111827;
            margin: 0 0 0.18rem 0;
            line-height: 1.25;
        }
        .compact-section-desc {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-bottom: 0.55rem;
            line-height: 1.45;
        }
        .module-preview-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.94));
            border: 1px solid rgba(96,165,250,0.28);
            border-radius: 16px;
            padding: 0.82rem 0.96rem 0.88rem;
            box-shadow: 0 8px 24px rgba(37,99,235,0.08);
            min-height: 112px;
        }
        .module-preview-card .eyebrow {
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #2563eb;
            margin-bottom: 0.14rem;
        }
        .module-preview-card .title {
            font-size: 1.04rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
            margin-bottom: 0.22rem;
        }
        .module-preview-card .summary {
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }
        .module-feature-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.34rem;
        }
        .module-feature-chip {
            display: inline-flex;
            align-items: center;
            background: rgba(37,99,235,0.08);
            border: 1px solid rgba(37,99,235,0.12);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 700;
            color: #1d4ed8;
            line-height: 1.1;
        }
        .module-feature-chip.muted {
            background: rgba(148,163,184,0.12);
            border-color: rgba(148,163,184,0.18);
            color: #475569;
        }
        .preview-hint-line {
            font-size: 0.77rem;
            color: #64748b;
            margin: 0.15rem 0 0.5rem;
        }
        .preview-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            padding: 0.24rem 0.58rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.08);
            border: 1px solid rgba(59,130,246,0.14);
            color: #1d4ed8;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1;
        }
        .preview-badge.warning {
            background: rgba(245,158,11,0.10);
            border-color: rgba(245,158,11,0.16);
            color: #b45309;
        }
        .preview-toolbar {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.18rem 0 0.24rem;
        }
        .preview-toolbar-main {
            min-width: 0;
        }
        .preview-toolbar-title {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 700;
            color: #2563eb;
            margin-bottom: 0.14rem;
        }
        .preview-toolbar-note {
            font-size: 0.79rem;
            color: #64748b;
            line-height: 1.35;
        }
        .preview-toolbar-note code {
            background: rgba(37,99,235,0.08);
            color: #1d4ed8;
            border-radius: 8px;
            padding: 0.08rem 0.34rem;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .inline-control-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0.18rem 0 0.18rem;
        }
        .subtle-preview-note {
            font-size: 0.78rem;
            color: #64748b;
            margin: 0.1rem 0 0.35rem;
            line-height: 1.35;
        }
        .mini-stat-card {
            background: #fff;
            border: 1px solid #dbeafe;
            border-left: 3px solid #60a5fa;
            border-radius: 12px;
            padding: 0.52rem 0.68rem;
            min-height: 48px;
        }
        .mini-stat-card .mini-label {
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 0.1rem;
        }
        .mini-stat-card .mini-value {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
        }
        .tiny-stat-card {
            background: #fff;
            border: 1px solid #dbeafe;
            border-left: 3px solid #60a5fa;
            border-radius: 10px;
            padding: 0.42rem 0.58rem;
            min-height: 42px;
        }
        .tiny-stat-card .tiny-label {
            font-size: 0.58rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 0.08rem;
        }
        .tiny-stat-card .tiny-value {
            font-size: 0.94rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.12;
        }
        .server-browser-box {
            background: rgba(255,255,255,0.96);
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 0.7rem 0.8rem 0.8rem;
            margin-top: 0.45rem;
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
        }
        .server-browser-path {
            font-size: 0.78rem;
            color: #334155;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.45rem 0.55rem;
            margin: 0.45rem 0 0.55rem;
            word-break: break-all;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800;900&display=swap');

        /* ============================================================
           EasyICU paper-figure visual skin
           Aligns the live web UI with the accepted image2 figure system.
           ============================================================ */
        :root {
            --figure-navy: #0b1f44;
            --figure-blue: #2563eb;
            --figure-cyan: #0891b2;
            --figure-teal: #0f766e;
            --figure-orange: #ea7a1a;
            --figure-bg: #f4f8fc;
            --figure-card: #ffffff;
            --figure-soft: #edf4fb;
            --figure-line: #cddbeb;
            --figure-line-strong: #b7cae2;
            --figure-muted: #60718a;
            --figure-shadow: 0 10px 30px rgba(15, 31, 68, 0.055), 0 1px 2px rgba(15, 31, 68, 0.05);
            --figure-shadow-hover: 0 16px 38px rgba(37, 99, 235, 0.12), 0 4px 14px rgba(15, 31, 68, 0.06);
            --gradient-primary: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
            --gradient-info: linear-gradient(135deg, #0891b2 0%, #14b8a6 100%);
            --gradient-hero: linear-gradient(135deg, #102a56 0%, #1d4f86 58%, #0f766e 100%);
            --shadow-card: var(--figure-shadow);
            --shadow-hover: var(--figure-shadow-hover);
            --shadow-glow: 0 10px 26px rgba(37, 99, 235, 0.16);
            --border-subtle: var(--figure-line);
            --border-light: rgba(37, 99, 235, 0.16);
            --bg-primary: var(--figure-bg);
            --bg-secondary: var(--figure-card);
            --bg-tertiary: var(--figure-soft);
            --primary-color: var(--figure-blue);
            --secondary-color: var(--figure-cyan);
            --accent-color: #14b8a6;
            --font-sans: 'Source Sans 3', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        html, body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
        .stAlert, div[data-testid="stMetric"], div[data-baseweb="select"],
        div[data-baseweb="input"], div[data-baseweb="textarea"],
        div[data-baseweb="tab-list"], h1, h2, h3, h4, h5, h6, label,
        input, textarea, select, option, td, th {
            font-family: var(--font-sans) !important;
        }

        html, body,
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"], [data-testid="stMain"], .main {
            background:
                radial-gradient(circle at 16% 0%, rgba(37, 99, 235, 0.055), transparent 32%),
                radial-gradient(circle at 92% 8%, rgba(20, 184, 166, 0.055), transparent 30%),
                var(--figure-bg) !important;
            color: #0f172a !important;
        }

        .block-container {
            max-width: clamp(1120px, 94vw, 1880px) !important;
            padding-left: clamp(0.7rem, 1vw, 1.6rem) !important;
            padding-right: clamp(0.7rem, 1vw, 1.6rem) !important;
        }

        div[data-baseweb="tab-list"] {
            background: rgba(237, 244, 251, 0.94) !important;
            border: 1px solid var(--figure-line) !important;
            border-radius: 999px !important;
            padding: 7px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.78), var(--figure-shadow) !important;
            backdrop-filter: blur(10px);
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
        }

        div[data-baseweb="tab-list"] button {
            color: #5d6f88 !important;
            border-radius: 999px !important;
            letter-spacing: 0.01em !important;
            min-height: 34px !important;
            font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
            white-space: nowrap !important;
            flex: 0 0 auto !important;
        }

        div[data-baseweb="tab-list"] button p {
            font-size: clamp(0.78rem, 0.08vw + 0.76rem, 0.92rem) !important;
            white-space: nowrap !important;
        }

        div[data-baseweb="tab-list"] button:hover {
            background: rgba(255,255,255,0.72) !important;
            color: var(--figure-navy) !important;
        }

        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18) !important;
        }

        div[data-baseweb="tab-list"] button[aria-selected="true"] * {
            color: #ffffff !important;
        }

        div[data-testid="stMetric"],
        .metric-card, .feature-card, .patient-card,
        .compact-summary-card, .module-preview-card,
        .mini-stat-card, .tiny-stat-card,
        details[data-testid="stExpander"],
        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"],
        [data-testid="stVegaLiteChart"] {
            background: var(--figure-card) !important;
            border: 1px solid var(--figure-line) !important;
            border-radius: 18px !important;
            box-shadow: var(--figure-shadow) !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }

        div[data-testid="stMetric"]::before,
        .compact-summary-card::before,
        .mini-stat-card::before,
        .tiny-stat-card::before {
            background: var(--gradient-primary) !important;
        }

        .compact-summary-card,
        .mini-stat-card,
        .tiny-stat-card {
            border-left: 4px solid #5aa8ff !important;
        }

        .module-preview-card {
            background: linear-gradient(135deg, #ffffff 0%, #f4f9ff 100%) !important;
        }

        .module-feature-chip,
        .preview-badge,
        div[data-baseweb="tag"] {
            background: #eaf2ff !important;
            border: 1px solid #cfe0f6 !important;
            color: var(--figure-navy) !important;
            box-shadow: none !important;
        }

        [data-testid="stMultiSelect"] [data-baseweb="tag"],
        [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg *,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] path {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }

        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
            border: 1px solid #1d4ed8 !important;
            color: #ffffff !important;
            box-shadow: 0 5px 14px rgba(37,99,235,0.22) !important;
        }

        div[data-testid="stMetric"] label,
        .compact-summary-card .summary-label,
        .mini-stat-card .mini-label,
        .tiny-stat-card .tiny-label,
        .inline-control-label {
            color: var(--figure-muted) !important;
            font-weight: 800 !important;
            letter-spacing: 0.075em !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        .compact-summary-card .summary-value,
        .mini-stat-card .mini-value,
        .tiny-stat-card .tiny-value {
            color: var(--figure-navy) !important;
            letter-spacing: 0 !important;
        }

        [data-testid="stDataFrame"] {
            overflow: hidden !important;
        }

        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] thead th {
            background: #f4f7fb !important;
            color: #5f6f84 !important;
            text-transform: none !important;
        }

        .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton button {
            background: var(--gradient-primary) !important;
            border: 1px solid rgba(37, 99, 235, 0.12) !important;
            border-radius: 12px !important;
            box-shadow: 0 9px 20px rgba(37, 99, 235, 0.16) !important;
        }

        div[data-testid="stButton"] > button[kind="secondary"],
        div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
            background: #ffffff !important;
            border: 1px solid var(--figure-line) !important;
            border-radius: 12px !important;
            color: var(--figure-navy) !important;
            box-shadow: 0 4px 12px rgba(15, 31, 68, 0.04) !important;
        }

        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] .stButton button * {
            color: #1f3b63 !important;
            fill: #1f3b63 !important;
            white-space: normal !important;
            line-height: 1.15 !important;
        }

        [data-testid="stSidebar"] .stButton button {
            background: linear-gradient(180deg, #ffffff 0%, #f4f8fd 100%) !important;
            border: 1px solid #bfd4ed !important;
            border-radius: 12px !important;
            min-height: 2.45rem !important;
            box-shadow: 0 6px 15px rgba(15, 23, 42, 0.055) !important;
        }

        [data-testid="stSidebar"] .stButton button[kind="primary"],
        [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] .stButton button:hover {
            background: linear-gradient(135deg, #1d7ef2 0%, #0b8fc7 100%) !important;
            border-color: #166fd0 !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22) !important;
        }

        [data-testid="stSidebar"] .stButton button[kind="primary"],
        [data-testid="stSidebar"] .stButton button[kind="primary"] *,
        [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] *,
        [data-testid="stSidebar"] .stButton button:hover,
        [data-testid="stSidebar"] .stButton button:hover * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: #ffffff !important;
            border: 1px solid var(--figure-line) !important;
            border-radius: 13px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8) !important;
        }

        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within {
            border-color: #7fb4ff !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10) !important;
        }

        div[data-testid="stAlert"] {
            border: 1px solid var(--figure-line) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }

        .highlight-card,
        .step-indicator.active,
        .mode-card-tag.blue {
            background: rgba(37, 99, 235, 0.07) !important;
            border-color: rgba(37, 99, 235, 0.16) !important;
            color: var(--figure-navy) !important;
        }

        .patient-card:hover,
        .metric-card:hover,
        .feature-card:hover {
            border-color: #a9c7ee !important;
            box-shadow: var(--figure-shadow-hover) !important;
        }

        .compact-section-title,
        .preview-toolbar-title,
        .module-preview-card .title,
        .app-page-title {
            color: var(--figure-navy) !important;
        }

        .compact-section-desc,
        .preview-toolbar-note,
        .subtle-preview-note,
        .app-page-subtitle {
            color: var(--figure-muted) !important;
        }

        .app-page-header {
            border-bottom-color: rgba(205, 219, 235, 0.82) !important;
        }

        .app-page-kicker {
            color: var(--figure-blue) !important;
        }

        .app-page-icon {
            background: rgba(37, 99, 235, 0.08) !important;
            border-color: rgba(37, 99, 235, 0.14) !important;
        }

        .workflow-figure-shell {
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 18px;
            box-shadow: var(--figure-shadow);
            padding: clamp(0.9rem, 0.5vw + 0.75rem, 1.25rem);
            margin: 0.55rem 0 1rem;
        }

        .workflow-figure-title {
            color: var(--figure-navy);
            font-weight: 900;
            font-size: clamp(1.08rem, 0.55vw + 0.95rem, 1.55rem);
            letter-spacing: -0.025em;
            margin-bottom: 0.2rem;
        }

        .workflow-figure-subtitle {
            color: var(--figure-muted);
            font-weight: 650;
            font-size: clamp(0.78rem, 0.15vw + 0.74rem, 0.92rem);
            margin-bottom: 0.9rem;
        }

        .workflow-pipeline-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 26px minmax(0, 1.25fr) 26px minmax(0, 1.25fr) 26px minmax(0, 1fr);
            gap: 0.55rem;
            align-items: stretch;
        }

        .workflow-card {
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 14px;
            padding: 0.9rem 0.95rem;
            min-height: 286px;
            box-shadow: 0 5px 16px rgba(15, 31, 68, 0.035);
            color: var(--figure-navy);
        }

        .workflow-card-head {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 0.65rem;
            align-items: start;
            margin-bottom: 0.72rem;
        }

        .workflow-badge {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #082957;
            color: #ffffff;
            font-size: 1.02rem;
            font-weight: 900;
            line-height: 1;
            box-shadow: 0 6px 16px rgba(8, 41, 87, 0.16);
        }

        .workflow-card-title {
            color: #082957;
            font-weight: 900;
            font-size: 1.02rem;
            line-height: 1.16;
            letter-spacing: -0.01em;
        }

        .workflow-card-kicker {
            color: var(--figure-muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-top: 0.08rem;
        }

        .workflow-field {
            margin: 0.52rem 0;
        }

        .workflow-label {
            color: #172b4d;
            font-size: 0.74rem;
            font-weight: 750;
            margin-bottom: 0.24rem;
        }

        .workflow-input {
            min-height: 34px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.45rem;
            border: 1px solid #d9e3f1;
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            color: #14233d;
            padding: 0.42rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 650;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.92);
            overflow-wrap: anywhere;
        }

        .workflow-button {
            background: linear-gradient(135deg, #2563eb 0%, #0d7fd1 100%);
            color: #ffffff;
            border-radius: 8px;
            min-height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.42rem;
            margin-top: 0.72rem;
            font-weight: 850;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.20);
        }

        .workflow-status {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-height: 38px;
            margin-top: 0.72rem;
            padding: 0.46rem 0.62rem;
            border: 1px solid #d7eadf;
            border-radius: 8px;
            background: #f2fbf6;
            color: #14532d;
            font-weight: 800;
            font-size: 0.76rem;
        }

        .workflow-status.warn {
            border-color: #f8ddb0;
            background: #fff8ed;
            color: #92400e;
        }

        .workflow-check-dot {
            width: 19px;
            height: 19px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #2ca25f;
            color: #ffffff;
            font-size: 0.76rem;
            font-weight: 900;
            flex: 0 0 auto;
        }

        .workflow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #2563eb;
            font-size: 1.7rem;
            font-weight: 900;
            padding-top: 2.5rem;
        }

        .workflow-concepts {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.4rem 0.48rem;
            margin-top: 0.45rem;
        }

        .workflow-concept {
            display: flex;
            align-items: center;
            gap: 0.36rem;
            color: #11213b;
            font-size: 0.74rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .workflow-tick {
            width: 15px;
            height: 15px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #2f70d8;
            color: #ffffff;
            font-size: 0.64rem;
            font-weight: 900;
            flex: 0 0 auto;
        }

        .workflow-summary-panel {
            margin-top: 0.95rem;
            border: 1px solid var(--figure-line);
            border-radius: 15px;
            background: #ffffff;
            padding: 0.9rem 1rem;
        }

        .workflow-summary-grid {
            display: grid;
            grid-template-columns: 1.35fr 0.9fr;
            gap: 1rem;
            align-items: stretch;
        }

        .workflow-success-strip {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            border: 1px solid #d7eadf;
            border-radius: 8px;
            background: linear-gradient(90deg, #edf9f2 0%, #f8fdfa 100%);
            color: #14532d;
            padding: 0.56rem 0.68rem;
            font-weight: 850;
            margin-bottom: 0.62rem;
        }

        .workflow-success-strip.warn {
            border-color: #f8ddb0;
            background: linear-gradient(90deg, #fff8ed 0%, #fffdf8 100%);
            color: #92400e;
        }

        .workflow-file-list {
            border: 1px solid #dce6f3;
            border-radius: 9px;
            background: #ffffff;
            padding: 0.48rem 0.62rem;
            color: #52647d;
            font-size: 0.72rem;
            line-height: 1.75;
        }

        .workflow-stat-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
        }

        .workflow-mini-stat {
            border: 1px solid #dce6f3;
            border-radius: 10px;
            background: #ffffff;
            padding: 0.55rem 0.62rem;
        }

        .workflow-mini-label {
            color: var(--figure-muted);
            font-size: 0.57rem;
            font-weight: 900;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .workflow-mini-value {
            color: #082957;
            font-weight: 900;
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }

        .workflow-guide-title {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            margin: 0.9rem 0 0.6rem;
            color: var(--figure-navy);
            font-size: 1.42rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }

        .workflow-guide-title::before {
            content: "";
            width: 6px;
            height: 28px;
            border-radius: 4px;
            background: linear-gradient(180deg, #2563eb 0%, #0891b2 100%);
            display: inline-block;
            flex: 0 0 auto;
        }

        .quality-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin-bottom: 1rem;
        }

        .quality-summary-card {
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            text-align: center;
            box-shadow: 0 7px 20px rgba(15, 31, 68, 0.04);
        }

        .quality-summary-label {
            color: var(--figure-muted);
            font-size: 0.64rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            margin-bottom: 0.22rem;
        }

        .quality-summary-value {
            color: var(--figure-navy);
            font-size: 1.28rem;
            font-weight: 900;
            line-height: 1.1;
        }

        @media (max-width: 1500px) {
            .workflow-pipeline-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .workflow-arrow {
                display: none;
            }
            .workflow-card {
                min-height: auto;
            }
        }

        @media (max-width: 900px) {
            .workflow-pipeline-grid {
                grid-template-columns: 1fr;
            }
            .workflow-summary-grid,
            .workflow-stat-row,
            .quality-summary-grid {
                grid-template-columns: 1fr;
            }
        }

        .audit-figure-panel {
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 18px;
            box-shadow: var(--figure-shadow);
            padding: 0.95rem 1rem 1.05rem;
            margin: 0.4rem 0 0.95rem;
        }

        .audit-panel-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--figure-navy);
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.65rem;
        }

        .audit-panel-letter {
            width: 24px;
            height: 24px;
            border-radius: 7px;
            background: var(--figure-navy);
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 0.76rem;
            line-height: 1;
        }

        .audit-summary-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.72rem;
            margin: 0.2rem 0 0.85rem;
        }

        .audit-summary-card {
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 14px;
            padding: 0.78rem 0.85rem;
            box-shadow: 0 6px 18px rgba(15, 31, 68, 0.04);
        }

        .audit-summary-label {
            color: var(--figure-muted);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.075em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }

        .audit-summary-value {
            color: var(--figure-navy);
            font-size: 1.35rem;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .audit-flow {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            padding: 0.25rem 0.15rem;
        }

        .audit-flow-step {
            border: 1px solid var(--figure-line);
            border-radius: 13px;
            background: #ffffff;
            padding: 0.58rem 0.75rem;
            text-align: center;
            color: var(--figure-navy);
            position: relative;
        }

        .audit-flow-step:not(:last-child)::after {
            content: '↓';
            position: absolute;
            left: 50%;
            bottom: -0.72rem;
            transform: translateX(-50%);
            color: var(--figure-orange);
            font-weight: 900;
            font-size: 0.85rem;
        }

        .audit-flow-label {
            font-size: 0.72rem;
            font-weight: 800;
            color: #53657c;
        }

        .audit-flow-value {
            font-size: 1.08rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }

        .audit-flow-excluded {
            color: #b45309;
            font-size: 0.72rem;
            font-weight: 800;
            margin-top: 0.12rem;
        }

        .audit-denominator-note {
            border: 1px solid #c9d9ee;
            border-radius: 12px;
            background: #f7fbff;
            color: #3d516a;
            padding: 0.65rem 0.8rem;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .cohort-demo-workspace {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 0.78rem;
            background: #ffffff;
            border: 1px solid var(--figure-line);
            border-radius: 16px;
            padding: 0.78rem 0.92rem;
            margin: 0.45rem 0 0.92rem;
            box-shadow: 0 8px 24px rgba(15, 31, 68, 0.045);
            color: var(--figure-navy);
        }

        .cohort-demo-badge {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background: var(--figure-navy);
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 0.92rem;
            line-height: 1;
        }

        .cohort-demo-title {
            color: var(--figure-navy);
            font-weight: 900;
            font-size: 0.96rem;
            letter-spacing: -0.015em;
            margin-bottom: 0.1rem;
        }

        .cohort-demo-subtitle {
            color: var(--figure-muted);
            font-size: 0.78rem;
            line-height: 1.42;
        }

        .cohort-demo-status {
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            padding: 0.32rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 850;
            white-space: nowrap;
        }

        @media (max-width: 900px) {
            .audit-summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .cohort-demo-workspace {
                grid-template-columns: auto 1fr;
            }
            .cohort-demo-status {
                grid-column: 1 / -1;
                width: fit-content;
            }
        }
    </style>
    """, unsafe_allow_html=True)
