"""Paper-style figure rendering entry points for the EasyICU webapp."""

from __future__ import annotations

from typing import Any


def _install_app_context(app_context: dict[str, Any]) -> None:
    """Expose app-level helpers/constants to extracted workflows."""
    protected = {'render_publication_composite_figure', '_render_paper_panel_css', 'render_quick_figure_panel', 'render_cohort_figure_panel', "_install_app_context"}
    for name, value in app_context.items():
        if not name.startswith("__") and name not in protected:
            globals()[name] = value


def render_publication_composite_figure(panel: str, app_context: dict[str, Any] | None = None) -> None:
    """Render the exact accepted manuscript composite figure in the web app."""
    if app_context is not None:
        _install_app_context(app_context)
    
    image_path = _publication_figure_image_path(panel)
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1536px !important;
            padding: 0.9rem 0.9rem 1.2rem !important;
        }
        .publication-composite-stage {
            width: min(100%, 1536px);
            margin: 0 auto;
            background: #f4f8fc;
            border-radius: 0;
        }
        .publication-composite-stage img {
            display: block;
            width: 100%;
            height: auto;
            border: 0;
            box-shadow: none;
            background: #f4f8fc;
        }
        .publication-composite-fallback {
            margin: 40px auto;
            padding: 28px;
            border: 1px solid #cddbeb;
            border-radius: 16px;
            background: #ffffff;
            color: #0b1f44;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not image_path:
        st.markdown(
            f'<div class="publication-composite-fallback">Missing publication composite image for {html.escape(panel)}.</div>',
            unsafe_allow_html=True,
        )
        return

    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    st.markdown(
        f'''
        <div class="publication-composite-stage" data-panel="{html.escape(panel)}">
            <img src="data:image/png;base64,{encoded}" alt="{html.escape(panel)} publication composite" />
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_panel_css(app_context: dict[str, Any] | None = None) -> None:
    """Shared paper-panel CSS for live figure routes."""
    if app_context is not None:
        _install_app_context(app_context)
    
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 780px !important;
            padding: 0.85rem 0.9rem 1rem !important;
        }
        .paper-panel {
            background: #ffffff;
            border: 1px solid #cddbeb;
            border-radius: 14px;
            box-shadow: none;
            padding: 0.85rem 0.95rem 0.95rem;
            color: #0b1f44;
            font-family: var(--font-sans);
        }
        .paper-panel-header {
            display: flex;
            align-items: flex-start;
            gap: 0.72rem;
            border-bottom: 1px solid #dfe7f2;
            padding-bottom: 0.48rem;
            margin-bottom: 0.7rem;
        }
        .paper-panel-letter {
            width: 32px;
            height: 32px;
            border-radius: 7px;
            background: #082957;
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 1.05rem;
            line-height: 1;
            flex: 0 0 auto;
        }
        .paper-panel-title {
            font-size: 1.0rem;
            font-weight: 900;
            color: #082957;
            letter-spacing: -0.01em;
            line-height: 1.16;
        }
        .paper-panel-subtitle {
            margin-top: 0.12rem;
            font-size: 0.72rem;
            color: #5c6d86;
            line-height: 1.3;
        }
        .paper-eyebrow {
            color: #2563eb;
            font-size: 0.58rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            margin-bottom: 0.16rem;
        }
        .paper-card {
            background: #ffffff;
            border: 1px solid #dce6f3;
            border-radius: 10px;
            padding: 0.62rem 0.7rem;
        }
        .paper-soft-card {
            background: #f6faff;
            border: 1px solid #cfe0f6;
            border-radius: 10px;
            padding: 0.6rem 0.72rem;
        }
        .paper-grid-3 {
            display: grid;
            grid-template-columns: 1.8fr 0.62fr 0.62fr;
            gap: 0.55rem;
            margin-bottom: 0.65rem;
        }
        .paper-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
            margin-bottom: 0.58rem;
        }
        .paper-grid-5 {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.45rem;
        }
        .paper-metric-label {
            color: #6b7c95;
            font-size: 0.56rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 900;
            margin-bottom: 0.14rem;
        }
        .paper-metric-value {
            color: #071f45;
            font-size: 1.04rem;
            line-height: 1.05;
            letter-spacing: -0.02em;
            font-weight: 900;
        }
        .paper-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.22rem;
            margin-top: 0.42rem;
        }
        .paper-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid #cfe0f6;
            background: #edf4ff;
            color: #0b1f44;
            border-radius: 999px;
            padding: 0.08rem 0.42rem;
            font-size: 0.6rem;
            line-height: 1.25;
            font-weight: 800;
        }
        .paper-control-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.55rem;
            align-items: center;
            margin: 0.55rem 0 0.48rem;
            font-size: 0.72rem;
        }
        .paper-radio-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 99px;
            border: 2px solid #6d6af2;
            margin-right: 0.32rem;
            vertical-align: -1px;
            box-shadow: inset 0 0 0 3px #ffffff;
            background: #6d6af2;
        }
        .paper-radio-empty {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 99px;
            border: 1px solid #b8c4d4;
            margin: 0 0.32rem 0 0.52rem;
            vertical-align: -1px;
            background: #ffffff;
        }
        .paper-select {
            min-width: 110px;
            border: 1px solid #d4dfed;
            border-radius: 8px;
            padding: 0.34rem 0.58rem;
            color: #0b1f44;
            background: #ffffff;
            text-align: right;
        }
        .paper-table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border: 1px solid #dce6f3;
            border-radius: 10px;
            font-size: 0.58rem;
        }
        .paper-table th {
            background: #f6f8fb;
            color: #53637a;
            font-weight: 700;
            text-align: center;
            padding: 0.32rem 0.28rem;
            border: 1px solid #e4ebf4;
        }
        .paper-table td {
            color: #0b1f44;
            padding: 0.34rem 0.3rem;
            border: 1px solid #e4ebf4;
            text-align: right;
        }
        .paper-table td:first-child,
        .paper-table th:first-child {
            text-align: center;
            color: #60718a;
        }
        .paper-note {
            color: #60718a;
            font-size: 0.62rem;
            line-height: 1.35;
            margin-top: 0.4rem;
        }
        .paper-chart-grid-2 {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.72rem 0.9rem;
        }
        .paper-mini-chart-title {
            font-size: 0.62rem;
            color: #263b58;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }
        .paper-legend {
            display: flex;
            justify-content: flex-end;
            gap: 1.0rem;
            align-items: center;
            color: #263b58;
            font-size: 0.58rem;
            margin: -0.2rem 0 0.4rem;
        }
        .paper-legend-line {
            display: inline-block;
            width: 22px;
            height: 0;
            border-top: 2px solid #2563eb;
            margin-right: 0.25rem;
            vertical-align: middle;
        }
        .paper-legend-line.dash {
            border-top-style: dashed;
            border-top-color: #8c98aa;
        }
        .paper-legend-line.low {
            border-top-style: dashed;
            border-top-color: #ef4444;
        }
        .paper-legend-line.high {
            border-top-style: dashed;
            border-top-color: #f97316;
        }
        .paper-tabs {
            display: flex;
            gap: 0.28rem;
            background: #eef4fb;
            border-radius: 8px;
            padding: 0.25rem;
            margin: 0.52rem 0 0.55rem;
        }
        .paper-tab {
            color: #5c6d86;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 0.28rem 0.52rem;
            border-radius: 7px;
        }
        .paper-tab.active {
            color: #ffffff;
            background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
        }
        .paper-bar-row {
            display: grid;
            grid-template-columns: 110px 1fr 38px;
            gap: 0.46rem;
            align-items: center;
            margin: 0.28rem 0;
            font-size: 0.58rem;
        }
        .paper-bar-track {
            height: 10px;
            background: #edf2f7;
            border-radius: 999px;
            overflow: hidden;
        }
        .paper-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: #ef4444;
        }
        .paper-db-card {
            border: 1px solid #dce6f3;
            border-radius: 9px;
            padding: 0.45rem 0.5rem;
            background: #ffffff;
            border-left-width: 3px;
        }
        .paper-db-name {
            color: #52647d;
            font-size: 0.56rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 800;
        }
        .paper-db-value {
            font-size: 0.9rem;
            font-weight: 900;
            color: #071f45;
        }
        .paper-two-col {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 0.65rem;
        }
        .paper-flow-step {
            border: 1px solid #d7dfeb;
            border-radius: 8px;
            padding: 0.34rem 0.5rem;
            text-align: center;
            font-size: 0.58rem;
            margin-bottom: 0.4rem;
            position: relative;
            background: #fffaf4;
        }
        .paper-flow-step:not(:last-child)::after {
            content: '↓';
            position: absolute;
            left: 50%;
            bottom: -0.46rem;
            transform: translateX(-50%);
            color: #f59e0b;
            font-weight: 900;
        }
        .paper-heatmap {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.54rem;
        }
        .paper-heatmap th,
        .paper-heatmap td {
            border: 1px solid #e4ebf4;
            padding: 0.26rem;
            text-align: center;
        }
        .paper-heatmap th {
            background: #f6f8fb;
            font-weight: 800;
        }
        @media (max-width: 760px) {
            .paper-grid-3,
            .paper-grid-4,
            .paper-grid-5,
            .paper-chart-grid-2,
            .paper-two-col {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_quick_figure_panel(panel: str, app_context: dict[str, Any] | None = None) -> None:
    """Render live, paper-style Figure 3 panels without full app chrome."""
    if app_context is not None:
        _install_app_context(app_context)
    
    if panel == "Data Tables":
        _render_paper_data_panel()
    elif panel == "Time Series":
        _render_paper_timeseries_panel()
    elif panel == "Patient Overview":
        _render_paper_patient_panel()
    elif panel == "Data Quality":
        _render_paper_quality_panel()


def render_cohort_figure_panel(panel: str, app_context: dict[str, Any] | None = None) -> None:
    """Render live, paper-style Supplementary Figure S1 panels."""
    if app_context is not None:
        _install_app_context(app_context)
    
    _render_paper_panel_css()
    if panel == "Group Contrast":
        _render_paper_group_panel()
    elif panel == "Coverage Audit":
        _render_paper_coverage_panel()
    elif panel == "Cross-DB Benchmark":
        _render_paper_crossdb_panel()
    elif panel == "Cohort Snapshot":
        _render_paper_snapshot_panel()
    elif panel == "SOFA-1 vs SOFA-2":
        _render_paper_sofa_panel()
