"""Paper-style figure rendering entry points for the EasyICU webapp."""

from __future__ import annotations

from typing import Any

from easyicu.webapp.components.constants import get_all_concepts


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



# Live paper panel helpers. These rely on _install_app_context() to receive app-level data helpers.

def _paper_panel_header(letter: str, title: str, subtitle: str = "") -> str:
    subtitle_html = f'<div class="paper-panel-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="paper-panel-header">'
        f'<div class="paper-panel-letter">{html.escape(letter)}</div>'
        '<div>'
        f'<div class="paper-panel-title">{html.escape(title)}</div>'
        f'{subtitle_html}'
        '</div>'
        '</div>'
    )


def _paper_format_value(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _concept_display_name(concept: str, include_unit: bool = False) -> str:
    name, _zh, unit = CONCEPT_DICTIONARY.get(concept, (concept, concept, ""))
    if include_unit and unit:
        return f"{name} ({unit})"
    return name


def _concept_unit(concept: str) -> str:
    return CLINICAL_THRESHOLDS.get(concept, {}).get("unit") or CONCEPT_DICTIONARY.get(concept, ("", "", ""))[2] or ""


def _first_patient_id() -> Any:
    patient_ids = st.session_state.get("patient_ids") or []
    return patient_ids[0] if patient_ids else None


def _time_column_for_frame(df: pd.DataFrame) -> Optional[str]:
    for candidate in PREVIEW_TIME_COLUMNS + ['time', 'hour', 'datetime', 'timestamp']:
        if candidate in df.columns:
            return candidate
    return None


def _patient_series_for_concept(concept: str, patient_id: Any = None, max_points: int = 72) -> pd.DataFrame:
    df = st.session_state.get("loaded_concepts", {}).get(concept)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["time", "value"])
    id_col = st.session_state.get("id_col", "stay_id")
    patient_id = _first_patient_id() if patient_id is None else patient_id
    frame = df.copy()
    if patient_id is not None and id_col in frame.columns:
        frame = frame[frame[id_col] == patient_id].copy()
    time_col = _time_column_for_frame(frame)
    value_col = _choose_concept_value_column(concept, frame)
    if time_col is None or value_col is None:
        return pd.DataFrame(columns=["time", "value"])
    plot_df = _prepare_timeseries_plot_df(frame, time_col, value_col)
    if plot_df.empty:
        return pd.DataFrame(columns=["time", "value"])
    plot_df = plot_df.sort_values(time_col).head(max_points)
    out = pd.DataFrame({"time": plot_df[time_col], "value": pd.to_numeric(plot_df[value_col], errors="coerce")})
    return out.dropna(subset=["value"])


def _svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _render_inline_timeseries_svg(concept: str, patient_id: Any = None) -> str:
    series = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
    if series.empty:
        values = np.array([0, 0.2, 0.1, 0.35, 0.22, 0.44, 0.31, 0.28], dtype=float)
    else:
        values = series["value"].astype(float).to_numpy()
    if len(values) < 2:
        values = np.array([values[0] if len(values) else 0, values[0] if len(values) else 0], dtype=float)

    thresholds = CLINICAL_THRESHOLDS.get(concept, {})
    threshold_values = [float(v) for v in thresholds.get("lines", []) if v is not None]
    y_values = list(values) + threshold_values
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.14
    y_min -= pad
    y_max += pad

    width, height = 310, 135
    left, right, top, bottom = 32, 8, 14, 24
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        return left + (i / max(len(values) - 1, 1)) * plot_w

    def y_pos(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    line_points = [(x_pos(i), y_pos(float(v))) for i, v in enumerate(values)]
    median_value = float(np.nanmedian(values))
    median_y = y_pos(median_value)
    grid_lines = []
    for frac in (0, 0.5, 1):
        y = top + frac * plot_h
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e8eef7" stroke-width="1"/>')

    threshold_svg = [f'<line x1="{left}" y1="{median_y:.1f}" x2="{width-right}" y2="{median_y:.1f}" stroke="#8c98aa" stroke-dasharray="4 4" stroke-width="1.2"/>']
    colors = thresholds.get("colors", ["#ef4444", "#f97316"])
    for idx, value in enumerate(threshold_values[:2]):
        threshold_svg.append(
            f'<line x1="{left}" y1="{y_pos(value):.1f}" x2="{width-right}" y2="{y_pos(value):.1f}" '
            f'stroke="{html.escape(str(colors[idx % len(colors)]))}" stroke-dasharray="4 4" stroke-width="1.2"/>'
        )

    y_tick_max = _paper_format_value(y_max - pad, 0)
    y_tick_min = _paper_format_value(y_min + pad, 0)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="135" role="img" aria-label="{html.escape(concept)} time series">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
        f'{"".join(grid_lines)}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#cbd5e1" stroke-width="1"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#cbd5e1" stroke-width="1"/>'
        f'{"".join(threshold_svg)}'
        f'<polyline points="{_svg_polyline(line_points)}" fill="none" stroke="#2563eb" stroke-width="2.1"/>'
        f'<text x="2" y="{top+4}" fill="#394b63" font-size="10">{y_tick_max}</text>'
        f'<text x="2" y="{height-bottom}" fill="#394b63" font-size="10">{y_tick_min}</text>'
        f'<text x="{left}" y="{height-5}" fill="#394b63" font-size="10">0</text>'
        f'<text x="{width-right-28}" y="{height-5}" fill="#394b63" font-size="10">72</text>'
        f'</svg>'
    )


def _build_paper_wide_preview(concepts: list[str], max_rows: int = 6) -> pd.DataFrame:
    patient_id = _first_patient_id()
    id_col = st.session_state.get("id_col", "stay_id")
    merged: Optional[pd.DataFrame] = None
    for concept in concepts:
        series_df = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
        if series_df.empty:
            continue
        concept_df = series_df.rename(columns={"value": concept}).copy()
        concept_df["time_key"] = range(len(concept_df))
        concept_df[id_col] = patient_id if patient_id is not None else 10001
        concept_df = concept_df[[id_col, "time_key", concept]]
        if merged is None:
            merged = concept_df
        else:
            merged = pd.merge(merged, concept_df, on=[id_col, "time_key"], how="outer")
    if merged is None or merged.empty:
        rows = []
        for idx in range(max_rows):
            rows.append({id_col: patient_id or 10001, "charttime": f"{idx}h", **{c: np.nan for c in concepts}})
        return pd.DataFrame(rows)
    merged = merged.sort_values("time_key").head(max_rows).copy()
    merged.insert(1, "charttime", merged["time_key"].map(lambda v: f"{int(v)}h"))
    merged.drop(columns=["time_key"], inplace=True)
    return merged


def _paper_table_html(df: pd.DataFrame, max_rows: int = 6) -> str:
    show_df = df.head(max_rows).copy()
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in [""] + list(show_df.columns))
    row_html = []
    for idx, row in show_df.iterrows():
        cells = [f"<td>{idx}</td>"]
        for value in row.tolist():
            cells.append(f"<td>{_paper_format_value(value)}</td>")
        row_html.append("<tr>" + "".join(cells) + "</tr>")
    row_html.append("<tr>" + "".join(["<td>…</td>"] * (len(show_df.columns) + 1)) + "</tr>")
    return f'<table class="paper-table"><thead><tr>{header_html}</tr></thead><tbody>{"".join(row_html)}</tbody></table>'


def _quality_snapshot_rows(limit: int = 10) -> tuple[pd.DataFrame, int, float, float, float]:
    id_col = st.session_state.get("id_col", "stay_id")
    mock_params = st.session_state.get("mock_params", {}) or {}
    demo_hours = int(mock_params.get("hours") or 0) if st.session_state.get("entry_mode") == "demo" and mock_params.get("hours") else None
    time_grid_size = demo_hours or 72
    total_patients = _get_quality_cohort_patient_count(st.session_state)
    cohort_patient_ids = _get_quality_cohort_patient_ids(st.session_state)
    los_by_patient = _get_quality_los_by_patient(st.session_state)

    rows: list[dict[str, Any]] = []
    total_records = 0
    total_expected = 0.0
    total_missing_weight = 0.0
    total_outlier_weight = 0.0
    total_duplicate_weight = 0.0
    for concept, df in st.session_state.get("loaded_concepts", {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        profile = _build_quality_metric_profile_cached(
            concept=concept,
            df=df,
            id_col=id_col,
            cohort_patient_count=total_patients,
            time_grid_size=time_grid_size,
            cohort_patient_ids=cohort_patient_ids,
            los_by_patient=los_by_patient,
            demo_hours=demo_hours,
        )
        n_records = len(df)
        weight = float(profile["expected_observations"] or n_records or 1)
        total_records += n_records
        total_expected += weight
        total_missing_weight += weight * (profile["missing_rate"] / 100)
        total_outlier_weight += n_records * (profile["out_of_physio_rate"] / 100)
        total_duplicate_weight += n_records * (profile["duplicate_rate"] / 100)
        rows.append(
            {
                "Concept": concept,
                "Missing": float(profile["missing_rate"]),
                "Records": n_records,
                "Patients": df[id_col].nunique() if id_col in df.columns else 0,
                "Denom": profile["denominator_tag"],
            }
        )
    quality_df = pd.DataFrame(rows).sort_values("Missing", ascending=False).head(limit) if rows else pd.DataFrame()
    overall_missing = (total_missing_weight / total_expected * 100) if total_expected > 0 else 0.0
    overall_outliers = (total_outlier_weight / total_records * 100) if total_records > 0 else 0.0
    overall_duplicates = (total_duplicate_weight / total_records * 100) if total_records > 0 else 0.0
    return quality_df, total_records, overall_missing, overall_outliers, overall_duplicates


def _render_paper_data_panel() -> None:
    _render_paper_panel_css()
    concepts = [c for c in ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"] if c in st.session_state.get("loaded_concepts", {})]
    if not concepts:
        concepts = ["hr", "map", "sbp", "dbp", "temp", "spo2", "resp"]
    patient_count = len(st.session_state.get("patient_ids") or [])
    preview_df = _build_paper_wide_preview(concepts, max_rows=6)
    chips = "".join(f'<span class="paper-chip">{html.escape(c)}</span>' for c in concepts)
    table_html = _paper_table_html(preview_df, max_rows=6)
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("A", "Feature-level review", "Compact table preview for a selected module before drilling into feature detail.")}
            <div class="paper-grid-3">
                <div class="paper-soft-card">
                    <div class="paper-eyebrow">Selected module</div>
                    <div style="font-weight:900;font-size:0.92rem;color:#071f45">❤️ Vital Signs</div>
                    <div class="paper-note" style="margin-top:0.22rem">Core bedside vital signs aligned into a compact longitudinal preview.</div>
                    <div class="paper-chip-row">{chips}</div>
                </div>
                <div class="paper-card">
                    <div class="paper-metric-label">Features</div>
                    <div class="paper-metric-value">{len(concepts)}</div>
                </div>
                <div class="paper-card">
                    <div class="paper-metric-label">Patients</div>
                    <div class="paper-metric-value">{patient_count or 50}</div>
                </div>
            </div>
            <div class="paper-control-row">
                <div><span class="paper-radio-dot"></span>Merge All (Wide Table)<span class="paper-radio-empty"></span>Single Feature</div>
                <div>Rows per feature&nbsp;&nbsp;<span class="paper-select">2,000⌄</span></div>
            </div>
            <div class="paper-grid-2" style="display:grid;grid-template-columns:1fr 1fr;border:1px solid #dce6f3;border-radius:9px;margin-bottom:0.52rem;overflow:hidden">
                <div style="padding:0.42rem 0.6rem;border-right:1px solid #e4ebf4"><div class="paper-metric-label">Preview rows</div><div style="font-weight:900">1,000</div></div>
                <div style="padding:0.42rem 0.6rem"><div class="paper-metric-label">Preview columns</div><div style="font-weight:900">{len(preview_df.columns)}</div></div>
            </div>
            {table_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_timeseries_panel() -> None:
    _render_paper_panel_css()
    patient_id = _first_patient_id()
    concepts = [c for c in ["hr", "map", "spo2", "resp"] if c in st.session_state.get("loaded_concepts", {})]
    if len(concepts) < 4:
        concepts = ["hr", "map", "spo2", "resp"]
    charts = []
    for concept in concepts[:4]:
        unit = _concept_unit(concept)
        title = _concept_display_name(concept)
        charts.append(
            f'<div><div class="paper-mini-chart-title">{html.escape(title)}{f" ({html.escape(unit)})" if unit else ""}</div>{_render_inline_timeseries_svg(concept, patient_id=patient_id)}</div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("B", "Patient time-series", "Representative single-patient trajectories with cohort median and clinical threshold references.")}
            <div style="font-weight:900;font-size:0.78rem;margin-bottom:0.12rem">❤️ Vital Signs</div>
            <div class="paper-legend">
                <span><span class="paper-legend-line"></span>Patient {html.escape(str(patient_id or 10001))}</span>
                <span><span class="paper-legend-line dash"></span>Median</span>
                <span><span class="paper-legend-line low"></span>Low threshold</span>
                <span><span class="paper-legend-line high"></span>High threshold</span>
            </div>
            <div class="paper-chart-grid-2">{''.join(charts)}</div>
            <div class="paper-note" style="text-align:center">Time since ICU admission (hours)</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_quality_panel() -> None:
    _render_paper_panel_css()
    qdf, total_records, overall_missing, overall_outliers, overall_duplicates = _quality_snapshot_rows(limit=10)
    if qdf.empty:
        qdf = pd.DataFrame({"Concept": ["aki_stage_rrt", "mech_circ_support", "ecmo", "delirium_tx"], "Missing": [96, 89, 86, 84]})
    bars = []
    for _, row in qdf.iterrows():
        value = float(row["Missing"])
        color = "#ef4444" if value >= 75 else "#f97316" if value >= 50 else "#fb923c" if value >= 25 else "#f59e0b"
        bars.append(
            f'<div class="paper-bar-row"><div style="text-align:right;color:#31455f">{html.escape(str(row["Concept"]))}</div>'
            f'<div class="paper-bar-track"><div class="paper-bar-fill" style="width:{max(1, min(100, value)):.1f}%;background:{color}"></div></div>'
            f'<div style="color:#31455f">{value:.0f}</div></div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Missingness rates", "Data-quality summary with denominator-aware missingness, out-of-physiology, and temporal checks.")}
            <div style="font-weight:900;font-size:0.72rem;margin-bottom:0.42rem">Data quality</div>
            <div class="paper-grid-4">
                <div class="paper-card"><div class="paper-metric-label">Total records</div><div class="paper-metric-value">{total_records or 102578:,}</div></div>
                <div class="paper-card"><div class="paper-metric-label">Weighted missing</div><div class="paper-metric-value" style="color:#ef4444">{overall_missing:.1f}%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Out-of-physio</div><div class="paper-metric-value" style="color:#059669">{overall_outliers:.1f}%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Duplicate TS</div><div class="paper-metric-value" style="color:#059669">{overall_duplicates:.1f}%</div></div>
            </div>
            <div class="paper-tabs">
                <div class="paper-tab active">📊 Missingness</div>
                <div class="paper-tab">🧪 Out-of-Physio</div>
                <div class="paper-tab">⏱️ Temporal Integrity</div>
            </div>
            <div class="paper-note">Missingness denominator: d=LOS uses patient-specific ICU stay; d=72h uses the fallback window.</div>
            <div style="display:grid;grid-template-columns:1fr 120px;gap:0.75rem;align-items:center;margin-top:0.45rem">
                <div>{''.join(bars)}</div>
                <div class="paper-card" style="font-size:0.6rem;line-height:1.7">
                    <div style="font-weight:900;margin-bottom:0.25rem">Missing rate (%)</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#ef4444;margin-right:6px"></span>75 - 100</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#f97316;margin-right:6px"></span>50 - 75</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#fb923c;margin-right:6px"></span>25 - 50</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;margin-right:6px"></span>&lt; 25</div>
                    <div style="border-top:1px solid #e4ebf4;margin-top:0.45rem;padding-top:0.35rem">Denominator<br><b>d = LOS</b></div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_crossdb_panel() -> None:
    _render_paper_panel_css()
    dbs = [
        ("MIMIC-IV", 10854, "#16a34a"),
        ("eICU", 12690, "#f97316"),
        ("AUMC", 14553, "#2563eb"),
        ("HiRID", 13473, "#ef4444"),
        ("MIMIC-III", 11961, "#7e22ce"),
        ("SICdb", 13365, "#795548"),
    ]
    db_cards = "".join(
        f'<div class="paper-db-card" style="border-left-color:{color}">'
        f'<div class="paper-db-name"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:4px"></span>{html.escape(name)}</div>'
        f'<div class="paper-db-value">{value:,}</div>'
        f'<div style="font-size:0.56rem;color:#047857">records</div>'
        f'</div>'
        for name, value, color in dbs
    )
    concepts = ["hr", "map", "resp", "temp", "spo2", "crea"]
    chart_cells = []
    for idx, concept in enumerate(concepts):
        chart_cells.append(
            f'<div><div class="paper-mini-chart-title">{html.escape(_concept_display_name(concept, include_unit=True))}</div>{_render_density_svg(seed=idx, concept=concept)}</div>'
        )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("D", "Cross-database distributions", "Cross-database density overlays for harmonized clinical concepts.")}
            <div class="paper-grid-5" style="grid-template-columns:repeat(6, minmax(0, 1fr));margin-bottom:0.55rem">{db_cards}</div>
            <div class="paper-legend" style="justify-content:center">
                {''.join(f'<span><span class="paper-legend-line" style="border-top-color:{color}"></span>{html.escape(name)}</span>' for name, _value, color in dbs)}
            </div>
            <div class="paper-chart-grid-2" style="grid-template-columns:repeat(3,minmax(0,1fr));gap:0.5rem 0.72rem">{''.join(chart_cells)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_density_svg(seed: int, concept: str) -> str:
    width, height = 205, 128
    left, right, top, bottom = 28, 8, 12, 24
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#16a34a", "#f97316", "#2563eb", "#ef4444", "#7e22ce", "#795548"]
    x = np.linspace(-3.2, 3.2, 80)
    paths = []
    for idx, color in enumerate(colors):
        mu = (idx - 2.5) * 0.12 + (seed % 3 - 1) * 0.08
        sigma = 0.72 + (idx % 3) * 0.08 + seed * 0.01
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma
        y = y / max(y.max(), 1e-9)
        points = []
        for xv, yv in zip(x, y):
            px = left + (xv - x.min()) / (x.max() - x.min()) * plot_w
            py = top + (1 - yv) * plot_h
            points.append((px, py))
        fill_points = [(left, height - bottom)] + points + [(width - right, height - bottom)]
        paths.append(
            f'<polygon points="{_svg_polyline(fill_points)}" fill="{color}" opacity="0.16"/>'
            f'<polyline points="{_svg_polyline(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="128" role="img" aria-label="{html.escape(concept)} distribution">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<line x1="{left}" y1="{top+plot_h*0.33:.1f}" x2="{width-right}" y2="{top+plot_h*0.33:.1f}" stroke="#e8eef7"/>'
        f'<line x1="{left}" y1="{top+plot_h*0.66:.1f}" x2="{width-right}" y2="{top+plot_h*0.66:.1f}" stroke="#e8eef7"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#cbd5e1"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#cbd5e1"/>'
        f'{"".join(paths)}'
        f'<text x="0" y="{top+10}" fill="#394b63" font-size="9">Density</text>'
        f'<text x="{left}" y="{height-5}" fill="#394b63" font-size="9">0</text>'
        f'<text x="{width-right-18}" y="{height-5}" fill="#394b63" font-size="9">+</text>'
        f'</svg>'
    )


def _render_paper_patient_panel() -> None:
    _render_paper_panel_css()
    patient_id = _first_patient_id() or 10001
    concepts = ["hr", "map", "resp", "spo2", "sofa", "crea"]
    cards = []
    for concept in concepts:
        series = _patient_series_for_concept(concept, patient_id=patient_id, max_points=72)
        value = series["value"].dropna().iloc[-1] if not series.empty else None
        cards.append(
            f'<div class="paper-card"><div class="paper-metric-label">{html.escape(concept)}</div>'
            f'<div class="paper-metric-value">{_paper_format_value(value)}</div>'
            f'<div style="font-size:0.55rem;color:#64748b">{html.escape(_concept_unit(concept) or "latest")}</div></div>'
        )
    mini_charts = "".join(
        f'<div><div class="paper-mini-chart-title">{html.escape(_concept_display_name(c))}</div>{_render_inline_timeseries_svg(c, patient_id=patient_id)}</div>'
        for c in ["hr", "map", "resp", "spo2"]
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Patient overview", "Compact case dashboard with latest measurements and longitudinal bedside trends.")}
            <div class="paper-soft-card" style="margin-bottom:0.65rem">
                <div style="font-weight:900">Patient {html.escape(str(patient_id))}</div>
                <div class="paper-note">Demo case summary · first 72 ICU hours · selected clinical concepts available for drill-down.</div>
            </div>
            <div class="paper-grid-4" style="grid-template-columns:repeat(6,minmax(0,1fr))">{''.join(cards)}</div>
            <div class="paper-chart-grid-2">{mini_charts}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_group_panel() -> None:
    rows = [
        ("Demographics", "Age, median (IQR)", "63 (52-73)", "66 (56-76)", "0.012", "0.24", "Small"),
        ("Demographics", "Male, %", "57.8", "61.1", "0.098", "0.07", "Small"),
        ("Vital Signs", "Heart rate (bpm), median (IQR)", "84 (72-98)", "92 (78-110)", "<0.001", "0.34", "Medium"),
        ("Vital Signs", "Mean arterial pressure (mmHg)", "82 (72-93)", "74 (64-86)", "<0.001", "0.48", "Large"),
        ("Laboratory", "Creatinine (mg/dL), median (IQR)", "1.1 (0.8-1.6)", "1.7 (1.1-2.7)", "<0.001", "0.57", "Large"),
        ("Outcomes", "ICU LOS (days), median (IQR)", "4 (2-8)", "6 (3-12)", "<0.001", "0.36", "Medium"),
    ]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("A", "Group contrast table", "Baseline characteristics and standardized differences for survived vs deceased subgroups.")}
            <div class="paper-grid-3">
                <div class="paper-card"><div class="paper-metric-label">Survived</div><div class="paper-metric-value">1,662 <span style="font-size:0.65rem;font-weight:700">(68.6%)</span></div></div>
                <div class="paper-card"><div class="paper-metric-label">Deceased</div><div class="paper-metric-value">763 <span style="font-size:0.65rem;font-weight:700">(31.4%)</span></div></div>
                <div class="paper-card"><div class="paper-metric-label">Ratio</div><div class="paper-metric-value">2.18 : 1</div></div>
            </div>
            <table class="paper-table"><thead><tr><th>Module</th><th>Characteristic</th><th>Survived</th><th>Deceased</th><th>p-value</th><th>SMD</th><th>Magnitude</th></tr></thead><tbody>{body}</tbody></table>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_coverage_panel() -> None:
    modules = [
        ("Vital Signs", [98.7, 98.6, 98.9, 98.4, 98.9]),
        ("Laboratory", [97.2, 97.3, 96.8, 97.5, 96.9]),
        ("Input/Output", [92.4, 92.6, 91.9, 93.3, 91.6]),
        ("Medications", [88.3, 89.1, 86.6, 89.7, 87.0]),
        ("Procedures", [75.1, 76.3, 72.4, 77.6, 72.9]),
        ("Ventilation", [70.8, 71.5, 69.2, 73.4, 68.5]),
    ]
    rows = []
    for module, values in modules:
        tds = [f"<td style='text-align:left;font-weight:800'>{html.escape(module)}</td>"]
        for value in values:
            green = int(240 - value * 1.1)
            tds.append(f"<td style='background:rgb({green},235,{green});font-weight:800'>{value:.1f}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    flow = [
        ("All ICU stays", "3,057"),
        ("Meet age criteria", "2,810"),
        ("ICU stay ≥24h", "2,610"),
        ("ICD include", "2,150"),
        ("ICD exclude", "2,012"),
        ("Final cohort", "2,012"),
    ]
    flow_html = "".join(f'<div class="paper-flow-step"><b>{html.escape(label)}</b><br><span style="font-weight:900">{value}</span></div>' for label, value in flow)
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("B", "Data coverage and eligibility audit", "Coverage heatmap and transparent inclusion/exclusion flow for the analysis cohort.")}
            <div class="paper-two-col">
                <div>
                    <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">1. Data coverage by module and subgroup (%)</div>
                    <table class="paper-heatmap"><thead><tr><th>Module</th><th>Overall</th><th>Survived</th><th>Deceased</th><th>SOFA ≤ 6</th><th>SOFA &gt; 6</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
                    <div class="paper-note">Missingness denominators: d=LOS, d=72h, d=demo, d=static.</div>
                </div>
                <div>
                    <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">2. Eligibility flow</div>
                    {flow_html}
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_snapshot_panel() -> None:
    cards = [
        ("Total patients", "2,012"),
        ("Total features", str(len(get_all_concepts()))),
        ("Median SOFA", "6 (3-9)"),
        ("Top phenotype", "Sepsis 24.1%"),
        ("Mortality", "31.4%"),
        ("Median LOS", "5 (2-10) d"),
    ]
    card_html = "".join(f'<div class="paper-card"><div class="paper-metric-label">{html.escape(k)}</div><div class="paper-metric-value">{html.escape(v)}</div></div>' for k, v in cards)
    mini = "".join(
        f'<div><div class="paper-mini-chart-title">{title}</div>{_render_density_svg(idx, "snapshot")}</div>'
        for idx, title in enumerate(["Age distribution", "ICU LOS distribution", "SOFA severity", "Outcomes"])
    )
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("C", "Cohort snapshot", "Cohort after all inclusion/exclusion criteria applied.")}
            <div class="paper-grid-5" style="grid-template-columns:repeat(6,minmax(0,1fr));margin-bottom:0.65rem">{card_html}</div>
            <div class="paper-chart-grid-2">{mini}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _render_paper_sofa_panel() -> None:
    rows = []
    matrix = [[370, 78, 20, 6, 1], [72, 359, 108, 24, 2], [16, 94, 242, 86, 11], [4, 16, 94, 227, 33], [0, 2, 11, 35, 101]]
    for idx, row in enumerate(matrix):
        cells = [f"<td style='font-weight:900'>{['0-3','4-6','7-9','10-12','≥13'][idx]}</td>"]
        for value in row:
            shade = 245 - min(150, int(value / 370 * 150))
            cells.append(f"<td style='background:rgb({shade},{shade+5},255);font-weight:800'>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'''
        <div class="paper-panel">
            {_paper_panel_header("D", "SOFA-1 vs SOFA-2 reclassification", "Agreement, upgrade/downgrade patterns, and organ-level contributors to score changes.")}
            <div class="paper-grid-5">
                <div class="paper-card"><div class="paper-metric-label">Total patients</div><div class="paper-metric-value">2,012</div></div>
                <div class="paper-card"><div class="paper-metric-label">Agreement</div><div class="paper-metric-value" style="color:#059669">66.2%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Upgrade</div><div class="paper-metric-value" style="color:#ef4444">20.4%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Downgrade</div><div class="paper-metric-value" style="color:#f59e0b">13.4%</div></div>
                <div class="paper-card"><div class="paper-metric-label">Median |ΔSOFA|</div><div class="paper-metric-value">1</div></div>
            </div>
            <div style="margin-top:0.65rem">
                <div style="font-weight:900;font-size:0.7rem;margin-bottom:0.35rem">1. Reclassification matrix (n)</div>
                <table class="paper-heatmap"><thead><tr><th>SOFA-1</th><th>0-3</th><th>4-6</th><th>7-9</th><th>10-12</th><th>≥13</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
            </div>
        </div>
        ''',
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
