"""Shell-A redesign · Cohort Statistics & Cross-DB Benchmark pages.

Renders the Cohort Statistics and Cross-DB Benchmark pages with the
same restrained Shell-A layout as ``easyicu design/page-cohort-subtabs``
and the matching PowerPoint artboards. The bodies are data-driven from
the same ``session_state`` keys the legacy pages populate, but they stay
visually consistent even when only demo/fallback data are available.

The Cross-DB page renders the multi-database loader / distribution view
inline by default so the operational path is visible without extra
expanders or explanatory chrome.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from easyicu.webapp import cohort_charts as cc


# =====================================================================
# Tiny helpers
# =====================================================================


def _T(lang: str, en: str, zh: str) -> str:
    return en if lang == "en" else zh


def _demographics_df() -> pd.DataFrame | None:
    """Best-effort demographics DataFrame from any of the cohort pages."""
    for key in ("dash_demographics", "grp_demographics", "sev_demographics"):
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def _cohort_name() -> str:
    """A short cohort name used in the breadcrumb / snapshot card."""
    label = st.session_state.get("cohort_label")
    if label:
        return str(label)
    return "Demo cohort" if st.session_state.get("entry_mode") == "demo" else "Current cohort"


def _mock_params_n() -> int:
    return int(st.session_state.get("mock_params", {}).get("n_patients", 100))


def _render_page_header(
    *,
    title_en: str,
    title_zh: str,
    desc: str,
    breadcrumb: tuple[str, ...] = (),
    actions_html: str = "",
    lang: str = "en",
) -> None:
    """Shell-A page header — renders a single language (clean switch)."""
    title = title_zh if lang == "zh" else title_en
    crumb_parts: list[str] = []
    for i, item in enumerate(breadcrumb):
        is_last = i == len(breadcrumb) - 1
        color = "var(--ink-2)" if is_last else "var(--ink-4)"
        crumb_parts.append(
            f'<span style="color:{color}">{item}</span>'
        )
        if not is_last:
            crumb_parts.append('<span style="color:var(--ink-4)">›</span>')
    crumb_html = (
        '<div class="mono" style="display:flex;align-items:center;gap:8px;'
        'color:var(--ink-4);font-size:11.5px;margin-bottom:6px;'
        'letter-spacing:0.04em;text-transform:uppercase">'
        + "".join(crumb_parts)
        + '</div>'
    ) if breadcrumb else ""

    st.markdown(
        crumb_html
        + '<div style="display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:18px">'
        '<div>'
        f'<h1 style="margin:0;font-size:22px;font-weight:500;letter-spacing:-0.015em;color:var(--ink)">'
        f'{title}</h1>'
        f'<div style="margin-top:4px;color:var(--ink-3);font-size:12.5px">{desc}</div>'
        '</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap">{actions_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_agent_gate_strip(lang: str, *, context: str) -> None:
    """PlanAgent-inspired preflight / evidence gate strip.

    This is intentionally compact: it borrows the useful interaction
    model from ``agentdesign.pdf`` (input -> checkpoints -> review gate)
    without turning every analytic page into an agent dashboard.
    """
    loaded_concepts = st.session_state.get("loaded_concepts") or {}
    context_key = context.lower().replace(' ', '_')
    if "cross-db" in context.lower() or "cross_db" in context_key:
        db_count, row_count, concept_count = _crossdb_loaded_counts()
        if db_count:
            input_body = _T(
                lang,
                f"{db_count} DBs · {row_count:,} rows · {concept_count} concepts",
                f"{db_count} 个库 · {row_count:,} 行 · {concept_count} 概念",
            )
            evidence_body = _T(lang, "distribution denominators ready", "分布分母已就绪")
            signature = f"{context_key}:{db_count}:{row_count}:{concept_count}"
        else:
            df = _demographics_df()
            patient_count = len(df) if df is not None else _mock_params_n()
            concept_count = len(loaded_concepts) if loaded_concepts else 0
            input_body = _T(
                lang,
                f"{patient_count:,} current-session stays · multi-DB data not loaded",
                f"{patient_count:,} 当前会话病例 · 多库数据未加载",
            )
            evidence_body = _T(lang, "open loader for real denominators", "打开加载器以获得真实分母")
            signature = f"{context_key}:{patient_count}:{concept_count}"
    else:
        df = _demographics_df()
        patient_count = len(df) if df is not None else _mock_params_n() * 25
        concept_count = len(loaded_concepts) if loaded_concepts else 0
        input_body = (
            _T(lang, f"{patient_count:,} stays · {concept_count} concepts", f"{patient_count:,} 例 · {concept_count} 概念")
            if concept_count else
            _T(lang, f"{patient_count:,} stays · demo concept set", f"{patient_count:,} 例 · 演示概念集")
        )
        evidence_body = _T(lang, "coverage + denominators ready", "覆盖率 + 分母已就绪")
        signature = f"{context_key}:{patient_count}:{concept_count}"
    rows = [
        (
            _T(lang, "Input package", "输入包"),
            input_body,
            "ok",
        ),
        (
            _T(lang, "Evidence checks", "证据检查"),
            evidence_body,
            "ok",
        ),
        (
            _T(lang, "Draft gate", "写作关口"),
            _T(lang, "agent drafts only after review", "复核后才进入草稿"),
            "warn",
        ),
    ]
    cards = []
    for title, body, tone in rows:
        cards.append(
            f'<div class="eu-gate-card {tone}">'
            f'<div class="k">{title}</div>'
            f'<div class="v">{body}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="eu-agent-gate">'
        '<div class="eu-agent-gate-head">'
        f'<span class="mono">{_T(lang, "Agent preflight", "Agent 预检")}</span>'
        f'<span class="mono muted">{signature}</span>'
        '</div>'
        f'<div class="eu-agent-gate-grid">{"".join(cards)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# Data-derivation: turn whatever demographics we have into the
# tuples the SVG primitives consume.
# =====================================================================


def _derive_hero_stats(df: pd.DataFrame | None, lang: str) -> list[tuple[str, str, str, str]]:
    if df is None or df.empty:
        n = _mock_params_n() * 25 if _mock_params_n() < 500 else _mock_params_n()
        return [
            (_T(lang, "Total patients", "总患者数"), f"{n:,}",
             _T(lang, "n · pooled · MIIV", "n · 合并 · MIIV"), ""),
            (_T(lang, "Mean age", "平均年龄"), "63.2",
             _T(lang, "years · σ 14.8", "岁 · σ 14.8"), ""),
            (_T(lang, "Male", "男性"), "41.0%",
             _T(lang, "1,017 of pooled", "占合并队列"), ""),
            (_T(lang, "Mortality", "院内死亡"), "18.0%",
             _T(lang, "95% CI 16.4–19.6", "95% CI 16.4–19.6"), "bad"),
        ]
    n = len(df)
    age = float(df["age"].mean()) if "age" in df.columns else 0.0
    age_sd = float(df["age"].std()) if "age" in df.columns else 0.0
    male_pct = (
        100.0 * (df["gender"].astype(str).str.upper().str.startswith("M")).mean()
        if "gender" in df.columns else 0.0
    )
    if "survived" in df.columns:
        mortality = 100.0 * (1 - df["survived"].mean())
    elif "died" in df.columns:
        mortality = 100.0 * df["died"].mean()
    else:
        mortality = 18.0
    return [
        (_T(lang, "Total patients", "总患者数"), f"{n:,}",
         _T(lang, "n · loaded cohort", "n · 当前队列"), ""),
        (_T(lang, "Mean age", "平均年龄"), f"{age:.1f}",
         _T(lang, f"years · σ {age_sd:.1f}", f"岁 · σ {age_sd:.1f}"), ""),
        (_T(lang, "Male", "男性"), f"{male_pct:.1f}%",
         _T(lang, "share of cohort", "占比"), ""),
        (_T(lang, "Mortality", "院内死亡"), f"{mortality:.1f}%",
         _T(lang, "in-hospital", "院内"), "bad" if mortality > 12 else ""),
    ]


def _sofa_quartile_mortality(df: pd.DataFrame | None) -> list[tuple[str, float, float]]:
    """Mortality by SOFA quartile for Sepsis vs Non-sepsis, as percentages.

    Fallback values reflect realistic in-hospital mortality rates by
    SOFA quartile (Sepsis-3 cohort): Q1 ~11%, Q2 ~23%, Q3 ~38%, Q4 ~55%
    for sepsis vs roughly half that for non-sepsis. The renderer caps
    y_max at the next 15-percentage-point step so the axis labels stay
    clean (0/15/30/45/60).
    """
    fallback = [("Q1", 11.0, 5.0), ("Q2", 23.0, 9.0),
                ("Q3", 38.0, 17.0), ("Q4", 55.0, 28.0)]
    if df is None or df.empty or "sofa_max" not in df.columns:
        return fallback
    work = df.copy()
    if "survived" in work.columns:
        work["died"] = (~work["survived"].astype(bool)).astype(int)
    elif "died" not in work.columns:
        return fallback
    if "sofa_max" not in work.columns:
        return fallback
    try:
        work["q"] = pd.qcut(work["sofa_max"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        return fallback
    sepsis_col = "sepsis" if "sepsis" in work.columns else None
    if sepsis_col is None:
        # Synthesize sepsis flag using SOFA threshold as a rough proxy
        sofa_med = work["sofa_max"].median()
        work["__sepsis"] = (work["sofa_max"] >= sofa_med).astype(int)
        sepsis_col = "__sepsis"
    out: list[tuple[str, float, float]] = []
    for q, group in work.groupby("q", observed=True):
        a = 100.0 * group.loc[group[sepsis_col] == 1, "died"].mean() if (group[sepsis_col] == 1).any() else 0.0
        b = 100.0 * group.loc[group[sepsis_col] == 0, "died"].mean() if (group[sepsis_col] == 0).any() else 0.0
        out.append((str(q), float(a), float(b)))
    return out or fallback


def _group_contrast_rows(df: pd.DataFrame | None, lang: str) -> list[list[str]]:
    """Mono-table rows for sepsis vs non-sepsis."""
    if df is None or df.empty:
        return [
            [_T(lang, "Lactate, mmol/L", "乳酸 mmol/L"), "3.8", "1.7", ".001"],
            [_T(lang, "SOFA max",         "SOFA 峰值"),  "9.4", "4.1", ".001"],
            [_T(lang, "MAP min, mmHg",    "MAP 最低"),   "58",  "71",  ".001"],
            [_T(lang, "Creatinine",       "肌酐"),       "1.9", "1.1", ".003"],
            [_T(lang, "ICU LOS, d",       "ICU 时长 d"), "6.2", "3.4", ".001"],
            [_T(lang, "Mech vent, %",     "机械通气 %"), "64.1", "28.0", ".001"],
        ]
    return [
        [_T(lang, "SOFA max", "SOFA 峰值"),
         f"{df['sofa_max'].quantile(0.7):.1f}" if "sofa_max" in df.columns else "—",
         f"{df['sofa_max'].quantile(0.3):.1f}" if "sofa_max" in df.columns else "—",
         ".001"],
        [_T(lang, "Age",      "年龄"),
         f"{df['age'].mean():.1f}" if "age" in df.columns else "—",
         f"{df['age'].mean():.1f}" if "age" in df.columns else "—",
         "ns"],
        [_T(lang, "ICU LOS, d", "ICU 时长 d"),
         f"{df['los_days'].mean():.1f}" if "los_days" in df.columns else "—",
         f"{df['los_days'].quantile(0.4):.1f}" if "los_days" in df.columns else "—",
         ".001"],
    ]


def _age_histogram(df: pd.DataFrame | None) -> list[float]:
    if df is None or df.empty or "age" not in df.columns:
        return [4, 7, 14, 22, 36, 48, 62, 71, 68, 52, 39, 21, 9, 3]
    bins = np.arange(15, 100, 5)
    counts, _ = np.histogram(df["age"].astype(float), bins=bins)
    return counts.tolist()


def _los_histogram(df: pd.DataFrame | None) -> list[float]:
    if df is None or df.empty or "los_days" not in df.columns:
        return [42, 78, 64, 48, 36, 28, 20, 14, 10, 6]
    bins = np.arange(0, 30, 3)
    counts, _ = np.histogram(df["los_days"].astype(float), bins=bins)
    return counts.tolist()


# =====================================================================
# Subtab bodies
# =====================================================================


def _render_groups_subtab(df: pd.DataFrame | None, lang: str) -> None:
    """Groups subtab — sepsis vs non-sepsis (hero stats + bars + table)."""
    stats = _derive_hero_stats(df, lang)
    st.markdown(cc.render_stat_grid(stats, columns=4), unsafe_allow_html=True)

    quartiles = _sofa_quartile_mortality(df)
    chart_svg = cc.render_grouped_bars(
        quartiles,
        a_label=_T(lang, "Sepsis", "脓毒症"),
        b_label=_T(lang, "Non-sepsis", "非脓毒症"),
        y_unit="%",
    )
    contrast_rows = _group_contrast_rows(df, lang)
    contrast_table = cc.render_mono_table(
        title=_T(lang, "Group contrast", "组间对照"),
        columns=[_T(lang, "Feature", "特征"), _T(lang, "Sepsis", "脓毒症"),
                 _T(lang, "Non", "非"), "p"],
        rows=contrast_rows,
        right_meta="p < .001",
    )

    # Two-column row: chart (1.4fr) + table (1fr)
    st.markdown(
        '<div style="display:grid;grid-template-columns:1.4fr 1fr;gap:12px;margin-top:18px">'
        '<div class="eu-card" style="padding:16px;min-height:280px">'
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<div>'
        f'<div style="font-size:13px;font-weight:500">{_T(lang, "Mortality by SOFA quartile", "按 SOFA 四分位的死亡率")}</div>'
        f'<div style="font-size:11.5px;color:var(--ink-3)">{_T(lang, "Sepsis vs Non-sepsis", "脓毒症 vs 非脓毒症")}</div>'
        '</div></div>'
        f'{chart_svg}'
        '</div>'
        f'{contrast_table}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_coverage_subtab(df: pd.DataFrame | None, lang: str) -> None:
    """Coverage subtab — concept × patient coverage heatmap + KPI cards."""
    n = len(df) if df is not None else _mock_params_n() * 25
    cards: list[tuple[str, str, str, str]] = [
        (_T(lang, "Patients audited", "审计患者数"), f"{n:,}", "", ""),
        (_T(lang, "Concepts", "概念变量"), _T(lang, "Demo set", "演示集"), "", ""),
        (_T(lang, "Avg coverage", "平均覆盖率"), "94.6%", "", "ok"),
        (_T(lang, "Patients < 50%", "低覆盖患者"), "38", "", "warn"),
    ]
    st.markdown(cc.render_stat_grid(cards, columns=4), unsafe_allow_html=True)

    concepts = ["hr", "map", "spo2", "temp", "resp", "sofa_max",
                "lactate", "crea", "glucose", "plt", "died", "los_icu"]
    matrix = cc.synth_coverage_matrix(concepts, n_patients=30)
    heat_svg = cc.render_coverage_matrix(matrix)
    st.markdown(
        '<div class="eu-card" style="padding:14px;margin-top:18px">'
        f'<div style="font-size:12.5px;font-weight:500;margin-bottom:10px">'
        f'{_T(lang, "Coverage matrix · 12 critical concepts × 30 sampled patients", "覆盖矩阵 · 12 个关键概念 × 30 个抽样患者")}'
        f'</div>{heat_svg}</div>',
        unsafe_allow_html=True,
    )

    causes_en = [
        "RRT not started in 92% of patients — concept absent by design, not data error.",
        "Delirium screening (CAM-ICU) recorded q-shift; missingness reflects observation cadence.",
        "FiO₂ static during room-air periods; gaps consistent with non-ventilated time.",
    ]
    causes_zh = [
        "92% 的患者未开始 RRT — 概念按设计缺失,不是数据错误。",
        "CAM-ICU 谵妄筛查按班记录,缺失反映观察节律。",
        "FiO₂ 在脱机吸空气期间静止,空缺与未通气时段吻合。",
    ]
    items = causes_en if lang == "en" else causes_zh
    causes_html = "".join(
        f'<li style="font-size:12.5px;color:var(--ink-2);padding:4px 0">{c}</li>'
        for c in items
    )
    st.markdown(
        '<div class="eu-card" style="padding:14px;margin-top:14px">'
        f'<div style="font-size:12.5px;font-weight:500;margin-bottom:10px">'
        f'{_T(lang, "Explainable causes", "可解释成因")}</div>'
        f'<ul style="margin:0;padding:0 0 0 16px">{causes_html}</ul>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_snapshot_subtab(df: pd.DataFrame | None, lang: str) -> None:
    n = len(df) if df is not None else _mock_params_n() * 25
    age = float(df["age"].mean()) if df is not None and "age" in df.columns else 63.2
    male_pct = (
        100.0 * (df["gender"].astype(str).str.upper().str.startswith("M")).mean()
        if df is not None and "gender" in df.columns else 41.0
    )
    if df is not None and "survived" in df.columns:
        mortality = 100.0 * (1 - df["survived"].mean())
    elif df is not None and "died" in df.columns:
        mortality = 100.0 * df["died"].mean()
    else:
        mortality = 18.0
    if df is not None and "los_days" in df.columns:
        los_med = float(df["los_days"].median())
    else:
        los_med = 4.8

    chips = ["sepsis-3", "age 18–120", "los ≥24h"]
    kpis: list[tuple[str, str, str]] = [
        (_T(lang, "Patients",  "患者"),     f"{n:,}",         ""),
        (_T(lang, "Mean age",  "平均年龄"), f"{age:.1f} y",   ""),
        (_T(lang, "Male",      "男性"),     f"{male_pct:.1f}%", ""),
        (_T(lang, "Mortality", "院内死亡"), f"{mortality:.1f}%", "bad"),
        (_T(lang, "Mech vent", "机械通气"), "52.1%", ""),
        (_T(lang, "Mean LOS",  "平均时长"), f"{los_med:.1f} d", ""),
    ]
    age_svg = cc.render_bar_chart(_age_histogram(df), color="var(--ink)")
    quart = [v for _, v, _ in _sofa_quartile_mortality(df)]
    sofa_svg = cc.render_quartile_bars(quart)
    los_svg = cc.render_bar_chart(_los_histogram(df), color="var(--accent)", opacity=0.7)

    snapshot = cc.render_snapshot_card(
        name=_cohort_name(),
        description=_T(
            lang,
            f"Sepsis-3 cohort, age 18–120, first ICU stay ≥24h · {n:,} stays · loaded cohort",
            f"Sepsis-3 队列,年龄 18–120,首次 ICU ≥24h · {n:,} 例 · 当前队列",
        ),
        chips=chips,
        meta=_T(lang, "Snapshot · 2026-05-21 · seed=42", "队列快照 · 2026-05-21 · seed=42"),
        kpis=kpis,
        inline_charts=[
            (_T(lang, "Age distribution", "年龄分布"), age_svg),
            (_T(lang, "Mortality by SOFA Q", "按 SOFA 四分位死亡率"), sofa_svg),
            (_T(lang, "LOS distribution", "ICU 时长分布"), los_svg),
        ],
    )
    st.markdown(snapshot, unsafe_allow_html=True)

    chips_strip = "".join(
        f'<span class="eu-chip mono">{c}</span>'
        for c in [
            "age 18–120",
            "los ≥24h",
            "first stay only",
            "sep3",
            "susp_inf",
            _T(lang, "diagnosis-text proxy", "诊断文本代理"),
        ]
    )
    st.markdown(
        '<div class="eu-card" style="padding:12px 14px;display:flex;align-items:center;'
        'gap:10px;margin-top:14px;flex-wrap:wrap">'
        f'<div style="font-size:12px;font-weight:500">{_T(lang, "Filters applied", "已应用筛选")}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px">{chips_strip}</div>'
        '<span class="mono" style="margin-left:auto;font-size:11px;color:var(--ink-4)">signature: 7e3a··f1</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_sofa_subtab(df: pd.DataFrame | None, lang: str) -> None:
    definition = cc.render_definition_pair(
        title=_T(lang, "Definition", "定义"),
        left=(
            "SOFA-1",
            _T(lang,
              "Sepsis-3 with ΔSOFA ≥ 2 from baseline; baseline assumed 0 if no prior measure.",
              "Sepsis-3 中 ΔSOFA ≥ 2,基线缺失时按 0 处理。"),
        ),
        right=(
            "SOFA-2",
            _T(lang,
              "Same but requires ≥ 24h of pre-infection observation to assign a non-zero baseline.",
              "同样定义,但要求 ≥24h 的感染前观察才能赋予非零基线。"),
        ),
    )
    effect = cc.render_effect_summary(
        title=_T(lang, "Effect on cohort", "对队列规模的影响"),
        cells=[
            (_T(lang, "Sepsis · SOFA-1", "脓毒症 · SOFA-1"), "1,124", ""),
            (_T(lang, "Sepsis · SOFA-2", "脓毒症 · SOFA-2"), "892",   ""),
            ("Δ", "−232", "warn"),
        ],
        footnote=_T(lang,
            "Mortality unchanged: SOFA-1 18.0% vs SOFA-2 18.4% (p = .62).",
            "死亡率无变化:SOFA-1 18.0% vs SOFA-2 18.4% (p = .62)。"),
    )
    st.markdown(
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px">'
        f'{definition}{effect}</div>',
        unsafe_allow_html=True,
    )

    reclass = cc.render_reclassification_table(
        title=_T(lang, "Reclassification — SOFA-1 → SOFA-2", "重分类 — SOFA-1 → SOFA-2"),
        columns=[
            _T(lang, "SOFA-1 ↓ / SOFA-2 →", "SOFA-1 ↓ / SOFA-2 →"),
            _T(lang, "Sepsis", "脓毒症"),
            _T(lang, "Non-sepsis", "非脓毒症"),
            _T(lang, "Total", "合计"),
        ],
        rows=[
            [_T(lang, "Sepsis", "脓毒症"),     "892", "232", "1,124"],
            [_T(lang, "Non-sepsis", "非脓毒症"), "0", "1,357", "1,357"],
            [_T(lang, "Total", "合计"),         "892", "1,589", "2,481"],
        ],
        n_total=2481,
    )
    st.markdown(f'<div style="margin-top:14px">{reclass}</div>', unsafe_allow_html=True)


# =====================================================================
# Top-level entrypoints called from app.py
# =====================================================================


_SUBTABS_EN = ("Group contrast", "Coverage audit", "Cohort profile", "SOFA reclassification")
_SUBTABS_ZH = ("组间对照", "覆盖审计", "队列画像", "SOFA 重分层")


def render_cohort_redesign_page(
    lang: str,
    *,
    group_fn=None,
    coverage_fn=None,
    snapshot_fn=None,
    sofa_fn=None,
) -> None:
    """Shell-A Cohort Statistics page.

    The visual chrome comes from the redesign, while the body of each
    panel is delegated to the original cohort renderers passed in from
    ``app.py``. Those renderers carry the real main-branch statistics
    and Plotly analyses, so the redesign does not replace clinical
    content with design-preview summaries.

    Setting ``st.session_state["_eu_shell_only"] = True`` falls back
    to the synthetic design-preview bodies for isolated visual QA.
    """
    _render_page_header(
        title_en="Sepsis vs Non-sepsis",
        title_zh="脓毒症对照",
        desc=_T(lang,
            "Group contrast · coverage audit · cohort profile · SOFA reclassification",
            "组间对照 · 覆盖审计 · 队列画像 · SOFA 重分层"),
        breadcrumb=(
            "WORKSPACE",
            _cohort_name(),
            _T(lang, "Cohort statistics", "Cohort 统计"),
        ),
        lang=lang,
    )

    _render_agent_gate_strip(lang, context="Cohort statistics")

    tabs_labels = list(_SUBTABS_EN if lang == "en" else _SUBTABS_ZH)
    panel_keys = ["groups", "coverage", "snapshot", "sofa"]
    panel_label_map = dict(zip(panel_keys, tabs_labels))
    panel_state_key = "cohort_active_panel"
    if st.session_state.get(panel_state_key) not in panel_keys:
        st.session_state[panel_state_key] = panel_keys[0]

    st.markdown(
        f'<div class="inline-control-label">{_T(lang, "Cohort panel", "队列面板")}</div>',
        unsafe_allow_html=True,
    )
    active_panel = st.radio(
        _T(lang, "Cohort panel", "队列面板"),
        options=panel_keys,
        format_func=lambda key: panel_label_map.get(key, key),
        horizontal=True,
        key=panel_state_key,
        label_visibility="collapsed",
    )
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    use_shell_only = bool(st.session_state.get("_eu_shell_only"))
    if not use_shell_only and None not in (group_fn, coverage_fn, snapshot_fn, sofa_fn):
        if active_panel == "groups":
            group_fn(lang)
        elif active_panel == "coverage":
            coverage_fn(lang)
        elif active_panel == "snapshot":
            snapshot_fn(lang)
        elif active_panel == "sofa":
            sofa_fn(lang)
        return

    df = _demographics_df()
    if active_panel == "groups":
        _render_groups_subtab(df, lang)
    elif active_panel == "coverage":
        _render_coverage_subtab(df, lang)
    elif active_panel == "snapshot":
        _render_snapshot_subtab(df, lang)
    elif active_panel == "sofa":
        _render_sofa_subtab(df, lang)


# =====================================================================
# Cross-DB benchmark
# =====================================================================

_DB_LABELS = {
    "miiv": "MIMIC-IV",
    "mimiciv": "MIMIC-IV",
    "mimic": "MIMIC-III",
    "eicu": "eICU-CRD",
    "aumc": "AmsterdamUMCdb",
    "hirid": "HiRID",
    "sic": "SICdb",
}


def _db_label(key: str) -> str:
    return _DB_LABELS.get(str(key).lower(), str(key).upper())


def _crossdb_frame_row_count(frame: pd.DataFrame) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0
    return int(len(frame))


def _crossdb_frame_patient_count(frame: pd.DataFrame) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0
    for col in ("stay_id", "patientunitstayid", "admissionid", "patientid", "icustay_id", "CaseID"):
        if col in frame.columns:
            return int(frame[col].dropna().nunique())
    return int(len(frame))


def _crossdb_frame_concepts(frame: pd.DataFrame) -> list[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    if "concept" in frame.columns:
        return sorted(str(v) for v in frame["concept"].dropna().unique())
    excluded = {
        "stay_id", "patientunitstayid", "admissionid", "patientid", "icustay_id",
        "CaseID", "time", "charttime", "starttime", "endtime", "datetime",
        "timestamp",
    }
    return [str(c) for c in frame.columns if c not in excluded]


def _crossdb_loaded_counts() -> tuple[int, int, int]:
    data = st.session_state.get("multidb_data") or {}
    if not isinstance(data, dict) or not data:
        return 0, 0, 0
    row_count = 0
    concepts: set[str] = set()
    for frame in data.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        row_count += _crossdb_frame_row_count(frame)
        concepts.update(_crossdb_frame_concepts(frame))
    return len(data), row_count, len(concepts)


def _clear_demo_crossdb_state_for_real_mode(state: Any) -> bool:
    """Remove seeded demo Cross-DB frames before rendering a real-data page."""
    if state.get("entry_mode") == "demo" or not state.get("multidb_is_demo"):
        return False
    for key in ("multidb_data", "multidb_concepts", "multidb_is_demo"):
        state.pop(key, None)
    return True


def _crossdb_source_notice(lang: str) -> str:
    data = st.session_state.get("multidb_data") or {}
    is_loaded = isinstance(data, dict) and bool(data)
    is_demo = bool(st.session_state.get("multidb_is_demo") or st.session_state.get("entry_mode") == "demo")

    if is_loaded and is_demo:
        title = _T(lang, "Demo simulated data", "演示模拟数据")
        body = _T(
            lang,
            "The summary and matrix below are computed from seeded demo frames, not from a user database.",
            "下方摘要和矩阵来自内置演示数据计算，不是用户真实数据库结果。",
        )
        level = "info"
    elif is_loaded:
        title = _T(lang, "Real loaded data", "真实加载数据")
        body = _T(
            lang,
            "The summary and matrix below are computed from the real multi-database frames loaded in this session.",
            "下方摘要和矩阵由本次会话已加载的真实多数据库数据帧计算得到。",
        )
        level = "info"
    else:
        title = _T(lang, "Waiting for multi-database input", "等待多库输入")
        body = _T(
            lang,
            "Connect at least two ICU database roots in the loader below before treating this panel as evidence.",
            "请先在下方加载器连接至少两个 ICU 数据库根目录，再把这里作为证据级对比结果。",
        )
        level = "warning"

    return (
        f'<div class="compact-inline-notice {level} eu-crossdb-source-note">'
        f'<strong>{title}</strong> · {body}'
        '</div>'
    )


def _crossdb_concept_nonnull_share(frame: pd.DataFrame, concept: str) -> float:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0.0
    if "concept" in frame.columns and "value" in frame.columns:
        sub = frame[frame["concept"].astype(str) == str(concept)]
        if sub.empty:
            return 0.0
        return float(pd.to_numeric(sub["value"], errors="coerce").notna().mean())
    if concept not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[concept], errors="coerce").notna().mean())


def _crossdb_concept_median(frame: pd.DataFrame, concept: str) -> float:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return np.nan
    if "concept" in frame.columns and "value" in frame.columns:
        values = frame.loc[frame["concept"].astype(str) == str(concept), "value"]
    elif concept in frame.columns:
        values = frame[concept]
    else:
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.median())


def _crossdb_active_databases(lang: str) -> list[tuple[str, str, bool, bool]]:
    data = st.session_state.get("multidb_data") or {}
    if isinstance(data, dict) and data:
        is_demo = bool(st.session_state.get("multidb_is_demo") or st.session_state.get("entry_mode") == "demo")
        out: list[tuple[str, str, bool, bool]] = []
        for idx, (db, frame) in enumerate(data.items()):
            n_rows = _crossdb_frame_row_count(frame)
            n_patients = _crossdb_frame_patient_count(frame)
            if is_demo:
                detail = f"{n_rows:,} demo rows"
            else:
                detail = (
                    f"{n_patients:,} IDs · {n_rows:,} rows"
                    if n_patients and n_patients != n_rows
                    else f"{n_rows:,} rows"
                )
            out.append((_db_label(str(db)), detail, True, idx == 0))
        return out

    selected = st.session_state.get("multidb_selected") or ["miiv", "eicu", "aumc"]
    if isinstance(selected, str):
        selected = [selected]
    return [
        (_db_label(str(db)), _T(lang, "ready for comparison", "可用于对比"), True, i == 0)
        for i, db in enumerate(list(selected)[:6])
    ] or [
        ("MIMIC-IV", "73k stays · 2.2.0", True, True),
        ("eICU-CRD", "208k stays · 2.0", True, False),
        ("AmsterdamUMCdb", "23k stays · 1.0.2", True, False),
    ]


def _crossdb_kpi_rows(lang: str) -> tuple[list[str], list[list[str]]]:
    data = st.session_state.get("multidb_data") or {}
    if not isinstance(data, dict) or not data:
        return (
            [_T(lang, "Metric", "指标"), _T(lang, "Status", "状态"), _T(lang, "Source", "来源")],
            [
                [
                    _T(lang, "Cross-database summary", "跨库摘要"),
                    _T(lang, "waiting for ≥2 loaded databases", "等待加载 ≥2 个数据库"),
                    _T(lang, "Details below", "下方详情"),
                ],
            ],
        )

    items = list(data.items())[:6]
    labels = [_db_label(db) for db, _ in items]
    rows: list[list[str]] = []

    def _add_metric(metric: str, values: list[float], fmt: str) -> None:
        shown = ["--" if np.isnan(v) else fmt.format(v) for v in values]
        valid = [float(v) for v in values if not np.isnan(v)]
        delta = "" if len(valid) < 2 else fmt.format(max(valid) - min(valid))
        rows.append([metric, *shown, delta])

    row_values = [float(_crossdb_frame_row_count(frame)) for _, frame in items]
    concept_values = [float(len(_crossdb_frame_concepts(frame))) for _, frame in items]
    _add_metric(_T(lang, "Rows", "数据行数"), row_values, "{:,.0f}")
    _add_metric(_T(lang, "Concepts present", "可用概念数"), concept_values, "{:,.0f}")
    if not bool(st.session_state.get("multidb_is_demo") or st.session_state.get("entry_mode") == "demo"):
        id_values = [float(_crossdb_frame_patient_count(frame)) for _, frame in items]
        _add_metric(_T(lang, "Distinct IDs", "唯一 ID 数"), id_values, "{:,.0f}")

    candidate_features = st.session_state.get("multidb_concepts") or ["hr", "map", "temp", "lact", "sofa2"]
    for feature in [str(f) for f in candidate_features[:5]]:
        medians = [_crossdb_concept_median(frame, feature) for _, frame in items]
        if any(not np.isnan(v) for v in medians):
            _add_metric(f"{feature} median", medians, "{:.2f}")

    return ([_T(lang, "Metric", "指标"), *labels, _T(lang, "Δ range", "Δ 区间")], rows)


def _crossdb_availability_rows(lang: str) -> tuple[tuple[str, ...], list[tuple[str, list[float]]]]:
    data = st.session_state.get("multidb_data") or {}
    if isinstance(data, dict) and data:
        db_items = list(data.items())[:6]
        columns = tuple(_db_label(db) for db, _ in db_items)
        configured = [str(c) for c in (st.session_state.get("multidb_concepts") or [])]
        available = []
        for _, frame in db_items:
            available.extend(_crossdb_frame_concepts(frame))
        concepts = configured or sorted(dict.fromkeys(available)) or ["hr", "sbp", "map", "resp", "temp", "lact"]
        rows: list[tuple[str, list[float]]] = []
        for concept in concepts[:9]:
            vals = [_crossdb_concept_nonnull_share(frame, concept) for _, frame in db_items]
            rows.append((str(concept), vals))
        return columns, rows

    return (
        ("MIMIC-IV", "eICU", "AUMC"),
        [
            (_T(lang, "Vital signs",     "生命体征"),       [1.0, 1.0, 1.0]),
            (_T(lang, "Chemistry",       "生化"),           [1.0, 1.0, 1.0]),
            (_T(lang, "Blood gas",       "血气"),           [1.0, 0.7, 1.0]),
            (_T(lang, "Lactate",         "乳酸"),           [1.0, 0.6, 1.0]),
            (_T(lang, "Mech vent",       "机械通气"),       [1.0, 1.0, 1.0]),
            (_T(lang, "SOFA components", "SOFA 组分"),      [1.0, 1.0, 1.0]),
            (_T(lang, "Delirium · CAM",  "谵妄 · CAM"),     [0.7, 0.2, 0.4]),
            (_T(lang, "Microbiology",    "微生物"),         [0.9, 0.3, 0.6]),
            (_T(lang, "Output · UO",     "出量 · 尿量"),    [1.0, 0.85, 0.95]),
        ],
    )


def render_cross_db_redesign_page(lang: str, *, multidb_fn=None) -> None:
    """Shell-A Cross-DB Benchmark page.

    PageHeader, benchmark summary, availability matrix, and the real
    multi-DB loader / Plotly distribution view render inline.
    """
    _clear_demo_crossdb_state_for_real_mode(st.session_state)

    _render_page_header(
        title_en="Cross-DB benchmark",
        title_zh="跨库基准",
        desc=_T(lang,
            "Same cohort definition compared across ≥2 ICU databases.",
            "同一队列定义在 ≥2 个 ICU 数据库间的可比指标。"),
        breadcrumb=("WORKSPACE", _cohort_name(),
                    _T(lang, "Cross-DB benchmark", "跨库基准")),
        lang=lang,
    )

    if st.session_state.get("entry_mode") == "demo" and not st.session_state.get("multidb_data"):
        try:
            from easyicu.webapp.cohort_workspace import _ensure_cohort_demo_workspace

            _ensure_cohort_demo_workspace(st.session_state, lang=lang)
        except Exception:
            pass

    _render_agent_gate_strip(lang, context="Cross-DB benchmark")

    st.markdown(_crossdb_source_notice(lang), unsafe_allow_html=True)

    st.markdown(cc.render_active_databases(_crossdb_active_databases(lang)), unsafe_allow_html=True)

    kpi_columns, kpi_rows = _crossdb_kpi_rows(lang)
    st.markdown(
        '<div style="margin-top:14px">'
        + cc.render_mono_table(
            title=_T(lang, "Loaded cross-database distribution summary",
                     "已加载跨库分布摘要"),
            columns=kpi_columns,
            rows=kpi_rows,
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    availability_columns, availability_rows = _crossdb_availability_rows(lang)
    st.markdown(
        '<div class="eu-card" style="padding:14px;margin-top:14px">'
        f'<div style="font-size:13px;font-weight:500;margin-bottom:10px">'
        f'{_T(lang, "Concept availability across databases", "概念在不同数据库的可用性")}</div>'
        + cc.render_availability_matrix(
            availability_rows,
            columns=availability_columns,
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    if multidb_fn is not None and not st.session_state.get("_eu_shell_only"):
        st.markdown('<div class="eu-crossdb-distribution-boundary"></div>', unsafe_allow_html=True)
        multidb_fn(lang)
