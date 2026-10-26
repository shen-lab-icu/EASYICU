"""Lay out the workflow demo's registered result as a journal-style report page.

Usage: python tools/build_demo_study_report.py

Reads only the demo's trace record (``assets/demo/sofa2-run-trace.json``) and
the figures redrawn from the run's registered tables, and writes one page per
language beside them. Every number on the page is a field of the trace record;
the page adds structure, not findings. The run's Writer withheld a manuscript
under strict evidence enforcement, so the page says it is a descriptive result
report and keeps the plan's claim ceiling and open limitation.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/easyicu/webserver/static/assets/demo"
TRACE = ASSETS / "sofa2-run-trace.json"
OUTPUT = {"en": ASSETS / "sofa2-study-report.en.html", "zh": ASSETS / "sofa2-study-report.zh.html"}

#: References in order of first citation; each key must be in the plan's literature.
CITED = (
    "singer_sepsis3_2016", "johnson_mimiciv_2023", "strobe_2007", "record_2015",
    "anderson_landmark_1983", "suissa_immortal_time_2008",
)

TEXT = {
    "kicker": ("Descriptive result report · laid out from one registered run", "描述性结果报告 · 由一次已登记运行排版"),
    "title": (
        "An experimental SOFA-2 Sepsis-3 phenotype and in-hospital mortality among adult ICU stays: a descriptive cohort analysis in MIMIC-IV",
        "成人 ICU 入住中实验性 SOFA-2 脓毒症表型与院内死亡：MIMIC-IV 描述性队列分析",
    ),
    "not_manuscript": (
        "This page is not a manuscript. The run’s Writer withheld one under strict evidence enforcement; the page arranges the run’s registered tables and figures and adds no finding of its own.",
        "本页不是论文。该次运行的论文生成被严格证据规则阻止；本页只排版运行登记的结果表与图，不增加任何结论。",
    ),
    "summary": ("Key results", "主要结果"),
    "methods": ("Methods", "方法"),
    "source": ("Data source", "数据来源"),
    "population": ("Study population", "研究人群"),
    "exposure": ("Exposure", "暴露"),
    "outcome": ("Outcome", "结局"),
    "analysis": ("Analysis", "分析"),
    "results": ("Results", "结果"),
    "boundary": ("How far the result reads", "结果的解读范围"),
    "traceability": ("Data and code traceability", "数据与代码可追溯性"),
    "references": ("References", "参考文献"),
    "supplementary": ("Supplementary figure", "补充图"),
    "group": ("Group", "分组"),
    "stays": ("ICU stays, n", "ICU 入住，n"),
    "share": ("Share, %", "占比，%"),
    "deaths": ("Deaths, n", "死亡，n"),
    "rate": ("In-hospital mortality, %", "院内死亡率，%"),
    "with": ("With the phenotype", "有表型"),
    "without": ("Without the phenotype", "无表型"),
    "all": ("All stays", "全部入住"),
}


def _t(key: str, lang: str) -> str:
    en, zh = TEXT[key]
    return zh if lang == "zh" else en


def _count(value: Any) -> str:
    return f"{int(value):,}"


def _pct(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _rows(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = trace["evidence"]["result_rows"]
    found = {
        "with": next(row for row in rows if row.get("level") == 1),
        "without": next(row for row in rows if row.get("level") == 0),
        "all": next(row for row in rows if row.get("role") == "overall"),
    }
    for row in found.values():
        if row["interval_method"] != "none_counts_only":
            raise SystemExit("the report layout is written for a counts-only result")
    return found


def _figure_url(record: dict[str, Any]) -> str:
    image = (ASSETS / record["png"]).read_bytes()
    if hashlib.sha256(image).hexdigest() != record["png_sha256"]:
        raise SystemExit(f"{record['png']} no longer matches its trace record")
    return f"{record['png']}?v={record['png_sha256'][:8]}"


def _reference(entry: dict[str, Any]) -> str:
    link = f' <a href="{escape(entry["source_url"])}">PMID {escape(str(entry["pmid"]))}</a>' if entry.get("pmid") else ""
    return f"{escape(entry['title'])} <i>{escape(entry['venue'])}</i>. {escape(str(entry['year']))}.{link}"


def page(trace: dict[str, Any], lang: str) -> str:
    zh = lang == "zh"
    rows = _rows(trace)
    cohort = trace["cohort"]
    criterion = cohort["inclusion"][0]
    literature = {entry["key"]: entry for entry in trace["literature"]}
    missing = [key for key in CITED if key not in literature]
    if missing:
        raise SystemExit(f"cited keys missing from the plan's literature: {missing}")
    cite = {key: index + 1 for index, key in enumerate(CITED)}
    primary = next(step for step in trace["plan"]["steps"] if step["role"] == "primary")
    sources = trace["sources"]
    main = trace["evidence"]["rerender"]
    supporting = trace["evidence"]["rerender_supporting"]
    records = trace["evidence"]["records"]
    table_id, code_id, figure_id = trace["evidence"]["chain"][:3]
    w, wo, al = rows["with"], rows["without"], rows["all"]

    def ref(*keys: str) -> str:
        return "<sup>" + ",".join(str(cite[key]) for key in keys) + "</sup>"

    if zh:
        summary = [
            f"{_count(al['stays'])} 个成人 ICU 入住中，{_count(w['stays'])} 个（{_pct(w['share_pct'])}%）在入 ICU 后 24 小时内符合实验性 SOFA-2 脓毒症表型。",
            f"有表型组院内死亡 {_count(w['deaths'])} 例（{_pct(w['rate_pct'])}%），无表型组 {_count(wo['deaths'])} 例（{_pct(wo['rate_pct'])}%）；全部入住 {_pct(al['rate_pct'])}%。",
            "仅为计数与比例：未作调整，没有置信区间或假设检验。",
        ]
        methods = {
            "source": f"MIMIC-IV 单中心重症监护数据库{ref('johnson_mimiciv_2023')}，以 EasyICU 概念字典提取（concept-dict 提交 <code>{escape(sources['concept_dict']['commit'][:9])}</code>，sofa2-dict 提交 <code>{escape(sources['sofa2_dict']['commit'][:9])}</code>）。",
            "population": f"年龄 ≥ {criterion['value']} 岁的 ICU 入住，共 {_count(cohort['n_stays'])} 个。分析单位为 ICU 入住；导出不含患者标识，同一患者的多次入住无法合并。",
            "exposure": f"入 ICU 后 0–{trace['dictionary']['window_hours'][1]:g} 小时内的实验性 Sepsis-3 表型：疑似感染前后 SOFA-2 评分升高 ≥ 2 分{ref('singer_sepsis3_2016')}。这是用 SOFA-2 替代原 SOFA 的敏感性定义，不是 2016 年的标准定义。",
            "outcome": "院内死亡，取自 <code>admissions.hospital_expire_flag</code>。",
            "analysis": f"按表型分组报告入住数、占比、死亡数和院内死亡率，分母均为各组全部入住。按 STROBE 与 RECORD 规范报告{ref('strobe_2007', 'record_2015')}。计划规定只作描述，不计算置信区间、检验或效应量。",
        }
        boundary = [
            "<b>仅描述性。</b>两组死亡率的差别不能解读为表型导致死亡，也没有调整任何混杂。",
            f"<b>暴露时间晚于随访起点。</b>表型在入 ICU 后 24 小时内判定，24 小时内死亡的患者获得表型的机会不同{ref('anderson_landmark_1983', 'suissa_immortal_time_2008')}；计划将此列为未解决的局限。",
            "<b>没有患者标识。</b>同一患者的多次入住按独立入住计数，因此没有给出区间估计。",
        ]
        captions = {
            "flow": f"<b>图 1</b> 队列流程。数据来自登记的队列流程表 <code>{escape(supporting['cohort_flow']['source_table'])}</code>。",
            "main": "<b>图 2</b> 表型占比（A）与各组院内死亡率（B）。B 中虚线为全部入住的死亡率。按登记的结果表重新绘制。",
            "table": "<b>表 1</b> 按表型分组的院内死亡",
            "table_note": "分母为各组全部入住；百分比由表中计数计算。",
            "dq": f"<b>补充图 S1</b> 分析变量的数据完整性。数据来自登记的测量审计表 <code>{escape(supporting['data_quality']['source_table'])}</code>。",
        }
        trace_rows = [
            ("结果表", table_id, records[table_id]["sha256"]),
            ("分析代码", code_id, records[code_id]["sha256"]),
            ("运行原图", figure_id, records[figure_id]["sha256"]),
            ("分析队列", "cohort", cohort["cohort_sha256"]),
            ("MIMIC-IV 导出清单", "export manifest", cohort["export_manifest_sha256"]),
        ]
        trace_note = f"运行 <code>{escape(trace['source_run_id'])}</code> 共登记 {trace['evidence']['count']} 条证据。重新绘制的图与原图摘要不同，原图保留不变。"
    else:
        summary = [
            f"Of {_count(al['stays'])} adult ICU stays, {_count(w['stays'])} ({_pct(w['share_pct'])}%) met the experimental SOFA-2 sepsis phenotype within 24 hours of ICU admission.",
            f"In-hospital mortality was {_pct(w['rate_pct'])}% ({_count(w['deaths'])} deaths) with the phenotype and {_pct(wo['rate_pct'])}% ({_count(wo['deaths'])}) without it; {_pct(al['rate_pct'])}% overall.",
            "Counts and proportions only: unadjusted, with no confidence intervals or hypothesis tests.",
        ]
        methods = {
            "source": f"MIMIC-IV, a single-centre critical care database{ref('johnson_mimiciv_2023')}, extracted with the EasyICU concept dictionary (concept-dict commit <code>{escape(sources['concept_dict']['commit'][:9])}</code>, sofa2-dict commit <code>{escape(sources['sofa2_dict']['commit'][:9])}</code>).",
            "population": f"ICU stays of patients aged ≥ {criterion['value']} years ({_count(cohort['n_stays'])} stays). The unit of analysis is the ICU stay; the export carries no patient identifier, so repeat stays of one patient cannot be grouped.",
            "exposure": f"An experimental Sepsis-3 phenotype within hours 0–{trace['dictionary']['window_hours'][1]:g} after ICU admission: a rise of ≥ 2 points in the SOFA-2 score around a suspected infection{ref('singer_sepsis3_2016')}. It substitutes SOFA-2 for the original SOFA and is a sensitivity definition, not the 2016 definition.",
            "outcome": "In-hospital death, from <code>admissions.hospital_expire_flag</code>.",
            "analysis": f"Stays, share, deaths and in-hospital mortality by phenotype group, each over all stays in the group, reported following STROBE and RECORD{ref('strobe_2007', 'record_2015')}. The plan specified a descriptive analysis without confidence intervals, tests or effect estimates.",
        }
        boundary = [
            "<b>Descriptive only.</b> The difference in mortality between the groups is not an effect of the phenotype, and nothing is adjusted for confounding.",
            f"<b>Exposure assessed after the start of follow-up.</b> The phenotype is assessed over the first 24 hours, so stays ending in early death had a different opportunity to meet it{ref('anderson_landmark_1983', 'suissa_immortal_time_2008')}; the plan records this as an open limitation.",
            "<b>No patient identifier.</b> Repeat stays of one patient count as separate stays, so no interval estimate is given.",
        ]
        captions = {
            "flow": f"<b>Figure 1</b> Cohort flow, from the registered cohort-flow table <code>{escape(supporting['cohort_flow']['source_table'])}</code>.",
            "main": "<b>Figure 2</b> Share of stays by phenotype (A) and in-hospital mortality by group (B). The dashed line in B is the mortality of all stays. Redrawn from the registered result table.",
            "table": "<b>Table 1</b> In-hospital mortality by phenotype group",
            "table_note": "Denominators are all stays in each group; percentages are computed from the counts shown.",
            "dq": f"<b>Supplementary Figure S1</b> Data completeness of the analysis variables, from the registered measurement audit <code>{escape(supporting['data_quality']['source_table'])}</code>.",
        }
        trace_rows = [
            ("Result table", table_id, records[table_id]["sha256"]),
            ("Analysis code", code_id, records[code_id]["sha256"]),
            ("Run figure", figure_id, records[figure_id]["sha256"]),
            ("Analysis cohort", "cohort", cohort["cohort_sha256"]),
            ("MIMIC-IV export manifest", "export manifest", cohort["export_manifest_sha256"]),
        ]
        trace_note = f"Run <code>{escape(trace['source_run_id'])}</code> registered {trace['evidence']['count']} evidence records. The redrawn figures have their own digests; the run’s figures are kept unchanged."
    del primary  # the claim ceiling and limitation are stated in ``boundary``

    table_rows = "".join(
        f"<tr><th scope=\"row\">{escape(_t(key, lang))}</th><td>{_count(row['stays'])}</td>"
        f"<td>{_pct(row['share_pct'])}</td><td>{_count(row['deaths'])}</td><td>{_pct(row['rate_pct'])}</td></tr>"
        for key, row in rows.items()
    )
    head = "".join(f"<th scope=\"col\">{escape(_t(key, lang))}</th>" for key in ("group", "stays", "share", "deaths", "rate"))
    methods_html = "".join(f"<h3>{escape(_t(key, lang))}</h3><p>{text}</p>" for key, text in methods.items())
    references = "".join(f"<li>{_reference(literature[key])}</li>" for key in CITED)
    trace_html = "".join(
        f"<tr><th scope=\"row\">{escape(label)}</th><td><code>{escape(ident)}</code></td><td><code>{escape(digest[:12])}</code></td></tr>"
        for label, ident, digest in trace_rows
    )
    return f"""<!doctype html>
<html lang="{'zh-CN' if zh else 'en'}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="easyicu-authority-class" content="descriptive_result_report">
<meta name="easyicu-source-run" content="{escape(trace['source_run_id'])}">
<title>{escape(_t('title', lang))}</title>
<style>
:root{{--ink:#1f2328;--ink-2:#4a4f57;--ink-3:#6e747d;--rule:#c9cdd3;--accent:#0f4d92;--paper:#fff;--wash:#f4f6f8}}
*{{box-sizing:border-box}}
html{{background:#eceef1}}
body{{margin:0;color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"PingFang SC","Microsoft YaHei",sans-serif}}
article{{max-width:820px;margin:28px auto;padding:56px 64px 64px;background:var(--paper);box-shadow:0 1px 3px #0000001a}}
.kicker{{color:var(--accent);font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}}
h1{{margin:10px 0 14px;font:600 28px/1.25 Georgia,"Times New Roman","Songti SC","SimSun",serif;letter-spacing:-.01em}}
.meta{{color:var(--ink-3);font-size:13px}}
.notice{{margin:18px 0 26px;padding:10px 14px;border-left:3px solid var(--rule);background:var(--wash);color:var(--ink-2);font-size:13.5px}}
h2{{margin:34px 0 10px;padding-top:12px;border-top:1px solid var(--rule);font:600 19px/1.3 Georgia,"Times New Roman","Songti SC","SimSun",serif}}
h3{{margin:16px 0 2px;font-size:14px;font-weight:650}}
p{{margin:4px 0 10px}}
.summary{{margin:0;padding:14px 18px 14px 34px;border:1px solid var(--rule);border-radius:4px}}
.summary li{{margin:4px 0}}
sup{{font-size:10px;color:var(--accent)}}
code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink-2);overflow-wrap:anywhere}}
figure{{margin:20px 0}}
figure img{{display:block;width:100%;height:auto}}
figure.flow img{{max-width:560px;margin:0 auto}}
figcaption{{margin-top:8px;color:var(--ink-2);font-size:13px;line-height:1.55}}
.table-caption{{margin:22px 0 6px;font-size:13.5px}}
table.three-line{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}}
.three-line th,.three-line td{{padding:6px 10px;text-align:right;white-space:nowrap}}
.three-line th:first-child{{text-align:left}}
.three-line thead th{{font-weight:600;border-bottom:.75px solid var(--ink)}}
.three-line tbody tr:last-child>*{{border-top:.5px solid var(--rule)}}
.three-line tbody th{{font-weight:400}}
.table-note{{margin-top:6px;color:var(--ink-3);font-size:12.5px}}
.boundary li{{margin:6px 0}}
table.trace{{width:100%;border-collapse:collapse;font-size:13px}}
.trace th,.trace td{{padding:5px 8px;border-bottom:1px solid #e4e7eb;text-align:left;vertical-align:top}}
.trace th{{font-weight:500;color:var(--ink-2);white-space:nowrap}}
ol.refs{{padding-left:22px;font-size:13px;color:var(--ink-2)}}
ol.refs li{{margin:4px 0}}
a{{color:var(--accent)}}
@media (max-width:720px){{article{{margin:0;padding:28px 18px 36px;box-shadow:none}}h1{{font-size:23px}}.three-line th,.three-line td{{padding:5px 6px;white-space:normal}}}}
@media print{{html{{background:none}}article{{margin:0;padding:0;box-shadow:none;max-width:none}}h2{{break-after:avoid}}figure,table{{break-inside:avoid}}}}
</style>
</head>
<body>
<article>
<div class="kicker">{escape(_t('kicker', lang))}</div>
<h1>{escape(_t('title', lang))}</h1>
<div class="meta">MIMIC-IV · {escape(trace['source_run_id'])} · {escape(trace['finished_at'][:10])}</div>
<p class="notice">{escape(_t('not_manuscript', lang))}</p>
<h2>{escape(_t('summary', lang))}</h2>
<ul class="summary">{''.join(f'<li>{escape(item)}</li>' for item in summary)}</ul>
<h2>{escape(_t('methods', lang))}</h2>
{methods_html}
<h2>{escape(_t('results', lang))}</h2>
<figure class="flow"><img src="{escape(_figure_url(supporting['cohort_flow']))}" alt=""><figcaption>{captions['flow']}</figcaption></figure>
<p class="table-caption">{captions['table']}</p>
<table class="three-line"><thead><tr>{head}</tr></thead><tbody>{table_rows}</tbody></table>
<p class="table-note">{escape(captions['table_note'])}</p>
<figure><img src="{escape(_figure_url(main))}" alt=""><figcaption>{captions['main']}</figcaption></figure>
<h2>{escape(_t('boundary', lang))}</h2>
<ul class="boundary">{''.join(f'<li>{item}</li>' for item in boundary)}</ul>
<h2>{escape(_t('traceability', lang))}</h2>
<p>{trace_note}</p>
<table class="trace"><tbody>{trace_html}</tbody></table>
<h2>{escape(_t('supplementary', lang))}</h2>
<figure><img src="{escape(_figure_url(supporting['data_quality']))}" alt=""><figcaption>{captions['dq']}</figcaption></figure>
<h2>{escape(_t('references', lang))}</h2>
<ol class="refs">{references}</ol>
</article>
</body>
</html>
"""


def main() -> int:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    for lang, path in OUTPUT.items():
        path.write_text(page(trace, lang), encoding="utf-8")
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
