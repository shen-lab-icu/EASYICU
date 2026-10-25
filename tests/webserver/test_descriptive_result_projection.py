"""Registered aggregate summaries survive display caps without recomputation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

import easyicu
import pytest

from easyicu.webserver.agent_pipeline_runs import _table_projection
from tests.support.node import run_node


HEADERS = ["variable", "statistic", "level", "value", "numerator", "denominator",
           "percentage", "unit", "source_column", "group_variable", "group", "row_role"]
PLAN = {"analysis_type": "descriptive_epidemiology", "steps": [
    {"step_id": "describe", "planned_analysis_role": "primary",
     "expected_outputs": ["table:cohort_summary", "table:distribution_summary"]},
    {"step_id": "support", "planned_analysis_role": "secondary"},
]}


def summary_rows(group: str = "North", *, n: str = "7.0", mean: bool = True,
                 median: bool = True) -> list[list[str]]:
    statistics = {"nonmissing_n": n, "missing_n": "3.0"}
    if mean:
        statistics.update(mean="18.1250", std="2.2500")
    if median:
        statistics.update(median="17.7500", q1="16.1250", q3="19.8750")
    return [["analyte", statistic, "", value, value if statistic.endswith("_n") else "",
             "10" if statistic.endswith("_n") else n, "", "mmol/L", "analyte",
             "ward", group, "exposure_level"] for statistic, value in statistics.items()]


def registered(tmp_path: Path, rows: list[list[str]], *, audits: int = 0,
               headers: list[str] | None = None) -> list[dict]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    records = []
    for i in range(audits):
        path = evidence / f"measurement_audit_{i}.csv"
        path.write_text("concept,n_total\nsynthetic,10\n", encoding="utf-8")
        records.append({"kind": "table", "evidence_id": f"audit_{i}",
                        "relative_path": f"evidence/{path.name}", "produced_by_step": "support"})
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers or HEADERS)
    writer.writerows(rows)
    path = evidence / "registered_summary.csv"
    path.write_text(output.getvalue(), encoding="utf-8", newline="")
    records.append({"kind": "table", "evidence_id": "registered_summary",
                    "relative_path": f"evidence/{path.name}", "produced_by_step": "describe",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (evidence / "evidence_index.json").write_text(json.dumps(records), encoding="utf-8")
    return records


def projected(tmp_path: Path) -> dict:
    return next(table for table in _table_projection(tmp_path, plan=PLAN)["tables"]
                if table["evidence_id"] == "registered_summary")


def test_primary_summary_precedes_audits_and_keeps_all_groups_beyond_the_preview(tmp_path):
    rows = [row for i in range(9) for row in summary_rows(f"Ward {i}")]
    registered(tmp_path, rows, audits=14)

    table = projected(tmp_path)

    assert len(table["rows"]) == 30
    assert table["preview_truncated"] is True
    compact = table["descriptive_summary"]
    assert compact["schema_version"] == "easyicu.web-descriptive-summary/1"
    assert [row["group"] for row in compact["continuous"]] == [f"Ward {i}" for i in range(9)]
    assert compact["continuous"][-1] == {
        "variable": "analyte", "group": "Ward 8", "row_role": "exposure_level",
        "group_variable": "ward", "n_nonmissing": "7.0", "unit": "mmol/L",
        "mean": "18.1250", "sd": "2.2500", "median": "17.7500",
        "q25": "16.1250", "q75": "19.8750",
    }


@pytest.mark.parametrize("mean,median", [(False, True), (True, False)])
def test_only_registered_statistics_are_projected(tmp_path, mean, median):
    registered(tmp_path, summary_rows(mean=mean, median=median))
    row = projected(tmp_path)["descriptive_summary"]["continuous"][0]
    assert ("mean" in row) is mean
    assert ("median" in row) is median
    assert row["n_nonmissing"] == "7.0"


def test_categorical_proportions_keep_the_registered_numerator_and_denominator(tmp_path):
    rows = [["status", "level_count", level, number, number, "7", percentage,
             "", "status", "ward", group, "exposure_level"]
            for group, level, number, percentage in [
                ("North", "yes", "2", "28.5714285714"), ("North", "no", "5", "71.4285714286"),
                ("South", "yes", "0", "0.0000"), ("South", "no", "7", "100.0000")]]
    registered(tmp_path, rows)
    compact = projected(tmp_path)["descriptive_summary"]
    assert compact["continuous"] == []
    assert [(r["group"], r["level"], r["numerator"], r["denominator"], r["percentage"])
            for r in compact["categorical"]] == [
                ("North", "yes", "2", "7", "28.5714285714"), ("North", "no", "5", "7", "71.4285714286"),
                ("South", "yes", "0", "7", "0.0000"), ("South", "no", "7", "7", "100.0000")]


@pytest.mark.parametrize("group_columns", [False, True])
def test_ungrouped_summary_does_not_acquire_an_invented_group(tmp_path, group_columns):
    rows = summary_rows()
    for row in rows:
        row[9:] = ["", "", "overall"]
    registered(tmp_path, rows if group_columns else [row[:9] for row in rows],
               headers=HEADERS if group_columns else HEADERS[:9])
    row = projected(tmp_path)["descriptive_summary"]["continuous"][0]
    assert row["group"] == row["group_variable"] == ""
    assert row["row_role"] == "overall"


@pytest.mark.parametrize("mode", ["duplicate_statistic", "mixed_units", "wrong_denominator", "too_many_groups"])
def test_inconsistent_or_oversized_summaries_have_no_compact_answer(tmp_path, mode):
    rows = summary_rows()
    if mode == "duplicate_statistic":
        rows.append(list(rows[-1]))
    elif mode == "mixed_units":
        rows[-1][7] = "mg/dL"
    elif mode == "wrong_denominator":
        rows[-1][5] = "10"
    else:
        rows = [row for i in range(65) for row in summary_rows(f"Ward {i}")]
    registered(tmp_path, rows)
    assert "descriptive_summary" not in projected(tmp_path)


@pytest.mark.parametrize("mode", ["missing_digest", "duplicate_id", "source_copy", "secondary"])
def test_unbound_copies_and_secondary_summaries_do_not_gain_an_answer(tmp_path, mode):
    records = registered(tmp_path, summary_rows())
    if mode == "missing_digest":
        records[-1].pop("sha256")
    elif mode == "duplicate_id":
        records.append(dict(records[-1]))
    elif mode == "source_copy":
        records[-1]["description"] = "Source data copied beside a promoted publication figure."
    else:
        records[-1]["produced_by_step"] = "support"
    (tmp_path / "evidence/evidence_index.json").write_text(json.dumps(records), encoding="utf-8")
    assert "descriptive_summary" not in projected(tmp_path)


def answer_payload(payload: dict, plan: dict, *, root: Path | None = None,
                   lang: str = "zh", context: dict | None = None) -> dict:
    """Exercise the browser owners in Node; never import Python from JS CI."""
    script = r"""
      const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
      const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
      global.window = global; global.EU_LANG = input.lang || 'zh';
      const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
      require(path.join(process.argv[1], 'src/easyicu/webserver/static/js/data-catalog.js'));
      for (const name of ['modules', 'result-summary', 'run-answer'])
        require(path.join(process.argv[1], 'src/easyicu/webserver/static/js', `screens-guided-pi-${name}.js`));
      const modules = global.EasyICU.guidedPi;
      const payloads = {'agent_plan.json': input.plan, 'result_tables.json': input.payload,
        'run_context.json': input.context || {}};
      const run = {run_id: 'aggregate_replay', analysis_results_available: true, artifact_refs:
        Object.entries(payloads).map(([artifact, value]) => ({artifact, run_id: 'aggregate_replay',
          kind: 'research_artifact', sha256: crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex')}))};
      const answer = modules.require('runAnswer').create({esc, tr: (en, zh) => global.EU_LANG === 'zh' ? zh : en,
        projectId: () => 'aggregate_replay', resourceButton: () => '',
        api: () => ({loadPiCopilotResearchArtifact: async (project, runId, artifact) => ({payload: payloads[artifact]})})});
      (async () => {
        await answer.load(run);
        process.stdout.write(JSON.stringify({html: answer.render(run),
          summary: modules.require('resultSummary').summarize(input.payload, input.plan, {})}));
      })().catch(error => { console.error(error); process.exitCode = 1; });
    """
    with tempfile.TemporaryDirectory(prefix="easyicu-answer-") as temporary:
        input_file = Path(temporary) / "aggregate.json"
        input_file.write_text(json.dumps({"payload": payload, "plan": plan, "lang": lang, "context": context}),
                              encoding="utf-8")
        result = run_node("node", script, str(root or Path(easyicu.__file__).resolve().parents[2]),
                          str(input_file), check=True, timeout=15)
    return json.loads(result.stdout)


@pytest.mark.parametrize("kind", ["median_only", "mean_only", "categorical", "ungrouped", "unanswerable"])
def test_registered_csv_reaches_the_answer_without_a_browser(tmp_path, kind):
    rows = summary_rows()
    if kind in {"median_only", "mean_only"}:
        statistic = "median" if kind == "median_only" else "mean"
        rows = [row for group in ["North", "South"] for row in summary_rows(group)
                if row[1] in {"nonmissing_n", statistic}]
    elif kind == "categorical":
        rows = [["status", "level_count", "yes", "2", "2", "7", "28.5714285714", "", "status",
                 "ward", group, "exposure_level"] for group in ["North", "South"]]
    elif kind == "ungrouped":
        for row in rows:
            row[9:] = ["", "", "overall"]
    else:
        rows = [row for row in rows if row[1] in {"nonmissing_n", "missing_n"}]
    registered(tmp_path, rows)

    html = answer_payload(_table_projection(tmp_path, plan=PLAN), PLAN)["html"]

    if kind == "unanswerable":
        assert "分析已完成，暂未生成结果摘要。请查看结果表。" in html
        return
    assert "暂未生成结果摘要" not in html
    if kind == "categorical":
        assert "28.6%（2/7）" in html
    else:
        assert "mmol/L" in html
        assert "n=7" in html
        assert ("中位数" in html) is (kind != "mean_only")
        assert ("均值" in html) is (kind == "mean_only")
        assert ("17.8" in html) is (kind != "mean_only")
        assert ("18.1" in html) is (kind == "mean_only")
    if kind in {"median_only", "mean_only"}:
        assert "四分位距" not in html and "标准差" not in html
    if kind == "ungrouped":
        assert "「」" not in html and "North" not in html
    else:
        assert "North" in html and "South" in html


def test_a_csv_parse_failure_after_the_preview_cannot_break_the_result_projection(tmp_path):
    rows = [row for i in range(9) for row in summary_rows(f"Ward {i}")]
    rows[-1][3] = "9" * 140_000
    registered(tmp_path, rows)
    table = projected(tmp_path)
    assert len(table["rows"]) == 30
    assert "descriptive_summary" not in table


@pytest.mark.parametrize("groups", [33, 65])
def test_wide_primary_summary_keeps_units_and_groups_behind_display_limits(tmp_path, groups):
    headers = ["variable", "group", "row_role", "group_n", "n_nonmissing", "missing_n",
               "missing_pct", "median", "q25", "q75", "mean", "sd", "unit"]
    rows = [["analyte", f"Ward {i}", "group", "10", "7", "3", "30", "17.7500",
             "16.1250", "19.8750", "18.1250", "2.2500", "mmol/L"] for i in range(groups)]
    registered(tmp_path, rows, headers=headers)
    table = projected(tmp_path)
    assert len(table["rows"]) == 30 and len(table["headers"]) == 12
    assert table["preview_truncated"] and table["preview_columns_truncated"]
    if groups > 64:
        assert "descriptive_summary" not in table
        return
    compact = table["descriptive_summary"]["continuous"]
    assert len(compact) == groups
    assert compact[-1]["unit"] == "mmol/L"
    assert compact[-1]["n_nonmissing"] == "7"
    result = answer_payload(_table_projection(tmp_path, plan=PLAN), PLAN)
    assert len(result["summary"]["continuousGroups"]) == groups
    assert "Ward 11" in result["html"] and "mmol/L" in result["html"]
    assert "Ward 12" not in result["html"] and "另有 21 组见结果表" in result["html"]


@pytest.mark.parametrize("wide", [False, True])
def test_missing_group_labels_are_not_invented_as_overall(tmp_path, wide):
    if wide:
        registered(tmp_path, [["analyte", "", "", "7", "17.7500", "mmol/L"]],
                   headers=["variable", "group", "row_role", "n", "median", "unit"])
    else:
        rows = summary_rows(group="")
        for row in rows:
            row[-1] = ""
        registered(tmp_path, rows)
    result = answer_payload(_table_projection(tmp_path, plan=PLAN), PLAN)
    assert "分析已完成，暂未生成结果摘要。请查看结果表。" in result["html"]


@pytest.mark.parametrize("mode", ["bound", "wrong_digest", "wrong_step", "outside", "duplicate", "absent"])
def test_descriptive_reader_metadata_comes_only_from_the_exact_bound_context(tmp_path, mode):
    registered(tmp_path, summary_rows())
    variable = {"name": "analyte", "source_concept": "glu", "description": "source glucose",
                "unit_normalization": "window_numeric_max", "analysis_window": "icu_admission[0,18]h"}
    variables = [variable, {"name": "unrelated", "description": "Unused metadata"}]
    if mode == "duplicate":
        variables.append(variable)
    raw = json.dumps({"variables": variables}).encode()
    (tmp_path / "research_context.json").write_bytes(raw)
    context = {"relative_path": "research_context.json", "sha256": hashlib.sha256(raw).hexdigest()}
    if mode == "wrong_digest":
        context["sha256"] = "0" * 64
    elif mode == "outside":
        context["relative_path"] = "../research_context.json"
    resolved = tmp_path / "resolved_inputs"
    resolved.mkdir()
    if mode != "absent":
        (resolved / "describe.json").write_text(json.dumps({
            "step_id": "other" if mode == "wrong_step" else "describe", "context": context,
        }))

    table = projected(tmp_path)
    compact = table["descriptive_summary"]
    assert compact["continuous"][0]["median"] == "17.7500"
    if mode != "bound":
        assert not compact.get("variable_display")
        return
    assert compact["variable_display"] == {"analyte": {key: variable[key] for key in (
        "source_concept", "description", "unit_normalization", "analysis_window")}}
    html = answer_payload(_table_projection(tmp_path, plan=PLAN), PLAN)["html"]
    assert "入 ICU 18 小时内最高血糖（mmol/L），中位数［四分位距］：North 17.8［16.1–19.9］，n=7。" in html
