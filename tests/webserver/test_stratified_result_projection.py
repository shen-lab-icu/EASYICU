"""Verified generated secondary products reach the compact answer end to end."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import easyicu
import pytest

from easyicu.webserver.agent_pipeline_runs import _table_projection
from tests.support.node import run_node


PLAN = {
    "analysis_type": "association",
    "research_question": "Compare the marker association stratified by ward.",
    "display_labels": {"ward": "Ward"},
    "steps": [
        {"step_id": "primary", "planned_analysis_role": "primary"},
        {"step_id": "modifier", "planned_analysis_role": "secondary",
         "scientific_action_id": "association.effect_modification",
         "method": "planned_modifier_regression", "inputs": ["marker", "event", "ward"],
         "expected_outputs": ["table:ward_comparison"]},
    ],
}
HEADERS = ["subgroup_column", "subgroup_value", "estimate", "ci_low", "ci_high",
           "effect_scale", "interaction_p_value", "fit_status", "contrast"]


def requested_secondary_tables(*args):
    # Keep the end-to-end regressions collectible against the read-only base,
    # which predates this owner, without replacing either checkout's code.
    from easyicu.webserver.stratified_result_projection import requested_secondary_tables as bind

    return bind(*args)


def fixture(tmp_path, strata=2, *, headers=None, rows=None):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    records = []

    def register(evidence_id, filename, step, columns, values):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(values)
        path = evidence / f"{evidence_id}__{filename}"
        path.write_text(stream.getvalue(), encoding="utf-8", newline="")
        records.append({"evidence_id": evidence_id, "kind": "table", "produced_by_step": step,
                        "relative_path": f"evidence/{path.name}", "generation_mode": "llm",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "metadata": {"diagnostic_only": False, "aliases_published": True}})

    # The previous preview's first twelve entries crowd out the requested result.
    for index in range(12):
        register(f"audit_{index}", f"missingness_audit_{index}.csv", "audit", ["concept", "n"], [["marker", 24]])
    register("primary_table", "primary.csv", "primary",
             ["fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "is_primary_contrast", "contrast"],
             [["fitted", "1.23456", "1.01", "1.55", "odds_ratio", "true", "B vs A"]])
    values = rows if rows is not None else [
        ["ward", level, estimate, low, high, "odds_ratio", "0.034567", "fitted", "B vs A"]
        for level, estimate, low, high in [
            ("North", "1.24567", "1.01567", "1.75678"),
            ("South", "0.84567", "0.61567", "1.15678"),
            ("West", "1.44567", "1.11567", "1.95678"),
        ][:strata]
    ]
    register("strata_table", "custom_result.csv", "modifier", headers or HEADERS, values)
    manifest = {"per_step_records": [{"step_id": "modifier", "status": "ok",
        "evidence_ids": ["strata_table"], "step_summary": {"status": "completed",
            "output_files": {"table:ward_comparison": "custom_result.csv"}}}]}
    save(tmp_path, records, manifest)
    return records, manifest


def save(tmp_path, records, manifest):
    (tmp_path / "evidence/evidence_index.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def answer(tmp_path, payload, plan=PLAN, lang="zh"):
    script = r"""
const fs = require('node:fs'), path = require('node:path');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
global.window = global; global.EU_LANG = input.lang;
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
global.EU_HTML = {esc};
require(path.join(process.argv[1], 'src/easyicu/webserver/static/js/screens-guided-contracts.js'));
for (const name of ['modules', 'result-summary', 'run-answer'])
  require(path.join(process.argv[1], 'src/easyicu/webserver/static/js', `screens-guided-pi-${name}.js`));
const modules = global.EasyICU.guidedPi;
const payloads = {'agent_plan.json': input.plan, 'result_tables.json': input.payload};
const run = {run_id: 'synthetic', analysis_results_available: true, artifact_refs:
  Object.keys(payloads).map(artifact => ({artifact, run_id: 'synthetic', sha256: 'a'.repeat(64)}))};
const owner = modules.require('runAnswer').create({esc, tr: (en, zh) => input.lang === 'zh' ? zh : en,
  projectId: () => 'synthetic', resourceButton: ref => `<button data-evidence="${esc(ref.evidence_id || '')}" data-sha="${esc(ref.evidence_sha256 || '')}">source</button>`,
  api: () => ({loadPiCopilotResearchArtifact: async (project, run, artifact) => ({payload: payloads[artifact]})})});
(async () => {await owner.load(run); process.stdout.write(owner.render(run));})()
  .catch(error => {console.error(error); process.exitCode = 1;});
"""
    path = tmp_path / "answer_input.json"
    path.write_text(json.dumps({"payload": payload, "plan": plan, "lang": lang}), encoding="utf-8")
    root = Path(easyicu.__file__).resolve().parents[2]
    return run_node("node", script, str(root), str(path), check=True, timeout=15).stdout


@pytest.mark.parametrize("strata", [2, 3])
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_requested_strata_survive_the_preview_cap_and_follow_the_primary(tmp_path, strata, lang):
    records, _ = fixture(tmp_path, strata)
    payload = _table_projection(tmp_path, plan=PLAN)
    assert len(payload["tables"]) == 12
    table = next(t for t in payload["tables"] if t["evidence_id"] == "strata_table")
    assert table["sha256"] == records[-1]["sha256"]
    assert len(table["stratified_summary"]["strata"]) == strata
    html = answer(tmp_path, payload, lang=lang)
    primary, subgroup, interaction = ("主要估计", "按 Ward 分层", "交互作用 p") if lang == "zh" else (
        "Primary estimate", "Stratified by Ward", "interaction p")
    assert html.index(primary) < html.index(subgroup) < html.index(interaction)
    assert "1.25" in html and "1.02–1.76" in html and "0.85" in html and "0.62–1.16" in html
    assert ("West" in html) is (strata == 3)
    assert "0.035" in html
    assert 'data-evidence="strata_table"' in html
    assert f'data-sha="{records[-1]["sha256"]}"' in html
    assert "0.034567" not in html and "1.24567" not in html
    assert html.count(subgroup) == 1 and "<h" not in html


@pytest.mark.parametrize("mode", ["failed", "superseded", "inactive", "unregistered", "digest", "duplicate_id",
    "wrong_owner", "undeclared", "wrong_file", "diagnostic", "unpublished", "omitted", "source_copy",
    "no_request", "not_secondary", "unrelated_action", "duplicate_product", "duplicate_step", "shared_file"])
def test_unverified_or_unrequested_secondary_products_never_enter_the_answer(tmp_path, mode):
    records, manifest = fixture(tmp_path)
    plan = copy.deepcopy(PLAN)
    record = manifest["per_step_records"][0]
    if mode == "failed":
        record["status"] = "failed"
    elif mode == "superseded":
        manifest["per_step_records"].append({**record, "status": "failed"})
    elif mode == "inactive":
        record["evidence_ids"] = []
    elif mode == "unregistered":
        records.pop()
    elif mode == "digest":
        records[-1]["sha256"] = "f" * 64
    elif mode == "duplicate_id":
        records.append(dict(records[-1]))
    elif mode == "wrong_owner":
        records[-1]["produced_by_step"] = "another"
    elif mode == "undeclared":
        plan["steps"][1]["expected_outputs"] = ["table:other"]
    elif mode == "wrong_file":
        record["step_summary"]["output_files"]["table:ward_comparison"] = "other.csv"
    elif mode == "diagnostic":
        records[-1]["metadata"]["diagnostic_only"] = True
    elif mode == "unpublished":
        records[-1]["metadata"]["aliases_published"] = False
    elif mode == "omitted":
        record["step_summary"]["effect_output_authorized"] = False
    elif mode == "source_copy":
        records[-1]["description"] = "Source data copied beside a promoted publication figure."
    elif mode == "no_request":
        plan["research_question"] = "Estimate the overall marker association."
    elif mode == "not_secondary":
        plan["steps"][1]["planned_analysis_role"] = "diagnostic"
    elif mode == "duplicate_product":
        plan["steps"].append({**plan["steps"][1], "step_id": "other"})
    elif mode == "duplicate_step":
        plan["steps"].append({**plan["steps"][1], "expected_outputs": ["table:other"]})
    elif mode == "shared_file":
        record["step_summary"]["output_files"]["table:other"] = "custom_result.csv"
    else:
        plan["steps"][1]["scientific_action_id"] = "association.descriptive_context"
    save(tmp_path, records, manifest)
    if mode != "source_copy":
        assert requested_secondary_tables(tmp_path, plan, records) == {}
    payload = _table_projection(tmp_path, plan=plan)
    assert all("stratified_summary" not in t for t in payload["tables"])
    html = answer(tmp_path, payload, plan)
    assert "主要估计" in html and "North" not in html and "0.035" not in html


@pytest.mark.parametrize("mode", ["failed_fit", "bad_ci", "nonfinite", "bad_p", "different_p", "duplicate_level",
                                  "mixed_contrast", "unknown_variable", "descriptive_only", "too_many", "missing_p",
                                  "bad_scale", "negative_ratio", "single_stratum", "wrong_ci_level",
                                  "duplicate_header", "duplicate_alias", "nonfinite_upper"])
def test_incomplete_or_ambiguous_stratum_values_are_not_summarized(tmp_path, mode):
    rows = [["ward", level, "1.2", "1.0", "1.5", "odds_ratio", "0.04", "fitted", "B vs A"]
            for level in ["North", "South"]]
    headers = HEADERS
    changes = {"failed_fit": (7, "failed"), "bad_ci": (3, "2.0"), "nonfinite": (2, "NaN"),
               "bad_p": (6, "1.2"), "different_p": (6, "0.2"), "duplicate_level": (1, "North"),
               "mixed_contrast": (8, "C vs A"), "unknown_variable": (0, "unplanned"), "missing_p": (6, ""),
               "bad_scale": (5, "mystery"), "nonfinite_upper": (4, "inf")}
    if mode in changes:
        index, value = changes[mode]
        rows[1][index] = value
        if mode in ("unknown_variable", "bad_scale", "bad_p"):
            rows[0][index] = value
    elif mode == "too_many":
        rows = [[*rows[0][:1], f"Ward {i}", *rows[0][2:]] for i in range(65)]
    elif mode == "negative_ratio":
        rows[1][2:5] = ["-1.2", "-1.5", "-1.0"]
    elif mode == "single_stratum":
        rows = rows[:1]
    elif mode in ("wrong_ci_level", "duplicate_header", "duplicate_alias"):
        column, values = {"wrong_ci_level": ("ci_level", ["90", "90"]),
                          "duplicate_header": ("estimate", ["1.2", "1.2"]),
                          "duplicate_alias": ("stratum", ["North", "South"])}[mode]
        headers = [*HEADERS, column]
        rows = [[*row, value] for row, value in zip(rows, values)]
    else:
        headers, rows = ["subgroup_value", "n", "event_rate"], [["North", 12, 0.2], ["South", 12, 0.3]]
    fixture(tmp_path, headers=headers, rows=rows)
    payload = _table_projection(tmp_path, plan=PLAN)
    assert all("stratified_summary" not in t for t in payload["tables"])
    assert "交互作用" not in answer(tmp_path, payload)


@pytest.mark.parametrize("mode", ["no_digest", "stale", "no_block", "wrong_product", "duplicate_table", "bad_ci", "bad_p",
                                  "wrong_role", "wrong_action", "wrong_input", "schema", "duplicate_level", "single_stratum"])
def test_browser_requires_the_verified_product_block_and_complete_values(tmp_path, mode):
    fixture(tmp_path)
    payload = _table_projection(tmp_path, plan=PLAN)
    plan = copy.deepcopy(PLAN)
    table = next(t for t in payload["tables"] if t["evidence_id"] == "strata_table")
    if mode == "no_digest":
        table.pop("sha256")
    elif mode == "stale":
        table["source_status"] = "stale"
    elif mode == "no_block":
        table.pop("stratified_summary")
    elif mode == "wrong_product":
        table["stratified_summary"]["product"] = "table:other"
    elif mode == "duplicate_table":
        payload["tables"].append(copy.deepcopy(table))
    elif mode == "bad_ci":
        table["stratified_summary"]["strata"][0]["ci_low"] = "8"
    elif mode == "bad_p":
        table["stratified_summary"]["interaction_p_value"] = "2"
    elif mode == "wrong_role":
        plan["steps"][1]["planned_analysis_role"] = "diagnostic"
    elif mode == "wrong_action":
        plan["steps"][1]["scientific_action_id"] = "association.descriptive_context"
    elif mode == "wrong_input":
        plan["steps"][1]["inputs"] = ["marker", "event"]
    elif mode == "schema":
        table["stratified_summary"]["schema_version"] = "unknown"
    elif mode == "duplicate_level":
        table["stratified_summary"]["strata"][1]["level"] = "North"
    else:
        table["stratified_summary"]["strata"] = table["stratified_summary"]["strata"][:1]
    html = answer(tmp_path, payload, plan)
    assert "主要估计" in html and "交互作用" not in html
    assert 'data-evidence="strata_table"' not in html


@pytest.mark.parametrize("question", ["比较按病区分层的标志物关联。", "Assess subgroup associations.",
                                      "Assess the marker interaction with ward.", "Compare the association within ward strata."])
def test_requested_results_keep_the_full_snapshot_when_columns_are_bounded(tmp_path, question):
    columns = [f"diagnostic_{i}" for i in range(14)] + HEADERS
    rows = [["audit"] * 14 + ["ward", level, "1.2", "1.0", "1.5", "odds_ratio", "0.00001", "fitted", "B vs A"]
            for level in ("North", "South", "West")]
    fixture(tmp_path, headers=columns, rows=rows)
    plan = {**PLAN, "research_question": question}
    payload = _table_projection(tmp_path, plan=plan)
    table = next(t for t in payload["tables"] if t["evidence_id"] == "strata_table")
    assert table["preview_columns_truncated"]
    assert [row["level"] for row in table["stratified_summary"]["strata"]] == ["North", "South", "West"]
    html = answer(tmp_path, payload, plan)
    assert "West" in html and "p &lt;0.001" in html


def test_safe_column_aliases_and_survival_scales_keep_the_declared_identity(tmp_path):
    columns = ["stratum_variable", "stratum", "estimate", "ci_low", "ci_high", "effect_scale",
               "p_interaction", "fit_status", "contrast"]
    rows = [["ward", level, "1.2", "1.0", "1.5", "hazard_ratio", "0.05", "fitted", "B vs A"]
            for level in ("North", "South")]
    fixture(tmp_path, headers=columns, rows=rows)
    plan = copy.deepcopy(PLAN)
    plan["steps"][1]["scientific_action_id"] = "time_to_event.subgroup_hr"
    html = answer(tmp_path, _table_projection(tmp_path, plan=plan), plan, lang="en")
    assert "North HR 1.20 (95% CI 1.00–1.50)" in html and "interaction p = 0.05" in html


def test_a_disappearing_registered_file_is_unavailable(tmp_path, monkeypatch):
    from easyicu.webserver import stratified_result_projection as owner

    records, _ = fixture(tmp_path)

    def disappeared(*_):
        raise FileNotFoundError("synthetic removed file")

    monkeypatch.setattr(owner, "verified_run_evidence_path", disappeared)
    assert requested_secondary_tables(tmp_path, PLAN, records) == {}


def test_failed_secondary_table_is_not_shown_even_without_preview_pressure(tmp_path):
    records, manifest = fixture(tmp_path)
    manifest["per_step_records"][0]["status"] = "failed"
    save(tmp_path, records[-2:], manifest)
    payload = _table_projection(tmp_path, plan=PLAN)
    assert [table["evidence_id"] for table in payload["tables"]] == ["primary_table"]
    assert "North" not in answer(tmp_path, payload)


def test_separate_stratum_fit_ids_preserve_the_shared_comparison(tmp_path):
    rows = [["ward", level, "1.2", "1.0", "1.5", "odds_ratio", "0.05", "fitted", "B vs A", model]
            for level, model in [("North", "fit_north"), ("South", "fit_south")]]
    fixture(tmp_path, headers=[*HEADERS, "model_id"], rows=rows)
    html = answer(tmp_path, _table_projection(tmp_path, plan=PLAN), lang="en")
    assert "North OR 1.20" in html and "South OR 1.20" in html
