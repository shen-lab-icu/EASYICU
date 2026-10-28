"""Only gate-passing equal-estimand aggregate products enter a pool."""

import json
import math
from xml.etree import ElementTree

import pytest

from easyicu.research_agent.replication.metrics import pool_database_estimates
from easyicu.research_agent.replication.report import render_database_meta_analysis


def inputs(unit="odds_ratio"):
    return [dict(database=name, source_sha256=str(i) * 64, estimand_sha256="a" * 64,
                 estimand_label="adjusted_association", unit=unit, gate_passed=True,
                 estimate=math.exp(value) if unit.endswith("ratio") else value,
                 standard_error=.5, ci_low=math.exp(value - .98) if unit.endswith("ratio") else value - .98,
                 ci_high=math.exp(value + .98) if unit.endswith("ratio") else value + .98)
            for i, (name, value) in enumerate(zip(["miiv", "eicu", "sic"], [0, 1, 2]), 1)]


@pytest.mark.parametrize("unit", ["odds_ratio", "hazard_ratio", "risk_ratio", "risk_difference", "mean_difference"])
def test_supported_scales_and_leave_one_out_recompute_tau(unit):
    rows = inputs(unit)
    result = pool_database_estimates(rows, reference_estimand="a" * 64)
    ratio = unit.endswith("ratio")
    assert result["status"] == "pooled"
    assert result["claim_ceiling"] == "pooled_observational_estimate"
    assert result["random"]["estimate"] == pytest.approx(math.e if ratio else 1)
    assert result["random"]["tau_squared"] == pytest.approx(.75, abs=1e-7)
    assert result["analysis_scale"] == ("log" if ratio else "identity")
    assert len(result["included"]) == len(result["leave_one_out"]) == 3
    for omitted in result["leave_one_out"]:
        subset = [row for row in rows if row["database"] != omitted["omitted_database"]]
        repeated = pool_database_estimates(subset, reference_estimand="a" * 64)
        assert omitted["random"] == repeated["random"]


@pytest.mark.parametrize("field,value,reason", [
    ("estimand_sha256", "b" * 64, "changed_estimand"),
    ("gate_passed", False, "result_gate_failed"),
    ("standard_error", None, "standard_error_unavailable"),
    ("standard_error", 0, "standard_error_invalid"),
    ("substitution_codes", ["outcome_changed"], "changed_estimand"),
    ("definition_codes", ["population_changed"], "changed_estimand"),
    ("unit", "hazard_ratio", "effect_measure_changed"),
])
def test_exclusion_is_explicit_and_never_changes_the_pool(field, value, reason):
    rows = inputs()
    rows[1][field] = value
    result = pool_database_estimates(rows, reference_estimand="a" * 64)
    assert [row["database"] for row in result["included"]] == ["miiv", "sic"]
    assert result["excluded"] == [{"database": "eicu", "reason_code": reason}]
    assert result["random"] == pool_database_estimates([rows[0], rows[2]], reference_estimand="a" * 64)["random"]


def test_unknown_gate_cannot_default_to_passed():
    rows = inputs()
    del rows[1]["gate_passed"]
    assert pool_database_estimates(rows, reference_estimand="a" * 64)["excluded"][0]["reason_code"] == "result_gate_failed"


def test_duplicate_sources_or_databases_are_never_independent_replications():
    for field in ("database", "source_sha256"):
        rows = inputs()
        rows[1][field] = rows[0][field]
        assert pool_database_estimates(rows, reference_estimand="a" * 64)["status"] == "unavailable"


def test_report_contains_aggregate_forest_diamond_prediction_and_exclusions():
    rows = inputs()
    rows.append({**rows[0], "database": "aumc", "source_sha256": "4" * 64, "estimand_sha256": "c" * 64})
    result = pool_database_estimates(rows, reference_estimand="a" * 64, hartung_knapp=True)
    report = render_database_meta_analysis(result)
    assert "REML" in report["markdown"] and "Hartung-Knapp" in report["markdown"]
    assert "Q" in report["markdown"] and "I²" in report["markdown"] and "Leave-one-database-out" in report["markdown"]
    assert "changed_estimand" in report["markdown"]
    root = ElementTree.fromstring(report["forest_svg"])
    ids = {element.get("id") for element in root.iter()}
    assert {"pooled-diamond", "prediction-interval", "database-miiv", "database-eicu", "database-sic"} <= ids
    assert "weight" in report["forest_svg"].lower()
    assert "aumc" not in report["forest_svg"]
    json.dumps(result, allow_nan=False)


def test_skill_catalogue_has_pooling_reporting_and_observational_ceiling():
    from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package

    for enabled in (True, False):
        row = next(row for row in method_skill_catalog(enabled=enabled)["components"] if row["id"] == "cross_database.meta_analysis")
        assert row["enabled"] is enabled
        assert row["claim_ceiling"] == "analysis_only"
        assert "pooled observational estimate" in row["description"].lower()
        assert {"Q with df and p", "I²", "tau²", "prediction interval", "exclusions", "leave-one-database-out"} <= set(row["reporting_items"])
        package = method_skill_package(row["id"], enabled=enabled)
        assert any("REML" in item["content"] for item in package["files"])


def test_native_replication_report_emits_pool_and_aggregate_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.replication import metrics, report

    rows = inputs()
    monkeypatch.setattr(metrics, "read_replication_estimate", lambda workdir, database:
                        next(row for row in rows if row["database"] == database))
    results = {row["database"]: SimpleNamespace(workdir="synthetic") for row in rows}
    text = report._render_cross_database_validation_report(question="Synthetic exposure and outcome",
        target_outcome="synthetic_event", rows=[], run_summaries=[], run_results=results, report_dir=tmp_path)
    assert "REML" in text and "Leave-one-database-out" in text
    assert json.loads((tmp_path / "meta_analysis.json").read_text())["status"] == "pooled"
    assert (tmp_path / "meta_analysis_forest.svg").read_text().startswith("<svg")
    assert "REML" in (tmp_path / "meta_analysis_report.md").read_text()


def test_native_replication_never_changes_reference_after_its_gate_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.replication import metrics, report

    rows = inputs()
    def read(workdir, database):
        if database == "miiv":
            raise ValueError("result_gate_failed")
        return next(row for row in rows if row["database"] == database)
    monkeypatch.setattr(metrics, "read_replication_estimate", read)
    report._replication_meta_analysis({row["database"]: SimpleNamespace(workdir="synthetic") for row in rows}, tmp_path)
    result = json.loads((tmp_path / "meta_analysis.json").read_text())
    assert result["status"] == "unavailable"
    assert len(result["excluded"]) == 3


def test_governed_survival_row_keeps_log_hr_se_and_rejects_another_exposure():
    from types import SimpleNamespace
    from easyicu.research_agent.replication.metrics import pooling_primary_row

    product = "table:survival_effect_estimates"
    requirement = SimpleNamespace(analysis_family="survival", exposure_source="synthetic_marker", outcome="synthetic_time",
                                  effect_scale="hazard_ratio")
    plan = SimpleNamespace(steps=[SimpleNamespace(planned_analysis_role="primary", expected_outputs=[product],
                                                  family_primary_result_requirement=requirement)])
    row = dict(exposure_source="synthetic_marker", outcome="synthetic_time", effect_scale="hazard_ratio",
               hazard_ratio=1.5, ci_low=1.1, ci_high=2, standard_error=.15, n_analysis_rows=30)
    normalized = pooling_primary_row(plan, product, [row])
    assert normalized["estimate"] == 1.5 and normalized["standard_error"] == .15
    for changed in ({**row, "exposure_source": "other_synthetic"}, {**row, "outcome": "other_time"},
                    {**row, "effect_scale": "risk_ratio"}):
        with pytest.raises(ValueError):
            pooling_primary_row(plan, product, [changed])
    with pytest.raises(ValueError):
        pooling_primary_row(plan, product, [row, row])


@pytest.mark.parametrize("field,value", [("estimate", float("inf")), ("ci_low", 100), ("ci_high", None)])
def test_invalid_aggregate_interval_is_excluded(field, value):
    rows = inputs()
    rows[1][field] = value
    result = pool_database_estimates(rows, reference_estimand="a" * 64)
    assert result["excluded"] == [{"database": "eicu", "reason_code": "estimate_invalid"}]


def test_random_hartung_knapp_uses_t_interval_and_keeps_prediction_convention():
    result = pool_database_estimates(inputs("mean_difference"), reference_estimand="a" * 64, hartung_knapp=True)
    assert result["random"]["ci_df"] == 2
    assert result["random"]["hartung_knapp_scale"] == pytest.approx(1, abs=1e-7)
    assert result["random"]["ci_high"] == pytest.approx(1 + 4.302652729696142 / math.sqrt(3))


def test_public_replication_owner_wires_pooling_and_hartung_knapp(ra, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.replication import metrics

    rows = inputs()
    owner = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    def run(**kwargs):
        database = kwargs["database"]
        return SimpleNamespace(workdir=str(tmp_path / database), run_id=database,
                               manifest_path="", report_path="", manuscript_path="")
    monkeypatch.setattr(owner, "run", run)
    monkeypatch.setattr(pipeline_module, "_extract_primary_effect_row", lambda database, result: {"database": database})
    monkeypatch.setattr(metrics, "read_replication_estimate", lambda workdir, database:
                        next(row for row in rows if row["database"] == database))
    value = owner.replicate(question="Synthetic marker and event", cohorts={row["database"]: object() for row in rows},
                            meta_analysis_hartung_knapp=True)
    from pathlib import Path

    calculation = json.loads(Path(value["meta_analysis_json"]).read_text())
    assert calculation["random"]["interval_method"] == "Hartung-Knapp"
    assert "meta_analysis_forest.svg" in Path(value["validation_report"]).read_text()


def test_csv_scalar_numbers_normalize_before_log_transform():
    rows = inputs()
    for row in rows:
        for key in ("estimate", "standard_error", "ci_low", "ci_high"):
            row[key] = str(row[key])
    assert pool_database_estimates(rows, reference_estimand="a" * 64)["random"]["estimate"] == pytest.approx(math.e)
