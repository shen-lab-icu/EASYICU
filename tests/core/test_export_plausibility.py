"""The administrator's screening report reads synthetic packages only."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from tests.support.plausibility_export import change_column, screening_export


def report(root: Path, **kwargs) -> dict:
    from tools.report_export_plausibility import SCREENING_RULES
    from easyicu.io.export_plausibility import build_report

    return build_report(root, rules=SCREENING_RULES, **kwargs)


def test_planted_export_defects_are_all_flagged(tmp_path) -> None:
    result = report(screening_export(tmp_path / "planted", planted=True))["current"]
    assert result["concepts"]["vitals.hr"]["out_of_bounds"] == 1
    assert result["concepts"]["vitals.hr"]["verdict"] == "fail"
    assert result["stays"]["demographics"]["duplicate_stays"] == 1
    assert result["stays"]["demographics"]["verdict"] == "fail"
    assert result["stays"]["icu_los"]["above_365_days"] == 1
    assert result["stays"]["icu_los"]["verdict"] == "fail"
    assert result["scores"]["sofa_comparison"]["lower_share"] == 1.0
    assert result["scores"]["sofa_comparison"]["verdict"] == "fail"
    assert result["scores"]["totals"]["sofa2"]["mismatches"] == 4
    assert result["scores"]["totals"]["sofa2"]["verdict"] == "fail"


def test_clean_export_has_no_fails_and_only_aggregates(tmp_path) -> None:
    result = report(screening_export(tmp_path / "clean"))
    current = result["current"]
    assert result["summary"]["fail"] == 0
    assert current["stay_count"] == 4
    assert current["concepts"]["vitals.hr"]["p50"] == 80.0
    assert current["coverage"]["modules"]["vitals"]["share"] == 1.0
    assert current["coverage"]["key_concepts"]["vasopressors"]["share"] == 1.0
    assert current["scores"]["sepsis"]["sep3_sofa1"]["positive_stays"] == 1
    assert current["scores"]["sepsis"]["sep3_sofa2"]["prevalence"] == 0.25
    payload = json.dumps(result, allow_nan=False)
    assert all(str(value) not in payload for value in (910001, 910002, 910003, 910004))
    assert str(tmp_path) not in payload
    assert '"stay_id"' not in payload


def test_missing_values_do_not_count_as_coverage_or_healthy_scores(tmp_path) -> None:
    root = screening_export(tmp_path / "missing")
    change_column(root, "vitals", "hr", [None] * 4)
    change_column(root, "sofa2_score", "sofa2", [None] * 4)
    change_column(root, "sepsis3_sofa2", "sep3_sofa2", [None] * 4)
    result = report(root)["current"]
    assert result["concepts"]["vitals.hr"]["null_share"] == 1.0
    assert result["concepts"]["vitals.hr"]["verdict"] == "warn"
    assert result["coverage"]["key_concepts"]["hr"]["share"] == 0.0
    assert result["coverage"]["key_concepts"]["hr"]["verdict"] == "warn"
    assert result["scores"]["sofa_comparison"]["lower_share"] is None
    assert result["scores"]["sofa_comparison"]["verdict"] == "warn"
    assert result["scores"]["totals"]["sofa2"]["verdict"] == "warn"
    assert result["scores"]["sepsis"]["sep3_sofa2"]["observed_prevalence"] is None


def test_los_is_in_days_and_duplicates_do_not_weight_stay_statistics(tmp_path) -> None:
    root = screening_export(tmp_path / "los")
    change_column(root, "outcome", "los_icu", [60.0, 180.0, 365.0, 500.0])
    path = root / "outcome.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[-1:]]).to_parquet(path, index=False)
    current = report(root)["current"]
    assert current["stays"]["outcome"]["duplicate_stays"] == 1
    assert current["stays"]["icu_los"]["above_60_days"] == 3
    assert current["stays"]["icu_los"]["above_180_days"] == 2
    assert current["stays"]["icu_los"]["above_365_days"] == 1
    assert current["stays"]["icu_los"]["p50"] == 272.5


def test_negative_times_and_discrete_score_ranges(tmp_path) -> None:
    root = screening_export(tmp_path / "ranges")
    change_column(root, "vitals", "charttime", [-1.0, 0.0, 1.0, 2.0])
    change_column(root, "outcome", "los_icu", [-1.0, 2.0, 2.0, 2.0])
    for name, maximum in (("qsofa", 3.0), ("sirs", 4.0), ("news", 20.0)):
        change_column(root, "other_scores", name, [0.0, maximum, -1.0, maximum + 1.0])
    result = report(root)["current"]
    assert result["negative_times"]["vitals"]["negative_count"] == 1
    assert result["negative_times"]["vitals"]["verdict"] == "warn"
    assert result["stays"]["icu_los"]["negative_count"] == 1
    assert result["stays"]["icu_los"]["verdict"] == "fail"
    for name in ("qsofa", "sirs", "news"):
        assert result["scores"]["ranges"][name]["out_of_bounds"] == 2
        assert result["scores"]["ranges"][name]["verdict"] == "fail"


def test_sofa_maxima_are_paired_per_stay_and_components_per_time(tmp_path) -> None:
    root = screening_export(tmp_path / "scores")
    for module, total in (("sofa1_score", "sofa"), ("sofa2_score", "sofa2")):
        path = root / f"{module}.parquet"
        frame = pd.read_parquet(path)
        extra = frame.copy()
        extra["charttime"] = 1.0
        if total == "sofa":
            extra[total] = 12.0
            extra["sofa_resp"] = 7.0
        else:
            frame.loc[0, "sofa2_resp"] = None
            frame.loc[0, "sofa2"] = 5.0
        pd.concat([frame, extra]).to_parquet(path, index=False)
    scores = report(root)["current"]["scores"]
    comparison = scores["sofa_comparison"]
    assert comparison["paired_stays"] == 4
    assert comparison["lower_share"] == 1.0
    assert comparison["higher_share"] == 0.0
    assert comparison["equal_share"] == 0.0
    assert comparison["sofa_p50"] == 12.0
    assert comparison["sofa2_p50"] == 6.0
    assert scores["totals"]["sofa2"]["compared_rows"] == 8
    assert scores["totals"]["sofa2"]["mismatches"] == 0


def test_compare_reports_all_metrics_and_module_row_deltas(tmp_path) -> None:
    old = screening_export(tmp_path / "old")
    current = screening_export(tmp_path / "new", planted=True)
    result = report(current, compare=old)
    comparisons = {item["metric"]: item for item in result["comparison"]["metrics"]}
    moved = comparisons["scores.sofa_comparison.lower_share"]
    assert (moved["old"], moved["current"]) == (0.0, 1.0)
    assert moved["verdict"] == "warn"
    assert result["comparison"]["module_rows"]["demographics"]["delta"] == 1
    assert result["comparison"]["module_rows"]["demographics"]["verdict"] == "warn"
    assert comparisons["concepts.vitals.hr.p50"]["verdict"] == "pass"


def test_duckdb_memory_limit_and_threads_are_effective(tmp_path, monkeypatch) -> None:
    observed = []
    connect = duckdb.connect

    def spy(*args, **kwargs):
        connection = connect(*args, **kwargs)
        observed.append(
            connection.execute(
                "SELECT current_setting('memory_limit'), current_setting('threads')"
            ).fetchone()
        )
        return connection

    monkeypatch.setattr(duckdb, "connect", spy)
    root = screening_export(tmp_path / "resources")
    default = report(root)
    custom = report(root, memory_limit="128MB", threads=1)
    assert default["runtime"]["requested_memory_limit"] == "8GB"
    assert observed == [("7.4 GiB", 2), ("122.0 MiB", 1)]
    assert custom["runtime"]["memory_limit"] == "122.0 MiB"


def test_cli_prints_bilingual_summary_and_json_even_when_checks_fail(
    tmp_path, capsys
) -> None:
    from tools.report_export_plausibility import main

    root = screening_export(tmp_path / "cli", planted=True)
    old = screening_export(tmp_path / "cli-old")
    output = tmp_path / "report.json"
    assert (
        main([str(root), "--compare", str(old), "--out", str(output), "--threads", "1"])
        == 0
    )
    printed = capsys.readouterr().out
    assert "筛查" in printed and "Screening" in printed
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["fail"] >= 4
    assert saved["runtime"]["requested_memory_limit"] == "8GB"
    assert saved["runtime"]["threads"] == 1
    assert saved["comparison"]["module_rows"]["demographics"]["delta"] == 1
    assert main([str(root), "--memory-limit", "128MB"]) == 0
    printed = capsys.readouterr().out
    console = json.loads(printed[printed.index("{") :])
    assert console["summary"]["fail"] >= 4
    assert console["runtime"]["memory_limit"] == "122.0 MiB"


@pytest.mark.parametrize("defect", ["outside", "schema", "broken_parquet"])
def test_invalid_inputs_have_aggregate_only_errors(tmp_path, capsys, defect) -> None:
    from tools.report_export_plausibility import main

    root = screening_export(tmp_path / "invalid")
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if defect == "outside":
        original = root / manifest["files"][0]["file"]
        (root.parent / "outside.parquet").write_bytes(original.read_bytes())
        manifest["files"][0]["file"] = "../outside.parquet"
    elif defect == "schema":
        manifest["schema_version"] = "unsupported"
    else:
        (root / "vitals.parquet").write_text("910001 confidential", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert main([str(root)]) == 0
    printed = capsys.readouterr().out
    assert "910001" not in printed and str(tmp_path) not in printed
    result = json.loads(printed[printed.index("{") :])
    assert result["status"] == "unavailable"


def test_nonfinite_values_are_counted_without_invalid_json(tmp_path) -> None:
    root = screening_export(tmp_path / "nonfinite")
    change_column(root, "vitals", "hr", [float("inf"), -float("inf"), None, 80.0])
    result = report(root)
    stats = result["current"]["concepts"]["vitals.hr"]
    assert stats["invalid_count"] == 2
    assert stats["null_share"] == 0.25
    assert stats["p50"] == 80.0
    assert stats["verdict"] == "fail"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("lower_stays", "verdict"), [(1, "pass"), (2, "warn"), (3, "fail")]
)
def test_sofa_lower_share_thresholds_are_screening_heuristics(
    tmp_path, lower_stays, verdict
) -> None:
    root = screening_export(tmp_path / "sofa-boundary")
    change_column(
        root, "sofa2_score", "sofa2", [5.0] * lower_stays + [6.0] * (4 - lower_stays)
    )
    comparison = report(root)["current"]["scores"]["sofa_comparison"]
    assert comparison["lower_share"] == lower_stays / 4
    assert comparison["verdict"] == verdict


@pytest.mark.parametrize(
    ("prevalence", "verdict"), [(0, "warn"), (1, "pass"), (4, "warn")]
)
def test_sepsis_case_mix_is_reviewed_and_not_failed(
    tmp_path, prevalence, verdict
) -> None:
    root = screening_export(tmp_path / "sepsis-boundary")
    change_column(
        root,
        "sepsis3_sofa1",
        "sep3_sofa1",
        [True] * prevalence + [False] * (4 - prevalence),
    )
    result = report(root)["current"]["scores"]["sepsis"]["sep3_sofa1"]
    assert result["prevalence"] == prevalence / 4
    assert result["verdict"] == verdict


def test_null_share_and_los_warning_thresholds(tmp_path) -> None:
    root = screening_export(tmp_path / "warnings")
    path = root / "vitals.parquet"
    frame = pd.read_parquet(path)
    frame = pd.concat([frame.iloc[:1]] * 25, ignore_index=True)
    frame["hr"] = [None] * 24 + [80.0]
    frame.to_parquet(path, index=False)
    change_column(root, "outcome", "los_icu", [61.0, 60.0, 2.0, 2.0])
    current = report(root)["current"]
    assert current["concepts"]["vitals.hr"]["null_share"] == 0.96
    assert current["concepts"]["vitals.hr"]["verdict"] == "warn"
    assert current["stays"]["icu_los"]["above_60_days"] == 1
    assert current["stays"]["icu_los"]["verdict"] == "warn"


def test_module_coverage_excludes_receipts_and_null_values(tmp_path) -> None:
    root = screening_export(tmp_path / "coverage")
    path = root / "sofa2_score.parquet"
    frame = pd.read_parquet(path)
    for column in frame:
        if column not in {"stay_id", "charttime"}:
            frame[column] = None
    frame["sofa2_available"] = True
    frame.to_parquet(path, index=False)
    coverage = report(root)["current"]["coverage"]["modules"]["sofa2_score"]
    assert coverage["stays_with_value"] == 0
    assert coverage["verdict"] == "warn"


def test_comparison_thresholds_and_availability_changes(tmp_path) -> None:
    old = screening_export(tmp_path / "drift-old")
    current = screening_export(tmp_path / "drift-new")
    change_column(current, "vitals", "hr", [100.0] * 4)
    change_column(current, "sepsis3_sofa1", "sep3_sofa1", [False] * 4)
    change_column(current, "blood_gas", "lact", [None] * 4)
    result = report(current, compare=old)
    metrics = {item["metric"]: item for item in result["comparison"]["metrics"]}
    assert metrics["concepts.vitals.hr.p50"]["relative_delta"] == 0.25
    assert metrics["concepts.vitals.hr.p50"]["verdict"] == "warn"
    assert metrics["scores.sepsis.sep3_sofa1.prevalence"]["delta"] == -0.25
    assert metrics["scores.sepsis.sep3_sofa1.prevalence"]["verdict"] == "warn"
    assert metrics["concepts.blood_gas.lact.p50"]["current"] is None
    assert metrics["concepts.blood_gas.lact.p50"]["verdict"] == "warn"
    change_column(current, "vitals", "hr", [96.0] * 4)
    metrics = {
        item["metric"]: item
        for item in report(current, compare=old)["comparison"]["metrics"]
    }
    assert metrics["concepts.vitals.hr.p50"]["verdict"] == "pass"


def test_discrete_score_rejects_fractional_values(tmp_path) -> None:
    root = screening_export(tmp_path / "fractional")
    for name in ("qsofa", "sirs", "news"):
        change_column(root, "other_scores", name, [0.5, 0.0, 1.0, 2.0])
    result = report(root)["current"]["scores"]["ranges"]
    for name in ("qsofa", "sirs", "news"):
        assert result[name]["out_of_bounds"] == 1
        assert result[name]["verdict"] == "fail"


def test_missing_modules_and_empty_exports_are_unavailable_not_healthy(
    tmp_path,
) -> None:
    root = screening_export(tmp_path / "partial")
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        item for item in manifest["files"] if item["module"] == "vitals"
    ]
    manifest_path.write_text(json.dumps(manifest))
    current = report(root)["current"]
    assert current["stays"]["outcome"]["verdict"] == "warn"
    assert current["scores"]["ranges"]["news"]["verdict"] == "warn"
    assert current["coverage"]["key_concepts"]["lactate"]["verdict"] == "warn"
    path = root / "vitals.parquet"
    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    result = report(root)
    assert result["current"]["stay_count"] == 0
    assert result["current"]["coverage"]["modules"]["vitals"]["share"] is None
    assert result["summary"]["fail"] == 0


def test_sepsis_is_deduplicated_per_stay_and_invalid_flags_fail(tmp_path) -> None:
    root = screening_export(tmp_path / "sepsis-grain")
    path = root / "sepsis3_sofa1.parquet"
    frame = pd.read_parquet(path)
    frame["sep3_sofa1"] = frame["sep3_sofa1"].astype(float)
    extra = frame.iloc[1:2].copy()
    extra["charttime"] = 1.0
    pd.concat([frame, extra]).to_parquet(path, index=False)
    stats = report(root)["current"]["scores"]["sepsis"]["sep3_sofa1"]
    assert stats["positive_stays"] == 1
    assert stats["prevalence"] == 0.25
    frame.loc[0, "sep3_sofa1"] = 2.0
    frame.to_parquet(path, index=False)
    stats = report(root)["current"]["scores"]["sepsis"]["sep3_sofa1"]
    assert stats["invalid_count"] == 1
    assert stats["verdict"] == "fail"


def test_nonfinite_score_components_cannot_pass_an_identity_check(tmp_path) -> None:
    root = screening_export(tmp_path / "invalid-component")
    change_column(root, "sofa2_score", "sofa2_resp", [float("inf"), 1.0, 1.0, 1.0])
    total = report(root)["current"]["scores"]["totals"]["sofa2"]
    assert total["invalid_count"] == 1
    assert total["verdict"] == "fail"
