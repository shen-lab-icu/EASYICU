"""Regression probes for audit-tool reporting and CLI failure boundaries."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    """Execute an audit tool in-process and roll back its import-time env edits.

    The extraction recipes set process-wide fast-path switches at module import
    (for example ``EASYICU_DISABLE_AUTO_CHUNK`` in
    ``scripts/r4_crossdb_sofa2_extract.py``).  Loading one here must not change
    the environment that later tests run under.
    """

    name = "audit_" + Path(relative).stem
    spec = spec_from_file_location(name, ROOT / relative)
    module = module_from_spec(spec)
    sys.modules[name] = module
    before = dict(os.environ)
    try:
        spec.loader.exec_module(module)
    finally:
        for key in set(os.environ) - set(before):
            del os.environ[key]
        for key, value in before.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
    return module


def test_skip_smoke_writes_only_readiness_and_dictionary_outputs(tmp_path, monkeypatch):
    tool = load("tools/build_top_level_mechanism_qc.py")
    monkeypatch.setattr(
        tool,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, sample_size=23, skip_smoke=True),
    )
    monkeypatch.setattr(tool, "load_concept_dictionary", lambda: {})
    monkeypatch.setattr(
        tool,
        "check_readiness",
        lambda: pd.DataFrame([{"dataset": "A", "ready": False}]),
    )
    monkeypatch.setattr(
        tool,
        "build_support_matrix",
        lambda _: pd.DataFrame(
            [{"dataset": "A", "concept": "peep", "dictionary_supported": True}]
        ),
    )

    def forbidden(*args):
        pytest.fail("skip-smoke must not extract data or build smoke figures")

    monkeypatch.setattr(tool, "run_smoke_extraction", forbidden)
    monkeypatch.setattr(tool, "build_figures", forbidden)
    assert tool.main() == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "crossdb_top_level_dataset_readiness.csv",
        "crossdb_top_level_support_matrix.csv",
        "crossdb_top_level_qc_status.csv",
    }
    assert pd.read_csv(
        tmp_path / "crossdb_top_level_qc_status.csv"
    ).qc_status.tolist() == ["supported_not_smoke_checked"]


def test_qc_report_reports_actual_failures_unsupported_rows_and_sample_limit(tmp_path):
    tool = load("tools/build_top_level_mechanism_qc.py")
    qc = pd.DataFrame(
        [
            {
                "dataset": "A",
                "concept": "peep",
                "qc_status": "error",
                "error": "failed",
            },
            {
                "dataset": "B",
                "concept": "fio2",
                "qc_status": "unsupported_by_dictionary",
                "tables": "none",
            },
            {
                "dataset": "C",
                "concept": "rrt",
                "qc_status": "supported_but_absent_in_smoke",
                "non_null": 0,
                "rows": 0,
                "patients": 0,
            },
        ]
    )
    tool.write_report(
        out_dir=tmp_path,
        readiness=pd.DataFrame(
            [{"dataset": "A", "ready": False}, {"dataset": "B", "ready": True}]
        ),
        qc=qc,
        warnings_df=pd.DataFrame(),
        figures={
            "support_heatmap_png": "support.png",
            "nonnull_heatmap_png": "nonnull.png",
        },
        sample_size=23,
    )
    report = (tmp_path / "crossdb_top_level_qc_report.md").read_text()
    assert "1/2 个数据库" in report
    assert "错误或输出列缺失 1 项" in report
    assert "字典不支持 1 项" in report
    assert "无非空值 1 项" in report
    assert "max_patients=23" in report
    assert "max_patients=10" not in report and "均稳定保留" not in report


@pytest.mark.parametrize(
    "successes,expected", [([True], 0), ([True, False], 1), ([], 1)]
)
def test_fullflow_exit_status_matches_saved_failure_denominator(
    tmp_path, monkeypatch, successes, expected
):
    tool = load("tools/run_openrouter_fullflow_validation.py")
    tasks = [
        tool.ValidationTask(
            str(i),
            "Synthetic",
            "prediction",
            "easy",
            "Synthetic question",
            lambda _: None,
        )
        for i in range(len(successes))
    ]
    monkeypatch.setattr(tool, "_default_tasks", lambda: tasks)

    def result(**kwargs):
        task = kwargs["task"]
        success = successes[int(task.key)]
        return {
            "task_key": task.key,
            "family": task.family,
            "difficulty": task.difficulty,
            "success": success,
            "summary": {"strict_success": success},
            "run_dir": "synthetic",
        }

    monkeypatch.setattr(tool, "_run_task", result)
    monkeypatch.setattr(
        sys, "argv", ["runner", "--out-root", str(tmp_path), "--max-retries", "1"]
    )
    assert tool.main() == expected
    payload = json.loads((tmp_path / "validation_results.json").read_text())
    assert payload["n_tasks"] == len(successes)
    assert payload["n_failed"] == successes.count(False)


@pytest.mark.parametrize(
    "aggregate_each,return_codes,expected_failures",
    [
        (False, [0, 1], ["final:single-model"]),
        (True, [0, 1, 0], ["model:mock"]),
    ],
)
def test_analysis_bench_aggregation_failures_set_exit_status_and_receipt(
    tmp_path, monkeypatch, aggregate_each, return_codes, expected_failures
):
    tool = load("tools/run_analysis_bench_overnight.py")
    commands = iter(return_codes)
    monkeypatch.setattr(tool, "_run_command", lambda **_kwargs: next(commands))
    argv = [
        "runner",
        "--provider",
        "mock",
        "--items",
        "analysis_sofa_multisignal_mortality",
        "--max-retries",
        "1",
        "--out-root",
        str(tmp_path),
    ]
    if aggregate_each:
        argv.append("--aggregate-after-each-model")
    monkeypatch.setattr(sys, "argv", argv)

    assert tool.main() == 1
    progress = json.loads((tmp_path / "overnight_progress.json").read_text())
    assert progress["failed_items"] == []
    assert progress["aggregation_failures"] == expected_failures


def test_crossdb_resume_normalizes_legacy_full_count_shapes():
    tool = load("scripts/r4_crossdb_sofa2_extract.py")
    normalized = tool._normalize_existing_databases(
        {
            "MIMIC-III": {"icu_stays_full": 7},
            "MIMIC-IV": {"icu_stays_full": [None, "missing table"]},
            "eICU": {"icu_stays_full": {"n": 11}},
        }
    )

    assert normalized["MIMIC-III"]["icu_stays_full"] == {"n": 7}
    assert normalized["MIMIC-IV"]["icu_stays_full"] == {
        "n": None,
        "error": "missing table",
    }
    assert normalized["eICU"]["icu_stays_full"] == {"n": 11}


@pytest.fixture
def figure_data():
    tool = load("scripts/r5_fig5_nature.py")
    records = []
    for i, db in enumerate(tool.DBS):
        for cat, rate in zip(tool.CATS, [15, 10, 9, 8, 7]):
            value = rate * (1 + i / 10)
            records.append(
                {
                    "database": db,
                    "bmi_category": cat,
                    "mortality_pct": value,
                    "ci_lo": value - 1,
                    "ci_hi": value + 1,
                }
            )
    return tool, pd.DataFrame(records)


def test_figure_annotations_follow_changed_inputs(figure_data):
    tool, df = figure_data
    spread, below = tool.validate_data(df)
    assert spread == pytest.approx(1.5) and below == 6
    mask = (df.database == tool.DBS[0]) & (df.bmi_category == "overweight")
    df.loc[mask, ["mortality_pct", "ci_lo", "ci_hi"]] = [12, 11, 13]
    assert tool.validate_data(df)[1] == 5


@pytest.mark.parametrize(
    "defect",
    [
        "missing_database",
        "missing_group",
        "duplicate",
        "nan",
        "infinity",
        "zero_reference",
        "invalid_ci",
    ],
)
def test_figure_rejects_incomplete_or_invalid_evidence(figure_data, defect):
    tool, df = figure_data
    if defect == "missing_database":
        df = df[df.database != tool.DBS[0]]
    elif defect == "missing_group":
        df = df.iloc[1:]
    elif defect == "duplicate":
        df = pd.concat([df, df.iloc[[0]]])
    elif defect == "nan":
        df.loc[0, "mortality_pct"] = np.nan
    elif defect == "infinity":
        df.loc[0, "ci_hi"] = np.inf
    elif defect == "zero_reference":
        df.loc[1, ["mortality_pct", "ci_lo", "ci_hi"]] = [0, 0, 1]
    elif defect == "invalid_ci":
        df.loc[0, "ci_lo"] = 99
    with pytest.raises(ValueError):
        tool.validate_data(df)
