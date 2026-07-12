"""Regression tests for locked-universe cohort attrition replay."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from easyicu.research_agent.cohort_schema import clear_cohort_concept_ids
from easyicu.research_agent.deterministic_cohort_flow import (
    primary_cohort_flow_code,
)


def _locked_definition() -> dict:
    return {
        "name": "primary",
        "inclusion": [
            {
                "concept_id": "age",
                "time_window": {
                    "anchor": "icu_admit",
                    "start_offset_hours": 0,
                    "end_offset_hours": 24,
                },
                "aggregation": "first",
                "op": ">=",
                "value": 18,
            },
            {
                "concept_id": "los_icu",
                "time_window": {
                    "anchor": "icu_admit",
                    "start_offset_hours": 0,
                    "end_offset_hours": 720,
                },
                "aggregation": "first",
                "op": ">=",
                "value": 1.0,
            },
        ],
        "exclusion": [],
        "derived_from_named": None,
        "locked_at": "not_locked",
    }


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": list(range(1, 9)),
            "age": [17, 18, 30, 40, 50, 60, 70, 80],
            "los_icu": [2.0, 0.5, 1.0, 2.0, 0.2, 3.0, 1.5, 0.8],
            "death": [0, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def _exec_runner(run_dir: Path, *, mismatched_analysis: bool = False):
    out_dir = run_dir / "steps" / "01_primary_cohort_flow" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = _universe()
    universe.to_parquet(run_dir / "cohort.parquet", index=False)
    analysis = universe[universe["stay_id"].isin([3, 4, 6, 7])].copy()
    if mismatched_analysis:
        analysis = pd.concat([analysis, universe.iloc[[1]]], ignore_index=True)
    analysis.to_parquet(run_dir / "cohort_analysis.parquet", index=False)
    (run_dir / "cohort_analysis_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.analysis_cohort/1",
                "n_universe": len(universe),
                "n_analysis_cohort": len(analysis),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cohort_locked.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.cohort_definition/1",
                "cohort": _locked_definition(),
            }
        ),
        encoding="utf-8",
    )

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(run_dir / "cohort_analysis.parquet")
    clear_cohort_concept_ids()
    try:
        code = primary_cohort_flow_code()
        try:
            exec(compile(code, "<det_cohort_flow>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        clear_cohort_concept_ids()
        os.environ.clear()
        os.environ.update(saved)
    return (
        json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8")),
        out_dir,
    )


def test_replays_universe_to_locked_analysis_cohort(tmp_path: Path):
    summary, out_dir = _exec_runner(tmp_path)
    assert summary["status"] == "ok"
    assert summary["n_universe"] == 8
    assert summary["n_analysis"] == 4
    assert summary["expected_analysis_n"] == 4

    flow = pd.read_csv(out_dir / "cohort_flow.csv")
    assert flow["n"].tolist() == [8, 7, 4]
    assert flow["n_removed_from_prior_stage"].tolist() == [0, 1, 3]
    assert flow.iloc[0]["stage"] == "universe"

    attrition = pd.read_csv(out_dir / "attrition.csv")
    assert attrition[attrition["status"] == "excluded"]["n"].tolist() == [1, 3]
    assert (out_dir / "cohort_attrition.csv").exists()
    assert (out_dir / "cohort_denominators.csv").exists()


def test_blocks_when_replay_disagrees_with_materialised_cohort(tmp_path: Path):
    summary, _out_dir = _exec_runner(tmp_path, mismatched_analysis=True)
    assert summary["status"] == "blocked"
    assert summary["n_analysis"] == 4
    assert summary["expected_analysis_n"] == 5
    assert "does not match" in summary["blocking_reason"]
