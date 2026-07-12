"""Regression: the deterministic missingness/measurement audit runner produces a
per-concept audit table without an LLM coder call.

Built 2026-07-08 for E3 + M1. The missingness/measurement audit is a pure count
(measured vs missing fraction per concept + a structural-vs-measurement split),
yet the LLM coder reliably exhausted its retry budget on it (~27.6 min, IDENTICAL
across two real E3 runs) and failed with no code, blocking the whole run on
``execution_complete``. The deterministic runner removes both the flakiness and
the dominant coder round-trip.

The tests exec the runner's code string against synthetic cohorts (no real data),
asserting: it uses the ``<concept>_measured`` indicator as the authoritative
measurement signal, narrowly distinguishes a complete binary event-status flag
from measurement availability, never imputes, distinguishes structural-no-source
from measurement missingness, and blocks gracefully on an empty cohort.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.deterministic_missingness import (
    missingness_measurement_audit_code,
)


def _exec_runner(
    run_dir: Path,
    cohort: pd.DataFrame,
    context: dict,
    *,
    requested_inputs: list[str] | None = None,
):
    out_dir = run_dir / "steps" / "02_missingness_measurement_audit" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_context.json").write_text(json.dumps(context), encoding="utf-8")
    if requested_inputs is not None:
        (run_dir / "analysis_plan.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "02_missingness_measurement_audit",
                            "inputs": requested_inputs,
                            "expected_outputs": [
                                "table:missingness_measurement_audit",
                                "table:analytic_denominators",
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = missingness_measurement_audit_code()
        try:
            exec(compile(code, "<det_missingness>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return (
        json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8")),
        out_dir,
    )


def _cohort(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # lactate: measured for 60% of stays; crea: measured for 90%; rrt: measured
    # for NO stay (structural no-source in this cohort).
    lact_measured = (rng.random(n) < 0.60).astype(int)
    crea_measured = (rng.random(n) < 0.90).astype(int)
    rrt_measured = np.zeros(n, dtype=int)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "age": rng.integers(40, 90, n),
            "sex": rng.choice(["Male", "Female"], n),
            "death": (rng.random(n) < 0.12).astype(int),
            "los_icu": rng.gamma(2.0, 2.0, n),
            "lactate": np.where(lact_measured == 1, rng.gamma(2.0, 1.5, n), np.nan),
            "lactate_measured": lact_measured,
            "crea": np.where(crea_measured == 1, rng.gamma(2.0, 0.6, n), np.nan),
            "crea_measured": crea_measured,
            "rrt": np.full(n, np.nan),
            "rrt_measured": rrt_measured,
        }
    )


def test_missingness_audit_counts_from_measured_indicator(tmp_path: Path):
    cohort = _cohort(n=1000, seed=1)
    summary, out_dir = _exec_runner(tmp_path, cohort, {})
    assert summary["status"] == "ok"
    assert summary["analysis_family"] == "data_quality"
    assert summary["adjusted_effect"] is None  # a descriptive audit, never an effect

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    concepts = set(audit["concept"])
    assert {"lactate", "crea", "rrt"} <= concepts
    # id / demographic / outcome columns are NOT audited as concepts.
    assert concepts.isdisjoint({"stay_id", "age", "sex", "death", "los_icu"})

    lact = audit[audit["concept"] == "lactate"].iloc[0]
    exp_measured = int((cohort["lactate_measured"] == 1).sum())
    assert lact["measured_one_n"] == exp_measured
    assert lact["value_missing_n"] == 1000 - exp_measured
    # counts and percentages are consistent (never imputed to 0 -> full denom).
    assert lact["measured_one_n"] + lact["value_missing_n"] == 1000
    assert abs(lact["value_missing_pct"] - 100.0 * (1000 - exp_measured) / 1000) < 1e-6
    # schema aliases the figure renderer resolves are all present.
    for col in ("n_total", "measured_n", "n_nonmissing", "missing_n", "missing_pct", "measured_pct"):
        assert col in audit.columns


def test_structural_no_source_distinguished_from_measurement_missing(tmp_path: Path):
    cohort = _cohort(n=800, seed=2)
    summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    # rrt is measured for NO stay -> structural no-source, not measurement-missing.
    rrt = audit[audit["concept"] == "rrt"].iloc[0]
    assert rrt["missingness_kind"] == "structural_no_source"
    assert rrt["measured_one_n"] == 0
    # lactate/crea are sourced (some measured) -> measurement missingness.
    lact = audit[audit["concept"] == "lactate"].iloc[0]
    assert lact["missingness_kind"] == "measurement_missing"
    assert summary["n_structural_no_source"] >= 1


def test_never_imputes_partial_concept(tmp_path: Path):
    # A concept measured for exactly 3/5 stays must report measured=3, missing=2
    # -- the two unmeasured stays are NOT counted as measured-zero.
    cohort = pd.DataFrame(
        {
            "stay_id": [0, 1, 2, 3, 4],
            "age": [60, 61, 62, 63, 64],
            "sex": ["Male"] * 5,
            "death": [0, 1, 0, 0, 1],
            "bili": [1.2, np.nan, 3.4, np.nan, 0.9],
            "bili_measured": [1, 0, 1, 0, 1],
        }
    )
    _summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    bili = audit[audit["concept"] == "bili"].iloc[0]
    assert bili["measured_one_n"] == 3
    assert bili["value_missing_n"] == 2
    assert bili["value_missing_pct"] == 40.0


def test_complete_binary_event_flag_is_not_misread_as_measurement_missingness(
    tmp_path: Path,
):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "sex": ["F", "M", "F", "M", "F"],
            "death": [0, 1, 0, 0, 1],
            "rrt_first": [0, 0, 1, 0, 1],
            "rrt_measured": [0, 0, 1, 0, 1],
            "rrt_n": [0, 0, 1, 0, 2],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["rrt_first", "rrt_measured"],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    rrt = audit[audit["concept"] == "rrt"].iloc[0]
    assert summary["n_binary_event_status"] == 1
    assert rrt["indicator_semantics"] == "binary_event_presence"
    assert rrt["missingness_kind"] == "binary_event_status_complete"
    assert rrt["raw_indicator_one_n"] == 2
    assert rrt["event_count_column"] == "rrt_n"
    assert rrt["measured_one_n"] == 5
    assert rrt["value_missing_n"] == 0
    assert rrt["value_present_but_measured_zero_n"] == 0


def test_binary_value_flag_without_event_count_remains_a_conflict(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [60, 61, 62, 63],
            "sex": ["F", "M", "F", "M"],
            "death": [0, 1, 0, 0],
            "screen_first": [0, 1, 0, 1],
            "screen_measured": [0, 1, 0, 1],
        }
    )
    _summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    screen = audit[audit["concept"] == "screen"].iloc[0]
    assert screen["indicator_semantics"] == "measurement_availability"
    assert screen["missingness_kind"] == "measurement_flag_conflict"
    assert screen["measured_one_n"] == 2
    assert screen["value_present_but_measured_zero_n"] == 2


def test_family_aggregate_and_declared_analytic_denominator(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "sex": ["F", "M", "F", "M", "F"],
            "death": [0, 1, 0, 0, 1],
            "aki_stage_first": [0, 0, 1, 1, 2],
            "aki_stage_max": [0, 1, 2, np.nan, 3],
            "aki_stage_measured": [1, 1, 1, 0, 1],
            "crea_first": [0.8, 1.1, np.nan, 2.0, 0.9],
            "crea_measured": [1, 1, 0, 1, 1],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "aki_stage_max",
            "aki_stage_measured",
            "crea_first",
            "crea_measured",
            "age",
            "sex",
            "death",
        ],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    stage = audit[audit["concept"] == "aki_stage"].iloc[0]
    crea = audit[audit["concept"] == "crea"].iloc[0]
    assert stage["value_column"] == "aki_stage_max"
    assert crea["value_column"] == "crea_first"
    assert stage["raw_value_missing_n"] == 1
    assert stage["measured_but_value_missing_n"] == 0
    assert {"age", "sex", "death"} <= set(audit["concept"])

    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    complete = denominators[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    assert complete["n_total"] == 5
    assert complete["n_complete"] == 3
    assert complete["n_excluded_missing"] == 2
    assert summary["all_requested_inputs_complete_n"] == 3
    assert summary["missing_declared_inputs"] == []


def test_bare_concept_declared_input_resolves_to_value_column_not_blocked(tmp_path: Path):
    # Regression: a plan may declare a time-series concept by its BARE name
    # (``crea``) while the cohort materialises it only as aggregates
    # (``crea_first``/``crea_measured``). The audit resolves it via
    # _representative_value_column, so the analytic-denominator loop must resolve
    # it the SAME way instead of flagging it as a missing declared input and
    # spuriously blocking an otherwise-complete audit.
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "crea_first": [0.8, 1.1, np.nan, 2.0, 0.9],
            "crea_measured": [1, 1, 0, 1, 1],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path, cohort, {}, requested_inputs=["crea", "age"]
    )
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    assert "crea" in set(audit["concept"])  # concept was genuinely audited
    assert summary["status"] == "ok"  # ... so it must not be blocked
    assert summary["missing_declared_inputs"] == []
    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    complete = denominators[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    # crea_first missing on row 3 only -> 4 complete on {crea, age}
    assert complete["n_complete"] == 4


def test_genuinely_absent_declared_input_still_blocks(tmp_path: Path):
    # The bare-name resolution must NOT mask a truly-missing declared input.
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "crea_first": [0.8, np.nan, 2.0],
            "crea_measured": [1, 0, 1],
        }
    )
    summary, _ = _exec_runner(
        tmp_path, cohort, {}, requested_inputs=["crea", "nonexistent_var"]
    )
    assert summary["status"] == "blocked"
    assert summary["missing_declared_inputs"] == ["nonexistent_var"]


def test_declared_inputs_scope_audit_instead_of_scanning_unrelated_wide_columns(
    tmp_path: Path,
):
    cohort = _cohort(n=20, seed=7)
    cohort["sofa_liver"] = np.arange(20, dtype=float)
    cohort["sofa_liver_measured"] = 1
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["lactate", "lactate_measured", "age", "death"],
    )
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    assert summary["n_concepts_audited"] == 3
    assert set(audit["concept"]) == {"lactate", "age", "death"}
    assert "sofa_liver" not in set(audit["concept"])


def test_missing_declared_input_blocks_joint_denominator(tmp_path: Path):
    cohort = _cohort(n=20, seed=3)
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["lactate", "column_not_in_cohort"],
    )
    assert summary["status"] == "blocked"
    assert summary["all_requested_inputs_complete_n"] is None
    assert summary["missing_declared_inputs"] == ["column_not_in_cohort"]

    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    joint = denominators[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    assert pd.isna(joint["n_complete"])


def test_blocks_on_empty_cohort(tmp_path: Path):
    cohort = _cohort(n=0)
    summary, _out = _exec_runner(tmp_path, cohort, {})
    assert summary["status"] == "blocked"
    assert summary["adjusted_effect"] is None
    assert "empty" in summary["blocking_reason"].lower()
