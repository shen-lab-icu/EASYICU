"""Deterministic Cox survival runner.

The primary time-to-event result must run WITHOUT an LLM coder so it is
reproducible: same cohort -> same hazard ratio, correct exposure name, and
DATA tables only (no inline figures that would trip the figure contract).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.deterministic_survival import survival_primary_analysis_code


def _synth_cohort(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vent = (rng.random(n) < 0.33).astype(int)
    age = rng.normal(63, 15, n).clip(18, 95)
    # ventilated patients die sooner (higher hazard) -> HR > 1
    base = rng.exponential(300, n)
    surv = base / (1.0 + 1.1 * vent + 0.01 * (age - 60))
    died = (surv < 250).astype(int)
    followup = np.where(died == 1, np.clip(surv, 1, 250), 250.0) + 24.0
    death_time = np.where(died == 1, np.clip(surv, 1, 250) + 24.0, np.nan)
    return pd.DataFrame(
        {
            "vent_24h_any": vent,
            "age": age,
            "sex": rng.choice(["M", "F"], n),
            "charlson_first": rng.integers(0, 8, n).astype(float),
            "death": died,
            "event_observed": died,
            "death_time": death_time,
            "followup_time_hours": followup,
            "mech_vent_max": vent.astype(float),
        }
    )


def test_deterministic_survival_runner_produces_hr_and_data_only(tmp_path: Path):
    run_dir = tmp_path
    (run_dir / "research_context.json").write_text(
        json.dumps({"primary_exposure": "vent_24h_any", "target_outcome": "death"})
    )
    out_dir = run_dir / "steps" / "01_survival_analysis" / "outputs"
    out_dir.mkdir(parents=True)
    cohort = run_dir / "cohort.parquet"
    _synth_cohort().to_parquet(cohort)

    script = run_dir / "gen.py"
    script.write_text(survival_primary_analysis_code())
    env = {
        **os.environ,
        "STEP_OUT_DIR": str(out_dir),
        "COHORT_PARQUET": str(cohort),
    }
    proc = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "ok"
    # exposure name matches the contract's required predictor
    assert summary["primary_predictor"] == "vent_24h_any"
    # a real, positive hazard ratio in the expected direction
    assert summary["hazard_ratio"] > 1.0
    assert summary["hazard_ratio_ci_low"] > 0
    assert summary["n_events"] > 0
    # DATA tables written, NO figures emitted by the analysis step
    for name in ("cox_summary", "hazard_ratio", "km_curve", "risk_table"):
        assert (out_dir / f"{name}.csv").exists()
    figs = [
        p
        for p in out_dir.iterdir()
        if p.suffix.lower() in {".png", ".svg", ".pdf", ".tiff"}
    ]
    assert not figs, f"analysis step must not emit figures, got {figs}"

    # determinism: a second run yields the identical hazard ratio
    out_dir2 = run_dir / "steps" / "01_survival_analysis_rerun" / "outputs"
    out_dir2.mkdir(parents=True)
    env2 = {**env, "STEP_OUT_DIR": str(out_dir2)}
    subprocess.run([sys.executable, str(script)], env=env2, check=True)
    summary2 = json.loads((out_dir2 / "step_summary.json").read_text())
    assert summary2["hazard_ratio"] == summary["hazard_ratio"]


def test_deterministic_survival_km_curve_has_two_groups(tmp_path: Path):
    run_dir = tmp_path
    (run_dir / "research_context.json").write_text(
        json.dumps({"primary_exposure": "vent_24h_any", "target_outcome": "death"})
    )
    out_dir = run_dir / "steps" / "01_survival_analysis" / "outputs"
    out_dir.mkdir(parents=True)
    cohort = run_dir / "cohort.parquet"
    _synth_cohort().to_parquet(cohort)
    script = run_dir / "gen.py"
    script.write_text(survival_primary_analysis_code())
    subprocess.run(
        [sys.executable, str(script)],
        env={
            **os.environ,
            "STEP_OUT_DIR": str(out_dir),
            "COHORT_PARQUET": str(cohort),
        },
        check=True,
    )
    km = pd.read_csv(out_dir / "km_curve.csv")
    assert set(km["group"].unique()) == {"Exposed", "Unexposed"}
    assert {"time", "survival", "at_risk"}.issubset(km.columns)


def _run(run_dir: Path, cohort: pd.DataFrame, ctx: dict) -> dict:
    (run_dir / "research_context.json").write_text(json.dumps(ctx))
    out_dir = run_dir / "steps" / "01_survival_analysis" / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path)
    script = run_dir / "gen.py"
    script.write_text(survival_primary_analysis_code())
    proc = subprocess.run(
        [sys.executable, str(script)],
        env={
            **os.environ,
            "STEP_OUT_DIR": str(out_dir),
            "COHORT_PARQUET": str(cohort_path),
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((out_dir / "step_summary.json").read_text())


def test_survival_runner_is_exposure_agnostic_and_config_driven(tmp_path: Path):
    """A NON-ventilation, NON-default-covariate question must be honored.

    Proves the skill carries no H1 (mechanical-ventilation) or fixed-covariate
    assumptions: the exposure, adjustment set, and landmark all come from
    research_context.json.
    """
    rng = np.random.default_rng(3)
    n = 4000
    high_lac = (rng.random(n) < 0.4).astype(int)
    base = rng.exponential(300, n)
    surv = base / (1.0 + 0.9 * high_lac)
    died = (surv < 250).astype(int)
    cohort = pd.DataFrame(
        {
            # a domain the H1 case knows nothing about:
            "high_lactate": high_lac,
            "sofa_total": rng.integers(0, 20, n).astype(float),
            "lactate_max": rng.normal(3.0, 1.5, n).clip(0.1, 20),
            # demographics present but NOT requested -> must be ignored:
            "age": rng.normal(63, 15, n).clip(18, 95),
            "sex": rng.choice(["M", "F"], n),
            "charlson_first": rng.integers(0, 8, n).astype(float),
            "event_observed": died,
            "death_time": np.where(died == 1, np.clip(surv, 1, 250) + 24.0, np.nan),
            "followup_time_hours": np.where(died == 1, np.clip(surv, 1, 250), 250.0) + 48.0,
        }
    )
    summary = _run(
        tmp_path,
        cohort,
        {
            "primary_exposure": "high_lactate",
            "target_outcome": "death",
            "user_preferences": {
                "covariates": ["sofa_total", "lactate_max"],
                "landmark_hours": 48.0,
            },
        },
    )
    assert summary["status"] == "ok"
    assert summary["primary_predictor"] == "high_lactate"
    # config landmark honored (not the 24h default)
    assert summary["landmark_hours"] == 48.0
    # config covariates honored; default demographics NOT silently used
    assert summary["adjustment_source"] == "config"
    assert set(summary["adjustment_covariates"]) == {"sofa_total", "lactate_max"}
    assert "age" not in summary["cox_terms"]
    assert "sex_M" not in summary["cox_terms"]
    assert summary["cox_terms"][0] == "high_lactate"


def test_survival_runner_blocks_instead_of_inventing_a_ventilation_exposure(tmp_path: Path):
    """If the DECLARED exposure column is absent, block — never guess vent.

    The old skill fell back to deriving a mechanical-ventilation surrogate,
    which would answer a different question than the one configured. The
    generalized skill must fail closed and keep the declared exposure name.
    """
    rng = np.random.default_rng(5)
    n = 500
    cohort = pd.DataFrame(
        {
            # NOTE: a stray ventilation column exists in the export, but it is
            # NOT the declared exposure and must NOT be used.
            "mech_vent_max": (rng.random(n) < 0.3).astype(float),
            "age": rng.normal(60, 12, n).clip(18, 95),
            "event_observed": (rng.random(n) < 0.3).astype(int),
            "followup_time_hours": rng.uniform(30, 400, n),
        }
    )
    summary = _run(
        tmp_path,
        cohort,
        {"primary_exposure": "renal_replacement_therapy", "target_outcome": "death"},
    )
    assert summary["status"] == "blocked"
    assert summary["primary_predictor"] == "renal_replacement_therapy"
    out_dir = tmp_path / "steps" / "01_survival_analysis" / "outputs"
    # no result artifacts produced from a stray ventilation column
    assert not (out_dir / "km_curve.csv").exists()
    assert not (out_dir / "hazard_ratio.csv").exists()
