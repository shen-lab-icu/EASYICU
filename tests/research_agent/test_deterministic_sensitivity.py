from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.deterministic_sensitivity import (
    cohort_definition_overlap_code,
    cohort_definition_sensitivity_comparison_code,
)


def test_cohort_definition_sensitivity_template_executes_from_parent_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    parent = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True)

    attrition = pd.DataFrame(
        {
            "definition_id": [
                "primary_adult_los1_all_vitals_sep3_measured",
                "alt_adult_no_los_all_vitals_sep3_measured",
                "alt_adult_los1_three_of_four_vitals_sep3_measured",
                "alt_adult_los1_no_temp_requirement_sep3_measured",
                "alt_adult_los2_all_vitals_sep3_measured",
            ],
            "definition_label": [
                "Primary cohort",
                "Relax ICU length-of-stay threshold",
                "Relax vital completeness to >=3 of 4",
                "Relax temperature requirement",
                "Tighten ICU length-of-stay threshold",
            ],
            "definition_type": [
                "primary",
                "alternative",
                "alternative",
                "alternative",
                "alternative",
            ],
            "criteria": [
                "age>=18 AND los_icu>=1 day AND map/hr/resp/temp measured AND sep3_sofa2_measured",
                "age>=18 AND map/hr/resp/temp measured AND sep3_sofa2_measured",
                "age>=18 AND los_icu>=1 day AND at least 3 of map/hr/resp/temp measured AND sep3_sofa2_measured",
                "age>=18 AND los_icu>=1 day AND map/hr/resp measured AND sep3_sofa2_measured",
                "age>=18 AND los_icu>=2 days AND map/hr/resp/temp measured AND sep3_sofa2_measured",
            ],
            "n_included": [128, 128, 144, 144, 96],
        }
    )
    attrition.to_csv(parent / "alternative_cohort_attrition.csv", index=False)
    pd.DataFrame(
        {
            "definition_a": ["primary_adult_los1_all_vitals_sep3_measured"],
            "definition_b": ["primary_adult_los1_all_vitals_sep3_measured"],
            "intersection_n": [128],
            "union_n": [128],
            "jaccard": [1.0],
        }
    ).to_csv(parent / "cohort_overlap_matrix.csv", index=False)

    rng = np.random.default_rng(11)
    n = 180
    sepsis = rng.binomial(1, 0.42, n)
    death_prob = 0.07 + 0.06 * sepsis + 0.015 * (np.arange(n) % 4 == 0)
    death = rng.binomial(1, death_prob)
    temp_measured = np.ones(n, dtype=int)
    temp_measured[::8] = 0
    lact = rng.lognormal(mean=0.2, sigma=0.45, size=n)
    lact[::7] = np.nan
    cohort = pd.DataFrame(
        {
            "stay_id": np.arange(n) + 1000,
            "age": rng.normal(66, 9, n),
            "sex": np.where(np.arange(n) % 2 == 0, "Male", "Female"),
            "los_icu": np.where(np.arange(n) < 120, 2.4, 1.2),
            "sep3_sofa2_max": sepsis,
            "sep3_sofa2_measured": 1,
            "hr_max": rng.normal(96, 15, n),
            "hr_measured": 1,
            "map_min": rng.normal(68, 11, n),
            "map_measured": 1,
            "resp_max": rng.normal(23, 5, n),
            "resp_measured": 1,
            "temp_max": rng.normal(37.3, 0.7, n),
            "temp_measured": temp_measured,
            "lact_max": lact,
            "lact_measured": (~pd.isna(lact)).astype(int),
            "bun_max": rng.normal(24, 8, n),
            "bun_measured": 1,
            "wbc_max": rng.normal(12, 4, n),
            "wbc_measured": 1,
            "death": death,
        }
    )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path)

    out_dir = (
        run_dir / "steps" / "05_sensitivity_comparison_across_definitions" / "outputs"
    )
    out_dir.mkdir(parents=True)
    script_path = tmp_path / "analysis.py"
    script_path.write_text(
        cohort_definition_sensitivity_comparison_code(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "COHORT_PARQUET": str(cohort_path),
            "STEP_OUT_DIR": str(out_dir),
        }
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    comparison = pd.read_csv(out_dir / "sensitivity_comparison.csv")
    covariates = pd.read_csv(out_dir / "sensitivity_model_covariates.csv")
    summary = (out_dir / "step_summary.json").read_text(encoding="utf-8")

    assert {"OR", "RD"} <= set(comparison["effect_scale"].astype(str))
    assert "full_export_step03_scope" in set(comparison["definition_id"])
    assert "primary_lactate_complete_case" in set(comparison["definition_id"])
    assert "primary_without_lactate_adjustment" in set(comparison["definition_id"])
    assert not covariates["covariates_used"].str.contains("map", case=False).any()
    assert "not identical to the stricter primary eligibility definition" in summary
    assert (out_dir / "noninformative_sensitivity_audit.csv").exists()


def test_cohort_definition_overlap_retains_sepsis3_negatives_when_measured_flag_is_positive_only(
    tmp_path: Path,
) -> None:
    n = 120
    sepsis = np.r_[np.ones(45, dtype=int), np.zeros(75, dtype=int)]
    cohort = pd.DataFrame(
        {
            "stay_id": np.arange(n) + 2000,
            "age": 65,
            "los_icu": 1.5,
            "sep3_sofa2_max": sepsis,
            "sep3_sofa2_measured": sepsis,
            "map_measured": 1,
            "hr_measured": 1,
            "resp_measured": 1,
            "temp_measured": 1,
        }
    )
    cohort_path = tmp_path / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path)
    out_dir = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    out_dir.mkdir(parents=True)
    script_path = tmp_path / "overlap.py"
    script_path.write_text(cohort_definition_overlap_code(), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "COHORT_PARQUET": str(cohort_path),
            "STEP_OUT_DIR": str(out_dir),
        }
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    attrition = pd.read_csv(out_dir / "alternative_cohort_attrition.csv")
    semantics = pd.read_csv(out_dir / "cohort_definition_semantics_audit.csv")
    primary = attrition[attrition["definition_type"] == "primary"].iloc[0]

    assert primary["n_included"] == n
    assert semantics["measured_flag_positive_only"].iloc[0] == np.True_
    assert "sep3_sofa2_measured == 1" in semantics["action"].iloc[0]


def test_cohort_definition_sensitivity_uses_declared_exposure_not_stray_sepsis_column(
    tmp_path: Path,
) -> None:
    """A NON-sepsis question must re-fit on its DECLARED exposure.

    Proves the skill is not bound to the Sepsis-3/mortality benchmark case: when
    research_context.json declares a different primary exposure, the re-fit uses
    that column even though a stray ``sep3_sofa2_max`` column is present in the
    export (which the old skill would have silently used instead).
    """
    run_dir = tmp_path
    (run_dir / "research_context.json").write_text(
        json.dumps({"primary_exposure": "vent_24h_any", "target_outcome": "death"})
    )
    parent = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "definition_id": ["primary_adult", "alt_adult_no_los"],
            "definition_label": ["Primary cohort", "Relax LOS"],
            "definition_type": ["primary", "alternative"],
            "criteria": [
                "age>=18 AND los_icu>=1 day AND map/hr/resp/temp measured",
                "age>=18 AND map/hr/resp/temp measured",
            ],
            "n_included": [150, 150],
        }
    ).to_csv(parent / "alternative_cohort_attrition.csv", index=False)

    rng = np.random.default_rng(21)
    n = 260
    vent = rng.binomial(1, 0.4, n)
    # ventilated patients die more; sepsis flag is UNcorrelated noise here
    death = rng.binomial(1, 0.08 + 0.12 * vent)
    cohort = pd.DataFrame(
        {
            "stay_id": np.arange(n) + 5000,
            "age": rng.normal(64, 10, n),
            "sex": np.where(np.arange(n) % 2 == 0, "Male", "Female"),
            "los_icu": 2.0,
            # DECLARED exposure:
            "vent_24h_any": vent,
            # stray column the old skill would have grabbed as "the exposure":
            "sep3_sofa2_max": rng.binomial(1, 0.5, n),
            "map_measured": 1,
            "hr_measured": 1,
            "resp_measured": 1,
            "temp_measured": 1,
            "death": death,
        }
    )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path)

    out_dir = (
        run_dir / "steps" / "05_sensitivity_comparison_across_definitions" / "outputs"
    )
    out_dir.mkdir(parents=True)
    script_path = tmp_path / "analysis.py"
    script_path.write_text(
        cohort_definition_sensitivity_comparison_code(), encoding="utf-8"
    )
    env = os.environ.copy()
    env.update({"COHORT_PARQUET": str(cohort_path), "STEP_OUT_DIR": str(out_dir)})
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_exposure"] == "vent_24h_any"
    assert summary["target_outcome"] == "death"
    # the exposure-semantics audit must reference the DECLARED exposure, not sep3
    audit = pd.read_csv(out_dir / "noninformative_sensitivity_audit.csv")
    exp_row = audit[audit["sensitivity_axis"] == "exposure_measurement_semantics"].iloc[
        0
    ]
    assert "vent_24h_any" in str(exp_row["evidence"])
    assert "sep3_sofa2" not in str(exp_row["evidence"])


def test_absent_alternative_attrition_degrades_to_clean_skip_not_block(tmp_path: Path):
    """No upstream alternative_cohort_attrition.csv -> clean skip, not a block.

    Regression for the H2 causal run (2026-07-06): outcome/denominator audit
    steps were routed to this runner; with no alternative cohort definition
    registered upstream, the runner used to emit a hard ``status="blocked"``
    with "no upstream file was available" phrasing. The LLM replanner read that
    as "produce the missing file" and spawned repair step after repair step
    (03b..03g), each re-matching this runner and re-blocking -- a runaway loop
    that burned ~50 min without converging. The runner now degrades cleanly to a
    diagnostic_only skip so nothing provokes the repair cascade; the
    ``max_replans`` budget is the deterministic backstop for any residual loop.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A cohort exists, but NO step anywhere produced alternative_cohort_attrition.csv.
    cohort = pd.DataFrame(
        {
            "stay_id": range(40),
            "death": np.tile([0, 1], 20),
            "age": np.linspace(40, 80, 40),
        }
    )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path)

    out_dir = run_dir / "steps" / "03b_outcome_binding_and_followup_audit" / "outputs"
    out_dir.mkdir(parents=True)
    script_path = tmp_path / "analysis.py"
    script_path.write_text(
        cohort_definition_sensitivity_comparison_code(), encoding="utf-8"
    )
    env = os.environ.copy()
    env.update({"COHORT_PARQUET": str(cohort_path), "STEP_OUT_DIR": str(out_dir)})
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    # Clean skip, not a block: nothing for the replanner to "repair".
    assert summary["status"] == "skipped", summary
    assert summary.get("diagnostic_only") is True
    assert summary.get("not_applicable") is True
    # The old hard-block signal must be gone.
    assert "blocking_reason" not in summary
    assert "skip_reason" in summary
