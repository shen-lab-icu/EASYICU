"""Synthetic prepared follow-up, with tied events and follow-up beyond the horizon."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..spec import SurvivalReferenceSpec


def example_spec() -> SurvivalReferenceSpec:
    return SurvivalReferenceSpec(
        title="Synthetic 28-day survival association", identity_column="stay_id",
        primary={
            "analysis_family": "survival", "exposure_source": "treatment", "outcome": "death",
            "expected_result_product": "table:survival_effect_estimates", "input_product": "table:analysis_cohort",
            "estimator": "cox_proportional_hazards", "effect_scale": "hazard_ratio",
            "uncertainty_method": "wald_95ci", "population": "synthetic prepared cohort",
            "time_origin": "synthetic admission", "time_column": "follow_up_days", "time_unit": "days",
            "event_column": "death", "event_value": 5, "event_definition": "synthetic event code 5",
            "censoring_strategy": "administrative at 28 days", "competing_risk_strategy": "none",
            "time_horizon": "28 days", "time_horizon_value": 28, "effect_measure": "hazard_ratio",
            "proportional_hazards_diagnostic": "schoenfeld_global_test",
            "proportional_hazards_alpha": 0.05, "proportional_hazards_policy": "block_paper_authorization",
            "covariates": ["age", "sex"], "exposure_encoding": "declared_model_terms",
            "missing_data_policy": "complete_case",
            "model_terms": [
                {"name": "treatment", "role": "exposure", "coding": "binary", "levels": ["0", "1"],
                 "reference_level": "1", "transform": "treatment_contrast"},
                {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
                {"name": "sex", "role": "covariate", "coding": "binary", "levels": ["female", "male"],
                 "reference_level": "male", "transform": "treatment_contrast"},
            ],
        },
    )


def example_cohort() -> pd.DataFrame:
    rng = np.random.default_rng(20261030)
    n = 240
    treatment = rng.integers(0, 2, n).astype(float)
    age = rng.normal(60, 11, n)
    times = rng.exponential(18 / np.exp(0.5 * treatment + 0.02 * (age - 60)))
    times = np.maximum(1, np.round(times))
    censor = rng.integers(8, 45, n)
    frame = pd.DataFrame({
        "stay_id": [f"synthetic-{i:03d}" for i in range(n)],
        "treatment": treatment, "age": age,
        "sex": np.where(rng.random(n) < 0.5, "female", "male"),
        "follow_up_days": np.minimum(times, censor), "death": np.where(times <= censor, 5., 0.),
    })
    frame.loc[0, "treatment"] = np.nan
    frame.loc[[1, 7, 12], "age"] = np.nan
    frame.loc[3, "sex"] = None
    frame.loc[9, "death"] = np.nan
    frame.loc[10, "follow_up_days"] = np.nan
    # Exercise the exact boundary: event at horizon kept; event after it censored.
    frame.loc[[20, 21], ["follow_up_days", "death"]] = [[28., 5.], [29., 5.]]
    frame.attrs["provenance"] = "synthetic_example"
    return frame


def write_example(out_dir: str | Path) -> tuple[Path, SurvivalReferenceSpec]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path, spec = out / "example_cohort.parquet", example_spec()
    example_cohort().to_parquet(path, index=False)
    (out / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path, spec
