"""Synthetic rows include missing predictors, reversed references, and patient groups."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..spec import AdjustedAssociationReferenceSpec


def example_spec() -> AdjustedAssociationReferenceSpec:
    return AdjustedAssociationReferenceSpec(
        title="Synthetic adjusted association", identity_column="stay_id",
        model={
            "requirement_id": "synthetic_primary", "outcome": "death",
            "outcome_type": "binary", "method_family": "statsmodels_logit_mle",
            "exposure_source": "treatment", "analysis_role": "primary",
            "analysis_set": "complete_case", "covariates": ["age", "sex"],
            "model_terms": [
                {"name": "treatment", "role": "exposure", "coding": "binary",
                 "levels": ["0", "1"], "reference_level": "1", "transform": "treatment_contrast"},
                {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
                {"name": "sex", "role": "covariate", "coding": "binary",
                 "levels": ["female", "male"], "reference_level": "male", "transform": "treatment_contrast"},
            ],
        },
    )


def example_cohort() -> pd.DataFrame:
    rng = np.random.default_rng(20261029)
    n = 240
    treatment = rng.integers(0, 2, n).astype(float)
    age = rng.normal(60, 11, n)
    sex = np.where(rng.random(n) < 0.5, "female", "male")
    patient_effect = np.repeat(rng.normal(0, 0.8, n // 3), 3)
    logit = -1 + 0.6 * treatment + 0.035 * (age - 60) + patient_effect
    frame = pd.DataFrame({
        "stay_id": [f"synthetic-{i:03d}" for i in range(n)],
        "patient_id": [f"synthetic-patient-{i // 3:03d}" for i in range(n)],
        "treatment": treatment, "age": age, "sex": sex,
        "death": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float),
        "continuous_outcome": 4 + 0.7 * treatment + 0.04 * age + patient_effect + rng.normal(0, 1, n),
    })
    frame.loc[0, "treatment"] = np.nan
    frame.loc[[1, 7, 12], "age"] = np.nan
    frame.loc[[3, 17], "sex"] = None
    frame.loc[9, "death"] = np.nan
    frame.attrs["provenance"] = "synthetic_example"
    return frame


def write_example(out_dir: str | Path) -> tuple[Path, AdjustedAssociationReferenceSpec]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path, spec = out / "example_cohort.parquet", example_spec()
    example_cohort().to_parquet(path, index=False)
    (out / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path, spec
