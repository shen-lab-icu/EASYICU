"""Small, deterministic prepared cohort made entirely from synthetic values."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..spec import TableOneReferenceSpec


def example_spec() -> TableOneReferenceSpec:
    return TableOneReferenceSpec(
        title="Synthetic baseline characteristics", identity_column="stay_id",
        table_one={
            "group_by": "arm", "group_levels": ["usual", "enhanced"],
            "missing_group_policy": "exclude_and_report",
            "variables": [
                {"name": "age", "variable_kind": "continuous", "summary": "both",
                 "test": "welch_t_or_anova"},
                {"name": "marker", "variable_kind": "continuous", "summary": "median_iqr",
                 "test": "mann_whitney_or_kruskal"},
                {"name": "sex", "variable_kind": "categorical", "summary": "count_percent",
                 "test": "chi_square_with_fisher_exact_for_sparse_2x2", "levels": ["female", "male"]},
                {"name": "rare_flag", "variable_kind": "categorical", "summary": "count_percent",
                 "test": "chi_square_with_fisher_exact_for_sparse_2x2", "levels": [0, 1]},
            ],
        },
    )


def example_cohort() -> pd.DataFrame:
    rng = np.random.default_rng(20261028)
    n = 120
    frame = pd.DataFrame({
        "stay_id": [f"synthetic-{i:03d}" for i in range(n)],
        "arm": np.where(np.arange(n) % 2, "enhanced", "usual"),
        "age": rng.normal(61, 12, n), "marker": rng.lognormal(0.4, 0.7, n),
        "sex": np.where(rng.random(n) < 0.5, "female", "male"),
        "rare_flag": (np.arange(n) == 1).astype(int),
    })
    frame.loc[0, "arm"] = None
    frame.loc[[3, 8, 14], "age"] = np.nan
    frame.loc[[5, 12], "marker"] = np.nan
    frame.loc[[9, 16], "sex"] = None
    frame.attrs["provenance"] = "synthetic_example"
    return frame


def write_example(out_dir: str | Path) -> tuple[Path, TableOneReferenceSpec]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path, spec = out / "example_cohort.parquet", example_spec()
    example_cohort().to_parquet(path, index=False)
    (out / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path, spec
