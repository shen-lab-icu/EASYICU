"""Plot the reported Cox HR; this never refits Cox or invents KM groups."""

from pathlib import Path

import pandas as pd

from ...reference_io import plot_intervals


def generate_all_plots(out_dir: str | Path) -> list[Path]:
    table = pd.read_csv(Path(out_dir) / "survival_effect_estimates.csv")
    return plot_intervals(table, out_dir, stem="survival_effect_forest",
                          estimate_column="hazard_ratio", label_column="exposure_design_column",
                          effect_scale="hazard_ratio")
