"""Plot the host's reported contrasts without recomputing an estimate."""

from pathlib import Path

import pandas as pd

from ...reference_io import plot_intervals


def generate_all_plots(out_dir: str | Path) -> list[Path]:
    table = pd.read_csv(Path(out_dir) / "adjusted_association_estimates.csv")
    table["label"] = table["contrast"].fillna(table["exposure"])
    return plot_intervals(table, out_dir, stem="adjusted_association_forest",
                          estimate_column="estimate", label_column="label",
                          effect_scale=str(table["effect_scale"].iloc[0]))
