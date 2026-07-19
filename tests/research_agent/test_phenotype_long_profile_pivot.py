"""The phenotype figure must pivot a LONG/tidy centroid table to wide.

Clustering code commonly emits one row per (cluster, variable) with a mean/
median value column, not a wide cluster x feature matrix. Without pivoting, the
downstream groupby(cluster).mean() collapses every clinical variable into the
aggregate stat columns (median / mean / sd / missing_pct), so the "phenotype
profile" heatmap plots cluster x summary-statistics -- scientifically
meaningless (M3 subphenotype: cluster_profiles.csv had 17 vars x 2 clusters).
Wide inputs must never regress.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.figures.base import resolve_column
from easyicu.research_agent.figures.phenotype import (
    _PROFILE_NAMES,
    _feature_columns,
    _to_wide_cluster_profiles,
    render_phenotype_figure,
)
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext

_CLUSTER_CANDS = ["cluster", "cluster_id", "phenotype", "label", "class", "group"]


def _long_profiles() -> pd.DataFrame:
    # 2 clusters x 3 clinical variables, tidy form, with a per-row size column.
    rows = []
    means = {
        (0, "lactate"): 2.1,
        (0, "creatinine"): 1.1,
        (0, "map"): 78.0,
        (1, "lactate"): 4.8,
        (1, "creatinine"): 2.3,
        (1, "map"): 62.0,
    }
    sizes = {0: 180, 1: 90}
    for (cl, var), mean in means.items():
        rows.append(
            {
                "cluster": cl,
                "variable": var,
                "label": var.title(),
                "median": mean,
                "mean": mean,
                "sd": 1.0,
                "n_total": sizes[cl],
            }
        )
    return pd.DataFrame(rows)


def _wide_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": [0, 1, 2],
            "lactate": [2.1, 4.8, 1.4],
            "creatinine": [1.1, 2.3, 0.9],
            "map": [78, 62, 85],
            "n": [180, 90, 230],
        }
    )


# --- pure pivot -------------------------------------------------------------


def test_long_table_pivots_to_wide_clinical_variables():
    long = _long_profiles()
    cc = resolve_column(long, _CLUSTER_CANDS)
    wide, size = _to_wide_cluster_profiles(long, cc)
    # one row per cluster now
    assert wide[cc].nunique() == 2
    assert len(wide) == 2
    # columns are the clinical variables, not stat columns
    feats = set(_feature_columns(wide, cc, None))
    assert {"lactate", "creatinine", "map"} <= feats
    assert "median" not in feats and "sd" not in feats and "missing_pct" not in feats
    # per-cluster size recovered before the pivot dropped it
    assert size is not None
    assert size.get("0") == 180 and size.get("1") == 90


def test_wide_table_is_returned_unchanged():
    wide = _wide_profiles()
    cc = resolve_column(wide, _CLUSTER_CANDS)
    out, size = _to_wide_cluster_profiles(wide, cc)
    assert size is None
    assert out.equals(wide)


def test_long_without_variable_or_value_column_is_left_alone():
    # duplicated clusters but no (variable, value) shape -> cannot pivot safely
    df = pd.DataFrame({"cluster": [0, 0, 1, 1], "x": [1, 2, 3, 4]})
    cc = resolve_column(df, _CLUSTER_CANDS)
    out, size = _to_wide_cluster_profiles(df, cc)
    assert size is None
    assert out.equals(df)


# --- integration through the renderer ---------------------------------------


def _register(evidence: EvidenceStore, run_dir: Path, name: str, df: pd.DataFrame):
    path = run_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    evidence.register_file(
        kind="table",
        description=f"{name} table.",
        source_path=path,
        evidence_id=name,
        aliases=[name],
        producer="coder",
        generation_mode="agent",
    )


def test_renderer_on_long_table_uses_clinical_variables(tmp_path: Path):
    import matplotlib.pyplot as plt

    evidence = EvidenceStore(tmp_path)
    _register(evidence, tmp_path, _PROFILE_NAMES[0], _long_profiles())
    context = ResearchContext(
        research_question="Identify sepsis subphenotypes by unsupervised clustering.",
        cohort={
            "cohort_name": "c",
            "database": "miiv",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[],
    )
    plan = AnalysisPlan(research_question="q", steps=[])
    fig = render_phenotype_figure(
        context=context, plan=plan, evidence=evidence, run_dir=tmp_path
    )
    assert fig is not None
    assert len(fig.panels) == 3
    profiles = fig.source_frames["cluster_profiles"]
    cols = set(profiles.columns)
    # the heatmap's feature axis is the clinical variables, not stat columns
    assert {"lactate", "creatinine", "map"} <= cols
    assert "sd" not in cols and "median" not in cols
    plt.close(fig.fig)
