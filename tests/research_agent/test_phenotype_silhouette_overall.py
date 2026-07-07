"""The phenotype stability panel reports the OVERALL silhouette, so a per-cluster
metrics table must be averaged, not read at row 0 (false-pass audit #14: taking
the first cluster's silhouette overstates overall cluster quality).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.figures.phenotype import _silhouette_value


def _register_metrics(tmp_path: Path, df: pd.DataFrame) -> EvidenceStore:
    ev = EvidenceStore(tmp_path)
    p = tmp_path / "clustering_metrics.csv"
    df.to_csv(p, index=False)
    ev.register_file(
        kind="table",
        description="metrics",
        source_path=p,
        evidence_id="clustering_metrics",
        aliases=["clustering_metrics"],
        producer="coder",
        generation_mode="agent",
    )
    return ev


def test_per_cluster_silhouette_is_averaged(tmp_path: Path):
    ev = _register_metrics(
        tmp_path, pd.DataFrame({"cluster": [0, 1, 2], "silhouette": [0.62, 0.31, 0.28]})
    )
    v = _silhouette_value(ev, tmp_path)
    assert v is not None
    assert abs(v - 0.403) < 0.01  # mean, not the first cluster's 0.62


def test_single_overall_silhouette_is_returned_as_is(tmp_path: Path):
    ev = _register_metrics(tmp_path, pd.DataFrame({"silhouette_score": [0.41]}))
    v = _silhouette_value(ev, tmp_path)
    assert v is not None and abs(v - 0.41) < 1e-9


def test_long_form_silhouette_rows_are_averaged(tmp_path: Path):
    ev = _register_metrics(
        tmp_path,
        pd.DataFrame(
            {
                "metric": ["silhouette", "silhouette", "inertia"],
                "value": [0.5, 0.3, 1200.0],
            }
        ),
    )
    v = _silhouette_value(ev, tmp_path)
    assert v is not None and abs(v - 0.4) < 1e-9
