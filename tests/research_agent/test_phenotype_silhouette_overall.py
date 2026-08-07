"""The phenotype stability panel accepts only a declared overall silhouette.

Per-cluster, k-sweep, fold, or resample values are not an overall score; an
unweighted average would manufacture a scientific result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
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


def test_per_cluster_silhouette_is_not_misreported_as_overall(tmp_path: Path):
    ev = _register_metrics(
        tmp_path, pd.DataFrame({"cluster": [0, 1, 2], "silhouette": [0.62, 0.31, 0.28]})
    )
    _, value = _silhouette_value(ev, tmp_path)
    assert value is None


def test_single_overall_silhouette_is_returned_as_is(tmp_path: Path):
    ev = _register_metrics(tmp_path, pd.DataFrame({"silhouette_score": [0.41]}))
    _, value = _silhouette_value(ev, tmp_path)
    assert value is not None and abs(value - 0.41) < 1e-9


def test_long_form_multiple_silhouette_rows_are_not_averaged(tmp_path: Path):
    ev = _register_metrics(
        tmp_path,
        pd.DataFrame(
            {
                "metric": ["silhouette", "silhouette", "inertia"],
                "value": [0.5, 0.3, 1200.0],
            }
        ),
    )
    _, value = _silhouette_value(ev, tmp_path)
    assert value is None


def test_empty_numeric_silhouette_returns_the_declared_tuple_shape(tmp_path: Path):
    ev = _register_metrics(tmp_path, pd.DataFrame({"silhouette_score": ["n/a"]}))

    record, value = _silhouette_value(ev, tmp_path)

    assert record is not None
    assert value is None
