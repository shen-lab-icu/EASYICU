"""The mixed-mode model keeps its level indicators sparse.

Each observed ordinal cell sets one level indicator.  The engine held them as
a dense float matrix and rebuilt it for every start.  At the design's limit
(16 coordinates over 48 windows, 5 levels, about 92,000 stays) that was 2.8 GB
per start, and 7.3 GB with GCS levels, on a 16 GB host.  The engine now keeps
one entry per observed ordinal cell, so the memory follows the observed cells,
not the levels.  The EM sums are the same, in another order.  Synthetic data
only.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from easyicu.research_agent.trajectory import mixed_mode_latent_class as engine

FIT = {"max_iter": 500, "tolerance": 1e-5, "regularization": 1e-6}


def _panel(*, n_levels: int, n: int = 400, windows: int = 12, seed: int = 11):
    """Two classes of whole-level courses, 15% missing, and one continuous column."""

    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    top = n_levels - 1
    centre = np.where(truth[:, None] == 0, 0.25 * top, 0.75 * top) * np.ones((n, windows))
    ordinal = np.clip(np.rint(centre + rng.normal(0.0, 0.7, centre.shape)), 0, top)
    continuous = rng.normal(truth.astype(float)[:, None], 1.0, size=(n, 1))
    x = np.concatenate([ordinal, continuous], axis=1)
    x[rng.random(x.shape) < 0.15] = np.nan
    column_levels = (tuple(range(n_levels)),) * windows + (None,)
    return x, column_levels, truth


def _indicators(x, column_levels, _truth=None):
    ordinal = [index for index, levels in enumerate(column_levels) if levels is not None]
    observed = np.isfinite(x)
    return engine._one_hot_levels(x, observed, ordinal, column_levels), observed, ordinal


@pytest.mark.parametrize("n_levels", [5, 13], ids=["sofa_like", "gcs_like"])
def test_the_indicators_hold_one_entry_per_observed_ordinal_cell(n_levels: int) -> None:
    x, column_levels, _truth = _panel(n_levels=n_levels, windows=48)
    (one_hot, blocks), observed, ordinal = _indicators(x, column_levels)

    assert sparse.issparse(one_hot)
    assert one_hot.shape == (len(x), 48 * n_levels)
    assert one_hot.nnz == int(observed[:, ordinal].sum())
    # The same cells as the dense indicators: one per observed level.
    expected = np.zeros(one_hot.shape)
    for (start, size), column in zip(blocks, ordinal, strict=True):
        rows = np.flatnonzero(observed[:, column])
        expected[rows, start + x[rows, column].astype(int)] = 1.0
    assert np.array_equal(one_hot.toarray(), expected)
    stored = one_hot.data.nbytes + one_hot.indices.nbytes + one_hot.indptr.nbytes
    assert stored < expected.nbytes / 3


def test_the_memory_follows_the_observed_cells_not_the_levels() -> None:
    (sofa, _), *_ = _indicators(*_panel(n_levels=5, windows=48))
    (gcs, _), *_ = _indicators(*_panel(n_levels=13, windows=48))

    def stored(matrix) -> int:
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

    # The same missingness: the same cells, and the same bytes at 13 levels.
    assert sofa.nnz == gcs.nnz
    assert stored(sofa) == stored(gcs)


def test_the_class_level_counts_follow_their_definition() -> None:
    x, column_levels, _truth = _panel(n_levels=5)
    (one_hot, blocks), observed, ordinal = _indicators(x, column_levels)
    # Three unequal classes, so a class out of place changes the counts.
    responsibilities = np.random.default_rng(3).dirichlet([4.0, 2.0, 1.0], size=len(x))

    counts = engine._class_level_counts(one_hot, responsibilities)

    expected = np.zeros((3, one_hot.shape[1]))
    for (start, _size), column in zip(blocks, ordinal, strict=True):
        for row in np.flatnonzero(observed[:, column]):
            expected[:, start + int(x[row, column])] += responsibilities[row]
    np.testing.assert_allclose(counts, expected, rtol=1e-12)


def test_the_row_level_scores_follow_their_definition() -> None:
    x, column_levels, _truth = _panel(n_levels=5)
    (one_hot, blocks), observed, ordinal = _indicators(x, column_levels)
    # Three classes with their own level probabilities in every column.
    draws = np.random.default_rng(5).dirichlet(np.ones(5), size=(3, len(blocks)))
    log_levels = np.log(draws).reshape(3, -1)

    scores = engine._row_level_log_probabilities(one_hot, log_levels)

    expected = np.zeros((len(x), 3))
    for (start, _size), column in zip(blocks, ordinal, strict=True):
        for row in np.flatnonzero(observed[:, column]):
            expected[row] += log_levels[:, start + int(x[row, column])]
    np.testing.assert_allclose(scores, expected, rtol=1e-12)


def test_the_fit_matches_the_dense_indicators_to_rounding(monkeypatch) -> None:
    x, column_levels, truth = _panel(n_levels=5)
    fit = dict(column_levels=column_levels, n_components=2, seed=7, **FIT)
    labels, trace = engine.fit_observed_data_mixed_mode_lca(x, **fit)
    # The model still finds the two classes the panel was drawn from.
    assert adjusted_rand_score(truth, labels) > 0.9

    build = engine._one_hot_levels

    def dense(*args, **kwargs):
        one_hot, blocks = build(*args, **kwargs)
        return one_hot.toarray(), blocks

    monkeypatch.setattr(engine, "_one_hot_levels", dense)
    dense_labels, dense_trace = engine.fit_observed_data_mixed_mode_lca(x, **fit)

    assert np.array_equal(labels, dense_labels)
    assert trace["n_iter"] == dense_trace["n_iter"]
    assert trace["final_log_likelihood"] == pytest.approx(
        dense_trace["final_log_likelihood"], rel=1e-12
    )


def test_a_design_without_ordinal_columns_has_no_indicators() -> None:
    x, _levels, _truth = _panel(n_levels=5)
    continuous = (None,) * x.shape[1]

    (one_hot, blocks), *_ = _indicators(x, continuous)
    labels, trace = engine.fit_observed_data_mixed_mode_lca(
        x, column_levels=continuous, n_components=2, seed=7, **FIT
    )

    assert one_hot.shape == (len(x), 0) and blocks == []
    assert set(labels) == {0, 1} and trace["converged"] is True
