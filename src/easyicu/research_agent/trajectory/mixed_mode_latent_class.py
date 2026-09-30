"""Observed-data EM for a mixed-mode latent class model.

Owner: the signed trajectory model engine for coordinates the host declares
ordinal.  Public contract: :func:`fit_observed_data_mixed_mode_lca` fits one
start and returns hard labels plus a trace; :func:`mixed_mode_parameter_count`
is the free-parameter count its BIC uses.

Each representation column is one indicator.  A declared ordinal coordinate
(an organ score with levels 0-4, a stage) is a categorical indicator with
class-specific level probabilities, the unrestricted form, which is what
class-specific thresholds amount to.  Tied integer levels are modelled as the
discrete values they are; a Gaussian fitted to them can keep adding narrow
components on single levels, so its likelihood, and a BIC built on it, keeps
improving with the class count.  A continuous coordinate is a class-specific
Gaussian on its pooled z-score with a variance floor.  A row contributes only
its observed indicators (full-information likelihood under missing at random).

A symmetric Dirichlet prior with total mass :data:`MIXED_MODE_CATEGORY_PRIOR`
on each indicator's level probabilities keeps estimates off the boundary
(posterior-mode EM).  The reported log-likelihood excludes the prior.  The
engine version fixes the prior, so it is not a design knob.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

#: Total pseudo-count spread evenly over one indicator's levels, per class.
MIXED_MODE_CATEGORY_PRIOR = 1.0

OrdinalLevels = tuple[int, ...]


def mixed_mode_parameter_count(
    column_levels: Sequence[OrdinalLevels | None], n_components: int
) -> int:
    """Mixture weights plus, per class, each indicator's free parameters.

    An ordinal indicator with L levels has L - 1 free level probabilities; a
    continuous indicator has a mean and a variance.
    """

    per_class = sum(
        (len(levels) - 1) if levels is not None else 2 for levels in column_levels
    )
    return (n_components - 1) + n_components * per_class


def _one_hot_levels(
    x: np.ndarray,
    observed: np.ndarray,
    ordinal_columns: Sequence[int],
    column_levels: Sequence[OrdinalLevels | None],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n_rows = x.shape[0]
    width = sum(len(column_levels[column]) for column in ordinal_columns)
    one_hot = np.zeros((n_rows, width), dtype=float)
    blocks: list[tuple[int, int]] = []
    start = 0
    for column in ordinal_columns:
        levels = column_levels[column]
        assert levels is not None
        seen = observed[:, column]
        values = x[seen, column]
        codes = np.searchsorted(np.asarray(levels, dtype=float), values)
        in_range = codes < len(levels)
        valid = np.zeros(values.shape, dtype=bool)
        valid[in_range] = np.asarray(levels, dtype=float)[codes[in_range]] == values[in_range]
        if not valid.all():
            raise ValueError(
                f"ordinal indicator {column} has a value outside its declared "
                f"levels {list(levels)}"
            )
        one_hot[np.flatnonzero(seen), start + codes] = 1.0
        blocks.append((start, len(levels)))
        start += len(levels)
    return one_hot, blocks


def fit_observed_data_mixed_mode_lca(
    x: np.ndarray,
    *,
    column_levels: Sequence[OrdinalLevels | None],
    n_components: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    regularization: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Fit one start of the mixed-mode latent class model by EM.

    ``x`` holds raw levels in ordinal columns and pooled z-scores in
    continuous ones; NaN marks a missing indicator.  The start is the random
    balanced assignment the Gaussian engine uses.  Convergence is judged on
    the log-posterior, which EM increases monotonically.
    """

    x = np.asarray(x, dtype=float)
    n_rows, n_columns = x.shape
    if len(column_levels) != n_columns:
        raise ValueError("column_levels must name one scale per indicator")
    if n_rows <= n_components:
        raise ValueError("fit sample is not larger than the class count")
    observed = np.isfinite(x)
    if not observed.any(axis=1).all():
        raise ValueError("a row has no observed indicator")
    if not observed.any(axis=0).all():
        raise ValueError("an indicator has no observed values")
    for levels in column_levels:
        if levels is not None and (
            len(levels) < 2 or list(levels) != sorted(set(levels))
        ):
            raise ValueError("ordinal levels must be two or more increasing values")
    ordinal = [index for index, levels in enumerate(column_levels) if levels is not None]
    continuous = [index for index, levels in enumerate(column_levels) if levels is None]
    one_hot, blocks = _one_hot_levels(x, observed, ordinal, column_levels)

    n_continuous = len(continuous)
    statistics = np.empty((n_rows, 3 * n_continuous), dtype=float)
    observed_float = statistics[:, :n_continuous]
    observed_float[:] = observed[:, continuous]
    values = statistics[:, n_continuous : 2 * n_continuous]
    values[:] = np.where(observed[:, continuous], x[:, continuous], 0.0)
    np.square(values, out=statistics[:, 2 * n_continuous :])
    if n_continuous:
        counts = observed_float.sum(axis=0)
        global_mean = values.sum(axis=0) / counts
        centered = values - global_mean
        global_variance = np.maximum(
            (observed_float * centered * centered).sum(axis=0) / counts,
            regularization,
        )

    rng = np.random.default_rng(seed)
    initial = np.arange(n_rows, dtype=int) % n_components
    rng.shuffle(initial)
    responsibilities = np.zeros((n_rows, n_components), dtype=float)
    responsibilities[np.arange(n_rows), initial] = 1.0

    previous = -np.inf
    converged = False
    log_level_probabilities = np.zeros((n_components, one_hot.shape[1]))
    means = variances = np.zeros((n_components, 0))
    for iteration in range(max_iter):
        weights = np.maximum(responsibilities.sum(axis=0), 1e-12)
        weights /= weights.sum()
        level_counts = responsibilities.T @ one_hot
        prior_term = 0.0
        for start, size in blocks:
            block = level_counts[:, start : start + size] + MIXED_MODE_CATEGORY_PRIOR / size
            log_block = np.log(block) - np.log(block.sum(axis=1, keepdims=True))
            log_level_probabilities[:, start : start + size] = log_block
            prior_term += float((MIXED_MODE_CATEGORY_PRIOR / size) * log_block.sum())
        log_prob = np.log(weights)[None, :] + one_hot @ log_level_probabilities.T
        if n_continuous:
            weighted = responsibilities.T @ statistics
            effective = weighted[:, :n_continuous]
            means = np.broadcast_to(global_mean, (n_components, n_continuous)).copy()
            np.divide(
                weighted[:, n_continuous : 2 * n_continuous],
                effective,
                out=means,
                where=effective > 0,
            )
            second = np.broadcast_to(
                global_variance + global_mean * global_mean,
                (n_components, n_continuous),
            ).copy()
            np.divide(
                weighted[:, 2 * n_continuous :], effective, out=second, where=effective > 0
            )
            variances = np.maximum(second - means * means, regularization)
            inverse = 1.0 / variances
            coefficients = np.concatenate(
                (
                    (np.log(2.0 * math.pi) + np.log(variances) + means * means * inverse).T,
                    (-2.0 * means * inverse).T,
                    inverse.T,
                ),
                axis=0,
            )
            log_prob = log_prob - 0.5 * (statistics @ coefficients)
        normalizer = logsumexp(log_prob, axis=1)
        likelihood = float(normalizer.sum())
        if not math.isfinite(likelihood):
            raise ValueError("mixed-mode latent class fit produced a non-finite likelihood")
        objective = likelihood + prior_term
        responsibilities = np.exp(log_prob - normalizer[:, None])
        if iteration > 0 and abs(objective - previous) <= tolerance * (1.0 + abs(previous)):
            converged = True
            break
        previous = objective

    if not converged:
        raise ValueError("mixed-mode latent class fit did not converge")
    labels = np.argmax(responsibilities, axis=1).astype(int)
    if np.unique(labels).size != n_components:
        raise ValueError("mixed-mode latent class fit did not realize every class")
    digest = hashlib.sha256()
    for array in (weights, log_level_probabilities, means, variances):
        digest.update(np.ascontiguousarray(array).tobytes())
    return labels, {
        "converged": True,
        "n_iter": iteration + 1,
        "final_log_likelihood": likelihood,
        "parameter_sha256": digest.hexdigest(),
    }


def coordinate_measurement_levels(
    measurement: Sequence[Mapping[str, Any]],
) -> dict[str, OrdinalLevels | None]:
    """Each coordinate concept's declared levels, or None for a continuous one."""

    levels_by_concept: dict[str, OrdinalLevels | None] = {}
    for entry in measurement:
        concept = str(entry.get("concept") or "")
        scale = entry.get("scale")
        if not concept or concept in levels_by_concept:
            raise ValueError("coordinate measurement names each concept once")
        if scale == "continuous" and entry.get("levels") is None:
            levels_by_concept[concept] = None
            continue
        levels = entry.get("levels")
        if (
            scale != "ordinal"
            or not isinstance(levels, (list, tuple))
            or any(isinstance(level, bool) or not isinstance(level, int) for level in levels)
            or len(levels) < 2
            or list(levels) != sorted(set(levels))
        ):
            raise ValueError(f"coordinate measurement for {concept!r} is malformed")
        levels_by_concept[concept] = tuple(int(level) for level in levels)
    return levels_by_concept


def representation_column_levels(
    columns: Sequence[str], measurement: Sequence[Mapping[str, Any]]
) -> list[OrdinalLevels | None]:
    """Per representation column (``<concept>__h<start>_<end>``), its levels."""

    levels_by_concept = coordinate_measurement_levels(measurement)
    levels: list[OrdinalLevels | None] = []
    for column in columns:
        concept = str(column).split("__h", 1)[0]
        if concept not in levels_by_concept:
            raise ValueError(f"representation column {column!r} has no declared measurement")
        levels.append(levels_by_concept[concept])
    return levels


__all__ = [
    "MIXED_MODE_CATEGORY_PRIOR",
    "coordinate_measurement_levels",
    "fit_observed_data_mixed_mode_lca",
    "mixed_mode_parameter_count",
    "representation_column_levels",
]
