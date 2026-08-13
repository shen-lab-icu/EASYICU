"""Shared helpers for the study-design-aware figure renderers.

This module has no dependency on :mod:`.skill`, so importing it from the
skill (``figures.skill`` -> ``figures`` -> ``figures.base``) creates no cycle.
It owns:

* :class:`RenderedFigure` -- the renderer -> skill hand-off object;
* table lookup/reading against the :class:`EvidenceStore`;
* small numeric helpers (Kaplan-Meier estimator, z-scored cluster profiles)
  the renderers need but that do not warrant a heavy dependency (lifelines /
  scikit-survival) inside the deterministic figure skill.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..authority.evidence_store import EvidenceStore
from ..schema import EvidenceRecord


@dataclass
class RenderedFigure:
    """A drawn matplotlib figure plus everything the skill needs to persist it.

    Renderers stay free of EvidenceStore registration logic: they draw the
    figure and describe its panels/sources, and the skill performs the
    save + audit + evidence registration in one shared place.
    """

    fig: Any
    figure_id: str
    core_claim: str
    generation_mode: str
    panels: List[Dict[str, Any]]
    source_evidence_ids: List[str]
    # name -> frame written next to the figure as ``publication_figure_source_<name>.csv``
    source_frames: Dict[str, "pd.DataFrame"] = field(default_factory=dict)
    statistics_note: str = (
        "The figure is generated after analysis validation from "
        "EvidenceStore-registered source tables; it is not drawn from writer prose."
    )


def close_leaked_figures() -> None:
    """Close matplotlib figures left open by a render that raised mid-way."""

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass


def read_table(path: Path) -> pd.DataFrame:
    """Read a registered table by suffix (csv/tsv/parquet/feather)."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"unsupported table format for figure renderer: {path.name}")


def find_table_records(
    evidence: EvidenceStore,
    names: Sequence[str],
) -> List[EvidenceRecord]:
    """All table records matching ``names`` by id/alias exact or path substring.

    Exact id/alias hits (via ``evidence.get``) rank ahead of substring hits so
    a precisely named ``cox_summary`` beats an incidental filename match.
    """

    name_set = {str(name).lower() for name in names if str(name).strip()}
    ordered: List[EvidenceRecord] = []
    seen: set[str] = set()
    for name in names:
        record = evidence.get(name)
        if (
            record is not None
            and record.kind == "table"
            and record.evidence_id not in seen
        ):
            ordered.append(record)
            seen.add(record.evidence_id)
    for record in evidence.records():
        if record.kind != "table" or record.evidence_id in seen:
            continue
        stem = Path(record.relative_path).stem.lower()
        if (
            any(token in stem for token in name_set)
            or record.evidence_id.lower() in name_set
        ):
            ordered.append(record)
            seen.add(record.evidence_id)
    return ordered


def load_table(
    evidence: EvidenceStore,
    run_dir: Path,
    names: Sequence[str],
    *,
    require_columns: Optional[Sequence[Sequence[str]]] = None,
    min_rows: int = 1,
) -> Tuple[Optional[EvidenceRecord], Optional[pd.DataFrame]]:
    """First matching table that reads, has ``>= min_rows`` and required columns.

    ``require_columns`` is a list of column *candidate groups*; every group must
    be satisfied by at least one column (case-insensitive substring). This lets
    a renderer demand e.g. "a time column AND an event column" without pinning
    exact spellings the coder may vary.
    """

    for record in find_table_records(evidence, names):
        try:
            frame = read_table(run_dir / record.relative_path)
        except Exception:
            continue
        if frame is None or frame.empty or len(frame) < min_rows:
            continue
        if require_columns and not all(
            resolve_column(frame, group) is not None for group in require_columns
        ):
            continue
        return record, frame
    return None, None


def _appears_as_token(key: str, column_lower: str) -> bool:
    """True when ``key`` occurs in ``column_lower`` as a whole token.

    A token boundary is the start/end of the string or any non-alphanumeric
    character (``_``, space, parens, ``%`` ...). This keeps ``lactate`` matching
    ``mean_lactate`` and ``n`` matching ``n_total`` while stopping the substring
    traps the false-pass audit found: ``n`` no longer matches ``media(n)`` and
    ``surv`` no longer matches ``survival_time`` (which had collided the time and
    survival-probability axes onto one column).
    """
    return (
        re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", column_lower)
        is not None
    )


def resolve_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """Return the actual column name matching any candidate.

    Matching is: exact (case-insensitive) first, then whole-TOKEN containment
    (not raw substring -- see :func:`_appears_as_token`). The first candidate that
    resolves wins, so callers should order candidates by preference and include
    the full form of any short abbreviation (e.g. list ``survival_prob`` as well
    as ``surv``) since a bare prefix no longer partial-matches a longer token.
    """

    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lookup:
            return lookup[key]
    for cand in candidates:
        key = str(cand).strip().lower()
        if not key:
            continue
        for lower, original in lookup.items():
            if _appears_as_token(key, lower):
                return original
    return None


def first_exact_column(
    columns: Mapping[str, str], candidates: Sequence[str]
) -> Optional[str]:
    """Return the first exact normalized column named by the caller."""

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to float, dropping unparseable values."""

    return pd.to_numeric(frame[column], errors="coerce")


def km_estimate(
    durations: Sequence[float],
    events: Sequence[float],
) -> Dict[str, Any]:
    """Kaplan-Meier survival estimate with a numpy-only implementation.

    Returns a dict with ``time`` (step boundaries incl. 0), ``survival``
    (step function), and ``at_risk`` at each unique event/censor time. Kept
    dependency-free so the deterministic figure skill does not require
    lifelines to draw a survival curve.
    """

    import numpy as np

    dur = np.asarray(list(durations), dtype=float)
    evt = np.asarray(list(events), dtype=float)
    mask = np.isfinite(dur) & np.isfinite(evt)
    dur = dur[mask]
    evt = (evt[mask] > 0).astype(float)
    if dur.size == 0:
        return {"time": [0.0], "survival": [1.0], "at_risk": [0], "n": 0}

    order = np.argsort(dur, kind="mergesort")
    dur = dur[order]
    evt = evt[order]
    unique_times = np.unique(dur)
    n = dur.size

    times: List[float] = [0.0]
    survival: List[float] = [1.0]
    at_risk_out: List[int] = [int(n)]
    surv = 1.0
    for t in unique_times:
        at_risk = int(np.sum(dur >= t))
        d = int(np.sum((dur == t) & (evt == 1)))
        if at_risk > 0 and d > 0:
            surv *= 1.0 - d / at_risk
        times.append(float(t))
        survival.append(float(surv))
        at_risk_out.append(at_risk)
    return {
        "time": times,
        "survival": survival,
        "at_risk": at_risk_out,
        "n": int(n),
        "n_events": int(np.sum(evt == 1)),
    }


def zscore_profiles(
    profiles: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Column-wise z-score of a cluster x feature centroid frame for a heatmap.

    Each feature (column) is standardised across clusters (rows) so the
    heatmap shows which clusters are high/low on each feature rather than raw
    unit scale. Zero-variance columns map to 0.
    """

    import numpy as np

    out = profiles[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    means = out.mean(axis=0)
    stds = out.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (out - means) / stds
    return z.fillna(0.0)


__all__ = [
    "RenderedFigure",
    "close_leaked_figures",
    "find_table_records",
    "km_estimate",
    "load_table",
    "numeric_series",
    "read_table",
    "resolve_column",
    "zscore_profiles",
]
