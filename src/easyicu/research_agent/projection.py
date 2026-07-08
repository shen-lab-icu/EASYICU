"""Deterministic source-data projection for primary/result publication figures.

Primary/result figures must not depend on LLM-written source-data tables (the
E3/M1/E2 fail-close class: the model hand-codes a ``*_source_data.csv`` that
drops the trace key or re-derives a wrong value). Instead a primary figure's
source-data table is a *literal, verbatim projection* of the validated upstream
analysis output the figure visualises — the exact table the deterministic
analysis runner wrote (``dose_response.csv``, ``primary_adjusted_odds_ratios``,
the parsed Cox/IPTW frame, cluster/trajectory characteristics, cohort-flow
attrition, ...).

Because the projection copies the key and value cells *verbatim* from that
upstream frame, the figure's source-data table passes
:meth:`FigureSourceDataValidator._compare_source_to_upstream` by construction:

* the trace **key** column is a verbatim slice of the upstream key column, so
  the row-subset check (source keys ⊆ upstream keys) holds by identity;
* every declared **value** column is a verbatim copy, so the 1e-9
  numeric-equality merge holds by identity.

The only way to fail is a *missing or non-whitelisted key column*, which this
module raises on **before** writing anything — turning a silent gate rejection
at report time into a loud, unit-testable construction error.

The module is intentionally dependency-light (pandas + the validator's column
tuples) so it can be imported by any figure renderer without pulling the
pipeline execution graph, and unit-tested in isolation against the real
validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from easyicu.research_agent.audits.validators import FigureSourceDataValidator

# Import the validator's own tuples so the projector's key/value whitelists can
# never drift from what the trace gate actually accepts.
KEY_COLUMNS: tuple[str, ...] = tuple(FigureSourceDataValidator._KEY_COLUMNS)
NUMERIC_COLUMNS: tuple[str, ...] = tuple(FigureSourceDataValidator._NUMERIC_COLUMNS)

# Provenance column carried on every projected table; not a trace key.
SOURCE_TABLE_COLUMN = "source_table"


class ProjectionError(ValueError):
    """Raised when a primary figure cannot be projected traceably.

    Callers for *primary/result* figures must treat this as fail-closed (do not
    fall back to an LLM-written source table); callers for supplementary figures
    may catch it and drop the figure with an advisory.
    """


@dataclass(frozen=True)
class ProjectionResult:
    path: Path
    key_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    n_rows: int


def project_source_data(
    *,
    upstream_frame: pd.DataFrame,
    upstream_path: Path,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    out_csv: Path,
    extra_display_columns: Mapping[str, Sequence] = (),
) -> ProjectionResult:
    """Emit ``out_csv`` as a verbatim, trace-safe projection of ``upstream_frame``.

    Parameters
    ----------
    upstream_frame:
        The exact parent analysis table (unmodified). Rows written to the figure
        source data are a subset (by default all) of these rows.
    upstream_path:
        Path of the parent table, recorded verbatim in ``source_table`` for
        provenance (not used as a trace key).
    key_columns:
        One or more per-row trace keys. Each MUST be present in
        ``upstream_frame`` and MUST be a member of the validator's
        ``_KEY_COLUMNS`` — otherwise the trace gate would not recognise it and
        this function raises :class:`ProjectionError`.
    value_columns:
        Numeric result columns copied verbatim (odds_ratio/ci_low/ci_high,
        point_estimate, retained_n, ...). Copied cell-for-cell so the validator's
        numeric-equality check is satisfied by identity.
    out_csv:
        Destination CSV (usually ``publication_figure_source_data.csv``).
    extra_display_columns:
        Optional non-load-bearing display columns (e.g. ``plot_label``). Each
        must align in length with ``upstream_frame`` and must NOT collide with a
        ``_KEY_COLUMNS`` name outside ``key_columns`` (that would let an
        unintended key shadow the chosen one) nor with any upstream column name.
    """
    if upstream_frame is None or len(upstream_frame.columns) == 0:
        raise ProjectionError("upstream_frame is empty; nothing to project")

    keys = tuple(dict.fromkeys(str(k) for k in key_columns))  # de-dup, keep order
    if not keys:
        raise ProjectionError("at least one key_column is required")

    for k in keys:
        if k not in upstream_frame.columns:
            raise ProjectionError(
                f"key column {k!r} not present in upstream table "
                f"{upstream_path.name}; available: {list(upstream_frame.columns)}"
            )
        if k not in KEY_COLUMNS:
            raise ProjectionError(
                f"key column {k!r} is not a recognised figure trace key "
                f"(_KEY_COLUMNS); the trace gate would reject it. "
                f"Choose one of: {KEY_COLUMNS}"
            )

    values = tuple(dict.fromkeys(str(v) for v in value_columns))
    for v in values:
        if v not in upstream_frame.columns:
            raise ProjectionError(
                f"value column {v!r} not present in upstream table "
                f"{upstream_path.name}; available: {list(upstream_frame.columns)}"
            )
        # A value column that is itself a _KEY_COLUMNS member (other than the
        # chosen keys) would be picked as the trace key ahead of our intended
        # key (the validator scans _KEY_COLUMNS in order). Refuse it.
        if v in KEY_COLUMNS and v not in keys:
            raise ProjectionError(
                f"value column {v!r} is a reserved trace-key name; it would "
                f"shadow the chosen key {keys!r}. Rename it or use it as the key."
            )

    out = pd.DataFrame(index=range(len(upstream_frame)))
    for k in keys:
        out[k] = upstream_frame[k].to_numpy()
    for v in values:
        out[v] = upstream_frame[v].to_numpy()

    for name, series in dict(extra_display_columns).items():
        name = str(name)
        if name in KEY_COLUMNS and name not in keys:
            raise ProjectionError(
                f"display column {name!r} is a reserved trace-key name and would "
                f"shadow the chosen key {keys!r}"
            )
        if name in upstream_frame.columns:
            raise ProjectionError(
                f"display column {name!r} collides with an upstream column; a "
                f"differing value would trip the numeric-equality trace check"
            )
        vals = list(series)
        if len(vals) != len(out):
            raise ProjectionError(
                f"display column {name!r} has length {len(vals)}, expected "
                f"{len(out)} (one per upstream row)"
            )
        out[name] = vals

    # Provenance column last (not a trace key). Guard the accidental collision
    # where an upstream table already carries a differing ``source_table``.
    if SOURCE_TABLE_COLUMN not in out.columns:
        out[SOURCE_TABLE_COLUMN] = upstream_path.name

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return ProjectionResult(
        path=out_csv,
        key_columns=keys,
        value_columns=values,
        n_rows=len(out),
    )
