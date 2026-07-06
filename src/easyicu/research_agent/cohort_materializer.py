"""Materialize an analysis cohort directly from the EasyICU data layer.

This is the trusted bridge that lets the research agent's *first step* be
real cohort extraction + inclusion/exclusion, rather than relying on a
hand-built cohort parquet. It runs **outside** the network-isolated
analysis sandbox (it needs data access), and produces the per-stay cohort
parquet the sandbox then consumes.

Two sources, auto-detected:

* **converted database** (a ``data_path`` to a ricu-style prepared MIMIC/eICU
  directory) -> concepts are extracted for all patients via
  :func:`easyicu.api.load_concepts`;
* **existing EasyICU export package** (a directory holding
  ``easyicu_export_manifest.json`` + one parquet per concept group) -> concepts
  are read from disk, *no re-extraction* (the "user already has the data, just
  filter" path).

For every time-series concept it emits a wide per-stay summary
(``<c>_max/_min/_mean/_first/_n/_measured``) over a window, matching the
data-quality-gate input contract (see ``docs/qc_eligibility_gate_design_v1``),
plus timing columns ``<c>_first_time/_last_time`` carrying the ``charttime``
(hours from ICU admission) of the first/last recorded value. The timing
columns are what make exposure-timing questions answerable — e.g. the first
``charttime`` where ``norepi_rate`` is recorded is the vasopressor initiation
time, so "early vs delayed" exposure groups can be constructed from the wide
cohort without re-reading the raw event stream.
Each outcome is emitted as a whole-stay binary ``<outcome>`` and, when its
source carries a timestamp, an event time ``<outcome>_time`` (e.g.
``death_time`` = time-of-death in hours from ICU admission, NaN when the event
never occurred) so survival models and immortal-time guards are possible
instead of being blocked by a timeless binary outcome.
For every concept named by a CTAS cohort predicate it additionally emits a
bare ``<concept_id>`` column carrying the predicate's declared aggregation, so
:func:`easyicu.research_agent.cohort_schema.build_cohort` can apply the
inclusion/exclusion (纳排) deterministically and auditably.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .easyicu_case_builder import (
    ID_COL,
    TIME_COL,
    _first_nonnull,
    _merge_left,
    _window,
    read_exported_concept,
)
from .cohort_schema import CohortDefinition, build_cohort

Window = Tuple[float, float]
_FALSE_TOKENS = {"", "0", "false", "f", "no", "n", "none", "nan", "na", "null", "off"}


def _truthy_series(values: pd.Series) -> pd.Series:
    """Coerce mixed boolean-like concept values to conservative truth flags."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype("string").str.strip().str.lower()
    missing = values.isna() | normalized.isna()

    out = pd.Series(False, index=values.index, dtype=bool)
    numeric_known = numeric.notna()
    out.loc[numeric_known] = numeric.loc[numeric_known] != 0

    text_known = ~numeric_known & ~missing
    out.loc[text_known] = ~normalized.loc[text_known].isin(_FALSE_TOKENS)
    return out


def _is_positive_only_boolean(series: pd.Series) -> bool:
    """True iff every non-NA value is boolean ``True`` (the positive level).

    Such a column is a sparse *event indicator*: the event is recorded as
    ``True`` and its absence is left as NA. A numeric column (e.g. a SOFA
    score) or a 0/1 column never matches, so the normalisation below cannot
    corrupt a measured variable.
    """
    nonnull = series.dropna()
    if nonnull.empty:
        return False
    return all(
        (v is True) or (isinstance(v, np.bool_) and bool(v))
        for v in nonnull.unique()
    )


# Summary suffixes emitted by `_summarize_timeseries` for a time-series concept.
_EVENT_SUMMARY_SUFFIXES = ("_max", "_min", "_mean", "_first")


def _normalize_event_indicator_columns(wide: pd.DataFrame) -> List[str]:
    """Decode sparse boolean event concepts so NA means "did not occur" (0).

    A concept like ``sep3_sofa2`` records only the positive event (``True``);
    a stay with no event has NA in ``<c>_max`` / ``<c>_first`` etc. Left as-is
    that NA reads as *measurement-missing* (e.g. an agent computing 66 %
    missingness and discarding the exposure), when it is really *structural
    absence* — the event did not happen. For any concept whose representative
    summary column is positive-only boolean, fill the negative level: integer
    summaries (max/min/first) become a clean 0/1 indicator and the mean
    becomes the within-window event fraction (0 when absent). Returns the list
    of normalised columns for provenance. Purely a representation decode — it
    never imputes a *measured* value (see ``_is_positive_only_boolean``).
    """
    normalized: List[str] = []
    bases: set[str] = set()
    for col in wide.columns:
        for suffix in _EVENT_SUMMARY_SUFFIXES:
            if col.endswith(suffix):
                bases.add(col[: -len(suffix)])
    for base in sorted(bases):
        probe = next(
            (
                f"{base}{s}"
                for s in ("_max", "_first", "_min")
                if f"{base}{s}" in wide.columns
            ),
            None,
        )
        if probe is None or not _is_positive_only_boolean(wide[probe]):
            continue
        for suffix in _EVENT_SUMMARY_SUFFIXES:
            col = f"{base}{suffix}"
            if col not in wide.columns:
                continue
            if suffix == "_mean":
                wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)
            else:
                # ``== True`` on a pandas *nullable* boolean/Int column keeps
                # <NA> in the result, which ``astype(int)`` rejects ("cannot
                # convert NA to integer"). Absence is the negative level for a
                # positive-only event indicator (docstring: "0 when absent"), so
                # fill the NA before the cast. Harmless no-op on numpy dtypes,
                # where NaN/None already compare False.
                wide[col] = (wide[col] == True).fillna(False).astype(int)  # noqa: E712
            normalized.append(col)
    return normalized


def _any_truthy(values: pd.Series) -> bool:
    return bool(_truthy_series(values).any())


def _all_truthy(values: pd.Series) -> bool:
    truth = _truthy_series(values.dropna())
    return bool(len(truth) > 0 and truth.all())

_AGG_FUNCS = {
    "max": lambda s: s.max(),
    "min": lambda s: s.min(),
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "sum": lambda s: s.sum(),
    "count": lambda s: s.count(),
    "first": _first_nonnull,
    "last": lambda s: _first_nonnull(s.iloc[::-1]),
    "any": _any_truthy,
    "all": _all_truthy,
}


def _coerce_int_stay(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or ID_COL not in df.columns:
        return df
    out = df.copy()
    out[ID_COL] = out[ID_COL].astype("float").astype("Int64").astype("int64")
    return out


def _is_export_dir(path: Path) -> bool:
    return (path / "easyicu_export_manifest.json").exists()


def _resolve_source(
    data_path: Union[str, Path], prefer_existing: bool
) -> Tuple[str, Path]:
    root = Path(data_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data_path does not exist: {root}")
    if _is_export_dir(root):
        return "export", root
    return "converted", root


def _timing_columns(w: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Per-stay ``<c>_first_time`` / ``<c>_last_time`` for one concept.

    The time index (``charttime``, hours from ICU admission) of the FIRST and
    LAST *recorded* (non-null) value of ``concept`` inside the window. Computed
    on the raw column **before** any presence-coercion, so a categorical event's
    ``_first_time`` is its true onset, not the window start.

    This is what makes timing-dependent questions answerable: e.g. the first
    ``charttime`` where ``norepi_rate`` is recorded IS the vasopressor
    initiation time, so "early vs delayed" exposure can be constructed. Without
    it the wide summary only exposes magnitude (``_max/_min/_mean/_first``) and
    an agent wrongly concludes no row-level timing exists and BLOCKs the study.

    A stay with no recorded value is absent here -> the column is NaN after the
    left-merge, which honestly reads as "never measured / event never occurred",
    i.e. no onset time.
    """
    if TIME_COL not in w.columns:
        return pd.DataFrame(columns=[ID_COL])
    recorded = w.loc[w[concept].notna(), [ID_COL, TIME_COL]]
    if recorded.empty:
        return pd.DataFrame(columns=[ID_COL])
    recorded = recorded.sort_values([ID_COL, TIME_COL])
    g = recorded.groupby(ID_COL)[TIME_COL]
    first_t = g.first()
    last_t = g.last()
    return pd.DataFrame(
        {
            ID_COL: first_t.index,
            f"{concept}_first_time": first_t.to_numpy(),
            f"{concept}_last_time": last_t.to_numpy(),
        }
    )


def _load_concept(
    source_mode: str,
    root: Path,
    concept: str,
    database: str,
    patient_ids: Optional[Sequence[int]],
    unavailable: List[str],
) -> pd.DataFrame:
    """Load one concept from an export package or a converted database.

    Shared by the wide-summary path and the long-trajectory path so they read
    the source identically. Appends to ``unavailable`` and returns an empty
    stay-keyed frame when the concept is absent (fail-soft); re-raises other
    errors with concept context.
    """
    try:
        if source_mode == "export":
            return _coerce_int_stay(read_exported_concept(root, concept))
        from ..api import load_concepts  # local import: heavy module

        return _coerce_int_stay(
            load_concepts(
                [concept], database=database, data_path=str(root), patient_ids=patient_ids
            )
        )
    except KeyError:
        unavailable.append(concept)
        return pd.DataFrame(columns=[ID_COL])
    except Exception as exc:  # noqa: BLE001 - add concept context before failing closed
        raise RuntimeError(f"failed to load concept {concept!r}") from exc


def _summarize_timeseries(df: pd.DataFrame, concept: str, window: Window) -> pd.DataFrame:
    """Per-stay summary columns for one time-series concept over ``window``."""
    w = _window(df, window[0], window[1])
    if w.empty or concept not in w.columns:
        return pd.DataFrame(columns=[ID_COL])
    # Timing (onset/last-record time) is taken from the RAW non-null values
    # before the presence-coercion below, so a categorical event keeps its true
    # onset charttime rather than the window start.
    timing = _timing_columns(w, concept)
    # The wide summary emits numeric _max/_min/_mean. A concept stored as object
    # (e.g. a ventilation status or a vasopressor drug name) cannot be reduced
    # with max/mean and would raise. Coerce: if any value parses as a number the
    # concept is numeric-stored-as-text (use the numeric view; unparseable rows
    # honestly become NaN/measurement-missing); otherwise it is a categorical /
    # event concept and we summarise its PRESENCE (1 = recorded in window), so
    # `_max` reads as "ever" and `_mean` as the within-window event fraction.
    col = w[concept]
    if not pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().any():
            w = w.assign(**{concept: numeric})
        else:
            w = w.assign(**{concept: _truthy_series(col).astype(float)})
    grp = w.groupby(ID_COL)[concept]
    out = grp.agg(["max", "min", "mean", "count"]).reset_index()
    out.columns = [ID_COL, f"{concept}_max", f"{concept}_min", f"{concept}_mean", f"{concept}_n"]
    first = (
        w.sort_values([ID_COL, TIME_COL])
        .groupby(ID_COL)[concept]
        .apply(_first_nonnull)
        .reset_index()
    )
    first.columns = [ID_COL, f"{concept}_first"]
    out = out.merge(first, on=ID_COL, how="left")
    out[f"{concept}_measured"] = (out[f"{concept}_n"].fillna(0) > 0).astype(int)
    if not timing.empty:
        out = out.merge(timing, on=ID_COL, how="left")
    return out


def _predicate_column(
    df: pd.DataFrame, concept: str, window: Window, aggregation: str
) -> pd.DataFrame:
    """A bare ``<concept>`` column carrying the CTAS-declared aggregation."""
    if TIME_COL not in df.columns:
        # static concept: one value per stay
        cols = [ID_COL, concept] if concept in df.columns else [ID_COL]
        return df[cols].drop_duplicates(ID_COL).copy()
    w = _window(df, window[0], window[1])
    if w.empty or concept not in w.columns:
        return pd.DataFrame(columns=[ID_COL, concept])
    agg_key = (aggregation or "").lower()
    if agg_key not in _AGG_FUNCS:
        raise ValueError(f"unsupported cohort predicate aggregation: {aggregation!r}")
    fn = _AGG_FUNCS[agg_key]
    if TIME_COL in w.columns:
        w = w.sort_values([ID_COL, TIME_COL])
    out = w.groupby(ID_COL)[concept].apply(fn).reset_index()
    out.columns = [ID_COL, concept]
    return out


def _static_column(df: pd.DataFrame, concept: str) -> pd.DataFrame:
    cols = [ID_COL, concept] if concept in df.columns else [ID_COL]
    return df[cols].drop_duplicates(ID_COL).copy()


def _binary_event_column(df: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Whole-stay binary: 1 if the stay has any event for ``concept`` (e.g. death)."""
    if ID_COL not in df.columns:
        return pd.DataFrame(columns=[ID_COL, concept])
    if concept not in df.columns:
        ids = pd.Series(df[ID_COL].dropna().unique(), name=ID_COL)
        return pd.DataFrame({ID_COL: ids, concept: 1})

    work = df[[ID_COL, concept]].dropna(subset=[ID_COL]).copy()
    if work.empty:
        return pd.DataFrame(columns=[ID_COL, concept])
    event = _truthy_series(work[concept])
    out = (
        pd.DataFrame({ID_COL: work[ID_COL], concept: event.astype(int)})
        .groupby(ID_COL, dropna=True)[concept]
        .max()
        .reset_index()
    )
    return out


def _event_time_column(df: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Per-stay ``<concept>_time``: the ``charttime`` of the event itself.

    ``_binary_event_column`` collapses an outcome to a whole-stay 0/1 and drops
    its time index. For an event concept whose source carries a timestamp (e.g.
    ``death`` is indexed by ``deathtime``), that time IS the time-of-event in
    hours from ICU admission. Surfacing it as ``<concept>_time`` (NaN when the
    event never occurred) is what lets a downstream analysis guard against
    immortal-time bias or fit a survival model — without it an agent sees only a
    binary outcome and must block any timing-aware effect estimate.

    Symmetric to ``_timing_columns`` for features. Returns an empty frame when
    the source has no usable time index (purely stay-level derived flags).
    """
    if TIME_COL not in df.columns or concept not in df.columns or ID_COL not in df.columns:
        return pd.DataFrame(columns=[ID_COL])
    work = df[[ID_COL, TIME_COL, concept]].dropna(subset=[ID_COL]).copy()
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    event = _truthy_series(work[concept])
    work = work[event & work[TIME_COL].notna()]
    if work.empty:
        return pd.DataFrame(columns=[ID_COL])
    out = (
        work.sort_values([ID_COL, TIME_COL])
        .groupby(ID_COL)[TIME_COL]
        .first()
        .reset_index()
    )
    out.columns = [ID_COL, f"{concept}_time"]
    return out


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()


def materialize_cohort(
    *,
    feature_concepts: Sequence[str],
    database: str = "miiv",
    data_path: Union[str, Path],
    cohort_definition: Optional[CohortDefinition] = None,
    cohort_window: Window = (0.0, 24.0),
    outcome_concepts: Sequence[str] = ("death",),
    static_concepts: Sequence[str] = ("age", "sex", "los_icu"),
    patient_ids: Optional[Sequence[int]] = None,
    prefer_existing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a per-stay analysis cohort from the EasyICU data layer.

    Returns ``(cohort_df, provenance)``. ``cohort_df`` is one row per ICU stay
    after applying ``cohort_definition`` (纳入排除); ``provenance`` records the
    source mode, concept list, window, attrition counts and hashes for audit.
    """
    t0 = time.time()
    source_mode, root = _resolve_source(data_path, prefer_existing)

    unavailable: List[str] = []

    def load(concept: str) -> pd.DataFrame:
        return _load_concept(
            source_mode, root, concept, database, patient_ids, unavailable
        )

    # concepts required by the CTAS cohort predicates (for 纳排)
    pred_specs: List[Tuple[str, Window, str]] = []
    if cohort_definition is not None:
        for pred in (*cohort_definition.inclusion, *cohort_definition.exclusion):
            tw = getattr(pred, "time_window", None)
            win: Window = (
                (float(tw.start_offset_hours), float(tw.end_offset_hours))
                if tw is not None
                else cohort_window
            )
            pred_specs.append((pred.concept_id, win, getattr(pred, "aggregation", "max")))

    static_set = list(dict.fromkeys(static_concepts))
    outcome_set = list(dict.fromkeys(outcome_concepts))
    feature_set = [c for c in dict.fromkeys(feature_concepts) if c not in static_set]

    # ---- base = every ICU stay (denominator); take from the first static concept
    base: Optional[pd.DataFrame] = None
    static_frames: List[pd.DataFrame] = []
    for c in static_set:
        df = load(c)
        static_frames.append(_static_column(df, c))
        if base is None and ID_COL in df.columns:
            base = df[[ID_COL]].drop_duplicates().copy()
    if base is None:
        raise RuntimeError("Could not establish a stay-level base from static_concepts")

    frames: List[pd.DataFrame] = [*static_frames]

    # ---- time-series features -> wide per-stay summaries (over cohort_window)
    for c in feature_set:
        df = load(c)
        if TIME_COL in df.columns:
            frames.append(_summarize_timeseries(df, c, cohort_window))
        else:
            frames.append(_static_column(df, c))

    # ---- outcomes -> whole-stay binary (a death after 24h still counts), plus
    # the event time (<c>_time, e.g. death_time = time-of-death hours from ICU
    # admission) when the source carries a timestamp, so timing-aware analyses
    # (immortal-time guards, survival models) are possible.
    for c in outcome_set:
        loaded = load(c)
        frames.append(_binary_event_column(loaded, c))
        event_time = _event_time_column(loaded, c)
        if not event_time.empty:
            frames.append(event_time)

    # ---- bare predicate columns for 纳排 (skip concepts already materialised bare)
    produced_bare = set(static_set) | set(outcome_set)
    for concept, win, agg in pred_specs:
        if concept in produced_bare:
            continue
        frames.append(_predicate_column(load(concept), concept, win, agg))
        produced_bare.add(concept)

    wide = _merge_left(base, frames)
    for c in outcome_set:
        if c in wide.columns:
            wide[c] = wide[c].fillna(0).astype(int)
    # A stay absent from a (sparse-event) concept's data has 0 measurements, not
    # an unknown count: fill `<c>_n` / `<c>_measured` with 0 so a sparse binary
    # event (e.g. sep3 onset) becomes a clean 0/1 cohort indicator rather than
    # NaN. Without this, non-event stays drop out of complete-case models and a
    # presence predictor collapses to a constant (singular design matrix).
    for col in wide.columns:
        if col.endswith("_n") or col.endswith("_measured"):
            wide[col] = wide[col].fillna(0)
            if col.endswith("_measured"):
                wide[col] = wide[col].astype(int)
    # Decode sparse boolean event concepts (e.g. sep3_sofa2): NA = event did
    # not occur -> 0, not measurement-missing. Prevents a downstream consumer
    # from misreading a structural absence as missing data and discarding the
    # exposure.
    event_indicator_columns = _normalize_event_indicator_columns(wide)
    n_all = int(len(wide))

    # ---- apply CTAS inclusion/exclusion (纳排), deterministic + auditable
    if cohort_definition is not None:
        cohort = build_cohort(cohort_definition, wide)
    else:
        cohort = wide
    n_after = int(len(cohort))

    provenance = {
        "schema_version": "easyicu.cohort_materializer/1",
        "source_mode": source_mode,
        "source": str(root),
        "database": database,
        "cohort_window_hours": list(cohort_window),
        "feature_concepts": list(feature_concepts),
        "outcome_concepts": list(outcome_concepts),
        "static_concepts": static_set,
        "cohort_definition": cohort_definition.to_dict() if cohort_definition else None,
        "n_stays_extracted": n_all,
        "n_stays_after_inclusion_exclusion": n_after,
        "unavailable_concepts": unavailable,
        "event_indicator_columns_normalized": event_indicator_columns,
        "columns": list(cohort.columns),
        "cohort_sha256": _hash_df(cohort.reset_index(drop=True)),
        "build_seconds": round(time.time() - t0, 2),
    }
    return cohort.reset_index(drop=True), provenance


def build_trajectory_long(
    *,
    data_path: Union[str, Path],
    concepts: Sequence[str],
    database: str = "miiv",
    window: Optional[Window] = None,
    patient_ids: Optional[Sequence[int]] = None,
    prefer_existing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Long-format trajectory ``(stay_id, charttime, concept, value_num, value_str)``.

    The wide per-stay summary in :func:`materialize_cohort` decides the temporal
    aggregation (max/min/first over a fixed window) up front, BEFORE the agent
    sees the data — which is lossy for any question that needs a threshold-
    crossing onset (first time MAP < 65), an incident-after-exposure endpoint
    (first AKI after first PEEP), or a time-varying exposure / landmark design.
    This emits the per-timepoint series for the named concepts so the agent can
    construct those temporal features itself in-sandbox, instead of being forced
    through the baseline-summary lens.

    Only rows with a recorded (non-null) value are kept. ``value_num`` is the
    numeric view (NaN when the concept is categorical/unparseable); ``value_str``
    preserves the raw value. ``window`` (hours from ICU admission) bounds the
    series; ``None`` keeps the full available trajectory.
    """
    source_mode, root = _resolve_source(data_path, prefer_existing)
    unavailable: List[str] = []
    frames: List[pd.DataFrame] = []
    materialized: List[str] = []
    for concept in dict.fromkeys(concepts):
        df = _load_concept(source_mode, root, concept, database, patient_ids, unavailable)
        if TIME_COL not in df.columns or concept not in df.columns:
            continue
        w = _window(df, window[0], window[1]) if window is not None else df
        sub = w.loc[w[concept].notna(), [ID_COL, TIME_COL, concept]]
        if sub.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    ID_COL: sub[ID_COL].to_numpy(),
                    TIME_COL: sub[TIME_COL].to_numpy(),
                    "concept": concept,
                    "value_num": pd.to_numeric(sub[concept], errors="coerce").to_numpy(),
                    "value_str": sub[concept].astype("string").to_numpy(),
                }
            )
        )
        materialized.append(concept)
    if frames:
        long_df = (
            pd.concat(frames, ignore_index=True)
            .sort_values([ID_COL, TIME_COL, "concept"])
            .reset_index(drop=True)
        )
    else:
        long_df = pd.DataFrame(
            columns=[ID_COL, TIME_COL, "concept", "value_num", "value_str"]
        )
    provenance = {
        "schema_version": "easyicu.cohort_trajectory/1",
        "source_mode": source_mode,
        "source": str(root),
        "database": database,
        "trajectory_window_hours": list(window) if window is not None else None,
        "trajectory_concepts_requested": list(dict.fromkeys(concepts)),
        "trajectory_concepts_materialized": materialized,
        "unavailable_concepts": unavailable,
        "n_rows": int(len(long_df)),
        "n_stays": int(long_df[ID_COL].nunique()) if len(long_df) else 0,
        "trajectory_sha256": _hash_df(long_df),
    }
    return long_df, provenance


def materialize_to_parquet(
    output_dir: Union[str, Path],
    *,
    stem: str = "cohort",
    emit_trajectory: bool = False,
    trajectory_concepts: Optional[Sequence[str]] = None,
    trajectory_window: Optional[Window] = None,
    **kwargs: Any,
) -> Dict[str, Path]:
    """Materialize and write ``<stem>.parquet`` + ``<stem>_provenance.json``.

    When ``emit_trajectory`` is set, also writes ``<stem>_trajectory.parquet``
    (+ ``_provenance.json``): the long per-timepoint series for
    ``trajectory_concepts`` (default: the outcome + feature concepts), so the
    agent can build onsets / incident endpoints / landmark designs that the wide
    summary cannot express. Default off — existing callers are unaffected.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort, provenance = materialize_cohort(**kwargs)
    parquet_path = out / f"{stem}.parquet"
    prov_path = out / f"{stem}_provenance.json"
    cohort.to_parquet(parquet_path, index=False)
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {"parquet": parquet_path, "provenance": prov_path}

    if emit_trajectory:
        concepts = trajectory_concepts
        if concepts is None:
            concepts = [
                *kwargs.get("outcome_concepts", ("death",)),
                *kwargs.get("feature_concepts", ()),
            ]
        long_df, traj_prov = build_trajectory_long(
            data_path=kwargs["data_path"],
            concepts=concepts,
            database=kwargs.get("database", "miiv"),
            window=trajectory_window,
            patient_ids=kwargs.get("patient_ids"),
            prefer_existing=kwargs.get("prefer_existing", True),
        )
        traj_path = out / f"{stem}_trajectory.parquet"
        traj_prov_path = out / f"{stem}_trajectory_provenance.json"
        long_df.to_parquet(traj_path, index=False)
        traj_prov_path.write_text(
            json.dumps(traj_prov, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        paths["trajectory"] = traj_path
        paths["trajectory_provenance"] = traj_prov_path
    return paths
