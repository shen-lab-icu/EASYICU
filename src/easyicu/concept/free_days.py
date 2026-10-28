"""Database-neutral free-day callback over verified support episodes.

An empty interval list means verified absence of support. Null/missing episodes
or insufficient survival follow-up refuse derivation; a sampled indicator or a
discharge time alone cannot establish 28-day ventilator-free survival.
"""

import math

import numpy as np
import pandas as pd


class FreeDaysDerivationError(ValueError):
    def __init__(self, code: str, message: str, *, database: str = "unknown"):
        self.code = code
        self.database = database
        super().__init__(f"{code} ({database}): {message}")


def _support_hours(intervals, horizon: float, database: str) -> float:
    if not isinstance(intervals, (list, tuple, np.ndarray)):
        raise FreeDaysDerivationError("support_intervals_unavailable", "verified episodes required", database=database)
    clipped = []
    for pair in intervals:
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise FreeDaysDerivationError("invalid_support_interval", "start/end hours required", database=database)
        try:
            start, end = map(float, pair)
        except (ValueError, TypeError) as exc:
            raise FreeDaysDerivationError("invalid_support_interval", "numeric hours required", database=database) from exc
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise FreeDaysDerivationError("invalid_support_interval", "finite ordered hours required", database=database)
        start, end = max(0.0, start), min(horizon, end)
        if end > start:
            clipped.append((start, end))
    total, right = 0.0, 0.0
    for start, end in sorted(clipped):
        total += max(0.0, end - max(right, start))
        right = max(right, end)
    return total


def derive_free_days(
    frame: pd.DataFrame, *, horizon_days: int, support_concept: str | list[str],
    death_score: int, intervals_column: str = "support_intervals",
    death_time_column: str = "death_hours", followup_time_column: str = "followup_hours",
    time_origin: str = "icu_admission",
) -> tuple[pd.Series, dict]:
    """Return complete free days and aggregate-only audit, without row export.

    All times are hours from the same declared origin. The input episode lists
    must come from concept/callback interval owners for the specified support.
    This is an explicit callback input contract, never an inference from a
    database name. Mixed databases are checked row by row; the refusal names
    the offending database when a ``database`` column is supplied.
    """
    if (time_origin != "icu_admission" or isinstance(horizon_days, bool) or not isinstance(horizon_days, int)
            or not 1 <= horizon_days <= 365 or death_score not in (-1, 0)
            or isinstance(death_score, bool)):
        raise ValueError("declare a positive integer horizon and death_score -1 or 0")
    concepts = [support_concept] if isinstance(support_concept, str) else support_concept
    if (not isinstance(concepts, list) or not concepts or len(concepts) != len(set(concepts))
            or not set(concepts) <= {"mech_vent", "vent_ind", "icu_stay", "vaso_ind"}):
        raise FreeDaysDerivationError("support_concept_unavailable", "unsupported support concept")
    for column, code in ((intervals_column, "support_intervals_unavailable"),
                         (death_time_column, "death_time_unavailable"),
                         (followup_time_column, "horizon_followup_unavailable")):
        if column not in frame:
            raise FreeDaysDerivationError(code, "required concept column absent")
    horizon = horizon_days * 24.0
    values, deaths = [], []
    for _, row in frame.iterrows():
        database = str(row.get("database", "unknown"))
        duration = _support_hours(row[intervals_column], horizon, database)
        try:
            death = float(row[death_time_column]) if pd.notna(row[death_time_column]) else None
            followup = float(row[followup_time_column])
        except (TypeError, ValueError) as exc:
            raise FreeDaysDerivationError("invalid_followup", "numeric hour coordinates required", database=database) from exc
        if (death is not None and (not math.isfinite(death) or death < 0)
                or not math.isfinite(followup) or followup < 0):
            raise FreeDaysDerivationError("invalid_followup", "finite nonnegative hours required", database=database)
        died = death is not None and death <= horizon
        if (death is not None and followup < death) or (not died and followup < horizon):
            raise FreeDaysDerivationError("horizon_followup_unavailable", "death ascertainment through the horizon required", database=database)
        values.append(death_score if died else math.floor((horizon - duration) / 24.0 + 1e-10))
        deaths.append(died)
    result = pd.Series(values, index=frame.index, dtype="int64")
    status = pd.Series(deaths, index=frame.index, dtype=bool)
    def describe(series):
        return {"n": len(series), "minimum": int(series.min()) if len(series) else None,
                "q1": float(series.quantile(0.25)) if len(series) else None,
                "median": float(series.median()) if len(series) else None,
                "q3": float(series.quantile(0.75)) if len(series) else None,
                "maximum": int(series.max()) if len(series) else None}
    audit = {
        "horizon_days": horizon_days, "support_concept": support_concept,
        "death_score": death_score, "day_rule": "complete_24h_free_of_support",
        "by_death_status": {"died_by_horizon": describe(result[status]),
                            "alive_at_horizon": describe(result[~status])},
        "distribution": describe(result),
        "frequencies": {str(int(k)): int(v) for k, v in result.value_counts().sort_index().items()},
    }
    return result, audit


def free_days_callback(tables, ctx):
    """Concept callback for plan-bound episodes plus observed follow-up.

    ``free_days_followup`` carries death_hours, followup_hours and a Boolean
    support_observed for each stay. The latter is a per-source coverage fact,
    supplied by the acquisition owner: absent episode rows only mean no support
    when observation was complete. A WinTbl supplies explicit interval validity;
    isolated indicators without durations are refused. No database is assumed
    to have episodes merely because it can record an admission-level flag.
    """
    from ..table import ICUTable, WinTbl

    options = dict(ctx.kwargs or {})
    requested = options.get("support_concept")
    concepts = [requested] if isinstance(requested, str) else requested
    database = str(getattr(getattr(ctx.data_source, "config", None), "name", "unknown"))
    if not isinstance(concepts, list) or not concepts:
        raise FreeDaysDerivationError("support_concept_unavailable", "declare the support concept roster", database=database)
    supports = [tables.get(concept) for concept in concepts]
    followup = tables.get("free_days_followup")
    if any(not isinstance(support, WinTbl) or not support.dur_var for support in supports):
        raise FreeDaysDerivationError("support_intervals_unavailable", "duration-bearing support concept required", database=database)
    if followup is None or "support_observed" not in followup.data:
        raise FreeDaysDerivationError("horizon_followup_unavailable", "support coverage and death follow-up required", database=database)
    ids = list(followup.id_columns)
    if not ids or any(list(support.id_columns) != ids for support in supports) or followup.data.duplicated(ids).any():
        raise FreeDaysDerivationError("concept_identity_mismatch", "unique same-grain stay identities required", database=database)
    frame = followup.data.copy()
    if not frame["support_observed"].eq(True).fillna(False).all():
        raise FreeDaysDerivationError("support_intervals_unavailable", "support not observed for every stay", database=database)
    def hours(series):
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds() / 3600
        # Concept interval coordinates carry timedeltas; bare numeric indices
        # may be minutes in a source and must never silently become hours.
        raise FreeDaysDerivationError("support_time_unit_unavailable", "timedelta concept intervals required", database=database)
    grouped = {}
    for concept, support in zip(concepts, supports):
        episodes = support.data.copy()
        if concept not in episodes or not episodes[concept].isin([0, 1, False, True]).all():
            raise FreeDaysDerivationError("invalid_support_indicator", "binary support indicator required", database=database)
        episodes = episodes.loc[episodes[concept].eq(1)]
        starts = hours(episodes[support.index_column])
        durations = hours(episodes[support.dur_var])
        for key, start, duration in zip(episodes[ids].itertuples(index=False, name=None), starts, durations):
            grouped.setdefault(key, []).append([start, start + duration])
    column = options.get("intervals_column", "support_intervals")
    frame[column] = [grouped.get(key, []) for key in frame[ids].itertuples(index=False, name=None)]
    frame["database"] = database
    values, audit = derive_free_days(frame, **options)
    frame[ctx.concept_name] = values
    output = ICUTable(frame[ids + [ctx.concept_name]], id_columns=ids, value_column=ctx.concept_name)
    output.data.attrs["free_days_derivation_audit"] = audit
    return output
