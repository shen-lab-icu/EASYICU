"""MIMIC chart support and medication evidence with explicit positive semantics.

These source callbacks do not infer exposure from the existence of a row.
CareVue vasoactive support reuses the bounded LOCF contract of mimic_rate_cv;
amount-only administrations remain points. Zero ECMO flow is unknown support,
not a decannulation event.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .schema import ConceptSource

logger = logging.getLogger(__name__)


def _numeric(values: pd.Series) -> pd.Series:
    """Use ordinary float/NaN semantics for nullable pandas source columns."""
    return pd.Series(pd.to_numeric(values, errors="coerce").to_numpy("float64", na_value=np.nan),
                     index=values.index)


def source_values(frame: pd.DataFrame, source: ConceptSource, column: str, concept: str) -> pd.Series:
    """Read a raw column even when the source loader renamed it to the concept."""
    if column in frame:
        return frame[column]
    if source.value_var == column and concept in frame:
        return frame[concept]
    return pd.Series(np.nan, index=frame.index)


def support_evidence(frame: pd.DataFrame, source: ConceptSource, concept: str) -> pd.DataFrame:
    """Positive settings/flows, nonzero circuit pressures, or declared text channels.

    Text channels are explicit source metadata, never a fallback for a failed
    numeric reading. Zero, missing and nonfinite readings cannot become support.
    """
    if frame.empty:
        return frame.assign(**{concept: pd.Series(index=frame.index, dtype=bool)})
    values = source_values(frame, source, "value", concept).astype("string").str.strip().str.casefold()
    numeric = _numeric(values)
    item = _numeric(frame[source.sub_var or "itemid"])
    pressure = item.isin(source.params.get("pressure_ids", []))
    positive = np.isfinite(numeric) & (numeric.gt(0) | (pressure & numeric.ne(0)))
    inactive = values.isin([
        "", "no", "none", "false", "off", "not applicable", "standby", "stand by",
        "ambient", "unknown", "other", "other/remarks", "nan", "na", "n/a", "null",
        "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    ])
    text = item.isin(source.params.get("text_ids", [])) & numeric.isna() & values.notna() & ~inactive
    return frame.loc[(positive | text).fillna(False)].assign(**{concept: True})


def ecmo_evidence(frame: pd.DataFrame, source: ConceptSource, concept: str) -> pd.DataFrame:
    """Parse flow, checkboxes and explicit state; ancillary site text is not on/off."""
    values = source_values(frame, source, "value", concept).astype("string").str.strip().str.casefold()
    item = _numeric(frame[source.sub_var or "itemid"])
    numeric = _numeric(values)
    checkbox = item.isin([228193, 229274])
    flow = item.isin([224660, 2957, 3265, 229270, 229277, 229278])
    config = item.eq(229268)
    positive = (flow & numeric.gt(0)) | (
        checkbox & (numeric.gt(0) | values.isin(["yes", "true", "on", "checked"]))
    ) | (config & values.isin(["vv", "va", "vav", "veno-venous", "veno-arterial"]))
    # Explicit off is a state; numeric zero flow can be a transient pump pause.
    negative = values.isin(["off", "no", "false", "unchecked", "discontinued"]) | (checkbox & numeric.eq(0))
    known = (positive | negative).fillna(False)
    out = frame.loc[known].copy()
    out[concept] = positive.loc[known].fillna(False).astype(bool)
    return out


def antibacterial_inputs(
    frame: pd.DataFrame, source: ConceptSource, concept: str, data_source: Any,
) -> pd.DataFrame:
    """Apply the prescription antibacterial whitelist to d_items labels as well."""
    from ..datasource import FilterOp, FilterSpec

    if frame.empty:
        return frame.assign(**{concept: pd.Series(dtype=bool)})
    if data_source is None:
        raise ValueError("mimic_antibacterial_inputs requires d_items labels")
    item_col = source.sub_var or "itemid"
    items = data_source.load_table(
        "d_items", columns=["itemid", "label"],
        filters=[FilterSpec("itemid", FilterOp.IN, frame[item_col].dropna().unique().tolist())],
        verbose=False,
    )
    items = items.data if hasattr(items, "data") else items
    labels = items.drop_duplicates("itemid").set_index("itemid")["label"]
    selected = frame[item_col].map(labels).astype("string").str.contains(
        str(source.params["antibacterial_regex"]), case=False, regex=True, na=False,
    )
    amount = _numeric(source_values(frame, source, "amount", concept))
    rate = _numeric(source_values(frame, source, "rate", concept))
    return frame.loc[selected & (amount.gt(0) | rate.gt(0))].assign(**{concept: True})


def other_vaso_points(
    frame: pd.DataFrame, source: ConceptSource, concept: str, *, icu_stays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return positive rate hours on the rate callback's grid, plus bolus points.

    CareVue shares mimic_rate_cv's per-linkorderid LOCF (six-hour maximum,
    one-hour final observation). Stops/zeros close support. This differs from
    the observed-span duration concept, whose singleton duration is unknown.
    """
    from ..io.ts_utils import expand
    from ..utils.callback_utils import mimic_rate_cv

    id_col = next(c for c in ["icustay_id", "stay_id"] if c in frame)
    time_col = source.index_var or ("charttime" if "charttime" in frame else "starttime")
    rate = _numeric(source_values(frame, source, "rate", concept))
    amount = _numeric(source_values(frame, source, "amount", concept))
    frame = frame.copy()
    frame["rate"] = rate
    parts = []
    if source.table == "inputevents_cv":
        stopped = frame["stopped"].astype("string").str.strip().str.casefold()
        terminal = stopped.isin(["stopped", "discontinued", "discontinue"]) | stopped.str.startswith("d/c", na=False)
        # Stop rows can retain the old positive pump setting. They close the
        # observed span but cannot start a course or count as a new bolus.
        frame.loc[terminal, "rate"] = 0.
        rate = rate.mask(terminal, 0.)
        # Match each rate source's drug and linkorderid grouping. Missing-rate
        # rows remain time references, exactly as in the numeric rate callback.
        for _, drug in frame.groupby(source.sub_var or "itemid", sort=False):
            drug = drug.assign(**{concept: drug["rate"]})
            hours = mimic_rate_cv(drug, val_col=concept, id_cols=[id_col],
                                  grp_var="linkorderid" if "linkorderid" in drug else None,
                                  admission_times=icu_stays)
            hours = hours.loc[hours[concept].notna()].copy()
            hours[concept] = _numeric(hours[concept]).gt(0)
            parts.append(hours)
    else:
        end_col = source.params.get("stop_var", "endtime")
        valid = rate.gt(0) & frame[end_col].gt(frame[time_col])
        spans = frame.loc[valid, [id_col, time_col, end_col]].assign(**{concept: True})
        epsilon = 1e-9 if pd.api.types.is_numeric_dtype(frame[time_col]) else pd.Timedelta(nanoseconds=1)
        spans[end_col] -= epsilon
        if not spans.empty:
            parts.append(expand(spans, start_var=time_col, end_var=end_col, id_cols=[id_col],
                                step_size=pd.Timedelta(hours=1), keep_vars=[concept],
                                admission_times=icu_stays))
    # A zero rate never becomes a bolus merely because a bag amount is recorded.
    bolus = rate.isna() & amount.gt(0)
    if bolus.any():
        parts.append(frame.loc[bolus, [id_col, time_col]].assign(**{concept: True}))
    negative = rate.eq(0) & ~amount.gt(0)
    if negative.any():
        parts.append(frame.loc[negative, [id_col, time_col]].assign(**{concept: False}))
    return (pd.concat(parts, ignore_index=True).drop_duplicates([id_col, time_col, concept])
            if parts else pd.DataFrame(columns=[id_col, time_col, concept]))


def rrt_procedure_hours(
    frame: pd.DataFrame, source: ConceptSource, concept: str, *, icu_stays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replay CareVue 582 starts/stops independently per stay and modality.

    Repeated starts keep the existing course open. Only an explicit start can
    open it, and a stop is exclusive. An unclosed start is point evidence for
    its own hour, with no invented treatment duration.
    """
    from ..io.ts_utils import expand

    if frame.empty:
        return frame.assign(**{concept: pd.Series(dtype=bool)})
    id_col = next(c for c in ("icustay_id", "stay_id") if c in frame)
    time_col = source.index_var or "charttime"
    text = source_values(frame, source, "value", concept).astype("string").str.strip().str.casefold()
    states = text.map({"hemodialysis st": "hd:start", "hemodialysis end": "hd:stop",
                       "cvvhd start": "cvvhd:start", "cvvhd d/c": "cvvhd:stop",
                       "cavh start": "cavh:start", "cavh d/c": "cavh:stop"})
    work = frame.loc[states.notna(), [id_col, time_col]].copy()
    work["__modality"] = states.str.split(":").str[0]
    work["__start"] = states.str.endswith(":start").fillna(False)
    work["__stop"] = states.str.endswith(":stop").fillna(False)
    keys = [id_col, "__modality"]
    work = work.dropna(subset=[id_col, time_col]).groupby([*keys, time_col], as_index=False).agg(
        __start=("__start", "max"), __stop=("__stop", "max"),
    ).sort_values([*keys, time_col], kind="stable")
    work["__course"] = work.groupby(keys)["__stop"].cumsum() - work["__stop"].astype(int)
    work["__begin"] = work[time_col].where(work["__start"])
    work["__end"] = work[time_col].where(work["__stop"])
    spans = work.groupby([*keys, "__course"], as_index=False).agg(
        **{time_col: ("__begin", "min"), "__end": ("__end", "min")},
    ).dropna(subset=[time_col])
    spans[concept] = True
    closed = spans["__end"].notna()
    parts = [expand(spans.loc[closed], start_var=time_col, end_var="__end", id_cols=[id_col],
                    step_size=pd.Timedelta(hours=1), keep_vars=[concept], admission_times=icu_stays,
                    end_inclusive=False)]
    points = spans.loc[~closed].copy()
    points["__end"] = points[time_col]
    parts.append(expand(points, start_var=time_col, end_var="__end", id_cols=[id_col],
                        step_size=pd.Timedelta(hours=1), keep_vars=[concept], admission_times=icu_stays))
    return pd.concat(parts, ignore_index=True).drop_duplicates([id_col, time_col])


def fentanyl_cv_mass_rate(
    frame: pd.DataFrame, source: ConceptSource, concept: str, *, resolver: Any = None, data_source: Any = None,
) -> pd.DataFrame:
    """30308 mcg/kg/hour -> mcg/hour using measured stay weight, never a default."""
    id_col = next(c for c in ["icustay_id", "stay_id"] if c in frame)
    out = frame.copy()
    if "weight" not in out and resolver is not None and data_source is not None:
        try:
            weights = resolver._load_single_concept(
                "weight", data_source, aggregator=False,
                patient_ids={id_col: out[id_col].dropna().unique().tolist()}, verbose=False,
            )
        except KeyError as exc:
            # A source without weight rows cannot convert per-kg rates; those rows
            # are then dropped and counted as missing_weight_rows below.
            logger.warning("fentanyl 30308: weight unavailable (%s); per-kg rows dropped", exc)
            weights = None
        weights = weights.data if hasattr(weights, "data") else weights
        if weights is not None and "weight" in weights and id_col in weights:
            valid = _numeric(weights["weight"]).where(lambda x: x.between(1, 500))
            out["weight"] = out[id_col].map(valid.groupby(weights[id_col]).median())
    weight = _numeric(out.get("weight", pd.Series(np.nan, index=out.index)))
    valid_weight = weight.between(1, 500)
    units = out[source.unit_var or "rateuom"].astype("string").str.strip().str.casefold()
    valid_unit = units.isin(["mcgkghr", "mcg/kg/hour", "mcg/kg/hr", "ug/kg/hr"])
    rate = _numeric(out[concept])
    stopped = source_values(out, source, "stopped", concept).astype("string").str.strip().str.casefold()
    terminal = stopped.isin(["stopped", "discontinued", "discontinue"]) | stopped.str.startswith("d/c", na=False)
    rate = rate.mask(terminal, 0.)
    # Keep explicit zero rates to terminate the CV expansion, even without weight.
    keep = (valid_weight & valid_unit) | rate.eq(0)
    audit = {"missing_weight_rows": int((rate.gt(0) & ~valid_weight).sum()),
             "unknown_unit_rows": int((rate.gt(0) & ~valid_unit).sum())}
    if any(audit.values()):
        logger.info("MIMIC CareVue mass-rate rejection concept=%s audit=%s", concept, audit)
    out[concept] = (rate * weight).where(rate.ne(0), 0.)
    out = out.loc[keep].copy()
    out[source.unit_var or "rateuom"] = "mcg/hour"
    out.attrs["mimic_mass_rate_audit"] = audit
    return out


def union_vaso_indicator(tables: dict, ctx: Any, duration_callback: Any) -> Any:
    """Union adjunct evidence with duration support in every source database.

    Boolean support is a point on its published grid, never a numeric duration.
    """
    from ..table import ICUTable

    duration_tables = {k: v for k, v in tables.items() if k.endswith("_dur")}
    base = duration_callback(duration_tables, ctx) if duration_tables else None
    pieces = []
    if base is not None and not base.data.empty:
        pieces.append(base.data.rename(columns={base.index_column: "__hour"}))
    id_cols = list(base.id_columns) if base is not None else []
    time_col = base.index_column if base is not None else None
    for name, other in tables.items():
        if name.endswith("_dur") or other.data.empty or not other.index_column:
            continue
        id_cols = list(other.id_columns)
        time_col = time_col or other.index_column
        values = other.data[name]
        positive = values.eq(True) if name == "other_vaso" else _numeric(values).gt(0)
        active = other.data.loc[positive, [*id_cols, other.index_column]].copy()
        clock = active.pop(other.index_column)
        active["__hour"] = (np.floor(_numeric(clock).to_numpy("float64", na_value=np.nan))
                            if pd.api.types.is_numeric_dtype(clock) else clock.dt.floor("h"))
        active["vaso_ind"] = True
        pieces.append(active)
    if not pieces:
        return base if base is not None else ICUTable(
            pd.DataFrame(columns=[*id_cols, time_col or "charttime", "vaso_ind"]),
            id_columns=id_cols, index_column=time_col or "charttime", value_column="vaso_ind",
        )
    active = pd.concat(pieces, ignore_index=True).drop_duplicates([*id_cols, "__hour"])
    active = active.sort_values([*id_cols, "__hour"]).rename(columns={"__hour": time_col}).reset_index(drop=True)
    return ICUTable(active, id_columns=id_cols, index_column=time_col, value_column="vaso_ind")
