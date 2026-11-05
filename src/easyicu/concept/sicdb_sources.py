"""SICdb source rules on the ICU clock, with inspectable row-level receipts.

Offsets are primary-admission seconds. Every explicit interval uses positive
overlap with ICU hours [h, h+1); exact stops never create another observation.
The coverage receipt retains each input row and its actual covered seconds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .immutable_receipts import freeze_sic_attrs

HOUR = 3600.
COVERAGE_ATTR = "sicdb_interval_coverage"


def preserves_raw_cvp(definition, database):
    """Only the declared SIC raw channel bypasses analytical CVP bounds."""
    return (
        database in {"sic", "sic_demo"} and definition is not None
        and definition.name == "cvp_raw"
        and any(s.callback == "sicdb_cvp_raw" for s in definition.sources.get("sic", []))
    )


def source_definition(definition, database):
    """Keep SIC raw values on direct and batched resolver entry points."""
    return replace(definition, minimum=None, maximum=None) if preserves_raw_cvp(definition, database) else definition


def restore_source_receipts(combined, frames):
    """Preserve compatible ICU clocks and interval receipts across source concat.

    Pandas discards all attrs when any source receipt differs. Only the small
    list of source frames is traversed here, never individual observations.
    """
    declared = [f.attrs.get("easyicu_time_columns", {}) for f in frames]
    clocks = {column: unit for mapping in declared for column, unit in mapping.items()}
    combined.attrs["easyicu_time_columns"] = {
        column: unit for column, unit in clocks.items()
        if all(f.empty or column not in f or m.get(column) == unit for f, m in zip(frames, declared))
    }
    receipts = []
    for index, frame in enumerate(frames):
        if COVERAGE_ATTR in frame.attrs:
            receipt = dict(frame.attrs[COVERAGE_ATTR])
            receipt["source_index"] = [index] * len(receipt["hour"])
            receipts.append(receipt)
    if receipts:
        combined.attrs[COVERAGE_ATTR] = {key: sum((r[key] for r in receipts), []) for key in receipts[0]}
    return freeze_sic_attrs(combined)


def _table(data_source, name, columns, filters=None):
    table = data_source.load_table(name, columns=columns, filters=filters, verbose=False)
    return table.data if hasattr(table, "data") else table


def icu_seconds(frame, data_source, columns):
    """Subtract the declared ICU origin, refusing missing or ambiguous anchors."""
    if frame.empty:
        return frame.copy()
    if data_source is None:
        raise ValueError("SICdb intervals require cases.ICUOffset")
    cases = _table(data_source, "cases", ["CaseID", "ICUOffset"])
    out = frame.drop(columns="ICUOffset", errors="ignore").merge(
        cases, on="CaseID", how="left", validate="many_to_one", sort=False,
    )
    origin = pd.to_numeric(out.ICUOffset, errors="coerce")
    if not np.isfinite(origin).all():
        raise ValueError("SICdb intervals require finite cases.ICUOffset for every case")
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce") - origin
    return out.drop(columns="ICUOffset")


def materialize_intervals(frame, *, start="Offset", stop="OffsetDrugEnd", data_source=None, relative=False):
    """Return hourly observations and per-interval overlap seconds, without aggregation."""
    raw = frame.copy() if relative else icu_seconds(frame, data_source, [start, stop])
    starts = pd.to_numeric(raw[start], errors="coerce").to_numpy(dtype=float)
    stops = pd.to_numeric(raw[stop], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(starts) & np.isfinite(stops) & (stops > starts)
    rows = np.flatnonzero(valid)
    first = np.floor(starts[valid] / HOUR).astype(np.int64)
    last = np.ceil(stops[valid] / HOUR).astype(np.int64)
    counts = last - first
    repeated = np.repeat(rows, counts)
    offsets = np.arange(len(repeated)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts) if len(rows) else np.array([], dtype=int)
    hours = np.repeat(first, counts) + offsets
    overlap_start = np.maximum(starts[repeated], hours * HOUR)
    overlap_stop = np.minimum(stops[repeated], (hours + 1) * HOUR)
    out = raw.iloc[repeated].copy().reset_index(drop=True)
    out[start] = hours.astype(float)
    out = out.drop(columns=[stop, "dur_var"], errors="ignore")
    out.attrs["easyicu_time_columns"] = {start: "icu_relative_hours"}
    out.attrs[COVERAGE_ATTR] = {
        "CaseID": out.CaseID.tolist(),
        "interval_row": repeated.tolist(),
        "interval_start_seconds": starts[repeated].tolist(),
        "interval_stop_seconds": stops[repeated].tolist(),
        "hour": hours.tolist(),
        "overlap_seconds": (overlap_stop - overlap_start).tolist(),
        "overlap_start_seconds": overlap_start.tolist(),
        "overlap_stop_seconds": overlap_stop.tolist(),
    }
    return freeze_sic_attrs(out)


def allocate_amounts(frame, *, concept_name, start, stop, single, data_source):
    """Allocate continuous amounts to ICU hours; preserve bolus event times."""
    raw = icu_seconds(frame, data_source, [start, stop])
    flags = pd.to_numeric(raw[single], errors="coerce")
    continuous = raw.loc[flags.eq(0)].copy()
    duration = pd.to_numeric(continuous[stop], errors="coerce") - continuous[start]
    continuous = continuous.loc[duration.gt(0) & np.isfinite(duration)].copy()
    continuous["_sic_duration"] = duration.loc[continuous.index]
    expanded = materialize_intervals(continuous, start=start, stop=stop, relative=True)
    coverage = expanded.attrs[COVERAGE_ATTR]
    expanded[concept_name] = expanded[concept_name] * np.asarray(coverage["overlap_seconds"]) / expanded["_sic_duration"]
    bolus = raw.loc[flags.eq(1)].copy()
    bolus[start] = bolus[start] / HOUR
    bolus = bolus.drop(columns=[stop, "dur_var"], errors="ignore")
    out = pd.concat([expanded, bolus], ignore_index=True).drop(columns="_sic_duration", errors="ignore")
    out.attrs["easyicu_time_columns"] = {start: "icu_relative_hours"}
    out.attrs[COVERAGE_ATTR] = coverage
    return out


def exposure(frame, value):
    """Observed zero is false, positive amount true, missing/negative unknown."""
    out = frame.copy()
    amount = pd.to_numeric(out[value], errors="coerce")
    out[value] = amount.gt(0).where(amount.ge(0) & np.isfinite(amount)).astype("boolean")
    return out


def amount_rate(frame, *, concept_name, source, data_source, factor):
    """Compute a continuous rate from an administered total and elapsed seconds."""
    value = concept_name if concept_name in frame else source.value_var
    start = source.index_var or "Offset"
    stop = source.params.get("stop_var", "OffsetDrugEnd")
    amount = pd.to_numeric(frame[value], errors="coerce")
    duration = pd.to_numeric(frame[stop], errors="coerce") - pd.to_numeric(frame[start], errors="coerce")
    if "IsSingleDose" in frame:
        continuous = pd.to_numeric(frame["IsSingleDose"], errors="coerce").eq(0)
    elif source.params.get("sic_continuous_filtered"):
        continuous = pd.Series(True, index=frame.index)
    else:
        raise ValueError("SICdb amount rates require IsSingleDose continuous evidence")
    valid = continuous & amount.gt(0) & duration.gt(0) & np.isfinite(amount) & np.isfinite(duration)
    out = frame.loc[valid].copy()
    out[concept_name] = amount.loc[valid] * factor / duration.loc[valid] * HOUR
    return materialize_intervals(out, start=start, stop=stop, data_source=data_source)


def _glucose_factors(frame, data_source):
    """Read amount units from the reference table; never infer them from values."""
    from ..datasource import FilterOp, FilterSpec
    from .errors import ConceptExtractionUnavailable

    ids = pd.unique(frame.DrugID).tolist()
    try:
        refs = _table(data_source, "d_references", ["ReferenceGlobalID", "ReferenceUnit", "ReferenceName", "ReferenceValue"], [FilterSpec("ReferenceGlobalID", FilterOp.IN, ids)])
        refs = refs.drop_duplicates().set_index("ReferenceGlobalID")
        if not refs.index.is_unique:
            raise ValueError("ambiguous reference units")
        unit = frame.DrugID.map(refs.ReferenceUnit).astype("string").str.strip().str.lower()
        # SICdb names the preparation in ReferenceValue ("Glukose 10%"); ReferenceName is its class ("Fluid").
        pattern = r"(?i)^\s*(?:glukose|glucose|dextrose)\s*(\d+(?:[.,]\d+)?)\s*%"
        percentage = refs.ReferenceValue.astype("string").str.extract(pattern, expand=False)
        percentage = percentage.fillna(refs.ReferenceName.astype("string").str.extract(pattern, expand=False))
        parsed = pd.to_numeric(percentage.str.replace(",", ".", regex=False), errors="coerce") / 100.
        concentration = frame.DrugID.map({2045: .33, 2148: .50})
        concentration = concentration.fillna(frame.DrugID.map(parsed))
        if not concentration.between(0, 1, inclusive="right").fillna(False).all():
            raise ValueError("unknown glucose concentration")
        factors = unit.map({"g": 10., "ml": 1.}).astype(float)
        factors = factors.where(unit.ne("ml"), concentration / .1)
        if factors.isna().any():
            raise ValueError("unknown glucose amount unit or concentration")
        return factors
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ConceptExtractionUnavailable(concept_id="dex", database="sic", stage="glucose_reference_unit", detail="SICdb glucose needs an unambiguous reference amount unit and known concentration") from exc


def precondition(frame, *, value, concept_name, data_source):
    """Decode a case precondition from reference labels, retaining unknowns.

    Reference identifiers are release-specific. Conflicting reference labels
    or conflicting Yes/No records for a case cannot establish its history.
    """
    from ..datasource import FilterOp, FilterSpec
    from .errors import ConceptExtractionUnavailable

    out = frame[["CaseID"]].copy()
    out[concept_name] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if frame.empty:
        return out
    if "FieldID" not in frame:
        raise ConceptExtractionUnavailable(
            concept_id=concept_name, database="sic", stage="precondition_field",
            detail="SICdb preconditions require data_ref.FieldID",
        )
    ids = pd.to_numeric(frame[value], errors="coerce")
    try:
        refs = _table(data_source, "d_references", ["ReferenceGlobalID", "ReferenceValue"],
                      [FilterSpec("ReferenceGlobalID", FilterOp.IN, ids.dropna().unique().tolist())])
        flags = refs.ReferenceValue.astype("string").str.strip().str.casefold().map({"yes": True, "no": False})
        lookup = pd.DataFrame({"reference": pd.to_numeric(refs.ReferenceGlobalID, errors="coerce"),
                               "flag": flags.astype("boolean")}).drop_duplicates()
        lookup = lookup.loc[~lookup.reference.duplicated(keep=False)].set_index("reference").flag
        out[concept_name] = ids.map(lookup).astype("boolean")
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
        raise ConceptExtractionUnavailable(
            concept_id=concept_name, database="sic", stage="precondition_reference",
            detail="SICdb preconditions require d_references Yes/No labels",
        ) from exc
    contradictory = out.groupby("CaseID")[concept_name].transform("nunique").gt(1)
    out[concept_name] = out[concept_name].mask(contradictory)
    return out.groupby("CaseID", as_index=False)[concept_name].first()


def apply_source(frame, *, source, concept_name, data_source):
    """Dispatch only explicitly declared SICdb source callbacks."""
    value = concept_name if concept_name in frame else source.value_var
    callback = source.callback
    if callback == "sicdb_precondition":
        return precondition(frame, value=value, concept_name=concept_name, data_source=data_source)
    if callback == "sicdb_cvp_raw":
        return frame
    if callback in {"sicdb_exposure", "sicdb_interval_exposure"}:
        out = exposure(frame, value)
        if callback == "sicdb_exposure":
            return out
        out = out.loc[out[value].fillna(False)].copy()
        return materialize_intervals(out, start=source.index_var or "Offset", stop=source.params.get("stop_var", "OffsetDrugEnd"), data_source=data_source)
    if callback in {"sicdb_airway_interval", "sicdb_airway_indicator", "sicdb_airway_rule"}:
        out = frame.copy()
        out[concept_name] = {"sicdb_airway_interval": "invasive", "sicdb_airway_indicator": True, "sicdb_airway_rule": "explicit_ett"}[callback]
        out = materialize_intervals(out, start=source.index_var or "Offset", stop="OffsetEnd", data_source=data_source)
        out.attrs["sicdb_ventilation_rule"] = "explicit_airway"
        return out
    if callback == "sicdb_amount_rate":
        factor = float(source.params["sic_amount_factor"])
        return amount_rate(frame, concept_name=concept_name, source=source, data_source=data_source, factor=factor)
    if callback == "sicdb_glucose_rate":
        out = frame.copy()
        # Zero/bolus rows need no unit lookup and cannot become an infusion.
        positive = pd.to_numeric(out[value], errors="coerce").gt(0)
        out = out.loc[positive].copy()
        if not out.empty:
            out[value] = pd.to_numeric(out[value], errors="coerce") * _glucose_factors(out, data_source)
        return amount_rate(out, concept_name=concept_name, source=source, data_source=data_source, factor=1.)
    if callback == "sicdb_minute_volume":
        out = frame.copy()
        raw = pd.to_numeric(out[value], errors="coerce")
        out[value] = raw.where(raw.ge(1))
        out.attrs["sicdb_minute_volume_qc"] = {"raw": raw.tolist(), "reason": _minute_volume_reasons(raw).tolist()}
        return out
    if callback == "sicdb_minute_volume_qc":
        start = source.index_var or "Offset"
        out = icu_seconds(frame, data_source, [start])
        out[start] = np.floor(out[start] / HOUR)
        raw = pd.to_numeric(out[value], errors="coerce")
        flags = pd.get_dummies(pd.Series(_minute_volume_reasons(raw), index=out.index), dtype=bool)
        flags = pd.concat([out[["CaseID", start]], flags], axis=1).groupby(["CaseID", start]).any()
        labels = pd.Series("", index=flags.index, dtype="string")
        # There are only four fixed reason codes; retain every source flag in
        # an hour without per-observation Python callbacks or arbitrary first.
        for reason in sorted(flags.columns):
            labels = labels.mask(flags[reason], labels + "|" + reason)
        out = labels.str.lstrip("|").rename(concept_name).reset_index()
        out.attrs["easyicu_time_columns"] = {start: "icu_relative_hours"}
        return out
    if callback == "sicdb_cvp_qc":
        from .callbacks import ConceptCallbackContext
        from ..table import ICUTable

        raw = icu_seconds(frame, data_source, [source.index_var or "Offset"])
        raw["Offset"] = np.floor(raw[source.index_var or "Offset"] / HOUR)
        raw = raw.rename(columns={value: "cvp_raw"})
        raw = raw.groupby(["CaseID", "Offset"], as_index=False).cvp_raw.median()
        table = ICUTable(raw, id_columns=["CaseID"], index_column="Offset", value_column="cvp_raw")
        ctx = ConceptCallbackContext(concept_name="cvp", target=None, interval=pd.Timedelta(hours=1), resolver=None, data_source=data_source, patient_ids=None)
        qc = filter_cvp({"cvp_raw": table}, ctx).data.attrs["sicdb_cvp_qc"]
        out = pd.DataFrame({"CaseID": qc["CaseID"], "Offset": qc["Offset"], concept_name: qc["reason"]})
        out.attrs["easyicu_time_columns"] = {"Offset": "icu_relative_hours"}
        return out
    if callback == "sicdb_ventilation_audit":
        from .sicdb_ventilation import apply_sicdb_attributed_ventilation

        main = apply_sicdb_attributed_ventilation(frame, concept_name="mech_vent", data_source=data_source)
        if source.params["sic_audit_kind"] == "opioid_only":
            rows = main.attrs.get("sicdb_opioid_only_hours", [])
            out = pd.DataFrame(rows, columns=["CaseID", "_h", "sicdb_ventilation_rule"]).rename(columns={"_h": "Offset"})
            out[concept_name] = True
            out = out[["CaseID", "Offset", concept_name]]
        else:
            out = pd.DataFrame(main.attrs.get("sicdb_ventilation_attribution", []), columns=["CaseID", "_h", "sicdb_ventilation_rule"])
            out = out.rename(columns={"_h": "Offset", "sicdb_ventilation_rule": concept_name})
        out.attrs["easyicu_time_columns"] = {"Offset": "icu_relative_hours"}
        return out
    if callback == "sicdb_labelled_value":
        out = frame.copy()
        label = str(source.params["source_label"])
        out.attrs["sicdb_source_labels"] = out[label].tolist()
        return out
    raise ValueError(f"Unknown SICdb source callback: {callback}")


def _minute_volume_reasons(raw):
    return np.select([raw.isna(), raw.lt(1), raw.gt(50)], ["missing", "below_analysis_min_1", "above_analysis_max_50"], default="kept")


def filter_cvp(tables, ctx):
    """Supply SICdb's invasive arterial item to the shared CVP/QC rule."""
    from ..datasource import FilterOp, FilterSpec
    from .callbacks import _cvp_quality_result, _merge_tables

    data, ids, time = _merge_tables(tables, ctx=ctx, how="outer")
    if data.empty or "cvp_raw" not in data:
        return _cvp_quality_result(data, ids, time, ctx, pd.Series(dtype=float))
    arterial = pd.Series(np.nan, index=data.index, dtype=float)
    try:
        refs = _table(ctx.data_source, "data_float_h", ["CaseID", "DataID", "Offset", "Val"], [FilterSpec("DataID", FilterOp.IN, [702]), FilterSpec("CaseID", FilterOp.IN, pd.unique(data.CaseID).tolist())])
        refs = icu_seconds(refs.loc[refs.DataID.eq(702)], ctx.data_source, ["Offset"])
        refs["_hour"] = np.floor(refs.Offset / HOUR)
        values = pd.to_numeric(refs.Val, errors="coerce")
        refs["Val"] = values.where(values.between(1, 200))
        lookup = refs.groupby(["CaseID", "_hour"]).Val.median()
        arterial = pd.Series(pd.MultiIndex.from_arrays([data.CaseID, np.floor(pd.to_numeric(data[time], errors="coerce"))]).map(lookup).to_numpy(), index=data.index, dtype=float)
    except (KeyError, ValueError, TypeError, AttributeError, OSError):
        pass  # Missing evidence is reported as QC; R2 still retains 30--50.
    dbp = pd.to_numeric(data.get("dbp", pd.Series(np.nan, index=data.index)), errors="coerce")
    return _cvp_quality_result(data, ids, time, ctx, arterial, arterial.isna() & dbp.notna())
