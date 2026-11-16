"""Native delivered-fluid records selected by the existing fluid_io sources.

No item codes live here. The typed accessor retains each source's exact
membership, status exclusions and row gates. Duration never determines mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .catalog import FLUID_IO_CONCEPTS
from .minute_resolution import grid_bin_hours, grid_floor_index
from .schema import ConceptDictionary, ConceptSource

_CLASSES = {
    "crystalloid_input_ml": "crystalloid", "colloid_input_ml": "colloid",
    "packed_rbc_input_ml": "blood_rbc", "ffp_input_ml": "blood_ffp",
    "platelet_input_ml": "blood_platelet", "parenteral_input_ml": "parenteral",
}
_CLOCKS = {
    "aumc": ("admissionid", "start", "stop", "admittedat", 60.),
    "miiv": ("stay_id", "starttime", "endtime", "intime", None),
    "sic": ("CaseID", "Offset", "OffsetDrugEnd", "ICUOffset", 3600.),
}


@dataclass(frozen=True)
class FluidAdministrationSource:
    concept: str
    fluid_class: str
    source: ConceptSource


def fluid_sources(dictionary: ConceptDictionary, database: str) -> tuple[FluidAdministrationSource, ...]:
    """Read exact fluid_io membership; HiRID counters cannot identify events."""
    if database not in _CLOCKS:
        return ()
    specs = []
    seen = set()
    for concept in FLUID_IO_CONCEPTS:
        if concept not in _CLASSES:
            continue
        for source in dictionary[concept].sources.get(database, ()):
            if not source.ids or source.regex or source.callback not in {"distribute_volume_hourly", "sicdb_volume_allocate"}:
                raise ValueError("native fluid events require exact delivered-volume sources")
            for item in source.ids:
                key = (source.table, source.sub_var, item)
                if key in seen:
                    raise ValueError("overlapping fluid_io administration sources")
                seen.add(key)
            specs.append(FluidAdministrationSource(concept, _CLASSES[concept], source))
    return tuple(specs)


def fluid_source_receipt(dictionary: ConceptDictionary, database: str) -> dict:
    """Bind the reused membership and gates without maintaining another list."""
    specs = fluid_sources(dictionary, database)
    declarations = [{"concept": s.concept, "fluid_class": s.fluid_class, "source": asdict(s.source)}
                    for s in specs]
    digest = hashlib.sha256(json.dumps(declarations, sort_keys=True, separators=(",", ":"),
                                       default=str).encode()).hexdigest()
    return {"source_membership": "fluid_io", "source_declarations_sha256": digest,
            "source_concepts": sorted({s.concept for s in specs}),
            "source_tables": sorted({s.source.table for s in specs}),
            "additional_gates": ["exclude_coded_flush", "exclude_coded_not_iv"] if database == "aumc" else []}


def administration_events(frame: pd.DataFrame, spec: FluidAdministrationSource, *,
                          database: str, origins: pd.DataFrame) -> pd.DataFrame:
    """Normalize source-table clocks/amounts without pooling native records.

    AUMC input is the datasource's exact absolute-minute clock, MIIV is
    datetime, SICdb seconds. Floor only after subtracting the stay origin.
    SICdb single doses have a documented artificial interval and end at their
    start. Other records retain their native end, including timed IV boluses.
    """
    from .callback_apply import _drop_declared_row_exclusions, _drop_excluded_statuses
    from ..utils.callback_utils import normalize_volume_to_ml

    source = spec.source
    raw = frame.loc[frame[source.sub_var].isin(source.ids)].copy()
    raw = _drop_excluded_statuses(raw, source)
    raw = _drop_declared_row_exclusions(raw, source, spec.concept)
    if database == "aumc":
        for column, pattern in (("action", r"(?i)^\s*flush\s*$"), ("ordercategory", r"(?i)^\s*niet iv")):
            if column in raw:
                raw = raw.loc[~raw[column].astype("string").str.contains(pattern, na=False)]
    ident, start, stop, origin, scale = _CLOCKS[database]
    lookup = origins[[ident, origin]].drop_duplicates()
    if lookup[ident].duplicated().any():
        raise ValueError("native fluid events require one origin per stay")
    anchor = raw[ident].map(lookup.set_index(ident)[origin])
    if scale is None:
        anchor = pd.to_datetime(anchor, errors="coerce", utc=True)
        begin = (pd.to_datetime(raw[start], errors="coerce", utc=True) - anchor).dt.total_seconds() / 3600
        finish = (pd.to_datetime(raw[stop], errors="coerce", utc=True) - anchor).dt.total_seconds() / 3600
    else:
        anchor = pd.to_numeric(anchor, errors="coerce")
        begin = (pd.to_numeric(raw[start], errors="coerce") - anchor) / scale
        finish = (pd.to_numeric(raw[stop], errors="coerce") - anchor) / scale
    amount = pd.to_numeric(raw[source.value_var], errors="coerce")
    if database == "miiv":
        amount = normalize_volume_to_ml(amount, raw[source.unit_var])
    # AUMC fluidin is documented mL; SICdb's selected Fluid references are mL.
    mode = pd.Series("unknown", index=raw.index, dtype="string")
    evidence = pd.Series("unrecognized_or_missing_mode", index=raw.index, dtype="string")
    if database == "miiv":
        labels = raw.get("ordercategorydescription", pd.Series(pd.NA, index=raw.index, dtype="string"))
        modes = labels.map({"Continuous IV": "continuous", "Continuous Med": "continuous",
                           "Bolus": "bolus", "Drug Push": "bolus"})
        mode = modes.fillna("unknown").astype("string")
        # Unknown labels are not copied: evidence is a closed coded vocabulary.
        evidence = labels.where(modes.notna(), evidence).astype("string")
    elif database == "aumc":
        flag = pd.to_numeric(raw.get("iscontinuous", pd.Series(np.nan, index=raw.index)), errors="coerce")
        mode.loc[flag.eq(1)] = "continuous"
        evidence.loc[flag.eq(1)] = "iscontinuous=1"
        # False means no restart on an empty bag, not necessarily a bolus.
        evidence.loc[flag.eq(0)] = "iscontinuous=0"
    else:
        flag = pd.to_numeric(raw["IsSingleDose"], errors="coerce")
        mode = flag.map({0: "continuous", 1: "bolus"}).fillna("unknown").astype("string")
        evidence.loc[flag.eq(0)] = "IsSingleDose=0"
        evidence.loc[flag.eq(1)] = "IsSingleDose=1"
        finish = finish.mask(flag.eq(1), begin)
    valid = raw[ident].notna() & np.isfinite(begin) & np.isfinite(amount) & amount.gt(0)
    valid &= finish.isna() | (np.isfinite(finish) & finish.ge(begin))
    if database == "sic":
        # Same eligibility as sicdb_volume_allocate, including malformed infusions.
        valid &= flag.eq(1) | (flag.eq(0) & finish.gt(begin))
    finish = finish.fillna(begin)
    grid = pd.Timedelta(minutes=1)
    step = 1 / 60
    return pd.DataFrame({
        "stay_id": raw.loc[valid, ident].astype("int64"),
        "start": grid_bin_hours(grid_floor_index(begin.loc[valid], step), grid),
        "end": grid_bin_hours(grid_floor_index(finish.loc[valid], step), grid),
        "volume_ml": amount.loc[valid], "fluid_class": spec.fluid_class,
        "mode": mode.loc[valid], "mode_evidence": evidence.loc[valid],
    }).reset_index(drop=True)


def load_event_batch(data_source, *, database: str, patient_ids: dict) -> pd.DataFrame:
    """One stay-disjoint batch through the normal storage/binding owner."""
    from ..datasource import FilterSpec, FilterOp
    from ..resources import load_dictionary
    from .native_events import NATIVE_EVENT_MODULES

    specs = fluid_sources(load_dictionary(), database)
    event_table = NATIVE_EVENT_MODULES["fluid_events"]
    if not specs:
        return event_table.empty()
    ident, start, stop, origin, _ = _CLOCKS[database]
    filters = [FilterSpec(ident, FilterOp.IN, patient_ids[ident])]
    origin_table = {"aumc": "admissions", "miiv": "icustays", "sic": "cases"}[database]
    origins = data_source.load_table(origin_table, columns=[ident, origin], filters=filters, verbose=False).data
    columns = {ident, start, stop}
    for spec in specs:
        source = spec.source
        columns.update(c for c in (source.sub_var, source.value_var, source.unit_var) if c)
        columns.update(source.params.get("extra_vars", ()))
    columns.update({"aumc": ["iscontinuous", "action", "ordercategory"],
                    "miiv": ["ordercategorydescription"], "sic": ["IsSingleDose"]}[database])
    sub_var = specs[0].source.sub_var
    filters.append(FilterSpec(sub_var, FilterOp.IN, sorted({i for s in specs for i in s.source.ids})))
    raw = data_source.load_table(specs[0].source.table, columns=sorted(columns), filters=filters, verbose=False).data
    pieces = [administration_events(raw, spec, database=database, origins=origins) for spec in specs]
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sort_values(list(frame.columns), kind="stable").reset_index(drop=True)
    frame["event_index"] = frame.groupby("stay_id", sort=False).cumcount()
    return event_table.canonical(frame)
