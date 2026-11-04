"""Loop-diuretic furosemide-equivalent dose.

Owner of the 2026-11-03 extraction-audit #19 rules.  The concept dictionary
declares which source rows are which drug; this module owns the clinical
conversion.

Furosemide equivalence (``loop_diuretic_furosemide_eq``, mg of intravenous
furosemide).  Source: Brater DC. Diuretic therapy. N Engl J Med
1998;339:387-395 -- Table 3 lists equipotent intravenous loading doses of
furosemide 40 mg, bumetanide 1 mg and torsemide 20 mg; the text states that
"the intravenous and oral doses of bumetanide and torsemide are similar" and
that for furosemide "the usual maximal oral dose is twice the intravenous
dose" (Table 1: oral bioavailability furosemide 10-100%, bumetanide and
torsemide 80-100%).  Hence::

    IV furosemide 1 mg   -> 1 mg     oral furosemide 1 mg -> 0.5 mg
    bumetanide 1 mg      -> 40 mg    (IV or oral)
    torasemide 1 mg      -> 2 mg     (IV or oral)

The ratios are approximate: Brater's infusion-rate and maximal-dose tables
imply bumetanide 20-40:1 and torsemide 2-4:1.  Ethacrynic acid and
fixed-dose combinations whose loop-diuretic content is not recorded per row
are exposures only and never enter the dose.

Each dose source declares ``fe_drug`` (``furosemide``, ``bumetanide`` or
``torasemide``) and either ``fe_route`` (``iv`` or ``oral``) or
``fe_route_var`` plus ``fe_oral_values`` (source values meaning an enteral
administration).  Amounts are read in the source's ``unit_var`` and converted
to mg (a table without a unit column declares ``fe_amount_unit`` from its
reference dictionary); a row whose unit is not a mass unit is dropped and counted in the
frame's ``attrs['furosemide_equivalent_audit']``, never guessed.  A source may
declare ``status_var`` and ``excluded_statuses`` (MIMIC ``Rewritten``
records are superseded entries, not doses). Declared continuous amounts use
half-open ICU-hour overlap, conserving the administered mg equivalent.
MetaVision item 228340 and AUMC item 7244 join SICdb interval allocation;
boluses remain point events. SICdb's unknown route retains the documented IV
upper bound.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

FUROSEMIDE_EQUIVALENT_CALLBACK = "furosemide_equivalent"
AUDIT_ATTR = "furosemide_equivalent_audit"

#: mg IV furosemide per mg of drug, by (drug, route); see the module docstring.
FUROSEMIDE_EQUIVALENT_FACTORS: Mapping[tuple, float] = {
    ("furosemide", "iv"): 1.0,
    ("furosemide", "oral"): 0.5,
    ("bumetanide", "iv"): 40.0,
    ("bumetanide", "oral"): 40.0,
    ("torasemide", "iv"): 2.0,
    ("torasemide", "oral"): 2.0,
}

#: mg per unit, after lower-casing and trimming the source unit.
_MASS_TO_MG: Mapping[str, float] = {
    "mg": 1.0,
    "g": 1000.0,
    "gram": 1000.0,
    "grams": 1000.0,
    "mcg": 0.001,
    "ug": 0.001,
    "µg": 0.001,
    "microgram": 0.001,
}

class DiureticSourceContractError(ValueError):
    """A diuretic source declaration cannot be evaluated."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _value_column(frame: pd.DataFrame, concept_name: str, value_var: Optional[str]) -> str:
    if concept_name in frame.columns:
        return concept_name
    if value_var and value_var in frame.columns:
        return value_var
    raise DiureticSourceContractError("diuretic_value_column_missing", str(value_var))


def _drop_excluded_statuses(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.DataFrame:
    excluded = params.get("excluded_statuses")
    if not excluded:
        return frame
    status_var = str(params.get("status_var") or "statusdescription")
    if status_var not in frame.columns:
        raise DiureticSourceContractError("diuretic_status_column_missing", status_var)
    statuses = frame[status_var].astype("string").str.strip()
    return frame.loc[~statuses.isin([str(s) for s in excluded]).fillna(False).to_numpy()]


def _route_series(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    route = params.get("fe_route")
    if route is not None:
        route = str(route)
        if route not in ("iv", "oral"):
            raise DiureticSourceContractError("diuretic_route_unknown", route)
        return pd.Series(route, index=frame.index, dtype="object")
    route_var = params.get("fe_route_var")
    oral_values: Iterable[object] = params.get("fe_oral_values") or ()
    if not route_var:
        raise DiureticSourceContractError("diuretic_route_undeclared")
    if route_var not in frame.columns:
        raise DiureticSourceContractError("diuretic_route_column_missing", str(route_var))
    oral = frame[route_var].astype("string").str.strip().isin([str(v) for v in oral_values])
    return pd.Series(oral.fillna(False).map({True: "oral", False: "iv"}).to_numpy(), index=frame.index)


def _spread_continuous_amounts(frame: pd.DataFrame, concept_name: str, params: Mapping[str, object], data_source=None) -> pd.DataFrame:
    """Allocate declared interval totals on the admission-anchored hour grid.

    SICdb keeps its source-specific single-dose flag and ICUOffset adapter.
    Other sources use the existing bounded-memory overlap allocator. Optional
    item selection keeps MetaVision bolus item 221794 at its administration
    hour, even when its order carries a nominal end time.
    """
    end_var = params.get("fe_spread_end_var")
    if not end_var or frame.empty:
        return frame
    index_var = str(params.get("fe_spread_index_var") or "")
    single_var = params.get("fe_spread_single_var")
    for column in (index_var, str(end_var)) + ((str(single_var),) if single_var else ()):
        if column not in frame.columns:
            raise DiureticSourceContractError("diuretic_spread_column_missing", column)
    if params.get("fe_icu_clock"):
        from .sicdb_sources import allocate_amounts

        return allocate_amounts(
            frame, concept_name=concept_name, start=index_var, stop=str(end_var),
            single=str(single_var), data_source=data_source,
        )
    from ..utils.callback_utils import distribute_volume_hourly

    work = frame.copy()
    item_ids = params.get("fe_spread_item_ids")
    if item_ids:
        item_var = str(params.get("fe_spread_item_var") or "itemid")
        if item_var not in work:
            raise DiureticSourceContractError("diuretic_spread_column_missing", item_var)
        work[str(end_var)] = work[str(end_var)].where(work[item_var].isin(item_ids))
    id_col = next((c for c in ("stay_id", "icustay_id", "admissionid") if c in work), None)
    if id_col is None:
        raise DiureticSourceContractError("diuretic_spread_identifier_missing")
    aumc = id_col == "admissionid"
    datetime_clock = pd.api.types.is_datetime64_any_dtype(work[index_var])
    origins, origin_col = None, None
    if aumc:
        from .aumc_intervals import admission_origins

        origins, origin_col = admission_origins(work, data_source), "admittedat"
    elif datetime_clock:
        if data_source is None:
            raise DiureticSourceContractError("diuretic_spread_admission_missing")
        table = data_source.load_table("icustays", columns=[id_col, "intime"], verbose=False)
        origins = table.data if hasattr(table, "data") else table
        origin_col = "intime"
    result = distribute_volume_hourly(
        work, val_col=concept_name, index_col=index_var, end_col=str(end_var),
        id_col=id_col, origin_times=origins, origin_col=origin_col,
        numeric_time_unit="minutes" if aumc else "hours",
        output_time_unit="absolute_minutes" if aumc else "relative_hours",
        aggregate=False,
    )
    if datetime_clock:
        lookup = origins.drop_duplicates(id_col).set_index(id_col)[origin_col]
        origin = pd.to_datetime(result[id_col].map(lookup), utc=True)
        result[index_var] = (origin + pd.to_timedelta(result[index_var], unit="h")).dt.tz_localize(None)
    return result


def furosemide_equivalent(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    value_var: Optional[str],
    unit_var: Optional[str],
    params: Mapping[str, object],
    data_source=None,
    maximum: Optional[float] = None,
) -> pd.DataFrame:
    """Convert one source's administered amounts to mg IV furosemide."""

    drug = str(params.get("fe_drug") or "")
    if drug not in {d for d, _ in FUROSEMIDE_EQUIVALENT_FACTORS}:
        raise DiureticSourceContractError("diuretic_drug_unknown", drug)
    if frame.empty:
        return frame.assign(**{concept_name: pd.Series(dtype="float64")})
    fixed_unit = params.get("fe_amount_unit")
    if fixed_unit is None and (not unit_var or unit_var not in frame.columns):
        raise DiureticSourceContractError("diuretic_unit_column_missing", str(unit_var))
    frame = _drop_excluded_statuses(frame, params)
    value_col = _value_column(frame, concept_name, value_var)
    amount = pd.to_numeric(frame[value_col], errors="coerce")
    if fixed_unit is not None:
        # A table without a unit column (SICdb medication) declares the unit
        # its own reference dictionary gives for the drug.
        units = pd.Series(str(fixed_unit).strip().lower(), index=frame.index, dtype="string")
    else:
        units = frame[unit_var].astype("string").str.strip().str.lower()
    to_mg = units.map(_MASS_TO_MG).astype("float64")
    routes = _route_series(frame, params)
    factor = routes.map(lambda r: FUROSEMIDE_EQUIVALENT_FACTORS[(drug, r)]).astype("float64")
    mg_eq = amount * to_mg * factor
    # A zero amount (HiRID pump start/stop marker) is no dose (2026-11-04
    # audit round 3, M R2-HIRID-42).
    keep = mg_eq.notna() & (mg_eq > 0)
    if maximum is not None and params.get("fe_spread_end_var") and not params.get("fe_icu_clock"):
        # Preserve the existing original-dose bound before the new adapters
        # split an interval into smaller amounts. Hourly sums have no such cap.
        keep &= mg_eq.le(maximum)
    audit = {
        "drug": drug,
        "rows_in": int(len(frame)),
        "rows_unit_not_mass": int(to_mg.isna().sum()),
        "rows_amount_missing": int(amount.isna().sum()),
        "rows_kept": int(keep.sum()),
    }
    out = frame.loc[keep.to_numpy()].copy()
    out[concept_name] = mg_eq.loc[keep].to_numpy()
    out = _spread_continuous_amounts(out, concept_name, params, data_source)
    out.attrs[AUDIT_ATTR] = audit
    return out


__all__ = [
    "AUDIT_ATTR",
    "DiureticSourceContractError",
    "FUROSEMIDE_EQUIVALENT_CALLBACK",
    "FUROSEMIDE_EQUIVALENT_FACTORS",
    "furosemide_equivalent",
]
