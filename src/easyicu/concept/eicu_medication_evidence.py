"""eICU route, order and charted-infusion evidence stay distinct.

Medication rows are orders. Only a matched, quantitatively consistent D50
infusion can corroborate an ordered dextrose dose. Loop-diuretic amounts come
from charted mass rates with an explicit IV order identifying the route, never
from the ordered quantity. No concentrations or unknown routes are guessed.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .diuretics import FUROSEMIDE_EQUIVALENT_FACTORS
from .eicu_source_qc import STAY, load_optional

_NONIV = re.compile(
    r"\b(?:po|oral|per\s+os|sl|sublingual|buccal|ngt?|ogt?|ft|peg|pegtube|"
    r"[gj]-?tube|gt|tube|gastric|enteral|nasogastric|sc|sq|subq|subcut\w*|"
    r"im|intramuscular|pr|rectal|intranasal|nasal|inh|inhal\w*|nebul\w*|"
    r"transdermal|patch|topical|epidural|intrathecal|intraosseous|io)\b", re.I,
)
_IV = re.compile(r"\b(?:iv(?:pb|p)?|intravenous)\b", re.I)


def normalize_routes(values):
    """Bounded aliases; a compound non-IV route cannot qualify as IV."""
    text = values.astype("string").fillna("").str.strip().str.lower()
    noniv = text.str.contains(_NONIV, na=False)
    has_iv = text.str.contains(_IV, na=False)
    rest = text.str.replace(_IV, "", regex=True).str.replace(r"\b(?:push|piggyback|infusion|drip|bolus|continuous)\b", "", regex=True)
    rest = rest.str.replace(r"[\s/,+;().-]", "", regex=True)
    return pd.Series(np.select([noniv, has_iv & rest.eq("")], ["noniv", "iv"], default="unknown"), index=values.index)


_DOSE_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?"
# eICU dosage text is "<amount> <unit>", where the unit is a word ("81 mg") or
# a numeric unit code ("81 3"); "1,000 mg" uses thousands separators and PRN
# orders give ranges ("2-4 mg"). A decimal comma ("0,5") stays unparsed.
_DOSE_TEXT = (
    rf"^\s*(?P<sign>[+-]?)(?P<low>{_DOSE_NUMBER})"
    rf"(?:\s*-\s*(?P<high>{_DOSE_NUMBER}))?(?=\s*(?:[a-zA-Zµμ%]|$)|\s+\d)"
)


def _positive_dose(values):
    """Read a numeric dose prefix; zero is false and invalid doses are unknown.

    A range is positive only when its lower bound is; a range from zero to a
    positive amount (a sliding scale) cannot say whether a dose was given.
    """
    parts = values.astype("string").str.extract(_DOSE_TEXT)
    low, high = (
        pd.to_numeric(parts[c].str.replace(",", "", regex=False), errors="coerce").astype("float64")
        for c in ("low", "high")
    )
    low = low.where(parts["sign"].ne("-"), -low)
    unknown = ~np.isfinite(low) | low.lt(0) | (low.eq(0) & high.gt(0)) | high.lt(low)
    return low.gt(0).astype("boolean").mask(unknown)


def positive_value(frame, column):
    """Positive numeric values or medication dose text establish exposure."""
    out = frame.copy()
    out[column] = _positive_dose(frame[column])
    return out


def iv_exposure(frame, column):
    route = normalize_routes(frame["routeadmin"])
    out = frame.loc[route.ne("noniv")].copy()
    positive = _positive_dose(frame.get("dosage", frame[column]))
    out[column] = positive.loc[out.index].mask(route.loc[out.index].eq("unknown"))
    out["eicu_route_status"] = route.loc[out.index]
    out["eicu_exposure_source"] = "order_exposure"
    out.attrs["eicu_medication_audit"] = {"source": "order_exposure", "route_unknown": int(route.eq("unknown").sum()), "noniv_dropped": int(route.eq("noniv").sum())}
    return out


def positive_infusion(frame, column):
    """Positive charted rate = exposure; zero = stop; unparseable = unknown."""
    out = frame.copy()
    values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    finite = np.isfinite(values)
    out[column] = values.gt(0).astype("boolean").mask(~finite | values.lt(0))
    paused = frame[column].astype("string").str.strip().str.casefold().isin(["paused", "held", "stopped", "off"])
    out["eicu_exposure_source"] = np.select([values.gt(0) & finite, values.eq(0), paused], ["infusion_positive", "infusion_stopped", "infusion_paused"], default="infusion_unknown")
    out.attrs["eicu_medication_audit"] = {"source": "charted_infusion", "positive": int((values.gt(0) & finite).sum()), "stopped": int(values.eq(0).sum()), "unknown": int(out[column].isna().sum())}
    return out


def _intervals(infusions):
    """Raw charted rate intervals, bounded by the next row and six hours.

    Last charting has at most one hour of support. Invalid and zero rows still
    end a preceding rate; they must survive until intervals have been built.
    """
    work = infusions.copy()
    work["_family"] = _drug_names(work.drugname).fillna(work.drugname)
    work["infusionoffset"] = pd.to_numeric(work.infusionoffset, errors="coerce")
    work = work.dropna(subset=[STAY, "infusionoffset"]).sort_values([STAY, "_family", "infusionoffset"])
    work = work.drop_duplicates([STAY, "_family", "infusionoffset"], keep="last")
    nxt = work.groupby([STAY, "_family"]).infusionoffset.shift(-1)
    work["_end"] = np.minimum(nxt.fillna(work.infusionoffset + 60), work.infusionoffset + 360)
    work["_rate"] = pd.to_numeric(work.drugrate, errors="coerce")
    return work


def dex_dose(frame, column, data_source):
    """D50 as D10 mL/h, corroborated by explicit D50 infusion volume."""
    out = frame.copy()
    tokens = out[column].astype("string").str.extract(r"^\s*(\d+(?:\.\d+)?)\s*(mg|g|ml)\s*$", flags=re.I)
    volume = pd.to_numeric(tokens[0], errors="coerce") * tokens[1].str.lower().map({"g": 2.0, "mg": .002, "ml": 1.0})
    start = pd.to_numeric(out.drugstartoffset, errors="coerce")
    stop = pd.to_numeric(out.drugstopoffset, errors="coerce")
    duration = stop - start
    cancelled = ~out.drugordercancelled.astype("string").str.strip().str.lower().isin(["false", "no", "0", "n"])
    route = normalize_routes(out.routeadmin)
    valid = volume.gt(0) & duration.gt(0) & duration.le(720) & ~cancelled & route.eq("iv")
    infusions = load_optional(data_source, "infusiondrug", out[STAY].dropna().unique())
    confirmed = pd.Series(False, index=out.index)
    if not infusions.empty:
        name = infusions.drugname.astype("string")
        d50 = name.str.contains(r"\bd50\b|dextrose\s*50\s*%", case=False, na=False)
        unit = name.str.contains(r"\(\s*ml/(?:hr|hour)\s*\)$", case=False, na=False)
        rates = _intervals(infusions.loc[d50 & unit])
        for idx in out.index[valid]:
            rows = rates.loc[rates[STAY].eq(out.at[idx, STAY])]
            overlap = (np.minimum(rows._end, stop[idx]) - np.maximum(rows.infusionoffset, start[idx])).clip(lower=0)
            actual = (overlap * rows._rate.where(rows._rate.gt(0), 0) / 60).sum()
            covered = overlap.where(rows._rate.gt(0), 0).sum()
            confirmed.at[idx] = bool(np.isclose(actual, volume[idx], rtol=1e-6, atol=1e-6) and np.isclose(covered, duration[idx]))
    kept = valid & confirmed
    out = out.loc[kept].copy()
    pieces = []
    for idx, row in out.iterrows():
        # Hourly mean D10-equivalent rate. Allocate exact overlap here so a
        # generic inclusive WinTbl expansion cannot add the stop hour.
        rate = volume[idx] * 5 / (duration[idx] / 60)
        for hour in range(int(np.floor(start[idx] / 60)), int(np.ceil(stop[idx] / 60))):
            overlap = min(stop[idx], (hour + 1) * 60) - max(start[idx], hour * 60)
            piece = row.to_dict()
            piece["drugstartoffset"] = hour * 60
            piece[column] = rate * overlap / 60
            pieces.append(piece)
    result = pd.DataFrame(pieces, columns=out.columns)
    result.attrs["eicu_medication_audit"] = {"source": "order_and_charted_infusion", "orders": len(frame), "execution_confirmed": int(kept.sum()), "order_only_or_invalid": int((~kept).sum()), "hourly_pieces": len(result)}
    return result


def _drug_names(values):
    text = values.astype("string")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    for drug, pattern in (("furosemide", r"furosem|frusem|lasix"), ("bumetanide", r"bumetan|bumex|burinex"), ("torasemide", r"torasem|torsem|demadex")):
        result.loc[text.str.contains(pattern, case=False, na=False)] = drug
    return result


def loop_dose(frame, column, data_source):
    """Integrate a charted mass rate only where a known IV route overlaps."""
    work = frame.copy()
    work["drugrate"] = frame[column]
    rates = _intervals(work)
    rates["_drug"] = _drug_names(rates.drugname)
    units = rates.drugname.astype("string").str.extract(r"\(\s*(mg|mcg|g)/(hr|hour|min)\s*\)$", flags=re.I)
    rates["_mg_hr"] = rates._rate * units[0].str.lower().map({"mg": 1.0, "mcg": .001, "g": 1000.0}) * units[1].str.lower().map({"hr": 1.0, "hour": 1.0, "min": 60.0})
    orders = load_optional(data_source, "medication", rates[STAY].dropna().unique())
    pieces = []
    if not orders.empty:
        orders = orders.copy()
        orders["_drug"] = _drug_names(orders.drugname)
        cancelled = orders.drugordercancelled.astype("string").str.lower().str.strip().isin(["true", "yes", "1", "y"])
        orders["drugstartoffset"] = pd.to_numeric(orders.drugstartoffset, errors="coerce")
        orders["drugstopoffset"] = pd.to_numeric(orders.drugstopoffset, errors="coerce")
        timed = np.isfinite(orders.drugstartoffset) & np.isfinite(orders.drugstopoffset) & orders.drugstopoffset.gt(orders.drugstartoffset)
        orders = orders.loc[normalize_routes(orders.routeadmin).eq("iv") & ~cancelled & timed]
        for _, row in rates.loc[rates._mg_hr.gt(0) & rates._drug.notna()].iterrows():
            matching = orders.loc[orders[STAY].eq(row[STAY]) & orders._drug.eq(row._drug)]
            # Union matching order intervals so duplicate orders cannot duplicate dose.
            spans = sorted((max(row.infusionoffset, float(o.drugstartoffset)), min(row._end, float(o.drugstopoffset))) for o in matching.itertuples())
            used_until = row.infusionoffset
            for start, end in spans:
                start = max(start, used_until)
                if not (np.isfinite(start) and np.isfinite(end) and end > start):
                    continue
                factor = FUROSEMIDE_EQUIVALENT_FACTORS[(row._drug, "iv")]
                for hour in range(int(np.floor(start / 60)), int(np.ceil(end / 60))):
                    duration = min(end, (hour + 1) * 60) - max(start, hour * 60)
                    pieces.append({STAY: row[STAY], "infusionoffset": hour * 60, column: row._mg_hr * factor * duration / 60})
                used_until = end
    out = pd.DataFrame(pieces, columns=[STAY, "infusionoffset", column])
    out.attrs["eicu_medication_audit"] = {"source": "charted_mass_rate_with_iv_order", "rate_rows": len(frame), "dose_pieces": len(out), "unknown_route_or_unit_not_dosed": True}
    return out


def insulin_units(frame, column):
    """Units/h, or mL/h with an explicit row-level units/mL concentration."""
    out = frame.copy()
    unit = frame.drugname.astype("string").str.extract(r"\(([^)]*)\)\s*$", expand=False).str.strip().str.lower()
    rate = pd.to_numeric(frame[column], errors="coerce")
    amount = pd.to_numeric(frame.drugamount, errors="coerce")
    volume = pd.to_numeric(frame.volumeoffluid, errors="coerce")
    concentration = (amount / volume).where(amount.gt(0) & volume.gt(0))
    value = rate.where(unit.isin(["units/hr", "units/hour", "u/hr", "u/hour"]))
    ml = unit.isin(["ml/hr", "ml/hour"])
    value.loc[ml] = rate.loc[ml] * concentration.loc[ml]
    out[column] = value.where(value.ge(0) & np.isfinite(value))
    out.attrs["eicu_insulin_unit_audit"] = {"unknown_dose": int(out[column].isna().sum()), "concentration_converted": int((ml & concentration.notna()).sum()), "assumed_concentration": False}
    return out
