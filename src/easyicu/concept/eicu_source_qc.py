"""Source-level eICU laboratory, anthropometric and diagnosis quality rules."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

STAY = "patientunitstayid"
logger = logging.getLogger(__name__)


def load_optional(data_source, table, stays, columns=None):
    """Read a stay-scoped optional channel; absent tables carry no evidence."""
    from ..datasource import FilterOp, FilterSpec

    if data_source is None or not len(stays):
        return pd.DataFrame(columns=columns)
    try:
        result = data_source.load_table(
            table, columns=columns,
            filters=[FilterSpec(STAY, FilterOp.IN, list(stays))], verbose=False,
        )
    except (FileNotFoundError, KeyError):
        return pd.DataFrame(columns=columns)
    return result.data if hasattr(result, "data") else result


def arterial_gases(frame, data_source):
    """Keep only confirmed arterial specimens at the exact stay/offset key.

    Unknown, unpaired and conflicting samples are excluded and counted. No
    forward/nearest-time specimen join or inference from the analyte name.
    """
    samples = load_optional(data_source, "customlab", frame[STAY].dropna().unique())
    keys = [STAY, "labresultoffset"]
    status = pd.DataFrame(columns=[*keys, "_nonarterial", "_arterial", "_unknown"])
    if not samples.empty:
        samples = samples.loc[samples.labothername.astype("string").str.strip().str.casefold().eq("sample type")].copy()
        text = samples.labothervaluetext.astype("string").fillna("").str.strip()
        numeric_text = samples.labotherresult.astype("string").fillna("").str.strip()
        text = text.where(text.ne(""), numeric_text)
        samples["_nonarterial"] = text.str.contains(r"\b(?:ven\w*|cap\w*|mixed|pa|pap|pulmonary\s+artery)\b", case=False, na=False)
        samples["_arterial"] = text.str.fullmatch(r"\s*(?:art(?:erial)?\.?)\s*", case=False, na=False)
        samples["_unknown"] = text.ne("") & ~samples["_arterial"] & ~samples["_nonarterial"]
        offset = samples.labotheroffset
        if pd.api.types.is_timedelta64_dtype(frame.labresultoffset):
            offset = offset if pd.api.types.is_timedelta64_dtype(offset) else pd.to_timedelta(pd.to_numeric(offset, errors="coerce"), unit="m")
        elif pd.api.types.is_datetime64_any_dtype(frame.labresultoffset):
            # An absolute clock cannot be paired to numeric admission offsets
            # without an admission-time anchor. Such specimens remain unknown.
            offset = pd.to_datetime(offset) if pd.api.types.is_datetime64_any_dtype(offset) else pd.Series(pd.NaT, index=samples.index)
        else:
            offset = pd.to_numeric(offset, errors="coerce")
        samples["labresultoffset"] = offset
        status = samples.groupby(keys, as_index=False)[["_nonarterial", "_arterial", "_unknown"]].any()
    if status.empty:
        out = frame.iloc[:0].copy()
        out["eicu_specimen_status"] = pd.Series(dtype="object")
        out.attrs["eicu_specimen_audit"] = {"nonarterial_dropped": 0, "unknown_kept": 0, "unknown_dropped": len(frame)}
        logger.info("eICU arterial specimen restriction audit=%s", out.attrs["eicu_specimen_audit"])
        return out
    keyed = frame[keys].reset_index(drop=True)
    joined = keyed.merge(status, on=keys, how="left", validate="many_to_one", sort=False)
    bad = joined["_nonarterial"].eq(True).fillna(False).to_numpy(dtype=bool)  # noqa: E712
    known = joined["_arterial"].eq(True).fillna(False).to_numpy(dtype=bool)  # noqa: E712
    unknown = joined["_unknown"].eq(True).fillna(False).to_numpy(dtype=bool)  # noqa: E712
    keep = known & ~bad & ~unknown
    out = frame.loc[keep].copy()
    out.attrs["eicu_specimen_audit"] = {"nonarterial_dropped": int(bad.sum()), "unknown_kept": 0, "unknown_dropped": int((~bad & ~keep).sum())}
    logger.info("eICU arterial specimen restriction audit=%s", out.attrs["eicu_specimen_audit"])
    out["eicu_specimen_status"] = "arterial"
    return out


def height_qc(frame, column):
    out = frame.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    age = pd.to_numeric(out["age"].astype("string").str.replace(">", "", regex=False).str.strip(), errors="coerce")
    bad = (age.ge(18) & ~values.between(100, 230)).fillna(False)
    out["admissionheight"] = values
    out["height_qc_reason"] = np.where(bad, "adult_height_outside_trusted_cm_domain", "")
    out[column] = values.mask(bad)
    out.attrs["eicu_height_audit"] = {"adult_height_dropped": int(bad.sum()), "unit_conversion": "none_without_explicit_unit"}
    return out


_INFECTION = re.compile(
    r"\b(?:sepsis|septic|infection|infectious|bacteremia|fungemia|pneumonia|"
    r"meningitis|encephalitis|endocarditis|peritonitis|abscess|empyema|cellulitis|"
    r"osteomyelitis|pyelonephritis|cholangitis|cholecystitis|appendicitis|"
    r"diverticulitis|bronchitis|epiglottitis|sinusitis|tuberculosis|fungal|candida)\b|necrotizing.fasciitis|"
    r"\b(?:bacterial|viral|infective)\s+(?:pericarditis|cystitis|colitis|enteritis)\b", re.I,
)
_NONINFECTION = re.compile(
    r"\bnon[- ]?infect\w*|\b(?:no|without|ruled out|rule out)\s+(?:evidence of\s+)?infect\w*|"
    r"\b(?:ischemic|ischaemic|ulcerative)\s+colitis\b|\b(?:crohn\w*|inflammatory bowel)\b", re.I,
)


def infection_evidence(frame, column):
    """Require a positive structured diagnostic phrase, applying negation per row."""
    text = frame[column].astype("string")
    positive = text.str.contains(_INFECTION, na=False)
    negative = text.str.contains(_NONINFECTION, na=False)
    # Negation applies to the named infection too, not just the word infection.
    # Keep this row-local so an independent positive diagnosis still contributes.
    negative |= text.str.contains(
        rf"\b(?:no|without|ruled out|rule out)\s+(?:evidence of\s+)?(?:{_INFECTION.pattern})|"
        rf"(?:{_INFECTION.pattern})\s+(?:was\s+)?(?:ruled out|excluded|absent)\b",
        flags=re.I, na=False,
    )
    out = frame.loc[positive & ~negative].copy()
    out[column] = True
    out.attrs["eicu_infection_audit"] = {"rows": len(frame), "noninfectious_dropped": int(negative.sum()), "positive_rows": len(out)}
    return out


def fio2_values(frame, column, table):
    """Percent/fraction normalization with explicit flow-unit rejection."""
    out = frame.copy()
    text = frame[column].astype("string").str.strip()
    flow = text.str.contains(r"(?:l\s*/\s*(?:min|m)|lpm)", case=False, na=False)
    if "respchartvaluelabel" in frame:
        flow |= frame.respchartvaluelabel.astype("string").str.contains(r"lpm|l\s*/\s*min", case=False, na=False)
    if "labmeasurenameinterface" in frame:
        flow |= frame.labmeasurenameinterface.astype("string").str.contains(r"lpm|l\s*/\s*min|mm\s*hg", case=False, na=False)
    value = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    value = value.where(~value.between(.21, 1), value * 100)
    out[column] = value.where(~flow & value.between(21, 100))
    out = out.loc[out[column].notna()].copy()
    out.attrs["eicu_fio2_source"] = "lab" if table == "lab" else "chart"
    return out


def fio2_parent(frame, column, table, data_source):
    """At identical raw times, prioritize lab, respiratory chart, then nursing."""
    out = fio2_values(frame, column, table)
    if out.empty or table == "lab":
        return out
    time = "respchartoffset" if table == "respiratorycharting" else "nursingchartoffset"
    higher = []
    labs = load_optional(data_source, "lab", out[STAY].unique())
    if not labs.empty:
        labs = labs.loc[labs.labname.eq("FiO2")]
        labs = fio2_values(labs, "labresult", "lab")
        higher.append(labs[[STAY, "labresultoffset"]].rename(columns={"labresultoffset": time}))
    if table == "nursecharting":
        resp = load_optional(data_source, "respiratorycharting", out[STAY].unique())
        if not resp.empty:
            resp = resp.loc[resp.respchartvaluelabel.astype("string").str.contains(r"fio2|O2 Percentage", case=False, na=False)]
            resp = fio2_values(resp, "respchartvalue", "respiratorycharting")
            higher.append(resp[[STAY, "respchartoffset"]].rename(columns={"respchartoffset": time}))
    if higher:
        occupied = pd.MultiIndex.from_frame(pd.concat(higher).drop_duplicates())
        out = out.loc[~pd.MultiIndex.from_frame(out[[STAY, time]]).isin(occupied)].copy()
    out.attrs["eicu_fio2_priority"] = "lab > respiratorycharting > nursecharting at exact offset"
    return out


def mode_values(frame, column, sequence=False):
    out = frame.copy()
    text = frame[column].astype("string").str.strip().str.casefold()
    value = pd.Series(pd.NA, index=frame.index, dtype="object")
    if sequence:
        value.loc[text.str.contains(r"\bsimv\b|^apvsimv$", na=False)] = "simv"
        value.loc[text.isin(["spontaneous", "spont", "spontaneous ventilation"])] = "spontaneous"
    else:
        value.loc[text.str.fullmatch(r"apv[- ]?(?:cmv|simv)?", na=False)] = "dual_adaptive"
    out[column] = value
    out.attrs["eicu_vent_mode_audit"] = {"unknown": int(value.isna().sum()), "controlled_breathing_inferred": False}
    return out


def rate_weights(frame, data_source, weight_var="patientweight"):
    """Charted infusion weight, admission kg, past charted kg, then past lb.

    Nonpositive weights are missing before fallback. A chart value never
    replaces an available higher-priority weight, and future charting is never
    borrowed. Explicit lb labels alone authorize conversion (R5-036/035).
    """
    weight = pd.to_numeric(frame.get(weight_var, pd.Series(np.nan, index=frame.index)), errors="coerce")
    weight = weight.where(weight.gt(0) & np.isfinite(weight))
    if data_source is None or STAY not in frame or not weight.isna().any():
        return weight.to_numpy(dtype=float)
    stays = frame[STAY].dropna().unique()
    patient = load_optional(data_source, "patient", stays, [STAY, "admissionweight"])
    if not patient.empty:
        patient = patient.copy()
        patient["_kg"] = pd.to_numeric(patient.admissionweight, errors="coerce")
        patient = patient.loc[patient._kg.gt(0) & np.isfinite(patient._kg)]
        consistent = patient.groupby(STAY)._kg.nunique().eq(1)
        lookup = patient.loc[patient[STAY].map(consistent)].drop_duplicates(STAY).set_index(STAY)._kg
        weight = weight.fillna(frame[STAY].map(lookup))
    if not weight.isna().any() or "infusionoffset" not in frame:
        return weight.to_numpy(dtype=float)
    chart = load_optional(data_source, "intakeoutput", stays)
    if chart.empty:
        return weight.to_numpy(dtype=float)
    chart = chart.loc[chart.celllabel.isin(["Bodyweight (kg)", "Bodyweight (lb)"])].copy()
    chart["_kg"] = pd.to_numeric(chart.cellvaluenumeric, errors="coerce") * chart.celllabel.map({"Bodyweight (kg)": 1.0, "Bodyweight (lb)": .45359237})
    chart["_t"] = pd.to_numeric(chart.intakeoutputoffset, errors="coerce").astype(float)
    chart = chart.loc[chart._kg.between(1, 500) & chart._t.notna()]
    grouped = chart.groupby([STAY, "_t"])._kg
    chart["_conflict"] = ~np.isclose(grouped.transform("min"), grouped.transform("max"), rtol=1e-6)
    targets = pd.DataFrame({STAY: frame[STAY].to_numpy(), "_t": pd.to_numeric(frame.infusionoffset, errors="coerce").to_numpy(dtype=float), "_row": np.arange(len(frame))})
    targets = targets.loc[weight.isna().to_numpy() & targets._t.notna()]
    if chart.empty or targets.empty:
        return weight.to_numpy(dtype=float)
    latest = pd.merge_asof(targets.sort_values("_t"), chart[[STAY, "_t", "_conflict"]].drop_duplicates([STAY, "_t"]).sort_values("_t"), by=STAY, on="_t", direction="backward")
    blocked = set(latest.loc[latest._conflict.eq(True), "_row"])  # noqa: E712
    for label, factor in [("Bodyweight (kg)", 1.0), ("Bodyweight (lb)", .45359237)]:
        candidates = chart.loc[chart.celllabel.eq(label)].copy()
        candidates["_kg"] = pd.to_numeric(candidates.cellvaluenumeric, errors="coerce") * factor
        candidates["_t"] = pd.to_numeric(candidates.intakeoutputoffset, errors="coerce").astype(float)
        candidates = candidates.loc[candidates._kg.between(1, 500) & candidates._t.notna()]
        # Same-time conflicting weights are unknown, never arbitrarily 'last'.
        conflicts = candidates.groupby([STAY, "_t"])._kg.transform("nunique").gt(1)
        candidates = candidates.loc[~conflicts].drop_duplicates([STAY, "_t"])
        target = pd.DataFrame({STAY: frame[STAY].to_numpy(), "_t": pd.to_numeric(frame.infusionoffset, errors="coerce").to_numpy(dtype=float), "_row": np.arange(len(frame))})
        target = target.loc[weight.isna().to_numpy() & target._t.notna() & ~target._row.isin(blocked)]
        if target.empty or candidates.empty:
            continue
        joined = pd.merge_asof(target.sort_values("_t"), candidates[[STAY, "_t", "_kg"]].sort_values("_t"), by=STAY, on="_t", direction="backward")
        weight.iloc[joined._row.to_numpy()] = joined._kg.to_numpy()
    return weight.to_numpy(dtype=float)
