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


#: eICU lab names whose analyte is itself arterial (R6 rule 2).
LABEL_DEFINED_ARTERIAL = frozenset({"pao2", "paco2"})
_NONARTERIAL_SAMPLE = r"\b(?:ven\w*|cap\w*|mixed|pa|pap|pulmonary\s+artery|central(?:\s+line)?)\b"
_ARTERIAL_SAMPLE = r"\s*(?:art(?:erial)?\.?|aterial|[lr]\.?\s*radial|radial(?:,?\s*(?:left|right))?)\s*"


def _offset_like(values, like):
    """Read a source offset on the clock of the gas frame's ``labresultoffset``."""
    if pd.api.types.is_timedelta64_dtype(like):
        return values if pd.api.types.is_timedelta64_dtype(values) else pd.to_timedelta(pd.to_numeric(values, errors="coerce"), unit="m")
    if pd.api.types.is_datetime64_any_dtype(like):
        # An absolute clock cannot be paired to numeric admission offsets
        # without an admission-time anchor; such evidence stays unpaired.
        return pd.to_datetime(values) if pd.api.types.is_datetime64_any_dtype(values) else pd.Series(pd.NaT, index=values.index)
    return pd.to_numeric(values, errors="coerce")


def _sample_types(data_source, stays, like):
    """Per stay/offset: recorded non-arterial, arterial and unknown Sample Type flags.

    ``_unknown`` marks a Sample Type record that names neither specimen
    (Unknown, Unspecified, blank): a sample was recorded, so the label and
    same-panel fallbacks do not apply to it (round-8 R7-F018).
    """
    keys = [STAY, "labresultoffset"]
    flags = ["_nonarterial", "_arterial", "_unknown"]
    samples = load_optional(data_source, "customlab", stays)
    if samples.empty:
        return pd.DataFrame(columns=[*keys, *flags])
    samples = samples.loc[samples.labothername.astype("string").str.strip().str.casefold().eq("sample type")].copy()
    text = samples.labothervaluetext.astype("string").fillna("").str.strip()
    text = text.where(text.ne(""), samples.labotherresult.astype("string").fillna("").str.strip())
    samples["_nonarterial"] = text.str.contains(_NONARTERIAL_SAMPLE, case=False, na=False)
    samples["_arterial"] = text.str.fullmatch(_ARTERIAL_SAMPLE, case=False, na=False) & ~samples["_nonarterial"]
    samples["_unknown"] = ~samples["_nonarterial"] & ~samples["_arterial"]
    samples["labresultoffset"] = _offset_like(samples.labotheroffset, like)
    return samples.groupby(keys, as_index=False)[flags].any()


def _arterial_anchor_keys(data_source, stays, like):
    """Stay/offset keys holding an admissible paO2 or paCO2 result."""
    from ..datasource import FilterOp, FilterSpec
    from .loader import _get_concept_bounds

    keys = [STAY, "labresultoffset"]
    if data_source is None or not len(stays):
        return pd.DataFrame(columns=keys)
    try:
        result = data_source.load_table(
            "lab", columns=[*keys, "labname", "labresult"],
            filters=[FilterSpec(STAY, FilterOp.IN, list(stays)),
                     FilterSpec("labname", FilterOp.IN, ["paO2", "paCO2"])],
            verbose=False,
        )
    except (FileNotFoundError, KeyError):
        return pd.DataFrame(columns=keys)
    lab = result.data if hasattr(result, "data") else result
    names = lab.labname.astype("string").str.strip().str.casefold()
    values = pd.to_numeric(lab.labresult, errors="coerce")
    admitted = pd.Series(False, index=lab.index)
    # A pressure rejected by its own concept cannot establish the specimen
    # of another analyte. Apply the same canonical bounds before pairing.
    for label, concept in (("pao2", "po2"), ("paco2", "pco2")):
        valid = names.eq(label) & np.isfinite(values)
        minimum, maximum = (_get_concept_bounds(concept, bound) for bound in ("min", "max"))
        if minimum is not None:
            valid &= values.ge(minimum)
        if maximum is not None:
            valid &= values.le(maximum)
        admitted |= valid.fillna(False)
    lab = lab.loc[admitted, keys].copy()
    lab["labresultoffset"] = _offset_like(lab.labresultoffset, like)
    return lab.dropna().drop_duplicates()


def arterial_gases(frame, data_source, analytes=()):
    """Admit eICU gases to arterial concepts by the R6 specimen order.

    1. A recorded Sample Type decides first: venous, capillary, mixed or
       central samples drop every gas at that stay/offset; an arterial sample
       admits them (``explicit_arterial``). A recorded sample that names
       neither (Unknown, Unspecified, blank) is ``excluded_recorded_unknown``;
       the fallbacks below apply only where no Sample Type was recorded.
    2. eICU has no per-sample specimen field otherwise. paO2 and paCO2 are
       arterial by definition (``label_defined_arterial``).
    3. pH, base excess/deficit and O2 Sat (%) do not name a specimen; they enter
       only beside a paO2/paCO2 result at the same stay and offset
       (``co_measured_arterial``) and are otherwise ``excluded_unknown``.
    No forward/nearest-time pairing is used.
    """
    keys = [STAY, "labresultoffset"]
    label_defined = bool(analytes) and all(
        str(a).strip().casefold() in LABEL_DEFINED_ARTERIAL for a in analytes
    )
    stays = frame[STAY].dropna().unique()
    work = frame[keys].reset_index(drop=True)
    def flag(evidence, column):
        if evidence.empty:
            return np.zeros(len(work), dtype=bool)
        evidence = evidence.astype({key: work[key].dtype for key in keys})
        joined = work.merge(evidence[[*keys, column]], on=keys, how="left", validate="many_to_one", sort=False)
        return joined[column].eq(True).fillna(False).to_numpy(dtype=bool)  # noqa: E712

    samples = _sample_types(data_source, stays, frame.labresultoffset)
    venous = flag(samples, "_nonarterial")
    explicit = flag(samples, "_arterial") & ~venous
    unknown = flag(samples, "_unknown") & ~venous & ~explicit
    if label_defined:
        anchored = np.ones(len(work), dtype=bool)
    else:
        anchors = _arterial_anchor_keys(data_source, stays, frame.labresultoffset)
        anchored = flag(anchors.assign(_anchor=True), "_anchor")
    status = np.select(
        [venous, explicit, unknown, anchored],
        ["excluded_venous", "explicit_arterial", "excluded_recorded_unknown",
         "label_defined_arterial" if label_defined else "co_measured_arterial"],
        default="excluded_unknown",
    )
    keep = ~pd.Series(status).str.startswith("excluded").to_numpy()
    out = frame.loc[keep].copy()
    out["specimen_status"] = status[keep]
    counts = pd.Series(status).value_counts()
    out.attrs["eicu_specimen_audit"] = {
        name: int(counts.get(name, 0))
        for name in ("explicit_arterial", "label_defined_arterial", "co_measured_arterial",
                     "excluded_venous", "excluded_recorded_unknown", "excluded_unknown")
    }
    logger.info("eICU arterial specimen restriction audit=%s", out.attrs["eicu_specimen_audit"])
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
